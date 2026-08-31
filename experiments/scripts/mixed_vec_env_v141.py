"""Two-context vector environment used by v141 specialist training."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict


class SpecialistMixedVecEnvV141:
    """Concatenate a target context and an F1-retention context."""

    def __init__(
        self,
        target_wrapper: Any,
        retention_wrapper: Any,
        *,
        target_context: str,
        dual_reward_scale: float,
    ) -> None:
        self.target_wrapper = target_wrapper
        self.retention_wrapper = retention_wrapper
        self.wrappers = (target_wrapper, retention_wrapper)
        self.target_context = str(target_context)
        self.dual_reward_scale = float(dual_reward_scale)
        self.device = target_wrapper.device
        self.num_actions = int(target_wrapper.num_actions)
        self.max_episode_length = int(target_wrapper.max_episode_length)
        self.cfg = target_wrapper.cfg
        if (
            retention_wrapper.device != self.device
            or int(retention_wrapper.num_actions) != self.num_actions
            or int(retention_wrapper.max_episode_length)
            != self.max_episode_length
        ):
            raise ValueError("v141 context wrappers have incompatible interfaces")
        self.target_count = int(target_wrapper.num_envs)
        self.retention_count = int(retention_wrapper.num_envs)
        self.num_envs = self.target_count + self.retention_count
        if self.target_count <= 0 or self.retention_count <= 0:
            raise ValueError("v141 context groups must both be non-empty")

    @property
    def target_environment_mask(self) -> torch.Tensor:
        return torch.arange(self.num_envs, device=self.device) < self.target_count

    @property
    def context_labels(self) -> torch.Tensor:
        return self.target_environment_mask.long()

    @property
    def step_dt(self) -> float:
        values = [float(wrapper.unwrapped.step_dt) for wrapper in self.wrappers]
        if values[0] != values[1]:
            raise RuntimeError("v141 context step durations differ")
        return values[0]

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return torch.cat(
            [wrapper.episode_length_buf for wrapper in self.wrappers], dim=0
        )

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        if value.shape != (self.num_envs,):
            raise ValueError("v141 episode length buffer has an invalid shape")
        self.target_wrapper.episode_length_buf[:] = value[: self.target_count]
        self.retention_wrapper.episode_length_buf[:] = value[self.target_count :]

    @staticmethod
    def _concat_tensordicts(values: list[TensorDict]) -> TensorDict:
        return torch.cat(values, dim=0)

    def get_observations(self) -> TensorDict:
        return self._concat_tensordicts(
            [wrapper.get_observations() for wrapper in self.wrappers]
        )

    def reset(self) -> tuple[TensorDict, dict[str, Any]]:
        observations = []
        for wrapper in self.wrappers:
            obs, _ = wrapper.reset()
            observations.append(obs)
        return self._concat_tensordicts(observations), {
            "v141_target_environment": self.target_environment_mask,
            "v141_context_index": self.context_labels,
        }

    def _merge_extras(self, extras: list[dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        common = set(extras[0]).intersection(extras[1])
        counts = (self.target_count, self.retention_count)
        for key in sorted(common):
            values = [extras[index][key] for index in range(2)]
            if not all(isinstance(value, torch.Tensor) for value in values):
                continue
            if not all(
                value.ndim >= 1 and value.shape[0] == count
                for value, count in zip(values, counts, strict=True)
            ):
                continue
            merged[key] = torch.cat(values, dim=0)
        merged["v141_target_environment"] = self.target_environment_mask
        merged["v141_context_index"] = self.context_labels
        return merged

    def get_termination_term(self, name: str) -> torch.Tensor:
        return torch.cat(
            [
                wrapper.unwrapped.termination_manager.get_term(name)
                for wrapper in self.wrappers
            ],
            dim=0,
        )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise ValueError(f"v141 action shape differs: {tuple(actions.shape)}")
        observations: list[TensorDict] = []
        rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        extras: list[dict[str, Any]] = []
        offset = 0
        for wrapper, count in zip(
            self.wrappers,
            (self.target_count, self.retention_count),
            strict=True,
        ):
            obs, reward, done, extra = wrapper.step(
                actions[offset : offset + count]
            )
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            extras.append(dict(extra))
            offset += count
        reward = torch.cat(rewards)
        done = torch.cat(dones)
        merged = self._merge_extras(extras)
        success = self.get_termination_term("reached_top").bool() & done.bool()
        raw_dual = merged.get("cbf_reward_dual_component")
        scaled_dual = (
            torch.zeros_like(reward)
            if raw_dual is None
            else raw_dual.to(reward) * self.dual_reward_scale * self.step_dt
        )
        merged["v141_success_terminal"] = success
        merged["v141_cbf_reward"] = scaled_dual
        merged["v141_task_reward"] = reward - scaled_dual
        return self._concat_tensordicts(observations), reward, done, merged

    def close(self) -> None:
        for wrapper in self.wrappers:
            wrapper.close()
