"""A fixed-episode, balanced F1/F2/F3 vector environment for v32."""

from __future__ import annotations

from typing import Any

import torch
from cbf_teacher_v32_protocol import (
    MIXED_CONTEXT_CAPACITY,
    MIXED_CONTEXTS,
    MIXED_EXPOSED_ENVS,
    mixed_context_env_counts,
)
from tensordict import TensorDict


class MixedContextVecEnvV32:
    """Expose 64 active environments from three fixed 22-env context groups.

    All underlying environments are reset at a round boundary.  During a round,
    an active environment keeps its terrain for every episode.  The context that
    contributes 22 rather than 21 environments rotates every round.
    """

    def __init__(self, wrappers: dict[str, Any], *, absolute_round: int) -> None:
        if tuple(wrappers) != MIXED_CONTEXTS:
            raise ValueError("v32 mixed wrappers must be ordered F1, F2, F3")
        self.wrappers = wrappers
        first = wrappers[MIXED_CONTEXTS[0]]
        self.device = first.device
        self.num_actions = int(first.num_actions)
        self.max_episode_length = int(first.max_episode_length)
        self.cfg = first.cfg
        for context, wrapper in wrappers.items():
            if int(wrapper.num_envs) != MIXED_CONTEXT_CAPACITY:
                raise ValueError(f"v32 {context} wrapper must contain 22 envs")
            if (
                wrapper.device != self.device
                or int(wrapper.num_actions) != self.num_actions
            ):
                raise ValueError("v32 mixed wrappers differ in device or action count")
            if int(wrapper.max_episode_length) != self.max_episode_length:
                raise ValueError("v32 mixed wrappers differ in episode length")
        self.num_envs = MIXED_EXPOSED_ENVS
        self.absolute_round = 0
        self.active_ids: dict[str, torch.Tensor] = {}
        self.context_counts: dict[str, int] = {}
        self.omitted_environment_ids: dict[str, int | None] = {}
        self.set_round(absolute_round)

    def set_round(self, absolute_round: int) -> None:
        counts = mixed_context_env_counts(absolute_round)
        active: dict[str, torch.Tensor] = {}
        omitted: dict[str, int | None] = {}
        for context_index, context in enumerate(MIXED_CONTEXTS):
            count = counts[context]
            if count == MIXED_CONTEXT_CAPACITY:
                ids = torch.arange(MIXED_CONTEXT_CAPACITY, device=self.device)
                omitted[context] = None
            elif count == MIXED_CONTEXT_CAPACITY - 1:
                omitted_id = (
                    absolute_round + context_index - 1
                ) % MIXED_CONTEXT_CAPACITY
                ids = torch.tensor(
                    [i for i in range(MIXED_CONTEXT_CAPACITY) if i != omitted_id],
                    dtype=torch.long,
                    device=self.device,
                )
                omitted[context] = omitted_id
            else:
                raise RuntimeError("v32 mixed context count must be 21 or 22")
            active[context] = ids
        if sum(len(ids) for ids in active.values()) != MIXED_EXPOSED_ENVS:
            raise RuntimeError("v32 mixed active environment count is not 64")
        self.absolute_round = absolute_round
        self.active_ids = active
        self.context_counts = counts
        self.omitted_environment_ids = omitted

    def assignment_metadata(self) -> dict[str, Any]:
        return {
            "absolute_round": self.absolute_round,
            "context_counts": dict(self.context_counts),
            "omitted_environment_ids": dict(self.omitted_environment_ids),
            "episode_context_fixed": True,
        }

    def _select_tensordicts(self, values: dict[str, TensorDict]) -> TensorDict:
        selected = [values[c][self.active_ids[c]] for c in MIXED_CONTEXTS]
        return torch.cat(selected, dim=0)

    @property
    def context_labels(self) -> torch.Tensor:
        return torch.cat(
            [
                torch.full(
                    (len(self.active_ids[context]),),
                    index,
                    dtype=torch.long,
                    device=self.device,
                )
                for index, context in enumerate(MIXED_CONTEXTS)
            ]
        )

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return torch.cat(
            [
                self.wrappers[c].episode_length_buf[self.active_ids[c]]
                for c in MIXED_CONTEXTS
            ]
        )

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        offset = 0
        for context in MIXED_CONTEXTS:
            ids = self.active_ids[context]
            self.wrappers[context].episode_length_buf[ids] = value[
                offset : offset + len(ids)
            ]
            offset += len(ids)

    def get_observations(self) -> TensorDict:
        return self._select_tensordicts(
            {
                context: wrapper.get_observations()
                for context, wrapper in self.wrappers.items()
            }
        )

    def reset(self) -> tuple[TensorDict, dict[str, Any]]:
        observations: dict[str, TensorDict] = {}
        for context, wrapper in self.wrappers.items():
            observations[context], _ = wrapper.reset()
        return self._select_tensordicts(observations), {
            "v32_context_index": self.context_labels,
            "v32_mixed_assignment": self.assignment_metadata(),
        }

    def _merge_extras(self, extras: dict[str, dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        common = set.intersection(*(set(item) for item in extras.values()))
        for key in sorted(common):
            values = [extras[c][key] for c in MIXED_CONTEXTS]
            if not all(isinstance(value, torch.Tensor) for value in values):
                continue
            if not all(
                value.ndim >= 1 and value.shape[0] == MIXED_CONTEXT_CAPACITY
                for value in values
            ):
                continue
            merged[key] = torch.cat(
                [
                    value[self.active_ids[c]]
                    for c, value in zip(MIXED_CONTEXTS, values, strict=True)
                ],
                dim=0,
            )
        merged["v32_context_index"] = self.context_labels
        merged["v32_mixed_assignment"] = self.assignment_metadata()
        return merged

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise ValueError(f"v32 mixed action shape differs: {tuple(actions.shape)}")
        observations: dict[str, TensorDict] = {}
        rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        extras: dict[str, dict[str, Any]] = {}
        offset = 0
        for context in MIXED_CONTEXTS:
            wrapper = self.wrappers[context]
            ids = self.active_ids[context]
            count = len(ids)
            full_actions = torch.zeros(
                MIXED_CONTEXT_CAPACITY,
                self.num_actions,
                dtype=actions.dtype,
                device=self.device,
            )
            full_actions[ids] = actions[offset : offset + count]
            obs, reward, done, extra = wrapper.step(full_actions)
            observations[context] = obs
            rewards.append(reward[ids])
            dones.append(done[ids])
            extras[context] = dict(extra)
            offset += count
        return (
            self._select_tensordicts(observations),
            torch.cat(rewards),
            torch.cat(dones),
            self._merge_extras(extras),
        )

    def get_termination_term(self, name: str) -> torch.Tensor:
        return torch.cat(
            [
                self.wrappers[c].unwrapped.termination_manager.get_term(name)[
                    self.active_ids[c]
                ]
                for c in MIXED_CONTEXTS
            ]
        )

    def close(self) -> None:
        for wrapper in self.wrappers.values():
            wrapper.close()
