"""Pure tensor/config tests for conservative online refinement."""

from __future__ import annotations

import torch

from src.tasks.stairs_cbf.config import (
  TARGET_RISE_PROFILE,
  TARGET_TREAD_PROFILE,
  g1_online_stairs_env_cfg,
  g1_online_stairs_runner_cfg,
)
from src.tasks.stairs_cbf.online import (
  CandidateGateThresholds,
  backtrack_actor_state,
  backward_intervention_credit,
  candidate_gate,
  candidate_precheck,
)
from src.tasks.stairs_cbf import mdp
from src.tasks.stairs_cbf.terrain import ForwardStairsTerrainCfg


def test_fixed_target_geometry_profile() -> None:
  cfg = ForwardStairsTerrainCfg(
    num_steps=18,
    step_height_profile=TARGET_RISE_PROFILE,
    step_width_profile=TARGET_TREAD_PROFILE,
  )
  rises, treads = cfg._geometry_profile(difficulty=0.91)
  assert rises.shape == (18,)
  assert treads.shape == (18,)
  assert abs(float(rises.mean()) - 0.145) < 0.002
  assert min(treads) >= 0.32
  assert max(treads) <= 0.34


def test_credit_does_not_cross_episode_boundary() -> None:
  magnitude = torch.zeros(5, 1)
  magnitude[4, 0] = 0.05
  intervened = torch.zeros(5, 1, dtype=torch.bool)
  intervened[4, 0] = True
  dones = torch.zeros(5, 1, dtype=torch.bool)
  dones[2, 0] = True
  credit = backward_intervention_credit(
    magnitude,
    intervened,
    dones,
    horizon=5,
    decay=0.8,
    magnitude_scale=0.05,
  )
  assert torch.allclose(credit[:, 0], torch.tensor([0.0, 0.0, 0.0, 0.8, 1.0]))


def test_precheck_rejects_large_policy_step() -> None:
  reasons = candidate_precheck(
    update_metrics={
      "mean_kl": 0.01,
      "clip_fraction": 0.5,
      "action_saturation_fraction": 0.0,
    },
    total_kl_from_base=0.01,
    parameters_finite=True,
  )
  assert "update KL exceeds target" in reasons
  assert "clip fraction exceeds limit" in reasons


def test_candidate_gate_accepts_non_regressing_paired_result() -> None:
  old = {
    "D0": {"success_rate": 0.95, "fall_rate": 0.01, "intervention_per_riser": 0.1},
    "D4": {"success_rate": 0.70, "fall_rate": 0.02, "intervention_per_riser": 0.5},
    "D5": {"success_rate": 0.60, "fall_rate": 0.03, "intervention_per_riser": 0.6},
  }
  candidate = {
    "D0": {"success_rate": 0.94, "fall_rate": 0.01, "intervention_per_riser": 0.1},
    "D4": {"success_rate": 0.75, "fall_rate": 0.01, "intervention_per_riser": 0.48},
    "D5": {"success_rate": 0.62, "fall_rate": 0.02, "intervention_per_riser": 0.58},
  }
  accepted, reasons = candidate_gate(
    update_metrics={
      "mean_kl": 0.001,
      "clip_fraction": 0.1,
      "action_saturation_fraction": 0.01,
    },
    old_eval=old,
    candidate_eval=candidate,
    base_d0_success=0.95,
    total_kl_from_base=0.01,
    parameters_finite=True,
    thresholds=CandidateGateThresholds(),
  )
  assert accepted
  assert reasons == []


def test_candidate_gate_rejects_unchanged_candidate() -> None:
  evaluation = {
    "D0": {"success_rate": 1.0, "fall_rate": 0.0, "intervention_per_riser": 0.1},
    "D4": {"success_rate": 0.5, "fall_rate": 0.5, "intervention_per_riser": 1.0},
    "D5": {"success_rate": 0.5, "fall_rate": 0.5, "intervention_per_riser": 1.0},
  }
  accepted, reasons = candidate_gate(
    update_metrics={
      "mean_kl": 0.001,
      "clip_fraction": 0.0,
      "action_saturation_fraction": 0.1,
    },
    old_eval=evaluation,
    candidate_eval=evaluation,
    base_d0_success=1.0,
    total_kl_from_base=0.001,
    parameters_finite=True,
  )
  assert not accepted
  assert "target metrics show no strict improvement" in reasons


def test_candidate_gate_rejects_neighbor_regression() -> None:
  old = {
    "D0": {"success_rate": 1.0, "fall_rate": 0.0, "intervention_per_riser": 0.1},
    "D4": {"success_rate": 0.5, "fall_rate": 0.5, "intervention_per_riser": 1.0},
    "D5": {"success_rate": 0.6, "fall_rate": 0.4, "intervention_per_riser": 1.0},
  }
  candidate = {
    "D0": old["D0"],
    "D4": {"success_rate": 0.6, "fall_rate": 0.4, "intervention_per_riser": 0.9},
    "D5": {"success_rate": 0.5, "fall_rate": 0.5, "intervention_per_riser": 1.0},
  }
  accepted, reasons = candidate_gate(
    update_metrics={
      "mean_kl": 0.001,
      "clip_fraction": 0.0,
      "action_saturation_fraction": 0.1,
    },
    old_eval=old,
    candidate_eval=candidate,
    base_d0_success=1.0,
    total_kl_from_base=0.001,
    parameters_finite=True,
  )
  assert not accepted
  assert "D5 success regressed" in reasons


def test_online_domain_and_ppo_config_are_conservative() -> None:
  d4 = g1_online_stairs_env_cfg("D4")
  terrain = d4.scene.terrain
  assert terrain is not None and terrain.terrain_generator is not None
  stairs = terrain.terrain_generator.sub_terrains["forward_stairs"]
  assert stairs.num_steps == 18
  assert stairs.step_height_profile == TARGET_RISE_PROFILE
  assert d4.observations["actor"].history_length == 5
  assert "online_privileged" in d4.observations

  runner = g1_online_stairs_runner_cfg()
  assert runner.algorithm.clip_param == 0.05
  assert runner.algorithm.num_learning_epochs == 2
  assert runner.algorithm.desired_kl == 0.003
  assert runner.obs_groups["critic"] == (
    "actor",
    "critic",
    "online_privileged",
  )
  quick = g1_online_stairs_env_cfg("DQ")
  quick_stairs = quick.scene.terrain.terrain_generator.sub_terrains["forward_stairs"]
  assert quick_stairs.num_steps == 9
  assert d4.rewards["is_terminated"].weight == 0.0
  assert d4.rewards["fall_termination"].weight == -200.0


def test_fall_reward_does_not_penalize_successful_top_termination() -> None:
  class _TerminationManager:
    def __init__(self, fell: torch.Tensor):
      self.fell = fell

    def get_term(self, name: str) -> torch.Tensor:
      assert name == "fell_over"
      return self.fell

  class _Env:
    def __init__(self):
      # env 0 fell; env 1 represents a reached-top-only termination.
      self.termination_manager = _TerminationManager(torch.tensor([True, False]))
      self.extras = {"log": {}}

  result = mdp.fall_termination(_Env())
  assert torch.equal(result, torch.tensor([1.0, 0.0]))


def test_actor_backtracking_only_interpolates_trainable_mlp() -> None:
  base = {
    "mlp.0.weight": torch.tensor([0.0, 2.0]),
    "obs_normalizer._mean": torch.tensor([10.0]),
    "distribution.std_param": torch.tensor([0.6]),
  }
  candidate = {
    "mlp.0.weight": torch.tensor([4.0, 6.0]),
    "obs_normalizer._mean": torch.tensor([11.0]),
    "distribution.std_param": torch.tensor([0.2]),
  }
  midpoint = backtrack_actor_state(base, candidate, 0.5)
  assert torch.equal(midpoint["mlp.0.weight"], torch.tensor([2.0, 4.0]))
  assert torch.equal(midpoint["obs_normalizer._mean"], candidate["obs_normalizer._mean"])
  assert torch.equal(midpoint["distribution.std_param"], candidate["distribution.std_param"])
  reference = backtrack_actor_state(base, candidate, 0.0)
  assert torch.equal(reference["mlp.0.weight"], base["mlp.0.weight"])
  assert torch.equal(reference["distribution.std_param"], candidate["distribution.std_param"])
