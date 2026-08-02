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
  CbfIndependenceThresholds,
  OnlineSafePPO,
  OnlineSafeRefinementRunner,
  SafeImprovementScoreWeights,
  backtrack_actor_state,
  backward_intervention_credit,
  candidate_gate,
  candidate_gate_intervals,
  candidate_precheck,
  cbf_independence_gate,
  cbf_corrected_mean_target,
  critic_calibration_by_riser,
  pre_event_value_delta,
  rollout_action_dataflow_metrics,
  paired_metric_delta_interval,
  safety_demand_per_riser,
  safe_improvement_score,
  adaptive_cbf_std_factor,
  validate_behavior_log_prob,
  validate_behavior_distribution_params,
)
from src.tasks.stairs_cbf.hard_cases import (
  HardCaseStateBank,
  curriculum_destination_ids,
  hard_case_destination_ids,
  perturb_joystick_command_state,
)
from src.tasks.stairs_cbf import mdp
from src.tasks.stairs_cbf.terrain import ForwardStairsTerrainCfg
from src.tasks.stairs_cbf.teleop_math import (
  centerline_feedback_command,
  signed_deadband,
)


def _with_paired_signatures(results: dict[str, dict]) -> dict[str, dict]:
  for domain, result in results.items():
    result["initial_state_signatures"] = [f"{domain}-seed42"]
  return results


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
      "actor_gradient_norm_pre_clip_max": 101.0,
    },
    total_kl_from_base=0.01,
    parameters_finite=True,
  )
  assert "update KL exceeds target" in reasons
  assert "clip fraction exceeds limit" in reasons
  assert "actor gradient norm exceeds limit" in reasons


def test_candidate_gate_accepts_non_regressing_paired_result() -> None:
  old = _with_paired_signatures({
    "D0": {"success_rate": 0.95, "fall_rate": 0.01, "intervention_per_riser": 0.1},
    "D4": {"success_rate": 0.70, "fall_rate": 0.02, "intervention_per_riser": 0.5},
    "D5": {"success_rate": 0.60, "fall_rate": 0.03, "intervention_per_riser": 0.6},
  })
  candidate = _with_paired_signatures({
    "D0": {"success_rate": 0.94, "fall_rate": 0.01, "intervention_per_riser": 0.1},
    "D4": {"success_rate": 0.75, "fall_rate": 0.0, "intervention_per_riser": 0.48},
    "D5": {"success_rate": 0.62, "fall_rate": 0.02, "intervention_per_riser": 0.58},
  })
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


def test_candidate_gate_requires_target_zero_fall_by_default() -> None:
  old = _with_paired_signatures({
    "D0": {"success_rate": 1.0, "fall_rate": 0.0, "intervention_per_riser": 0.1},
    "D4": {"success_rate": 0.7, "fall_rate": 0.1, "intervention_per_riser": 0.6},
    "D5": {"success_rate": 0.7, "fall_rate": 0.1, "intervention_per_riser": 0.6},
  })
  candidate = _with_paired_signatures({
    "D0": old["D0"] | {},
    "D4": {"success_rate": 0.8, "fall_rate": 0.05, "intervention_per_riser": 0.5},
    "D5": {"success_rate": 0.72, "fall_rate": 0.08, "intervention_per_riser": 0.58},
  })
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
  assert "D4 candidate fall rate exceeds safety limit" in reasons


def test_safe_improvement_score_rewards_task_and_penalizes_cbf_drift() -> None:
  weights = SafeImprovementScoreWeights()
  old = {
    "success_rate": 0.6,
    "mean_return": 8.0,
    "fall_rate": 0.2,
    "intervention_per_riser": 1.0,
    "runtime_filter": True,
  }
  candidate = {
    "success_rate": 0.7,
    "mean_return": 9.0,
    "fall_rate": 0.1,
    "intervention_per_riser": 0.7,
    "runtime_filter": True,
  }
  old_score = safe_improvement_score(
    old, total_kl_from_base=0.0, weights=weights
  )
  candidate_score = safe_improvement_score(
    candidate, total_kl_from_base=0.01, weights=weights
  )
  assert candidate_score["total"] > old_score["total"]
  assert candidate_score["fall"] < 0.0
  assert candidate_score["intervention_per_riser"] < 0.0
  assert candidate_score["policy_drift"] == -0.01


def test_critic_calibration_and_pre_event_value_diagnostics() -> None:
  values = torch.tensor(
    [[1.0, 2.0], [0.9, 2.1], [0.7, 2.2], [0.4, 2.3], [0.1, 2.4]]
  )
  returns = values + torch.tensor(
    [[0.1, -0.1], [0.1, -0.1], [0.2, -0.2], [0.2, -0.2], [0.3, -0.3]]
  )
  stair_indices = torch.tensor(
    [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2]]
  )
  calibration = critic_calibration_by_riser(values, returns, stair_indices)
  assert set(calibration) == {"0", "1", "2"}
  assert calibration["0"]["count"] == 3
  assert calibration["2"]["rmse"] > 0.0

  events = torch.zeros(5, 2, dtype=torch.bool)
  events[4, 0] = True
  dones = torch.zeros_like(events)
  count, delta = pre_event_value_delta(values, events, dones, horizon=3)
  assert count == 1
  assert abs(delta - (0.1 - 0.9)) < 1.0e-6
  dones[2, 0] = True
  count, delta = pre_event_value_delta(values, events, dones, horizon=3)
  assert count == 0 and delta is None


def test_candidate_gate_rejects_unchanged_candidate() -> None:
  evaluation = _with_paired_signatures({
    "D0": {"success_rate": 1.0, "fall_rate": 0.0, "intervention_per_riser": 0.1},
    "D4": {"success_rate": 0.5, "fall_rate": 0.5, "intervention_per_riser": 1.0},
    "D5": {"success_rate": 0.5, "fall_rate": 0.5, "intervention_per_riser": 1.0},
  })
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
  old = _with_paired_signatures({
    "D0": {"success_rate": 1.0, "fall_rate": 0.0, "intervention_per_riser": 0.1},
    "D4": {"success_rate": 0.5, "fall_rate": 0.5, "intervention_per_riser": 1.0},
    "D5": {"success_rate": 0.6, "fall_rate": 0.4, "intervention_per_riser": 1.0},
  })
  candidate = _with_paired_signatures({
    "D0": old["D0"],
    "D4": {"success_rate": 0.6, "fall_rate": 0.4, "intervention_per_riser": 0.9},
    "D5": {"success_rate": 0.5, "fall_rate": 0.5, "intervention_per_riser": 1.0},
  })
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


def test_candidate_gate_rejects_unpaired_initial_state() -> None:
  old = _with_paired_signatures({
    "D0": {"success_rate": 1.0, "fall_rate": 0.0, "intervention_per_riser": 0.1},
    "D4": {"success_rate": 0.5, "fall_rate": 0.5, "intervention_per_riser": 1.0},
    "D5": {"success_rate": 0.5, "fall_rate": 0.5, "intervention_per_riser": 1.0},
  })
  candidate = _with_paired_signatures({
    "D0": old["D0"] | {},
    "D4": {"success_rate": 0.6, "fall_rate": 0.4, "intervention_per_riser": 0.9},
    "D5": old["D5"] | {},
  })
  candidate["D4"]["initial_state_signatures"] = ["wrong-state"]
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
  assert "D4 paired initial-state signature differs" in reasons


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
  assert runner.algorithm.clip_param == 0.03
  assert runner.algorithm.num_learning_epochs == 1
  assert runner.algorithm.desired_kl == 0.003
  assert runner.algorithm.base_anchor_weight == 0.01
  assert runner.algorithm.intervention_advantage_weight == 0.075
  assert runner.algorithm.use_counterfactual_cbf_credit is False
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

  mixed = g1_online_stairs_env_cfg("DQM")
  mixed_terrain = mixed.scene.terrain.terrain_generator
  assert mixed_terrain is not None
  assert mixed_terrain.num_cols == 2
  assert set(mixed_terrain.sub_terrains) == {"dq_stairs", "dqn_stairs"}
  assert sum(
    terrain.proportion for terrain in mixed_terrain.sub_terrains.values()
  ) == 1.0

  human = g1_online_stairs_env_cfg("DQNH")
  assert human.commands["twist"].closed_loop_centering is True
  assert human.actions["joint_pos"].num_steps == 9
  human_stairs = human.scene.terrain.terrain_generator.sub_terrains[
    "forward_stairs"
  ]
  assert human_stairs.step_height_range == (0.132, 0.132)
  assert human_stairs.step_width == 0.355

  formal_human = g1_online_stairs_env_cfg("D4H")
  assert formal_human.commands["twist"].closed_loop_centering is True
  formal_stairs = formal_human.scene.terrain.terrain_generator.sub_terrains[
    "forward_stairs"
  ]
  assert formal_stairs.step_height_profile == TARGET_RISE_PROFILE
  assert formal_stairs.step_width_profile == TARGET_TREAD_PROFILE


def test_signed_deadband_is_continuous_and_symmetric() -> None:
  values = torch.tensor([-0.10, -0.04, 0.0, 0.04, 0.10])
  result = signed_deadband(values, 0.04)
  assert torch.allclose(result, torch.tensor([-0.06, 0.0, 0.0, 0.0, 0.06]))


def test_human_centerline_feedback_corrects_drift_with_bounded_sticks() -> None:
  command = centerline_feedback_command(
    torch.tensor([0.4, 0.4, 0.4]),
    # Centerline is left, centered, and right of the robot, respectively.
    torch.tensor([0.50, 0.0, -0.50]),
    torch.tensor([0.40, 0.0, -0.40]),
    lateral_gain=0.8,
    heading_gain=1.4,
    lateral_deadband=0.04,
    heading_deadband=0.03,
    max_lateral_velocity=0.16,
    max_yaw_velocity=0.45,
  )
  assert torch.allclose(command[:, 0], torch.full((3,), 0.4))
  assert command[0, 1] > 0.0 and command[0, 2] > 0.0
  assert command[2, 1] < 0.0 and command[2, 2] < 0.0
  assert torch.equal(command[1, 1:], torch.zeros(2))
  assert torch.all(torch.abs(command[:, 1]) <= 0.16)
  assert torch.all(torch.abs(command[:, 2]) <= 0.45)


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


def test_cbf_independence_gate_requires_off_equivalence_and_near_zero_use() -> None:
  filter_on = {
    "success_rate": 0.80,
    "fall_rate": 0.20,
    "intervention_per_riser": 0.08,
    "correction_mean": 4.0e-4,
  }
  filter_off = {
    "success_rate": 0.79,
    "fall_rate": 0.21,
    "intervention_per_riser": 0.0,
    "correction_mean": 0.0,
  }
  accepted, reasons = cbf_independence_gate(
    filter_on_eval=filter_on,
    filter_off_eval=filter_off,
  )
  assert accepted and reasons == []

  dependent = dict(filter_on, intervention_per_riser=0.11)
  accepted, reasons = cbf_independence_gate(
    filter_on_eval=dependent,
    filter_off_eval=dict(filter_off, success_rate=0.70, fall_rate=0.30),
    thresholds=CbfIndependenceThresholds(),
  )
  assert not accepted
  assert "CBF-off success gap exceeds limit" in reasons
  assert "CBF-off fall gap exceeds limit" in reasons
  assert "runtime intervention per riser exceeds near-zero limit" in reasons


def test_safe_bc_target_cancels_sampled_exploration_noise() -> None:
  mean = torch.tensor([[0.1, -0.2]])
  exploration = torch.tensor([[0.7, -0.5]])
  correction = torch.tensor([[-0.03, 0.04]])
  nominal = mean + exploration
  safe = nominal + correction
  target = cbf_corrected_mean_target(mean, nominal, safe)
  assert torch.allclose(target, mean + correction)
  assert not torch.allclose(target, safe)


def test_unshielded_gate_uses_counterfactual_safety_demand() -> None:
  result = {
    "runtime_filter": False,
    "intervention_per_riser": 0.0,
    "would_intervene_per_riser": 0.75,
  }
  assert safety_demand_per_riser(result) == 0.75
  result["runtime_filter"] = True
  assert safety_demand_per_riser(result) == 0.0


def test_paired_interval_reports_consistent_improvement() -> None:
  old = {
    "replicates": [
      {"success_rate": 0.70},
      {"success_rate": 0.75},
      {"success_rate": 0.80},
    ]
  }
  candidate = {
    "replicates": [
      {"success_rate": 0.80},
      {"success_rate": 0.85},
      {"success_rate": 0.90},
    ]
  }
  mean, lower, upper = paired_metric_delta_interval(
    old, candidate, "success_rate"
  )
  assert abs(mean - 0.10) < 1.0e-12
  assert lower > 0.0
  assert upper > 0.0


def test_gate_intervals_expose_noisy_intervention_uncertainty() -> None:
  def result(success: float, fall: float, demand: float) -> dict:
    return {
      "success_rate": success,
      "fall_rate": fall,
      "intervention_per_riser": demand,
      "runtime_filter": True,
    }

  old_target = [
    result(0.9, 0.1, 0.60),
    result(0.9, 0.1, 0.82),
    result(0.9, 0.1, 0.65),
  ]
  candidate_target = [
    result(0.9, 0.1, 0.80),
    result(0.9, 0.1, 0.62),
    result(0.9, 0.1, 0.67),
  ]
  old = {
    "D4": dict(old_target[0], replicates=old_target),
    "D5": dict(old_target[0], replicates=old_target),
  }
  candidate = {
    "D4": dict(candidate_target[0], replicates=candidate_target),
    "D5": dict(candidate_target[0], replicates=candidate_target),
  }
  intervals = candidate_gate_intervals(
    old_eval=old, candidate_eval=candidate
  )
  _, lower, upper = intervals["target_intervention_delta_95"]
  assert lower < 0.0 < upper


def test_paired_bootstrap_is_deterministic_and_preserves_pairing() -> None:
  old = {
    "replicates": [
      {"fall_rate": 0.10},
      {"fall_rate": 0.20},
      {"fall_rate": 0.30},
    ]
  }
  candidate = {
    "replicates": [
      {"fall_rate": 0.20},  # +0.10
      {"fall_rate": 0.20},  #  0.00
      {"fall_rate": 0.20},  # -0.10
    ]
  }
  first = paired_metric_delta_interval(old, candidate, "fall_rate")
  second = paired_metric_delta_interval(old, candidate, "fall_rate")
  assert first == second
  mean, lower, upper = first
  assert abs(mean) < 1.0e-12
  assert lower < 0.0 < upper


def test_adaptive_std_reduces_shield_demand_and_falls() -> None:
  high_demand = adaptive_cbf_std_factor(
    0.80, target_intervention_per_riser=0.10, adaptation_rate=0.10
  )
  low_demand = adaptive_cbf_std_factor(
    0.00, target_intervention_per_riser=0.10, adaptation_rate=0.10
  )
  after_fall = adaptive_cbf_std_factor(
    0.00,
    target_intervention_per_riser=0.10,
    adaptation_rate=0.10,
    fall_count=1.0,
  )
  assert 0.80 <= high_demand < 1.0
  assert 1.0 < low_demand <= 1.05
  assert after_fall == 0.80


def test_rejection_preserves_anchored_exploration_distribution() -> None:
  class AlgorithmStub:
    def __init__(self) -> None:
      self.actor_lr_scales: list[float] = []
      self.std_scales: list[float] = []

    def scale_actor_learning_rate(self, factor: float) -> None:
      self.actor_lr_scales.append(factor)

    def scale_exploration_std(self, factor: float) -> None:
      self.std_scales.append(factor)

  runner = object.__new__(OnlineSafeRefinementRunner)
  runner.alg = AlgorithmStub()
  runner.reduce_after_rejection()
  assert runner.alg.actor_lr_scales == [0.5]
  assert runner.alg.std_scales == []


def test_hard_case_bank_prioritizes_and_roundtrips() -> None:
  bank = HardCaseStateBank(capacity=2)
  batched = {
    "feature": torch.tensor([[1.0], [2.0], [3.0]]),
    "terrain/type": torch.tensor([0, 1, 0]),
  }
  added = bank.add_batched(
    batched,
    torch.tensor([0, 1]),
    torch.tensor([0.10, 0.20]),
    torch.tensor([4, 5]),
  )
  assert added == 2 and len(bank) == 2
  replaced = bank.add_batched(
    batched,
    torch.tensor([2]),
    torch.tensor([0.90]),
    torch.tensor([6]),
  )
  assert replaced == 1
  priorities = torch.tensor(sorted(entry.priority for entry in bank.entries))
  assert torch.allclose(priorities, torch.tensor([0.20, 0.90]))

  restored = HardCaseStateBank(capacity=1)
  restored.load_state_dict(bank.state_dict())
  assert len(restored) == 2
  generator = torch.Generator().manual_seed(7)
  sample = restored.sample(2, device="cpu", generator=generator)
  assert sample["feature"].shape == (2, 1)
  assert set(sample["feature"].flatten().tolist()) == {2.0, 3.0}


def test_hard_case_destination_ids_are_reproducible() -> None:
  a = hard_case_destination_ids(
    20, 0.25, device="cpu", generator=torch.Generator().manual_seed(9)
  )
  b = hard_case_destination_ids(
    20, 0.25, device="cpu", generator=torch.Generator().manual_seed(9)
  )
  assert torch.equal(a, b)
  assert len(a) == 5
  assert len(torch.unique(a)) == 5


def test_three_way_curriculum_ids_are_disjoint_and_reproducible() -> None:
  first = curriculum_destination_ids(
    20,
    hard_case_fraction=0.25,
    neighbor_command_fraction=0.15,
    device="cpu",
    generator=torch.Generator().manual_seed(17),
  )
  second = curriculum_destination_ids(
    20,
    hard_case_fraction=0.25,
    neighbor_command_fraction=0.15,
    device="cpu",
    generator=torch.Generator().manual_seed(17),
  )
  assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
  hard_ids, neighbor_ids = first
  assert len(hard_ids) == 5
  assert len(neighbor_ids) == 3
  assert len(torch.unique(torch.cat([hard_ids, neighbor_ids]))) == 8


def test_neighbor_command_curriculum_is_bounded_and_local() -> None:
  class _Command:
    def __init__(self):
      self.raw_command = torch.tensor(
        [[0.4, 0.0, 0.0], [0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]
      )
      self.delivered_command = torch.ones(3, 3)
      self.command_derivative = torch.ones(3, 3)
      self.delay_steps = torch.tensor([3, 3, 3])
      self._max_delay_steps = 8
      self._delay_queue = torch.ones(3, 9, 3)

  class _Manager:
    def __init__(self, command):
      self.command = command

    def get_term(self, name: str):
      assert name == "twist"
      return self.command

  class _Env:
    def __init__(self):
      self.command_manager = _Manager(_Command())

  env = _Env()
  metrics = perturb_joystick_command_state(
    env,
    torch.tensor([0, 2]),
    generator=torch.Generator().manual_seed(3),
    forward_scale_range=(0.9, 1.1),
    delay_step_offset_range=(-2, 2),
  )
  command = env.command_manager.command
  assert torch.all(command.raw_command[[0, 2], 0] >= 0.36)
  assert torch.all(command.raw_command[[0, 2], 0] <= 0.44)
  assert command.raw_command[1, 0] == 0.4
  assert torch.equal(command.delivered_command[[0, 2]], torch.zeros(2, 3))
  assert torch.equal(command.command_derivative[[0, 2]], torch.zeros(2, 3))
  assert torch.equal(command._delay_queue[[0, 2]], torch.zeros(2, 9, 3))
  assert 1 <= int(command.delay_steps[[0, 2]].min())
  assert int(command.delay_steps[[0, 2]].max()) <= 5
  assert 0.9 <= metrics["neighbor_forward_scale_mean"] <= 1.1


def test_fixed_delay_queue_preserves_newest_first_state_and_padding() -> None:
  class _Command:
    _delay_queue = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3)

  state = mdp.fixed_delay_queue_state(
    _Command(), num_envs=2, command_dim=3, queue_length=5, device="cpu"
  )
  assert state.shape == (2, 5, 3)
  assert torch.equal(state[:, :3], _Command._delay_queue)
  assert torch.equal(state[:, 3:], torch.zeros(2, 2, 3))

  class _WaypointCommand:
    pass

  waypoint_state = mdp.fixed_delay_queue_state(
    _WaypointCommand(), num_envs=2, command_dim=3, queue_length=5, device="cpu"
  )
  assert torch.equal(waypoint_state, torch.zeros(2, 5, 3))


def test_rollout_action_dataflow_keeps_policy_action_out_of_cbf_projection() -> None:
  policy = torch.tensor(
    [
      [[1.2, 0.1], [0.2, -0.3]],
      [[0.4, 0.5], [-0.2, 0.1]],
    ]
  )
  stored = policy.clone()
  nominal = policy.clamp(-1.0, 1.0)
  safe = nominal.clone()
  safe[0, 0, 1] -= 0.05
  safe[1, 1, 0] += 0.03
  enabled = torch.tensor([[True, True], [False, False]])
  executed = torch.where(enabled.unsqueeze(-1), safe, nominal)
  metrics = rollout_action_dataflow_metrics(
    policy, stored, nominal, safe, executed, enabled
  )
  assert metrics["policy_storage_max_abs_error"] == 0.0
  assert metrics["executed_action_routing_max_abs_error"] == 0.0
  assert metrics["policy_to_nominal_clip_fraction"] > 0.0
  assert metrics["counterfactual_safe_action_fraction"] == 0.5
  assert metrics["executed_action_change_fraction"] == 0.25
  assert metrics["runtime_filter_enabled_fraction"] == 0.5


def test_rollout_action_dataflow_detects_safe_action_in_ppo_storage() -> None:
  policy = torch.zeros(2, 1, 2)
  nominal = policy.clone()
  safe = policy.clone()
  safe[0, 0, 0] = 0.1
  enabled = torch.ones(2, 1, dtype=torch.bool)
  metrics = rollout_action_dataflow_metrics(
    policy,
    safe,  # This is precisely the invalid PPO-buffer substitution.
    nominal,
    safe,
    safe,
    enabled,
  )
  assert abs(metrics["policy_storage_max_abs_error"] - 0.1) < 1.0e-6


def test_behavior_log_prob_tolerance_accepts_float32_reduction_noise() -> None:
  stored = torch.tensor([-12.0, -3.0], dtype=torch.float32)
  recomputed = stored + torch.tensor([1.2e-4, -1.5e-4])
  error = validate_behavior_log_prob(stored, recomputed)
  assert 1.0e-4 < error < 2.0e-4

  invalid = stored + torch.tensor([3.0e-4, 0.0])
  try:
    validate_behavior_log_prob(stored, invalid)
  except RuntimeError as exc:
    assert "inconsistent with a_policy" in str(exc)
  else:
    raise AssertionError("a true behavior log-prob mismatch was not rejected")


def test_behavior_distribution_parameter_audit_is_strict() -> None:
  stored = (torch.zeros(4, 12), torch.ones(4, 12))
  recomputed = tuple(value.clone() for value in stored)
  assert validate_behavior_distribution_params(stored, recomputed) == 0.0

  mismatched = (stored[0].clone(), stored[1].clone())
  mismatched[0][0, 0] = 2.0e-5
  try:
    validate_behavior_distribution_params(stored, mismatched)
  except RuntimeError as exc:
    assert "distribution is inconsistent" in str(exc)
  else:
    raise AssertionError("a true Gaussian parameter mismatch was not rejected")


def test_intervention_advantage_shaping_is_policy_only_and_immediate() -> None:
  class _Storage:
    advantages = torch.zeros(3, 2, 1)

  algorithm = object.__new__(OnlineSafePPO)
  algorithm.storage = _Storage()
  algorithm.cbf_magnitude = torch.tensor(
    [[0.0, 0.05], [0.10, 0.025], [0.05, 0.05]]
  )
  algorithm.cbf_intervened = torch.tensor(
    [[False, True], [True, False], [False, True]]
  )
  algorithm.intervention_magnitude_scale = 0.05
  algorithm.intervention_advantage_weight = 0.075
  metrics = algorithm.shape_intervention_advantages()
  expected = torch.tensor(
    [[0.0, -0.075], [-0.075, 0.0], [0.0, -0.075]]
  )
  assert torch.allclose(algorithm.storage.advantages.squeeze(-1), expected)
  assert abs(metrics["intervention_advantage_penalty_mean"] - 0.0375) < 1.0e-7


def test_base_actor_reference_replaces_only_pretrained_mean_network() -> None:
  class _Actor(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.mlp = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
      self.distribution = torch.nn.Linear(1, 1, bias=False)

  algorithm = object.__new__(OnlineSafePPO)
  algorithm.actor = _Actor()
  with torch.no_grad():
    algorithm.actor.mlp[0].weight.fill_(3.0)
    algorithm.actor.distribution.weight.fill_(7.0)
  base_state = {
    key: value.detach().clone() for key, value in algorithm.actor.state_dict().items()
  }
  base_state["mlp.0.weight"].fill_(1.0)
  base_state["distribution.weight"].fill_(99.0)
  algorithm.set_base_actor_reference(base_state)
  reference = algorithm.base_actor_reference
  assert torch.equal(reference.mlp[0].weight, torch.ones(1, 2))
  assert torch.equal(reference.distribution.weight, torch.full((1, 1), 7.0))
  assert all(not parameter.requires_grad for parameter in reference.parameters())
