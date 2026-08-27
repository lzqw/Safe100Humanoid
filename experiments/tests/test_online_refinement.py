"""Pure tensor/config tests for conservative online refinement."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch

from src.tasks.stairs_cbf.config import (
  TARGET_RISE_PROFILE,
  TARGET_TREAD_PROFILE,
  g1_online_stairs_env_cfg,
  g1_online_stairs_runner_cfg,
)
from src.tasks.stairs_cbf.online import (
  BriefPpoGateThresholds,
  CandidateGateThresholds,
  CbfIndependenceThresholds,
  FailureFocusedGateThresholds,
  SpecialistGateThresholds,
  OnlineSafePPO,
  OnlineSafeRefinementRunner,
  SafeImprovementScoreWeights,
  backtrack_actor_state,
  backward_intervention_credit,
  brief_actor_layer_profile_is_valid,
  brief_candidate_gate,
  brief_candidate_precheck,
  brief_d0_retention_gate,
  brief_dual_reward_weight,
  brief_target_score,
  candidate_gate,
  candidate_gate_intervals,
  candidate_precheck,
  cbf_independence_gate,
  cbf_corrected_mean_target,
  binary_risk_metrics,
  critic_readiness_reasons,
  critic_calibration_by_riser,
  future_event_labels,
  failure_focused_candidate_gate,
  failure_focused_candidate_precheck,
  failure_focused_target_score,
  hierarchical_specialist_macro_interval,
  generalized_cost_advantage,
  mask_legacy_actor_input_gradient,
  projected_lagrange_update,
  redistributed_fall_credit,
  pre_event_value_delta,
  rollout_action_dataflow_metrics,
  paired_metric_delta_interval,
  safety_demand_per_riser,
  safe_improvement_score,
  success_gated_correction_mask,
  adaptive_cbf_std_factor,
  validate_behavior_log_prob,
  validate_behavior_distribution_params,
  local_matched_success_actor_loss,
  normalize_v19_grouped_advantages,
  specialist_candidate_gate,
  specialist_candidate_precheck,
  specialist_d0_retention_gate,
  specialist_target_score,
)
from src.tasks.stairs_cbf.hard_cases import (
  HardCaseEntry,
  HardCaseStateBank,
  LATERAL_HEADING_DRIFT_FAILURE_TYPE,
  LateFailureCandidate,
  NON_LATERAL_BALANCE_FAILURE_TYPE,
  NON_LATERAL_HIGH_CBF_FAILURE_TYPE,
  RESTART_BALANCE_PROFILE_LATERAL_STAGE_SUPPORT_GROWTH,
  SPECIALIST_FAILURE_BANK_KIND,
  SPECIALIST_SUCCESS_BANK_KIND,
  SPECIALIST_SUCCESS_POOL_KIND,
  SpecialistBankCandidate,
  classify_target_failure_mode,
  curriculum_destination_ids,
  hard_case_destination_ids,
  hard_case_state_shape_mismatches,
  finalize_v19_replay_bank_update,
  perturb_joystick_command_state,
  match_specialist_success_counterexamples,
  match_v19_success_counterexamples,
  select_late_failure_candidate,
  select_specialist_failure_candidates,
  select_specialist_success_candidates,
  specialist_destination_ids,
  specialist_history_window,
  classify_v19_failure_mode,
  select_v19_contact_candidates,
  select_v19_lateral_failure_candidates,
  select_v19_lateral_success_candidates,
  select_v19_balanced_restart_pairs,
  v19_restart_pair_feasibility,
)
from src.tasks.stairs_cbf.deployment_context import (
  FAILURE_FOCUSED_CALIBRATION_KIND,
  SPECIALIST_CALIBRATION_KIND,
  SPECIALIST_FAILURE_TYPES,
  apply_frozen_deployment_context,
  generate_failure_focused_context,
  generate_specialist_context,
  validate_calibrated_deployment_context,
  validate_calibrated_specialist_context,
  validate_frozen_deployment_context,
  V19_CALIBRATION_KIND,
  V19_SPECIALIST_FAILURE_TYPES,
  generate_v19_specialist_context,
  validate_calibrated_v19_context,
)
from src.tasks.stairs_cbf.retention import (
  RETENTION_BANK_KIND,
  RETENTION_BANK_SCHEMA_VERSION,
  actor_observation_sha256,
  balanced_stage_quotas,
  cyclic_retention_batch,
  increase_anchor_weight_on_budget_violation,
  interleave_stage_observations,
  validate_retention_observation_bank,
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


def test_max_credit_bounds_repeated_intervention_over_full_swing() -> None:
  magnitude = torch.full((8, 1), 0.05)
  intervened = torch.ones(8, 1, dtype=torch.bool)
  dones = torch.zeros(8, 1, dtype=torch.bool)
  credit = backward_intervention_credit(
    magnitude,
    intervened,
    dones,
    horizon=8,
    decay=0.95,
    magnitude_scale=0.05,
    aggregation="max",
  )
  assert torch.allclose(credit, torch.ones_like(credit))
  assert float(credit.max()) == 1.0


def test_brief_dual_reward_schedule_is_task_first() -> None:
  assert [brief_dual_reward_weight(index) for index in range(1, 6)] == [
    0.0,
    0.0,
    0.02,
    0.02,
    0.02,
  ]
  with pytest.raises(ValueError, match="positive"):
    brief_dual_reward_weight(0)


def test_failure_focused_gate_uses_success_minus_fall_and_catastrophic_cbf_cap() -> None:
  old = {
    "success_rate": 0.80,
    "fall_rate": 0.15,
    "intervention_per_riser": 0.40,
    "initial_state_signatures": ["paired-v15"],
  }
  candidate = {
    "success_rate": 0.82,
    "fall_rate": 0.14,
    "intervention_per_riser": 0.49,
    "initial_state_signatures": ["paired-v15"],
  }
  assert failure_focused_target_score(candidate) == pytest.approx(
    {"success": 0.82, "fall": -0.14, "total": 0.68}
  )
  accepted, reasons, scores = failure_focused_candidate_gate(
    update_metrics={"mean_kl": 0.003},
    old_eval=old,
    candidate_eval=candidate,
    parameters_finite=True,
  )
  assert accepted
  assert reasons == []
  assert scores["candidate"]["total"] > scores["old"]["total"]

  accepted, reasons, _ = failure_focused_candidate_gate(
    update_metrics={"mean_kl": 0.003},
    old_eval=old,
    candidate_eval={**candidate, "intervention_per_riser": 0.501},
    parameters_finite=True,
  )
  assert not accepted
  assert any("25 percent" in reason for reason in reasons)
  thresholds = FailureFocusedGateThresholds()
  assert failure_focused_candidate_precheck(
    update_metrics={"mean_kl": 0.01},
    parameters_finite=True,
    thresholds=thresholds,
  ) == ["update KL is not below 0.01"]


def test_specialist_gate_is_diagonal_score_fall_and_kl_only() -> None:
  old = {
    "success_rate": 0.76,
    "fall_rate": 0.22,
    "intervention_per_riser": 0.1,
    "initial_state_signatures": ["paired-v17"],
  }
  candidate = {
    "success_rate": 0.79,
    "fall_rate": 0.20,
    # Deliberately catastrophic by the removed v15 CBF-demand gate.
    "intervention_per_riser": 100.0,
    "initial_state_signatures": ["paired-v17"],
  }
  assert specialist_target_score(candidate) == pytest.approx(
    {"success": 0.79, "fall": -0.20, "total": 0.59}
  )
  accepted, reasons, scores = specialist_candidate_gate(
    update_metrics={"mean_kl": 0.009},
    old_eval=old,
    candidate_eval=candidate,
    parameters_finite=True,
  )
  assert accepted
  assert reasons == []
  assert scores["candidate"]["total"] > scores["old"]["total"]
  assert specialist_candidate_precheck(
    update_metrics={"mean_kl": 0.01},
    parameters_finite=True,
    thresholds=SpecialistGateThresholds(),
  ) == ["update KL is not below 0.01"]

  d0_passed, d0_reasons = specialist_d0_retention_gate(
    baseline_eval={
      "success_rate": 0.90,
      "initial_state_signatures": ["d0-pair"],
    },
    candidate_eval={
      "success_rate": 0.85,
      "initial_state_signatures": ["d0-pair"],
    },
  )
  assert d0_passed
  assert d0_reasons == []


def test_specialist_contexts_are_hashed_mode_specific_and_base_only_calibrated() -> None:
  legacy = generate_failure_focused_context(1701)
  assert validate_frozen_deployment_context(legacy)["parameters_sha256"] == legacy[
    "parameters_sha256"
  ]
  contexts = {
    mode: generate_specialist_context(mode, seed)
    for mode, seed in (("lateral", 2100), ("cbf", 2200), ("balance", 2300))
  }
  assert len({context["parameters_sha256"] for context in contexts.values()}) == 3
  for mode, context in contexts.items():
    weights = {
      key.removesuffix("_signal_weight"): value
      for key, value in context["scenario"].items()
      if key.endswith("_signal_weight")
    }
    assert sum(weights.values()) == pytest.approx(1.0)
    attempt = {
      "candidate_seed": context["calibration_candidate_seed"],
      "parameters_sha256": context["parameters_sha256"],
      "base_policy_only": True,
      "num_episodes": 512,
      "success_rate": 0.80,
      "fall_count": 103,
      "target_failure_type": SPECIALIST_FAILURE_TYPES[mode],
      "target_failure_fraction": 0.65,
      "second_failure_fraction": 0.25,
      "qualifies": True,
    }
    context["calibration"] = {
      "kind": SPECIALIST_CALIBRATION_KIND,
      "success_rate_bounds": [0.70, 0.85],
      "minimum_target_failure_fraction": 0.60,
      "maximum_second_failure_fraction": 0.30,
      "minimum_fall_count": 100,
      "episodes_per_candidate": 512,
      "candidate_seeds": [context["calibration_candidate_seed"]],
      "attempts": [attempt],
      "selected_candidate_seed": context["calibration_candidate_seed"],
      "selected_parameters_sha256": context["parameters_sha256"],
      "adapted_policy_evaluations_used": False,
    }
    validated = validate_calibrated_specialist_context(context)
    assert validated["specialist_mode"] == mode


def _specialist_test_histories(length: int = 161):
  time = torch.linspace(0.0, 1.0, length)
  riser = torch.floor(5.0 + 6.0 * time).long()
  phase = (torch.arange(length, dtype=torch.float32) * 0.07) % 1.0
  support = (torch.arange(length) % 2).long()
  command = torch.stack((0.4 + 0.1 * time, 0.05 * time, 0.1 * time), dim=1)
  velocity = torch.stack((0.3 + 0.1 * time, 0.02 * time, 0.0 * time), dim=1)
  cbf_active = time > 0.55
  components = {
    "centerline": time,
    "heading": time.square(),
    "edge": (time > 0.75).float(),
    "intervention": time,
    "nominal_margin": time.square(),
    "roll": time,
    "pitch": 0.8 * time,
    "angular_velocity": time.square(),
    "slip": (time > 0.65).float(),
    "contact_mismatch": 0.6 * time,
  }
  return riser, components, phase, support, command, velocity, cbf_active


@pytest.mark.parametrize(
  ("mode", "window"),
  (("lateral", (50, 150)), ("cbf", (10, 50)), ("balance", (20, 100))),
)
def test_specialist_selectors_use_frozen_windows_and_balanced_buckets(
  mode: str, window: tuple[int, int]
) -> None:
  histories = _specialist_test_histories()
  assert specialist_history_window(mode) == window
  candidates = select_specialist_failure_candidates(
    mode,
    *histories,
    minimum_riser=5,
    maximum_candidates=4,
  )
  assert candidates
  assert all(
    window[0] <= candidate.steps_before_terminal <= window[1]
    for candidate in candidates
  )
  assert len({candidate.balance_bucket for candidate in candidates}) == len(
    candidates
  )
  if mode == "lateral":
    assert len({candidate.riser_index for candidate in candidates}) >= 2
  if mode == "balance":
    assert {candidate.support_foot for candidate in candidates} == {0, 1}
  successes = select_specialist_success_candidates(
    mode,
    *histories,
    minimum_riser=5,
    maximum_candidates=4,
  )
  assert successes
  assert all(candidate.outcome == "success" for candidate in successes)
  if mode in {"lateral", "cbf"}:
    assert len({candidate.riser_index for candidate in successes}) == len(
      successes
    )
  if mode == "balance":
    assert {candidate.support_foot for candidate in successes} == {0, 1}


def test_specialist_banks_match_successes_and_serialize_without_cross_mode_state() -> None:
  mode = "lateral"
  failure_bank = HardCaseStateBank(
    capacity=4,
    bank_kind=SPECIALIST_FAILURE_BANK_KIND,
    source_domain="DQHMED",
    context_sha256="context",
    specialist_mode=mode,
  )
  success_pool = HardCaseStateBank(
    capacity=4,
    bank_kind=SPECIALIST_SUCCESS_POOL_KIND,
    source_domain="DQHMED",
    context_sha256="context",
    specialist_mode=mode,
  )
  success_bank = HardCaseStateBank(
    capacity=4,
    bank_kind=SPECIALIST_SUCCESS_BANK_KIND,
    source_domain="DQHMED",
    context_sha256="context",
    specialist_mode=mode,
  )
  state = {
    "robot/root_pose_relative": torch.arange(14, dtype=torch.float32).reshape(2, 7),
    "terrain/type": torch.tensor([0, 1]),
  }
  for env_id in range(2):
    failure = SpecialistBankCandidate(
      history_index=0,
      steps_before_terminal=60 + env_id,
      riser_index=7 + env_id,
      gait_phase=0.2 + 0.4 * env_id,
      support_foot=env_id,
      delivered_command=(0.4, 0.0, 0.0),
      root_velocity=(0.3, 0.0, 0.0),
      cbf_active=bool(env_id),
      priority=10.0 + env_id,
      balance_bucket=f"riser:{7 + env_id}",
      selection_signal=0.5,
      outcome="failure",
      failure_type=LATERAL_HEADING_DRIFT_FAILURE_TYPE,
    )
    success = SpecialistBankCandidate(
      **{
        **failure.__dict__,
        "outcome": "success",
        "failure_type": "mixed",
        "priority": 8.0 + env_id,
      }
    )
    observation = torch.tensor([float(env_id), 0.5, -0.5])
    assert failure_bank.add_specialist_candidate(
      state, env_id, failure, observation
    ) == 1
    assert success_pool.add_specialist_candidate(
      state, env_id, success, observation + 0.01
    ) == 1
  matching = match_specialist_success_counterexamples(
    failure_bank, success_pool, success_bank
  )
  assert matching["one_match_per_replayed_failure"]
  assert len(success_bank) == len(failure_bank) == 2
  audit = success_bank.audit_metadata()
  assert audit["outcome_counts"] == {"success": 2}
  assert audit["matched_entry_count"] == 2
  restored = HardCaseStateBank(capacity=1)
  restored.load_state_dict(success_bank.state_dict())
  assert restored.specialist_mode == mode
  assert restored.audit_metadata()["matched_entry_count"] == 2


def test_specialist_start_ids_realize_integer_70_15_15_allocation() -> None:
  generator = torch.Generator().manual_seed(170)
  failure, success = specialist_destination_ids(
    64,
    failure_fraction=0.15,
    success_fraction=0.15,
    device="cpu",
    generator=generator,
  )
  assert len(failure) == len(success) == 10
  assert len(torch.unique(torch.cat((failure, success)))) == 20
  assert 64 - len(failure) - len(success) == 44


def test_hierarchical_specialist_macro_bootstrap_is_deterministic() -> None:
  groups = [
    [torch.full((32,), 0.03 + 0.002 * seed) for seed in range(3)]
    for _scene in range(3)
  ]
  first = hierarchical_specialist_macro_interval(
    groups, bootstrap_samples=1000, bootstrap_seed=17
  )
  second = hierarchical_specialist_macro_interval(
    groups, bootstrap_samples=1000, bootstrap_seed=17
  )
  assert first == second
  assert first[0] > 0.03
  assert first[1] > 0.0


def test_failure_focused_fall_credit_preserves_two_units_without_crossing_reset() -> None:
  falls = torch.zeros(205, 2, dtype=torch.bool)
  dones = torch.zeros_like(falls)
  dones[30, 0] = True
  falls[160, 0] = True
  dones[160, 0] = True
  falls[10, 1] = True
  dones[10, 1] = True
  credit = redistributed_fall_credit(falls, dones, horizon=100, decay=0.97)
  assert float(credit[:, 0].sum()) == pytest.approx(2.0, abs=1e-6)
  assert float(credit[:, 1].sum()) == pytest.approx(2.0, abs=1e-6)
  assert torch.count_nonzero(credit[:, 0]) == 100
  assert torch.count_nonzero(credit[:, 1]) == 11
  assert torch.count_nonzero(credit[:61, 0]) == 0
  assert float(credit[159, 0] / credit[160, 0]) == pytest.approx(0.97)


def test_late_failure_selector_and_bank_are_target_failure_only() -> None:
  length = 170
  risers = torch.arange(length).div(16, rounding_mode="floor").clamp_max(6)
  lateral = torch.zeros(length)
  heading = torch.zeros(length)
  correction = torch.zeros(length)
  lateral[90:] = 0.25
  heading[100:] = 0.30
  correction[110:] = 0.02
  candidate = select_late_failure_candidate(
    risers,
    lateral,
    heading,
    correction,
    minimum_steps_before_fall=50,
    maximum_steps_before_fall=150,
    minimum_riser=5,
  )
  assert candidate is not None
  assert 50 <= candidate.steps_before_fall <= 150
  assert candidate.riser_index >= 5
  assert max(
    candidate.lateral_drift_fraction,
    candidate.heading_drift_fraction,
    candidate.large_correction_fraction,
  ) > 0.0

  crossed_after_window = torch.full((160,), 5, dtype=torch.long)
  crossed_after_window[-20:] = 6
  assert select_late_failure_candidate(
    crossed_after_window,
    torch.zeros(160),
    torch.zeros(160),
    torch.zeros(160),
    minimum_steps_before_fall=50,
    maximum_steps_before_fall=150,
    minimum_riser=5,
  ) is None

  bank = HardCaseStateBank(
    capacity=2,
    bank_kind="target_late_failure",
    source_domain="DQHMED",
    context_sha256="context-hash",
  )
  state = {
    "terrain/type": torch.tensor([0]),
    "dummy": torch.tensor([[1.0, 2.0]]),
  }
  assert bank.add_late_failure(state, 0, candidate) == 1
  audit = bank.audit_metadata()
  assert audit["bank_kind"] == "target_late_failure"
  assert audit["late_failure_entry_count"] == 1
  assert 50 <= audit["steps_before_fall_min"] <= 150
  assert audit["successful_crossing_exclusion_passed"] is True

  general_bank = HardCaseStateBank(capacity=2)
  with pytest.raises(ValueError, match="target_late_failure"):
    general_bank.add_late_failure(
      state,
      0,
      LateFailureCandidate(0, 50, 5, 0.0, 0.0, 0.0, 5.0),
    )


def test_branch_b_failure_classifier_and_bank_freeze_one_dominant_type() -> None:
  assert classify_target_failure_mode(
    side_edge_breach=True,
    max_abs_centerline_error=0.1,
    max_abs_heading_error=0.1,
    correction_max=0.1,
  ) == LATERAL_HEADING_DRIFT_FAILURE_TYPE
  assert classify_target_failure_mode(
    side_edge_breach=False,
    max_abs_centerline_error=0.8,
    max_abs_heading_error=0.1,
    correction_max=0.1,
  ) == LATERAL_HEADING_DRIFT_FAILURE_TYPE
  assert classify_target_failure_mode(
    side_edge_breach=False,
    max_abs_centerline_error=0.1,
    max_abs_heading_error=0.1,
    correction_max=0.5,
  ) == NON_LATERAL_HIGH_CBF_FAILURE_TYPE
  assert classify_target_failure_mode(
    side_edge_breach=False,
    max_abs_centerline_error=0.1,
    max_abs_heading_error=0.1,
    correction_max=0.49,
  ) == NON_LATERAL_BALANCE_FAILURE_TYPE

  length = 170
  risers = torch.full((length,), 10, dtype=torch.long)
  lateral = torch.linspace(0.0, 1.0, length)
  candidate = select_late_failure_candidate(
    risers,
    lateral,
    torch.zeros(length),
    torch.full((length,), 0.2),
    failure_type=LATERAL_HEADING_DRIFT_FAILURE_TYPE,
  )
  mixed_candidate = select_late_failure_candidate(
    risers,
    lateral,
    torch.zeros(length),
    torch.full((length,), 0.2),
  )
  assert candidate is not None
  assert mixed_candidate is not None
  assert candidate.failure_type == LATERAL_HEADING_DRIFT_FAILURE_TYPE
  assert candidate.history_index == mixed_candidate.history_index
  assert candidate.priority == mixed_candidate.priority
  bank = HardCaseStateBank(
    capacity=2,
    bank_kind="target_late_failure",
    source_domain="DQHMED",
    dominant_failure_type=LATERAL_HEADING_DRIFT_FAILURE_TYPE,
  )
  state = {"terrain/type": torch.tensor([0]), "dummy": torch.ones(1, 2)}
  assert bank.add_late_failure(state, 0, candidate) == 1
  audit = bank.audit_metadata()
  assert audit["dominant_failure_type_purity_passed"] is True
  assert audit["failure_type_counts"] == {
    LATERAL_HEADING_DRIFT_FAILURE_TYPE: 1
  }
  wrong_type = LateFailureCandidate(
    0,
    50,
    10,
    0.0,
    0.0,
    1.0,
    10.0,
    failure_type=NON_LATERAL_HIGH_CBF_FAILURE_TYPE,
  )
  with pytest.raises(ValueError, match="frozen dominant failure"):
    bank.add_late_failure(state, 0, wrong_type)
  with pytest.raises(ValueError, match="only labeled late-failure"):
    bank.add_batched(
      state,
      torch.tensor([0]),
      torch.tensor([1.0]),
      torch.tensor([10]),
    )
  serialized = bank.state_dict()
  serialized["entries"][0]["failure_type"] = NON_LATERAL_HIGH_CBF_FAILURE_TYPE
  with pytest.raises(ValueError, match="outside its dominant failure"):
    HardCaseStateBank().load_state_dict(serialized)


def test_failure_focused_context_is_deterministic_hidden_and_calibration_auditable() -> None:
  first = generate_failure_focused_context(1001)
  assert first == generate_failure_focused_context(1001)
  assert first != generate_failure_focused_context(1002)
  assert validate_frozen_deployment_context(first)["parameters_sha256"] == first[
    "parameters_sha256"
  ]
  tampered = copy.deepcopy(first)
  tampered["target"]["action_gain"] += 0.001
  with pytest.raises(ValueError, match="hash mismatch"):
    validate_frozen_deployment_context(tampered)

  cfg = g1_online_stairs_env_cfg("DQHMED")
  actor_terms_before = tuple(cfg.observations["actor"].terms)
  metadata = apply_frozen_deployment_context(cfg, first, role="target")
  assert tuple(cfg.observations["actor"].terms) == actor_terms_before
  assert metadata["actor_context_fields_added"] == 0
  assert cfg.actions["joint_pos"].deployment_action_gain == first["target"][
    "action_gain"
  ]
  assert cfg.commands["twist"].command_delay_range_s == (
    first["target"]["command_delay_s"],
    first["target"]["command_delay_s"],
  )

  previous = generate_failure_focused_context(1000)
  calibrated = copy.deepcopy(first)
  calibrated["calibration"] = {
    "kind": FAILURE_FOCUSED_CALIBRATION_KIND,
    "selection_metric_fields": ["success_rate"],
    "adapted_policy_evaluations_used": False,
    "success_rate_bounds": [0.75, 0.85],
    "candidate_seeds": [1000, 1001],
    "attempts": [
      {
        "candidate_seed": 1000,
        "parameters_sha256": previous["parameters_sha256"],
        "base_policy_only": True,
        "num_episodes": 128,
        "success_rate": 0.70,
      },
      {
        "candidate_seed": 1001,
        "parameters_sha256": first["parameters_sha256"],
        "base_policy_only": True,
        "num_episodes": 128,
        "success_rate": 0.80,
      },
    ],
    "selected_candidate_seed": 1001,
    "selected_parameters_sha256": first["parameters_sha256"],
  }
  assert validate_calibrated_deployment_context(calibrated)["calibration"][
    "selected_candidate_seed"
  ] == 1001
  invalid = copy.deepcopy(calibrated)
  invalid["calibration"]["attempts"][0]["success_rate"] = 0.80
  with pytest.raises(ValueError, match="skipped an earlier qualifying"):
    validate_calibrated_deployment_context(invalid)


def test_brief_target_gate_uses_only_point_score_fall_and_small_kl() -> None:
  old = {
    "success_rate": 0.50,
    "fall_rate": 0.10,
    "intervention_per_riser": 0.20,
    "initial_state_signatures": ["paired-seed"],
  }
  candidate = {
    "success_rate": 0.53,
    "fall_rate": 0.11,
    "intervention_per_riser": 0.10,
    "initial_state_signatures": ["paired-seed"],
  }
  score = brief_target_score(candidate)
  assert score == pytest.approx(
    {
      "success": 0.53,
      "fall": -0.11,
      "intervention_per_riser": -0.001,
      "total": 0.419,
    }
  )
  accepted, reasons, scores = brief_candidate_gate(
    update_metrics={"mean_kl": 0.005},
    old_eval=old,
    candidate_eval=candidate,
    parameters_finite=True,
  )
  assert accepted
  assert reasons == []
  assert scores["candidate"]["total"] > scores["old"]["total"]

  rejected, reasons, _ = brief_candidate_gate(
    update_metrics={"mean_kl": 0.005},
    old_eval=old,
    candidate_eval={**candidate, "fall_rate": 0.131},
    parameters_finite=True,
  )
  assert not rejected
  assert any("3 percentage points" in reason for reason in reasons)


def test_brief_precheck_requires_strict_kl_below_one_percent() -> None:
  thresholds = BriefPpoGateThresholds()
  assert brief_candidate_precheck(
    update_metrics={"mean_kl": 0.009999},
    parameters_finite=True,
    thresholds=thresholds,
  ) == []
  assert brief_candidate_precheck(
    update_metrics={"mean_kl": 0.01},
    parameters_finite=True,
    thresholds=thresholds,
  ) == ["update KL is not below 0.01"]
  assert brief_candidate_precheck(
    update_metrics={"mean_kl": 0.001},
    parameters_finite=False,
    thresholds=thresholds,
  ) == ["non-finite model parameters"]


def test_brief_d0_gate_checks_baseline_minus_five_points_periodically() -> None:
  baseline = {
    "success_rate": 0.80,
    "initial_state_signatures": ["d0-pair"],
  }
  passed, reasons = brief_d0_retention_gate(
    baseline_eval=baseline,
    candidate_eval={
      "success_rate": 0.75,
      "initial_state_signatures": ["d0-pair"],
    },
  )
  assert passed
  assert reasons == []
  passed, reasons = brief_d0_retention_gate(
    baseline_eval=baseline,
    candidate_eval={
      "success_rate": 0.749,
      "initial_state_signatures": ["d0-pair"],
    },
  )
  assert not passed
  assert any("5 percentage points" in reason for reason in reasons)


def test_brief_failure_and_success_samples_weight_one_reward_advantage() -> None:
  class _Storage:
    advantages = torch.tensor([[[1.0], [2.0]], [[-3.0], [4.0]]])

  algorithm = object.__new__(OnlineSafePPO)
  algorithm.storage = _Storage()
  algorithm.hard_case_transitions = torch.tensor(
    [[False, True], [True, False]]
  )
  algorithm.success_counterexample_transitions = torch.tensor(
    [[True, False], [False, True]]
  )
  algorithm.hard_case_policy_weight = 0.5
  algorithm.success_counterexample_policy_weight = 1.5
  algorithm.last_update_metrics = {}
  metrics = algorithm.prepare_brief_advantages()
  assert torch.equal(
    algorithm.storage.advantages.squeeze(-1),
    torch.tensor([[1.5, 1.0], [-1.5, 6.0]]),
  )
  assert metrics["brief_single_reward_advantage"] == 1.0
  assert metrics["hard_case_transition_fraction"] == 0.5
  assert metrics["success_counterexample_transition_fraction"] == 0.5


def test_cost_gae_is_separate_and_stops_at_episode_boundaries() -> None:
  costs = torch.tensor([[0.0], [1.0], [0.0], [2.0]])
  values = torch.zeros_like(costs)
  dones = torch.tensor([[False], [True], [False], [False]])
  advantages, returns = generalized_cost_advantage(
    costs,
    values,
    torch.zeros(1),
    dones,
    gamma=1.0,
    lam=1.0,
  )
  assert torch.equal(advantages[:, 0], torch.tensor([1.0, 1.0, 2.0, 2.0]))
  assert torch.equal(returns, advantages)


def test_projected_dual_update_respects_budget_and_bounds() -> None:
  assert abs(
    projected_lagrange_update(
      0.2, 0.3, 0.1, learning_rate=2.0, maximum=1.0
    )
    - 0.6
  ) < 1.0e-12
  assert projected_lagrange_update(
    0.2, 0.0, 0.5, learning_rate=2.0, maximum=1.0
  ) == 0.0
  assert projected_lagrange_update(
    0.9, 1.0, 0.0, learning_rate=2.0, maximum=1.0
  ) == 1.0


def test_stage_balanced_retention_bank_is_actor_only_and_round_robin() -> None:
  quotas = balanced_stage_quotas(20_003, 6)
  assert sum(quotas) == 20_003
  assert max(quotas) - min(quotas) == 1
  stages = [
    torch.stack(
      (
        torch.full((count,), float(stage)),
        torch.arange(count, dtype=torch.float32),
      ),
      dim=1,
    )
    for stage, count in enumerate(quotas)
  ]
  bank = interleave_stage_observations(
    stages, generator=torch.Generator().manual_seed(12)
  )
  # Every complete group of six is exactly stage-balanced. Stage labels are
  # used only in this construction test; production payloads store no labels.
  for start in range(0, 120, 6):
    assert set(bank[start : start + 6, 0].tolist()) == set(range(6))
  payload = {
    "schema_version": RETENTION_BANK_SCHEMA_VERSION,
    "kind": RETENTION_BANK_KIND,
    "domain": "D0",
    "task": "Unitree-G1-Stairs-Online-D0",
    "seed": 17,
    "runtime_filter": True,
    "policy_mode": "deterministic_mean",
    "actor_observation_key": "actor",
    "actor_observation_dim": 2,
    "contains_privileged_observations": False,
    "num_stages": 6,
    "stage_counts": list(quotas),
    "ordering": "stage_round_robin_v1",
    "checkpoint_sha256": "abc",
    "observation_sha256": actor_observation_sha256(bank),
    "observations": bank,
  }
  validated, metadata = validate_retention_observation_bank(
    payload, expected_actor_dim=2, expected_domain="D0"
  )
  assert torch.equal(validated, bank)
  assert metadata["contains_privileged_observations"] is False
  assert set(metadata).isdisjoint({"observations", "online_privileged"})
  privileged = dict(payload, contains_privileged_observations=True)
  with pytest.raises(ValueError, match="exclude privileged"):
    validate_retention_observation_bank(privileged)


def test_retention_bank_cyclic_batch_and_budget_adaptation_are_deterministic() -> None:
  observations = torch.arange(15, dtype=torch.float32).reshape(5, 3)
  batch, cursor = cyclic_retention_batch(
    observations, cursor=4, batch_size=3
  )
  assert torch.equal(batch, observations[torch.tensor([4, 0, 1])])
  assert cursor == 2
  assert increase_anchor_weight_on_budget_violation(
    0.02, 0.001, 0.002, learning_rate=10.0, maximum=0.2
  ) == 0.02
  assert abs(
    increase_anchor_weight_on_budget_violation(
      0.02, 0.004, 0.002, learning_rate=10.0, maximum=0.2
    )
    - 0.04
  ) < 1.0e-12
  assert increase_anchor_weight_on_budget_violation(
    0.19, 0.2, 0.0, learning_rate=10.0, maximum=0.2
  ) == 0.2


def test_retention_anchor_gradient_does_not_advance_policy_rng() -> None:
  class _Actor(torch.nn.Module):
    obs_groups = ("actor",)
    obs_dim = 2

    def __init__(self):
      super().__init__()
      self.mlp = torch.nn.Linear(2, 1, bias=False)
      self._params = None

    def forward(self, obs, stochastic_output=False):
      mean = self.mlp(obs["actor"])
      std = torch.full_like(mean, 0.2)
      self._params = (mean, std)
      if stochastic_output:
        return mean + std * torch.randn_like(mean)
      return mean

    @property
    def output_distribution_params(self):
      return self._params

    @staticmethod
    def get_kl_divergence(old_params, new_params):
      old_mean, old_std = old_params
      new_mean, new_std = new_params
      return (
        torch.log(new_std / old_std)
        + (old_std.square() + (old_mean - new_mean).square())
        / (2.0 * new_std.square())
        - 0.5
      ).sum(dim=-1)

  algorithm = object.__new__(OnlineSafePPO)
  algorithm.actor = _Actor()
  algorithm.retention_actor_reference = copy.deepcopy(algorithm.actor)
  for parameter in algorithm.retention_actor_reference.parameters():
    parameter.requires_grad_(False)
  with torch.no_grad():
    algorithm.actor.mlp.weight.add_(0.1)
  algorithm.device = "cpu"
  algorithm.retention_anchor_banks = {
    "d0": torch.arange(40, dtype=torch.float32).reshape(20, 2) / 40.0
  }
  algorithm.retention_anchor_cursors = {"d0": 0, "neighbor": 0}
  algorithm.retention_anchor_batch_size = 8

  torch.manual_seed(99)
  before = torch.get_rng_state().clone()
  loss = algorithm._retention_anchor_loss("d0")
  assert torch.equal(torch.get_rng_state(), before)
  assert loss > 0.0
  loss.backward()
  assert algorithm.actor.mlp.weight.grad is not None
  assert float(algorithm.actor.mlp.weight.grad.abs().sum()) > 0.0
  before = torch.get_rng_state().clone()
  assert algorithm._full_retention_anchor_kl("d0") > 0.0
  assert torch.equal(torch.get_rng_state(), before)


def test_future_risk_and_success_gates_do_not_cross_resets() -> None:
  events = torch.tensor([[False], [False], [True], [False], [True]])
  dones = torch.tensor([[False], [True], [False], [False], [False]])
  labels = future_event_labels(events, dones, horizon=3)
  assert torch.equal(
    labels[:, 0], torch.tensor([False, False, True, True, True])
  )

  intervened = torch.tensor([[True], [True], [True], [False], [True]])
  risers = torch.tensor([[0], [0], [1], [1], [2]])
  task_advantages = torch.tensor([[-1.0], [-1.0], [-1.0], [1.0], [1.0]])
  falls = torch.tensor([[False], [False], [False], [False], [True]])
  correction = success_gated_correction_mask(
    intervened,
    risers,
    task_advantages,
    dones,
    falls,
    horizon=3,
  )
  # Step 0 cannot borrow progress across the step-1 reset; step 1 can see the
  # next episode's current terminal transition only after its own boundary.
  assert torch.equal(
    correction[:, 0], torch.tensor([False, False, True, False, False])
  )


def test_risk_metrics_and_local_critic_readiness_are_explicit() -> None:
  risk = binary_risk_metrics(
    torch.tensor([-5.0, -2.0, 2.0, 5.0]),
    torch.tensor([False, False, True, True]),
  )
  assert risk["auc"] == 1.0
  assert risk["brier"] < 0.01
  tied = binary_risk_metrics(
    torch.zeros(4), torch.tensor([False, False, True, True])
  )
  assert tied["auc"] == 0.5
  diagnostics = {
    "critic_calibration_by_riser": {
      "7": {"count": 32},
      "8": {"count": 32},
      "9": {"count": 32},
    },
    "pre_fall_value_event_count": 4,
    "pre_fall_cost_value_delta_after_update": 0.2,
    "risk_prediction_after_update": risk,
  }
  assert critic_readiness_reasons(
    diagnostics,
    late_risers=(7, 8, 9),
    minimum_samples_per_riser=16,
    minimum_fall_events=2,
    maximum_risk_brier=0.25,
    minimum_risk_auc=0.60,
    minimum_pre_fall_cost_rise=0.0,
  ) == []


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


def test_task_first_gate_rejects_intervention_only_candidate() -> None:
  old = _with_paired_signatures({
    "D0": {
      "success_rate": 0.95,
      "fall_rate": 0.01,
      "mean_return": 10.0,
      "intervention_per_riser": 0.2,
    },
    "DQH": {
      "success_rate": 0.90,
      "fall_rate": 0.10,
      "mean_return": 8.0,
      "intervention_per_riser": 1.0,
    },
    "DQNH": {
      "success_rate": 0.90,
      "fall_rate": 0.10,
      "mean_return": 8.0,
      "intervention_per_riser": 1.0,
    },
  })
  candidate = _with_paired_signatures({
    domain: dict(result) for domain, result in old.items()
  })
  candidate["DQH"]["intervention_per_riser"] = 0.5
  accepted, reasons = candidate_gate(
    update_metrics={
      "mean_kl": 0.001,
      "clip_fraction": 0.0,
      "action_saturation_fraction": 0.1,
    },
    old_eval=old,
    candidate_eval=candidate,
    base_d0_success=0.95,
    total_kl_from_base=0.001,
    parameters_finite=True,
    thresholds=CandidateGateThresholds(
      maximum_target_fall_rate=1.0,
      require_task_improvement=True,
    ),
    target_domain="DQH",
    neighbor_domain="DQNH",
  )
  assert not accepted
  assert "target task metrics show no strict improvement" in reasons


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


def test_candidate_gate_rejects_d0_fall_regression() -> None:
  old = _with_paired_signatures({
    "D0": {"success_rate": 0.95, "fall_rate": 0.0, "intervention_per_riser": 0.1},
    "D4": {"success_rate": 0.5, "fall_rate": 0.5, "intervention_per_riser": 1.0},
    "D5": {"success_rate": 0.6, "fall_rate": 0.4, "intervention_per_riser": 1.0},
  })
  candidate = _with_paired_signatures({
    "D0": {"success_rate": 0.96, "fall_rate": 0.1, "intervention_per_riser": 0.1},
    "D4": {"success_rate": 0.6, "fall_rate": 0.4, "intervention_per_riser": 0.9},
    "D5": old["D5"] | {},
  })
  accepted, reasons = candidate_gate(
    update_metrics={
      "mean_kl": 0.001,
      "clip_fraction": 0.0,
      "action_saturation_fraction": 0.1,
    },
    old_eval=old,
    candidate_eval=candidate,
    base_d0_success=0.95,
    total_kl_from_base=0.001,
    parameters_finite=True,
    thresholds=CandidateGateThresholds(maximum_target_fall_rate=1.0),
  )
  assert not accepted
  assert "D0 fall rate increased" in reasons


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
  assert runner.algorithm.d0_retention_anchor_weight == 0.0
  assert runner.algorithm.neighbor_retention_anchor_weight == 0.0
  assert runner.algorithm.retention_anchor_batch_size == 4096
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
  assert d4.rewards["online_safety_telemetry"].weight == 1.0
  assert d4.rewards["online_safety_telemetry"].func is mdp.online_safety_telemetry

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


def test_human_centerline_feedback_applies_a_wrapped_heading_reference_bias() -> None:
  command = centerline_feedback_command(
    torch.tensor([0.4, 0.4]),
    torch.zeros(2),
    torch.tensor([0.0, 3.0]),
    lateral_gain=0.8,
    heading_gain=1.4,
    lateral_deadband=0.04,
    heading_deadband=0.03,
    max_lateral_velocity=0.16,
    max_yaw_velocity=0.45,
    heading_reference_bias=0.4,
  )
  assert torch.allclose(command[:, 2], torch.tensor([0.45, -0.45]))
  assert torch.equal(command[:, 1], torch.zeros(2))


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
  extrapolated = backtrack_actor_state(base, candidate, 1.5)
  assert torch.equal(extrapolated["mlp.0.weight"], torch.tensor([6.0, 8.0]))
  assert torch.equal(
    extrapolated["distribution.std_param"], candidate["distribution.std_param"]
  )
  reference = backtrack_actor_state(base, candidate, 0.0)
  assert torch.equal(reference["mlp.0.weight"], base["mlp.0.weight"])
  assert torch.equal(reference["distribution.std_param"], candidate["distribution.std_param"])
  with pytest.raises(ValueError, match="line-search fraction"):
    backtrack_actor_state(base, candidate, 1.5001)


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
      self.optimizer_resets = 0

    def reset_online_optimizer(self) -> None:
      self.optimizer_resets += 1

  runner = object.__new__(OnlineSafeRefinementRunner)
  runner.alg = AlgorithmStub()
  runner.reduce_after_rejection()
  assert runner.alg.optimizer_resets == 1


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


def test_legacy_hard_case_shape_mismatch_is_detected_before_restore() -> None:
  current = {
    "actor/history": torch.zeros(4, 5, 81),
    "privileged/history": torch.zeros(4, 1, 150),
  }
  legacy = {
    "actor/history": torch.zeros(2, 5, 81),
    "privileged/history": torch.zeros(2, 1, 138),
  }
  mismatches = hard_case_state_shape_mismatches(current, legacy)
  assert mismatches == [
    "privileged/history: replay (1, 138) != current (1, 150)"
  ]


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
  recomputed = stored + torch.tensor([2.7e-4, -3.5e-4])
  error = validate_behavior_log_prob(stored, recomputed)
  assert 2.0e-4 < error < 5.0e-4

  legal_v31_reduction_noise = stored + torch.tensor([7.5e-4, 0.0])
  assert validate_behavior_log_prob(stored, legal_v31_reduction_noise) < 1.0e-3

  invalid = stored + torch.tensor([1.2e-3, 0.0])
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

  gpu_roundoff = (stored[0].clone(), stored[1].clone())
  gpu_roundoff[0][0, 0] = 1.1e-5
  assert (
    1.0e-5
    < validate_behavior_distribution_params(stored, gpu_roundoff)
    < 2.0e-5
  )

  mismatched = (stored[0].clone(), stored[1].clone())
  mismatched[0][0, 0] = 3.0e-5
  try:
    validate_behavior_distribution_params(stored, mismatched)
  except RuntimeError as exc:
    assert "distribution is inconsistent" in str(exc)
  else:
    raise AssertionError("a true Gaussian parameter mismatch was not rejected")


def test_failure_focused_brief_ppo_accepts_only_declared_layer_profiles() -> None:
  assert brief_actor_layer_profile_is_valid(
    (1.0, 1.0, 1.0, 1.0), failure_focused=False
  )
  assert brief_actor_layer_profile_is_valid(
    (1.0, 1.0, 1.0, 1.0), failure_focused=True
  )
  assert brief_actor_layer_profile_is_valid(
    (0.10, 0.25, 0.50, 1.0), failure_focused=True
  )
  assert not brief_actor_layer_profile_is_valid(
    (0.10, 0.25, 0.50, 1.0), failure_focused=False
  )
  assert not brief_actor_layer_profile_is_valid(
    (0.10, 0.25, 0.50), failure_focused=True
  )


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


def test_retention_reference_restore_ignores_transient_inference_distribution() -> None:
  class _Distribution(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self._distribution = None

  class _Actor(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.mlp = torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))
      self.distribution = _Distribution()

  algorithm = object.__new__(OnlineSafePPO)
  algorithm.actor = _Actor()
  algorithm.retention_actor_reference = None
  with torch.inference_mode():
    cached = algorithm.actor.mlp[0].weight.view(-1)
    algorithm.actor.distribution._distribution = cached
  with torch.no_grad():
    algorithm.actor.mlp[0].weight.add_(1.0)
  state = {
    key: value.detach().clone()
    for key, value in algorithm.actor.state_dict().items()
  }
  algorithm._set_retention_actor_reference(state)
  reference = algorithm.retention_actor_reference
  assert reference is not None
  assert algorithm.actor.distribution._distribution is cached
  assert all(not parameter.requires_grad for parameter in reference.parameters())
  identity = id(reference)
  algorithm._set_retention_actor_reference(state)
  assert id(algorithm.retention_actor_reference) == identity


def test_v19_grouped_advantages_normalize_before_replay_weighting() -> None:
  advantages = torch.tensor([1.0, 3.0, 10.0, 14.0, -5.0, 1.0])
  failure = torch.tensor([False, False, True, True, False, False])
  success = torch.tensor([False, False, False, False, True, True])
  weighted, metrics = normalize_v19_grouped_advantages(
    advantages, failure, success
  )
  normal = ~(failure | success)
  assert float(weighted[normal].mean()) == pytest.approx(0.0, abs=1e-7)
  assert float(weighted[failure].mean()) == pytest.approx(0.0, abs=1e-7)
  assert float(weighted[success].mean()) == pytest.approx(0.0, abs=1e-7)
  assert float(weighted[normal].std(unbiased=False)) == pytest.approx(1.0)
  assert float(weighted[failure].std(unbiased=False)) == pytest.approx(1.0)
  assert float(weighted[success].std(unbiased=False)) == pytest.approx(1.25)
  assert metrics["v19_failure_policy_weight"] == 1.0
  assert metrics["v19_success_policy_weight"] == 1.25


def test_v21_local_preservation_excludes_success_only_when_beta_is_positive() -> None:
  surrogate = torch.tensor([1.0, 3.0, 50.0, 70.0], requires_grad=True)
  kl = torch.tensor([0.1, 0.2, 0.5, 0.7], requires_grad=True)
  success = torch.tensor([False, False, True, True])

  control_ppo, control_kl, control_total = local_matched_success_actor_loss(
    surrogate, kl, success, beta=0.0
  )
  assert float(control_ppo) == pytest.approx(31.0)
  assert float(control_kl) == 0.0
  assert control_total is control_ppo

  ppo, preservation, total = local_matched_success_actor_loss(
    surrogate, kl, success, beta=4.0
  )
  assert float(ppo) == pytest.approx(2.0)
  assert float(preservation) == pytest.approx(0.6)
  assert float(total) == pytest.approx(4.4)
  total.backward()
  assert torch.equal(surrogate.grad, torch.tensor([0.5, 0.5, 0.0, 0.0]))
  assert torch.equal(kl.grad, torch.tensor([0.0, 0.0, 2.0, 2.0]))


def test_v21_local_preservation_rejects_missing_actor_groups() -> None:
  terms = torch.ones(4)
  with pytest.raises(RuntimeError, match="empty group"):
    local_matched_success_actor_loss(
      terms, terms, torch.ones(4, dtype=torch.bool), beta=1.0
    )
  with pytest.raises(ValueError, match="non-negative"):
    local_matched_success_actor_loss(
      terms, terms, torch.zeros(4, dtype=torch.bool), beta=-1.0
    )


def test_v19_zero_column_actor_expansion_preserves_every_legacy_output() -> None:
  torch.manual_seed(1901)
  legacy = torch.nn.Sequential(
    torch.nn.Linear(405, 8),
    torch.nn.Tanh(),
    torch.nn.Linear(8, 3),
  )
  expanded = torch.nn.Sequential(
    torch.nn.Linear(410, 8),
    torch.nn.Tanh(),
    torch.nn.Linear(8, 3),
  )
  legacy_actor = torch.nn.Module()
  legacy_actor.mlp = legacy
  expanded_actor = torch.nn.Module()
  expanded_actor.mlp = expanded
  runner = SimpleNamespace(alg=SimpleNamespace(actor=expanded_actor))
  proof = OnlineSafeRefinementRunner._load_actor_with_expansion(
    runner, legacy_actor.state_dict()
  )
  old_observation = torch.randn(32, 405)
  arbitrary_new_features = torch.randn(32, 5) * 100.0
  with torch.no_grad():
    expected = legacy_actor.mlp(old_observation)
    actual = expanded_actor.mlp(
      torch.cat((old_observation, arbitrary_new_features), dim=1)
    )
  # The parameter mapping is bit-exact.  Different-width GEMM kernels may
  # reorder floating-point accumulation even though the five added products
  # are exactly zero, so compare the realized outputs at one-ULP scale.
  torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-6)
  assert proof["pi0_exact_preservation_proof"] is True
  assert proof["legacy_tensor_copy_max_abs_error"] == 0.0
  assert proof["new_first_layer_column_max_abs"] == 0.0


def test_v19_contexts_are_observable_and_contact_context_is_mechanism_pure() -> None:
  contexts = {
    "lateral": generate_v19_specialist_context("lateral", 4107),
    "contact_stability": generate_v19_specialist_context(
      "contact_stability", 4207
    ),
  }
  assert contexts["lateral"]["parameters_sha256"] != contexts[
    "contact_stability"
  ]["parameters_sha256"]
  contact = contexts["contact_stability"]
  target = contact["target"]
  scenario = contact["scenario"]
  assert target["num_steps"] == 24
  assert target["action_bias"] == (0.0,) * 12
  assert target["encoder_bias"] == 0.0
  assert target["action_delay_steps"] == 0
  assert target["command_delay_s"] == 0.0
  assert target["command_forward_scale"] == 1.0
  assert scenario["foot_friction"] >= 0.35
  assert scenario["contact_observation_delay_steps"] in (1, 2)
  assert abs(scenario["left_response_scale"] - 1.0) <= 0.04
  assert abs(scenario["right_response_scale"] - 1.0) <= 0.04
  assert abs(scenario["left_response_scale"] - scenario["right_response_scale"]) <= 0.04
  assert scenario["lateral_command_bias"] == 0.0
  assert scenario["yaw_command_bias"] == 0.0

  for mode, context in contexts.items():
    minimum_purity = 0.80 if mode == "lateral" else 0.75
    maximum_second = 0.30 if mode == "lateral" else 0.20
    attempt = {
      "candidate_seed": context["calibration_candidate_seed"],
      "parameters_sha256": context["parameters_sha256"],
      "base_policy_only": True,
      "num_episodes": 512,
      "success_rate": 0.80,
      "fall_count": 103,
      "target_failure_type": V19_SPECIALIST_FAILURE_TYPES[mode],
      "target_failure_fraction": minimum_purity + 0.01,
      "second_failure_fraction": maximum_second - 0.01,
      "qualifies": True,
    }
    context["calibration"] = {
      "kind": V19_CALIBRATION_KIND,
      "success_rate_bounds": [0.70, 0.85],
      "minimum_target_failure_fraction": minimum_purity,
      "maximum_second_failure_fraction": maximum_second,
      "minimum_fall_count": 100,
      "episodes_per_candidate": 512,
      "candidate_seeds": [context["calibration_candidate_seed"]],
      "attempts": [attempt],
      "selected_candidate_seed": context["calibration_candidate_seed"],
      "selected_parameters_sha256": context["parameters_sha256"],
      "adapted_policy_evaluations_used": False,
    }
    assert validate_calibrated_v19_context(context)["specialist_mode"] == mode


def test_v19_failure_classifier_uses_contact_mechanism_not_attitude_outcome() -> None:
  common = {
    "side_edge_breach": False,
    "max_abs_centerline_error": 0.1,
    "max_abs_heading_error": 0.1,
    "correction_max": 0.1,
    "stair_half_width": 1.2,
  }
  assert classify_v19_failure_mode(
    **common,
    specialist_mode="contact_stability",
    maximum_left_slip_speed=0.35,
    maximum_right_slip_speed=0.1,
    mean_contact_mismatch=0.1,
  ) == "contact_stability"
  assert classify_v19_failure_mode(
    **common,
    specialist_mode="contact_stability",
    maximum_left_slip_speed=0.8,
    maximum_right_slip_speed=0.7,
    mean_contact_mismatch=0.4,
    first_contact_event_step=20,
    first_lateral_event_step=80,
  ) == "contact_stability"
  assert classify_v19_failure_mode(
    **common,
    specialist_mode="contact_stability",
    maximum_left_slip_speed=0.1,
    maximum_right_slip_speed=0.1,
    mean_contact_mismatch=0.35,
  ) == "contact_stability"
  assert classify_v19_failure_mode(
    **{**common, "correction_max": 0.6},
    specialist_mode="contact_stability",
    maximum_left_slip_speed=0.1,
    maximum_right_slip_speed=0.1,
    mean_contact_mismatch=0.1,
  ) == "non_lateral_high_cbf_demand"

  # Stair slip can mediate an injected lateral disturbance. In the lateral
  # family an observed geometric lateral event remains primary even if the
  # severe slip threshold happened a few steps earlier.
  assert classify_v19_failure_mode(
    **common,
    specialist_mode="lateral",
    maximum_left_slip_speed=0.8,
    maximum_right_slip_speed=0.7,
    mean_contact_mismatch=0.4,
    first_contact_event_step=20,
    first_lateral_event_step=80,
  ) == "lateral_heading_drift"


def test_v19_banks_select_lateral_strata_and_touchdown_centered_contact_states() -> None:
  length = 201
  time = torch.linspace(0.0, 1.0, length)
  riser = torch.floor(1.0 + 10.0 * time).long()
  phase = (torch.arange(length, dtype=torch.float32) * 0.01) % 1.0
  support = (torch.arange(length) % 2).long()
  command = torch.stack((0.4 + 0.05 * time, 0.03 * time, 0.1 * time), dim=1)
  velocity = torch.stack((0.3 + 0.05 * time, 0.02 * time, 0.0 * time), dim=1)
  cbf = torch.zeros(length, dtype=torch.bool)
  signed_wave = torch.sin(torch.linspace(-2.0 * math.pi, 2.0 * math.pi, length))
  components = {
    "centerline": signed_wave.abs(),
    "heading": torch.cos(torch.linspace(0.0, 3.0 * math.pi, length)).abs(),
    "centerline_signed": signed_wave,
    "heading_signed": torch.cos(torch.linspace(0.0, 3.0 * math.pi, length)),
    "centerline_rate": torch.gradient(signed_wave)[0] * 30.0,
    "heading_rate": torch.gradient(torch.cos(torch.linspace(0.0, 3.0 * math.pi, length)))[0] * 30.0,
    "contact_mismatch": 0.1 + 0.5 * time,
    "left_slip": torch.zeros(length),
    "right_slip": torch.zeros(length),
  }
  lateral = select_v19_lateral_failure_candidates(
    riser,
    components,
    phase,
    support,
    command,
    velocity,
    cbf,
    minimum_riser=1,
    total_risers=11,
    maximum_candidates=16,
  )
  assert lateral
  assert {candidate.centerline_sign for candidate in lateral} == {-1, 1}
  assert {candidate.heading_sign for candidate in lateral} == {-1, 1}
  successes = select_v19_lateral_success_candidates(
    riser,
    components,
    phase,
    support,
    command,
    velocity,
    cbf,
    minimum_riser=1,
    total_risers=11,
  )
  assert successes

  touchdown = torch.zeros(length, 2, dtype=torch.bool)
  touchdown[60, 0] = True
  touchdown[140, 1] = True
  phase[60] = 0.90  # early relative to left touchdown at phase zero
  phase[140] = 0.60  # delayed relative to right touchdown at phase 0.5
  components["left_slip"][60:121] = 0.6
  components["right_slip"][140:201] = 0.7
  contact = select_v19_contact_candidates(
    riser,
    components,
    phase,
    support,
    command,
    velocity,
    cbf,
    touchdown,
    minimum_riser=1,
    outcome="failure",
    maximum_candidates=16,
  )
  assert contact
  assert {candidate.touchdown_foot for candidate in contact} == {0, 1}
  assert {candidate.slip_foot for candidate in contact} == {0, 1}
  assert {candidate.contact_timing for candidate in contact} == {"early", "delayed"}
  assert {candidate.contact_window_side for candidate in contact} == {"pre", "post"}


def test_v19_restart_pairs_are_exact_and_balance_sparse_lateral_signs() -> None:
  count = 14
  state = {
    "robot/root_pose_relative": torch.arange(
      count * 7, dtype=torch.float32
    ).reshape(count, 7),
    "terrain/type": torch.zeros(count, dtype=torch.long),
  }
  failure_bank = HardCaseStateBank(
    capacity=32,
    bank_kind=SPECIALIST_FAILURE_BANK_KIND,
    source_domain="DQHMED",
    context_sha256="context",
    specialist_mode="lateral",
  )
  success_pool = HardCaseStateBank(
    capacity=32,
    bank_kind=SPECIALIST_SUCCESS_POOL_KIND,
    source_domain="DQHMED",
    context_sha256="context",
    specialist_mode="lateral",
  )
  success_bank = HardCaseStateBank(
    capacity=32,
    bank_kind=SPECIALIST_SUCCESS_BANK_KIND,
    source_domain="DQHMED",
    context_sha256="context",
    specialist_mode="lateral",
  )
  for env_id in range(count):
    rare = env_id >= 12
    centerline_sign, heading_sign = ((-1, 1) if rare else (1, -1))
    common = {
      "history_index": 0,
      "steps_before_terminal": 50 + env_id,
      "riser_index": 4 + env_id % 3,
      "gait_phase": 0.1 + 0.25 * (env_id % 3),
      "support_foot": env_id % 2,
      "delivered_command": (0.4, 0.02, 0.1),
      "root_velocity": (0.3, 0.01, 0.0),
      "cbf_active": False,
      "balance_bucket": f"pair:{env_id}",
      "selection_signal": 0.5,
      "centerline_sign": centerline_sign,
      "heading_sign": heading_sign,
      "error_growth_rate": 0.1 if env_id % 2 else 0.4,
      "riser_stage": ("early", "mid", "late")[env_id % 3],
    }
    failure = SpecialistBankCandidate(
      **common,
      priority=10.0 + env_id,
      outcome="failure",
      failure_type=LATERAL_HEADING_DRIFT_FAILURE_TYPE,
    )
    success = SpecialistBankCandidate(
      **common,
      priority=8.0 + env_id,
      outcome="success",
      failure_type="mixed",
    )
    observation = torch.tensor([float(env_id), 0.5, -0.5])
    assert failure_bank.add_specialist_candidate(
      state, env_id, failure, observation
    ) == 1
    assert success_pool.add_specialist_candidate(
      state, env_id, success, observation + 0.001
    ) == 1
  matching = match_v19_success_counterexamples(
    failure_bank, success_pool, success_bank
  )
  assert matching["one_match_per_replayed_failure"]

  first = select_v19_balanced_restart_pairs(
    failure_bank,
    success_bank,
    8,
    generator=torch.Generator().manual_seed(19),
  )
  second = select_v19_balanced_restart_pairs(
    failure_bank,
    success_bank,
    8,
    generator=torch.Generator().manual_seed(19),
  )
  assert first == second
  failure_indices, success_indices, audit = first
  assert audit["pair_count"] == 8
  assert audit["exact_match_passed"] is True
  assert sum(audit["primary_stratum_counts"].values()) == 8
  assert audit["primary_marginal_counts"] == {
    "centerline_sign": {"-1": 4, "1": 4},
    "heading_sign": {"-1": 4, "1": 4},
    "riser_stage": {"early": 3, "late": 2, "mid": 3},
    "support_foot": {"0": 4, "1": 4},
    "error_growth_bin": {"high": 4, "low": 4},
  }
  for failure_index, success_index in zip(
    failure_indices, success_indices, strict=True
  ):
    assert (
      success_bank.entries[success_index].matched_failure_index
      == failure_index
    )


def test_v19_input_adapter_freezes_legacy_gradient_columns() -> None:
  gradient = torch.arange(28, dtype=torch.float32).reshape(4, 7)
  masked = mask_legacy_actor_input_gradient(gradient, 3)
  assert torch.count_nonzero(masked[:, :4]) == 0
  assert torch.equal(masked[:, 4:], gradient[:, 4:])
  assert torch.equal(gradient, torch.arange(28, dtype=torch.float32).reshape(4, 7))
  with pytest.raises(ValueError, match="two-dimensional"):
    mask_legacy_actor_input_gradient(torch.ones(7), 3)
  with pytest.raises(ValueError, match="inside the input width"):
    mask_legacy_actor_input_gradient(gradient, 7)


def test_v19_contact_restart_quotas_balance_each_observable_margin() -> None:
  failure_bank = HardCaseStateBank(
    capacity=16,
    bank_kind=SPECIALIST_FAILURE_BANK_KIND,
    source_domain="DQHMED",
    context_sha256="context",
    specialist_mode="contact_stability",
  )
  success_bank = HardCaseStateBank(
    capacity=16,
    bank_kind=SPECIALIST_SUCCESS_BANK_KIND,
    source_domain="DQHMED",
    context_sha256="context",
    specialist_mode="contact_stability",
  )
  strata = (
    (0, 0, "early", 0),
    (0, 0, "delayed", 1),
    (0, 1, "early", 1),
    (1, 0, "delayed", 0),
    (1, 1, "early", 0),
    (1, 1, "delayed", 1),
  )
  for index, (touchdown, slip, timing, support) in enumerate(strata):
    common = {
      "state": {"state": torch.tensor([[float(index)]])},
      "priority": 1.0 + index,
      "riser_index": 5,
      "terrain_type": 0,
      "specialist_mode": "contact_stability",
      "gait_phase": 0.25,
      "support_foot": support,
      "delivered_command": (0.4, 0.0, 0.0),
      "root_velocity": (0.3, 0.0, 0.0),
      "cbf_active": False,
      "actor_observation": torch.tensor([float(index)]),
      "touchdown_foot": touchdown,
      "slip_foot": slip,
      "contact_timing": timing,
      "contact_window_side": "pre" if index % 2 else "post",
    }
    failure_bank.entries.append(HardCaseEntry(**common, outcome="failure"))
    success_bank.entries.append(
      HardCaseEntry(
        **common,
        outcome="success",
        matched_failure_index=index,
        match_distance=0.01,
        success_pool_index=index,
      )
    )
  _, _, audit = select_v19_balanced_restart_pairs(
    failure_bank,
    success_bank,
    6,
    generator=torch.Generator().manual_seed(29),
  )
  assert audit["exact_match_passed"] is True
  assert all(
    sorted(counts.values()) == [3, 3]
    for counts in audit["primary_marginal_counts"].values()
  )


def test_v19_exact_quota_fallback_escapes_a_balanced_local_optimum() -> None:
  failure_bank = HardCaseStateBank(
    capacity=16,
    bank_kind=SPECIALIST_FAILURE_BANK_KIND,
    source_domain="DQHMED",
    context_sha256="context",
    specialist_mode="lateral",
  )
  success_bank = HardCaseStateBank(
    capacity=16,
    bank_kind=SPECIALIST_SUCCESS_BANK_KIND,
    source_domain="DQHMED",
    context_sha256="context",
    specialist_mode="lateral",
  )
  strata = (
    (-1, 1, "late", 1, "low"),
    (-1, 1, "mid", 0, "high"),
    (-1, 1, "mid", 0, "low"),
    (1, -1, "early", 0, "high"),
    (1, -1, "late", 1, "high"),
    (1, -1, "mid", 1, "low"),
  )
  for index, (centerline, heading, stage, support, growth) in enumerate(strata):
    common = {
      "state": {"state": torch.tensor([[float(index)]])},
      "priority": 1.0 + index,
      "riser_index": 5,
      "terrain_type": 0,
      "specialist_mode": "lateral",
      "gait_phase": 0.25,
      "support_foot": support,
      "delivered_command": (0.4, 0.0, 0.0),
      "root_velocity": (0.3, 0.0, 0.0),
      "cbf_active": False,
      "actor_observation": torch.tensor([float(index)]),
      "centerline_sign": centerline,
      "heading_sign": heading,
      "riser_stage": stage,
      "error_growth_rate": 0.4 if growth == "high" else 0.1,
    }
    failure_bank.entries.append(HardCaseEntry(**common, outcome="failure"))
    success_bank.entries.append(
      HardCaseEntry(
        **common,
        outcome="success",
        matched_failure_index=index,
        match_distance=0.01,
        success_pool_index=index,
      )
    )
  _, _, audit = select_v19_balanced_restart_pairs(
    failure_bank,
    success_bank,
    12,
    generator=torch.Generator().manual_seed(31),
  )
  assert audit["quota_solver"] == "exact_search_fallback"
  assert audit["maximum_marginal_imbalance"] == 0
  assert audit["primary_marginal_counts"] == {
    "centerline_sign": {"-1": 6, "1": 6},
    "heading_sign": {"-1": 6, "1": 6},
    "riser_stage": {"early": 4, "late": 4, "mid": 4},
    "support_foot": {"0": 6, "1": 6},
    "error_growth_bin": {"high": 6, "low": 6},
  }


def test_v19_infeasible_joint_marginals_trigger_transactional_bank_rollback() -> None:
  failure_bank = HardCaseStateBank(
    capacity=32,
    bank_kind=SPECIALIST_FAILURE_BANK_KIND,
    source_domain="DQHMED",
    context_sha256="context",
    specialist_mode="lateral",
  )
  success_pool = HardCaseStateBank(
    capacity=32,
    bank_kind=SPECIALIST_SUCCESS_POOL_KIND,
    source_domain="DQHMED",
    context_sha256="context",
    specialist_mode="lateral",
  )
  success_bank = HardCaseStateBank(
    capacity=32,
    bank_kind=SPECIALIST_SUCCESS_BANK_KIND,
    source_domain="DQHMED",
    context_sha256="context",
    specialist_mode="lateral",
  )

  def load_strata(strata: tuple[tuple[int, int, str, int, str], ...]) -> None:
    failure_bank.clear()
    success_bank.clear()
    for index, (centerline, heading, stage, support, growth) in enumerate(strata):
      common = {
        "state": {"state": torch.tensor([[float(index)]])},
        "priority": 1.0 + index,
        "riser_index": 5,
        "terrain_type": 0,
        "specialist_mode": "lateral",
        "gait_phase": 0.25,
        "support_foot": support,
        "delivered_command": (0.4, 0.0, 0.0),
        "root_velocity": (0.3, 0.0, 0.0),
        "cbf_active": False,
        "actor_observation": torch.tensor([float(index)]),
        "centerline_sign": centerline,
        "heading_sign": heading,
        "riser_stage": stage,
        "error_growth_rate": 0.4 if growth == "high" else 0.1,
      }
      failure_bank.entries.append(HardCaseEntry(**common, outcome="failure"))
      success_bank.entries.append(
        HardCaseEntry(
          **common,
          outcome="success",
          matched_failure_index=index,
          match_distance=0.01,
          success_pool_index=index,
        )
      )

  feasible = (
    (-1, 1, "late", 1, "low"),
    (-1, 1, "mid", 0, "high"),
    (-1, 1, "mid", 0, "low"),
    (1, -1, "early", 0, "high"),
    (1, -1, "late", 1, "high"),
    (1, -1, "mid", 1, "low"),
  )
  load_strata(feasible)
  snapshots = (
    failure_bank.state_dict(),
    success_pool.state_dict(),
    success_bank.state_dict(),
  )
  assert v19_restart_pair_feasibility(failure_bank, success_bank, 12)[
    "passed"
  ]

  infeasible = (
    (-1, 1, "mid", 0, "low"),
    (-1, 1, "mid", 1, "high"),
    (-1, -1, "early", 0, "low"),
    (-1, -1, "late", 1, "high"),
    (1, -1, "early", 1, "high"),
    (1, -1, "late", 0, "low"),
  )
  load_strata(infeasible)
  failed = v19_restart_pair_feasibility(failure_bank, success_bank, 12)
  assert failed["passed"] is False
  assert failed["error"] == (
    "matched restart strata cannot realize balanced observable marginals"
  )
  mechanism_aligned = v19_restart_pair_feasibility(
    failure_bank,
    success_bank,
    12,
    balance_profile=RESTART_BALANCE_PROFILE_LATERAL_STAGE_SUPPORT_GROWTH,
  )
  assert mechanism_aligned["passed"] is True
  assert mechanism_aligned["audit"]["primary_marginal_counts"] == {
    "riser_stage": {"early": 4, "late": 4, "mid": 4},
    "support_foot": {"0": 6, "1": 6},
    "error_growth_bin": {"high": 6, "low": 6},
  }
  assert mechanism_aligned["audit"][
    "diagnostic_direction_marginal_counts"
  ] == {
    "centerline_sign": {"-1": 8, "1": 4},
    "heading_sign": {"-1": 8, "1": 4},
  }

  transaction = finalize_v19_replay_bank_update(
    failure_bank,
    success_pool,
    success_bank,
    snapshots,
    12,
  )
  assert transaction["attempted"] is True
  assert transaction["committed"] is False
  assert transaction["post_update_preflight"]["passed"] is False
  assert transaction["restored_preflight"]["passed"] is True
  assert transaction["usable_preflight"]["passed"] is True
  assert v19_restart_pair_feasibility(failure_bank, success_bank, 12)[
    "passed"
  ]
