"""Protocol and source-boundary regression tests for v22 effect-first."""

from __future__ import annotations

import ast
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/scripts"))

from specialist_v22_protocol import (
  CALIBRATION_EPISODES,
  CALIBRATION_MINIMUM_FAILURES,
  CALIBRATION_MINIMUM_PURITY,
  CALIBRATION_SUCCESS_BOUNDS,
  CANDIDATE_CONFIRM_EPISODES,
  CANDIDATE_D0_EPISODES,
  CANDIDATE_FRACTIONS,
  CANDIDATE_SCREEN_EPISODES,
  CONTEXT_ADAPTATION_SEEDS,
  CONTEXT_CALIBRATION_CANDIDATE_SEEDS,
  CONTEXT_REPORT_BOOTSTRAP_SEEDS,
  CONTEXTS,
  DUAL_ROLLOUT_BATCHES,
  FAILURE_DISCOVERY_MAX_ROLLOUTS,
  FINAL_D0_EPISODES,
  FINAL_TARGET_EPISODES,
  NORMAL_FAILURE_SUCCESS_SLOTS,
  REPORT_BOOTSTRAP_SAMPLES,
  ROUNDS,
  VALIDATION_EPISODES,
  all_v22_random_seeds,
  calibration_evaluation_seed,
  candidate_confirmation_seed,
  candidate_d0_seed,
  candidate_confirmation_gate,
  candidate_screen_seed,
  configure_v22_policy_evaluation_algorithm,
  development_success_gate,
  dual_rollout_seed,
  failure_discovery_seed,
  fresh_randomness_report,
  select_best_so_far,
)
from online_refine_stairs import _evaluate_state
from src.tasks.stairs_cbf.config import g1_online_stairs_env_cfg
from src.tasks.stairs_cbf.deployment_context import (
  OBSERVABLE_SPECIALIST_CONTEXT_KINDS,
  V22_CALIBRATION_KIND,
  V22_CONTEXT_KIND,
  V22_CONTEXT_SCHEMA_VERSION,
  V22_CONTEXT_SPECS,
  apply_frozen_deployment_context,
  generate_v22_specialist_context,
  validate_calibrated_v22_context,
  validate_frozen_deployment_context,
)


def test_v22_has_exactly_two_pure_conditional_contexts() -> None:
  assert V22_CONTEXT_SCHEMA_VERSION == 2
  assert CONTEXTS == ("L_effect", "C_effect")
  assert set(CONTEXTS) == set(V22_CONTEXT_SPECS)
  assert V22_CONTEXT_SPECS["L_effect"]["family"] == (
    "pure_lateral_bias_and_pulse"
  )
  assert V22_CONTEXT_SPECS["C_effect"]["family"] == "pure_low_foot_friction"
  assert V22_CONTEXT_SPECS["L_effect"]["effect_axes"] == [
    "lateral_command_bias",
    "lateral_disturbance_pulse",
  ]
  assert V22_CONTEXT_SPECS["C_effect"]["effect_axes"] == ["foot_friction"]


def test_v22_candidates_are_deterministic_hashed_and_valid() -> None:
  hashes = set()
  for context_id, seeds in CONTEXT_CALIBRATION_CANDIDATE_SEEDS.items():
    first = generate_v22_specialist_context(context_id, seeds[0])
    last = generate_v22_specialist_context(context_id, seeds[-1])
    assert first["kind"] == V22_CONTEXT_KIND
    assert first["candidate_severity"] == 0.0
    assert last["candidate_severity"] == 1.0
    assert validate_frozen_deployment_context(first) == first
    assert validate_frozen_deployment_context(json.loads(json.dumps(last))) == last
    assert first["parameters_sha256"] != last["parameters_sha256"]
    hashes.update((first["parameters_sha256"], last["parameters_sha256"]))
  assert len(hashes) == 4


def test_v22_lateral_varies_only_bias_and_lateral_pulse_magnitude() -> None:
  seeds = CONTEXT_CALIBRATION_CANDIDATE_SEEDS["L_effect"]
  first = generate_v22_specialist_context("L_effect", seeds[0])
  last = generate_v22_specialist_context("L_effect", seeds[-1])
  assert first["target"] == last["target"]
  target = first["target"]
  assert target["num_steps"] == 9
  assert target["rise_profile"] == (0.13,) * 9
  assert target["tread_profile"] == (0.35,) * 9
  assert target["episode_length_s"] == 35.0
  assert target["action_gain"] == 1.0
  assert target["action_delay_steps"] == 0
  assert target["encoder_bias"] == 0.0
  assert target["command_delay_s"] == 0.10
  assert target["command_low_pass_s"] == 0.08
  assert first["nominal_command_delay_range_s"] == [0.04, 0.16]
  assert not any(target["action_bias"])
  changed = {
    key
    for key in first["scenario"]
    if first["scenario"][key] != last["scenario"][key]
  }
  assert changed == {
    "lateral_command_bias",
    "lateral_pulse_min",
    "lateral_pulse_max",
  }
  for payload in (first, last):
    scenario = payload["scenario"]
    assert scenario["yaw_command_bias"] == 0.0
    assert scenario["yaw_pulse_min"] == 0.0
    assert scenario["yaw_pulse_max"] == 0.0
    assert scenario["foot_friction"] == 0.60
    assert scenario["toe_margin"] == 0.08
    assert scenario["stair_half_width"] == 1.2
    assert scenario["centerline_lateral_gain"] == 0.8
    assert scenario["centerline_heading_gain"] == 1.4


def test_v22_contact_varies_only_foot_friction() -> None:
  seeds = CONTEXT_CALIBRATION_CANDIDATE_SEEDS["C_effect"]
  first = generate_v22_specialist_context("C_effect", seeds[0])
  last = generate_v22_specialist_context("C_effect", seeds[-1])
  assert first["target"] == last["target"]
  assert first["target"]["num_steps"] == 9
  assert first["target"]["rise_profile"] == (0.13,) * 9
  assert first["target"]["tread_profile"] == (0.35,) * 9
  assert first["target"]["command_delay_s"] == 0.10
  assert first["target"]["command_low_pass_s"] == 0.08
  assert first["target"]["episode_length_s"] == 35.0
  assert first["nominal_command_delay_range_s"] == [0.04, 0.16]
  changed = {
    key
    for key in first["scenario"]
    if first["scenario"][key] != last["scenario"][key]
  }
  assert changed == {"foot_friction"}
  for payload in (first, last):
    scenario = payload["scenario"]
    assert scenario["disturbance_pulses_with_centering"] is False
    assert (scenario["lateral_pulse_min"], scenario["lateral_pulse_max"]) == (
      0.02,
      0.08,
    )
    assert (scenario["yaw_pulse_min"], scenario["yaw_pulse_max"]) == (
      0.05,
      0.20,
    )
    assert scenario["centerline_lateral_gain"] == 0.80
    assert scenario["centerline_heading_gain"] == 1.40
    assert scenario["centerline_max_lateral_velocity"] == 0.16
    assert scenario["centerline_max_yaw_velocity"] == 0.45
    assert scenario["toe_margin"] == 0.08
    assert scenario["contact_observation_delay_steps"] == 0
    assert scenario["gait_phase_offset"] == 0.0
    assert scenario["left_response_scale"] == 1.0
    assert scenario["right_response_scale"] == 1.0
    assert scenario["lateral_command_bias"] == 0.0


def test_v22_application_changes_only_declared_physical_effect_axes() -> None:
  common_command_fields = (
    "forward_velocity_range",
    "command_delay_range_s",
    "low_pass_time_constant_s",
    "closed_loop_centering",
    "stair_half_width",
    "centerline_lateral_gain",
    "centerline_heading_gain",
    "centerline_max_lateral_velocity",
    "centerline_max_yaw_velocity",
    "centerline_heading_reference_bias",
  )
  all_command_fields = common_command_fields + (
    "disturbance_pulses_with_centering",
    "fixed_lateral_bias",
    "fixed_yaw_bias",
    "lateral_pulse_abs_range",
    "yaw_pulse_abs_range",
    "pulse_interval_range_s",
    "pulse_duration_range_s",
  )
  for context_id, seeds in CONTEXT_CALIBRATION_CANDIDATE_SEEDS.items():
    for seed in (seeds[0], seeds[-1]):
      nominal = g1_online_stairs_env_cfg("DQHMED")
      cfg = g1_online_stairs_env_cfg("DQHMED")
      payload = generate_v22_specialist_context(context_id, seed)
      metadata = apply_frozen_deployment_context(cfg, payload, role="target")
      command = cfg.commands["twist"]
      nominal_command = nominal.commands["twist"]
      action = cfg.actions["joint_pos"]
      stairs = next(
        iter(cfg.scene.terrain.terrain_generator.sub_terrains.values())
      )

      assert stairs.num_steps == 9
      assert stairs.step_height_profile == (0.13,) * 9
      assert stairs.step_width_profile == (0.35,) * 9
      assert cfg.episode_length_s == 35.0
      assert action.deployment_action_gain == 1.0
      assert action.deployment_action_scale is None
      assert action.deployment_action_bias is None
      assert action.deployment_action_delay_steps == 0
      assert action.deployment_contact_delay_steps == 0
      assert action.deployment_contact_phase_offset == 0.0
      assert action.toe_margin == 0.08
      assert cfg.events["encoder_bias"].params["bias_range"] == (0.0, 0.0)
      for field in common_command_fields:
        assert getattr(command, field) == getattr(nominal_command, field)
      assert metadata["applied_command_delay_range_s"] == (0.04, 0.16)

      friction = cfg.events["specialist_foot_friction"].params["ranges"]
      scenario = payload["scenario"]
      assert friction == (scenario["foot_friction"],) * 2
      if context_id == "L_effect":
        assert metadata["declared_effect_axes"] == [
          "lateral_command_bias",
          "lateral_disturbance_pulse",
        ]
        assert command.disturbance_pulses_with_centering is True
        assert command.fixed_lateral_bias == scenario["lateral_command_bias"]
        assert command.lateral_pulse_abs_range == (
          scenario["lateral_pulse_min"],
          scenario["lateral_pulse_max"],
        )
        assert command.fixed_yaw_bias == 0.0
        assert command.yaw_pulse_abs_range == (0.0, 0.0)
        assert friction == (0.60, 0.60)
      else:
        assert metadata["declared_effect_axes"] == ["foot_friction"]
        for field in all_command_fields:
          assert getattr(command, field) == getattr(nominal_command, field)


def test_v22_is_accepted_by_isolated_observable_evaluation_boundary(
  tmp_path: Path,
) -> None:
  assert V22_CONTEXT_KIND in OBSERVABLE_SPECIALIST_CONTEXT_KINDS
  payload = generate_v22_specialist_context(
    "L_effect", CONTEXT_CALIBRATION_CANDIDATE_SEEDS["L_effect"][0]
  )
  context_path = tmp_path / "v22_context.json"
  context_path.write_text(json.dumps(payload))

  class _Algorithm:
    @staticmethod
    def save() -> dict:
      return {}

  runner = SimpleNamespace(alg=_Algorithm())
  assert _evaluate_state(
    runner,
    {},
    domains=(),
    num_envs=1,
    num_episodes=1,
    seed=1,
    device="cpu",
    deployment_context=context_path,
    v19_context=context_path,
  ) == {}
  for relative in (
    "experiments/scripts/online_refine_stairs.py",
    "experiments/scripts/evaluate_online_stairs.py",
  ):
    source = (REPO / relative).read_text()
    assert "OBSERVABLE_SPECIALIST_CONTEXT_KINDS" in source
    assert "v19/v21/v22" in source


def _calibrated_context() -> dict:
  context_id = "L_effect"
  seeds = list(CONTEXT_CALIBRATION_CANDIDATE_SEEDS[context_id])
  payload = generate_v22_specialist_context(context_id, seeds[0])
  success_count = 358
  failure_count = 512 - success_count
  target_failure_count = 139
  non_fall_failure_count = 10
  fall_count = failure_count - non_fall_failure_count
  attempt = {
    "candidate_seed": seeds[0],
    "base_policy_only": True,
    "num_episodes": 512,
    "success_count": success_count,
    "success_rate": success_count / 512,
    "failure_count": failure_count,
    "fall_count": fall_count,
    "non_fall_failure_count": non_fall_failure_count,
    "failure_type_counts": {
      "lateral_heading_drift": target_failure_count,
      "contact_stability": 0,
      "non_lateral_high_cbf_demand": 0,
      "non_lateral_balance_or_phase": fall_count - target_failure_count,
      "other_non_lateral": 0,
    },
    "target_failure_fraction": target_failure_count / failure_count,
    "target_failure_type": "lateral_heading_drift",
    "parameters_sha256": payload["parameters_sha256"],
    "qualifies": True,
  }
  payload["calibration"] = {
    "kind": V22_CALIBRATION_KIND,
    "adapted_policy_evaluations_used": False,
    "success_rate_bounds": [0.65, 0.75],
    "minimum_target_failure_fraction": 0.85,
    "minimum_failure_count": 100,
    "episodes_per_candidate": 512,
    "candidate_seeds": seeds,
    "attempts": [attempt],
    "selected_candidate_seed": seeds[0],
    "selected_parameters_sha256": payload["parameters_sha256"],
  }
  return payload


def test_v22_calibration_validator_enforces_first_base_only_qualifier() -> None:
  payload = _calibrated_context()
  assert validate_calibrated_v22_context(payload) == payload
  adapted = deepcopy(payload)
  adapted["calibration"]["adapted_policy_evaluations_used"] = True
  with pytest.raises(ValueError, match="adapted policy"):
    validate_calibrated_v22_context(adapted)
  fall_only_purity = deepcopy(payload)
  attempt = fall_only_purity["calibration"]["attempts"][0]
  attempt["target_failure_fraction"] = (
    attempt["failure_type_counts"]["lateral_heading_drift"]
    / attempt["fall_count"]
  )
  with pytest.raises(ValueError, match="purity denominator"):
    validate_calibrated_v22_context(fall_only_purity)
  skipped = deepcopy(payload)
  first = deepcopy(skipped["calibration"]["attempts"][0])
  first["qualifies"] = True
  second_seed = skipped["calibration"]["candidate_seeds"][1]
  second_context = generate_v22_specialist_context("L_effect", second_seed)
  second = deepcopy(first)
  second.update(
    candidate_seed=second_seed,
    parameters_sha256=second_context["parameters_sha256"],
  )
  skipped["calibration"]["attempts"] = [first, second]
  skipped["calibration"]["selected_candidate_seed"] = second_seed
  skipped["calibration"]["selected_parameters_sha256"] = second_context[
    "parameters_sha256"
  ]
  skipped.update(second_context)
  skipped["calibration"] = deepcopy(payload["calibration"])
  skipped["calibration"].update(
    attempts=[first, second],
    selected_candidate_seed=second_seed,
    selected_parameters_sha256=second_context["parameters_sha256"],
  )
  with pytest.raises(ValueError, match="skipped an earlier"):
    validate_calibrated_v22_context(skipped)


def test_v22_counts_and_gates_match_effect_first_design() -> None:
  assert CALIBRATION_EPISODES == 512
  assert CALIBRATION_MINIMUM_FAILURES == 100
  assert CALIBRATION_SUCCESS_BOUNDS == (0.65, 0.75)
  assert CALIBRATION_MINIMUM_PURITY == 0.85
  assert ROUNDS == 8
  assert DUAL_ROLLOUT_BATCHES == 2
  assert NORMAL_FAILURE_SUCCESS_SLOTS == (40, 12, 12)
  assert FAILURE_DISCOVERY_MAX_ROLLOUTS == 12
  assert CANDIDATE_FRACTIONS == (0.5, 1.0, 1.5)
  assert CANDIDATE_SCREEN_EPISODES == 64
  assert CANDIDATE_CONFIRM_EPISODES == 128
  assert CANDIDATE_D0_EPISODES == 128
  assert VALIDATION_EPISODES == 256
  assert FINAL_TARGET_EPISODES == 512
  assert FINAL_D0_EPISODES == 256
  assert REPORT_BOOTSTRAP_SAMPLES == 2_000
  accepted, reasons = candidate_confirmation_gate(
    success_delta=0.01, fall_delta=0.03, finite=True
  )
  assert accepted is True
  assert not reasons
  rejected, _ = candidate_confirmation_gate(
    success_delta=0.0, fall_delta=0.0, finite=True
  )
  assert rejected is False
  gate = development_success_gate(
    target_success_delta=0.03,
    target_fall_delta=0.01,
    d0_success_delta=-0.05,
  )
  assert gate["passed"] is True
  assert gate["strict_zero_fall_increase_passed"] is False


def test_v22_best_so_far_uses_validation_success_fall_and_earlier_tie() -> None:
  rows = [
    {"round": 0, "success_rate": 0.70, "fall_rate": 0.30, "d0_safe": True},
    {"round": 2, "success_rate": 0.75, "fall_rate": 0.31, "d0_safe": True},
    {"round": 4, "success_rate": 0.76, "fall_rate": 0.33, "d0_safe": True},
    {"round": 6, "success_rate": 0.75, "fall_rate": 0.31, "d0_safe": True},
  ]
  assert select_best_so_far(rows)["round"] == 2
  rows[1]["d0_safe"] = False
  assert select_best_so_far(rows)["round"] == 6


def test_v22_evaluation_configuration_is_beta_zero_v20_v21_core() -> None:
  cfg = SimpleNamespace()
  configure_v22_policy_evaluation_algorithm(cfg)
  assert cfg.matched_success_preservation_beta == 0.0
  assert cfg.brief_ppo_refinement is True
  assert cfg.observable_failure_conditioned_refinement is True
  assert cfg.actor_new_feature_count == 5
  assert cfg.freeze_legacy_actor_input_columns is True
  assert cfg.num_learning_epochs == 1
  assert cfg.num_mini_batches == 4


def test_v22_expanded_randomness_is_fresh_and_unique_against_history() -> None:
  assert CONTEXT_REPORT_BOOTSTRAP_SEEDS == {
    "L_effect": {"target": 97_000_000, "D0": 97_000_010},
    "C_effect": {"target": 97_100_000, "D0": 97_100_010},
  }
  seeds = all_v22_random_seeds()
  for context_id, role_seeds in CONTEXT_REPORT_BOOTSTRAP_SEEDS.items():
    assert set(role_seeds.values()) <= seeds
    assert CONTEXT_ADAPTATION_SEEDS[context_id] + 900_000 not in seeds
    assert candidate_d0_seed(context_id) in seeds
    assert calibration_evaluation_seed(context_id, 0) in seeds
    assert failure_discovery_seed(context_id, 11) in seeds
    assert dual_rollout_seed(context_id, 8, 1) in seeds
    assert candidate_screen_seed(context_id, 8) in seeds
    assert candidate_confirmation_seed(context_id, 8) in seeds
  assert len(seeds) > 100
  report = fresh_randomness_report(REPO)
  assert report["passed"] is True
  assert report["collisions"] == []


def test_v22_training_is_fixed_budget_and_monitor_is_not_a_candidate_gate() -> None:
  path = REPO / "experiments/scripts/refine_effect_first_v22.py"
  tree = ast.parse(path.read_text())
  loops = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.For)
    and isinstance(node.target, ast.Name)
    and node.target.id == "round_index"
  ]
  training_loops = [
    node
    for node in loops
    if any(
      isinstance(child, ast.Call)
      and isinstance(child.func, ast.Name)
      and child.func.id == "_save_checkpoint"
      for child in ast.walk(node)
    )
  ]
  assert len(training_loops) == 1
  assert not any(isinstance(node, ast.Break) for node in ast.walk(training_loops[0]))
  source = path.read_text()
  assert "validation_monitor_used_for_candidate_or_training" in source
  assert '"validation_monitor_used_for_candidate_or_training": False' in source
  assert "candidate_confirmation_gate(" in source
  assert "confirmation_block_gate" not in source
  assert "matched_success_preservation_beta = 0.0" in source


def test_v22_final_test_and_plot_scope_are_minimal() -> None:
  final_source = (REPO / "experiments/scripts/test_effect_first_v22.py").read_text()
  plot_source = (REPO / "experiments/scripts/plot_effect_first_v22.py").read_text()
  assert "FINAL_TARGET_EPISODES" in final_source
  assert "FINAL_D0_EPISODES" in final_source
  assert 'POLICY_ROLES = ("base", "best")' in final_source
  assert '"confidence_intervals_are_report_only": True' in final_source
  assert plot_source.count("_save(figure, output_dir, stem)") == 1
  for stem in (
    "validation_learning_curve",
    "base_vs_best_final",
    "repair_vs_regression",
    "failure_specific_telemetry",
  ):
    assert f'("{stem}",' in plot_source


def test_v22_contact_freeze_requires_passed_lateral_result() -> None:
  source = (
    REPO / "experiments/scripts/freeze_effect_first_v22_protocol.py"
  ).read_text()
  assert 'if args.context_id == "C_effect":' in source
  assert 'lateral.get("development_gate", {}).get("passed") is not True' in source
  assert "contact cannot start because the lateral gate did not pass" in source
  assert '"--superseded-before-any-base-evaluation"' in source
  assert '"--supersession-reason"' in source
  assert '"superseded_before_first_base_policy_episode"' in source
  assert '"verified_protocol_chain_depth"' in source
  assert "_verified_supersession_depth" in source
  assert '"base_policy_episode_outcomes_observed": False' in source
  assert '"context_schema_version": V22_CONTEXT_SCHEMA_VERSION' in source
  calibration_source = (
    REPO / "experiments/scripts/calibrate_effect_first_v22.py"
  ).read_text()
  assert '"calibration_negative_no_candidate_qualified"' in calibration_source
  assert 'output_dir / "calibration_negative.json"' in calibration_source
  assert '"conditional_disposition": "stop_v22_before_lateral_adaptation"' in (
    calibration_source
  )
