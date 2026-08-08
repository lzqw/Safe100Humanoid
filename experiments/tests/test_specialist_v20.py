"""Pure regression tests for fixed-budget independent specialist v20."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/scripts"))

from audit_specialist_diagonal_v20 import _pair_raw_rows
from collect_mechanism_telemetry_v20 import MECHANISM_FIELDS
from plot_specialist_v20 import _validate_inputs
from specialist_v20_protocol import (
  CALIBRATION_CANDIDATE_SEEDS,
  FORMAL_ADAPTATION_SEEDS,
  fixed_budget_status,
  fresh_randomness_report,
)
from specialist_v20_tables import (
  candidate_metric_rows,
  replay_metric_rows,
  round_metric_rows,
)


def test_v20_zero_and_two_retained_updates_are_formally_valid() -> None:
  zero = fixed_budget_status(actual_rounds=8, retained_update_count=0)
  two = fixed_budget_status(actual_rounds=8, retained_update_count=2)
  assert zero.protocol_valid is True
  assert two.protocol_valid is True
  assert zero.retained_update_count_is_gate is False
  assert zero.stop_reason == "fixed_round_budget_completed"


def test_v20_formal_run_requires_exactly_eight_rounds() -> None:
  with pytest.raises(ValueError, match="exactly 8 rounds"):
    fixed_budget_status(actual_rounds=7, retained_update_count=2)
  with pytest.raises(ValueError, match="cannot exceed"):
    fixed_budget_status(actual_rounds=8, retained_update_count=9)


def test_v20_randomness_is_fresh_and_collision_check_is_effective() -> None:
  report = fresh_randomness_report(REPO)
  assert report["passed"] is True
  assert report["collisions"] == {
    "adaptation": [],
    "audit_or_bootstrap": [],
    "calibration": [],
  }
  used = fresh_randomness_report(
    REPO,
    adaptation_seeds=(43, 143, 243, 343, 443),
  )
  assert used["passed"] is False
  assert used["collisions"]["adaptation"] == [43, 143, 243, 343, 443]


def test_v20_calibration_candidates_are_valid_v19_generator_ids() -> None:
  combined = []
  for seeds in CALIBRATION_CANDIDATE_SEEDS.values():
    assert len(seeds) == 8
    assert all(0 <= seed % 100 <= 19 for seed in seeds)
    combined.extend(seeds)
  assert len(combined) == len(set(combined))


def test_v20_calibration_uses_revision4_full_rate_adapter() -> None:
  source = (
    REPO / "experiments/scripts/calibrate_specialist_contexts_v20.py"
  ).read_text()
  for assignment in (
    "alg_cfg.actor_new_feature_count = 5",
    "alg_cfg.actor_new_feature_learning_rate_multiplier = 1.0",
    "alg_cfg.freeze_legacy_actor_input_columns = True",
    "alg_cfg.std_scale_from_base = 0.35",
    "alg_cfg.entropy_coef = 0.0",
  ):
    assert assignment in source


def test_v20_pairs_raw_rows_by_seed_and_environment_not_finish_order() -> None:
  baseline = [
    {"evaluation_seed": "10", "environment_id": "1"},
    {"evaluation_seed": "10", "environment_id": "0"},
  ]
  final = [
    {"evaluation_seed": "10", "environment_id": "0"},
    {"evaluation_seed": "10", "environment_id": "1"},
  ]
  old, new = _pair_raw_rows(baseline, final)
  assert [row["environment_id"] for row in old] == ["0", "1"]
  assert [row["environment_id"] for row in new] == ["0", "1"]
  with pytest.raises(RuntimeError, match="seed/environment identity"):
    _pair_raw_rows(baseline, final[:1])


def _evaluation(success: float = 0.75, fall: float = 0.25) -> dict:
  return {
    "success_rate": success,
    "fall_rate": fall,
    "mean_return": 1.0,
    "mean_reached_riser": 8.0,
    "intervention_per_riser": 0.2,
    "correction_mean": 0.01,
    "seeds": [1],
    "num_episodes": 128,
  }


def _synthetic_summary() -> dict:
  baseline_target = _evaluation()
  baseline_d0 = _evaluation(0.9, 0.02)
  rounds = []
  for round_index in range(1, 9):
    variants = []
    for fraction in (0.5, 1.0, 1.5):
      variants.append(
        {
          "fraction": fraction,
          "update_metrics": {"mean_kl": 0.001 * fraction},
          "screen_eval": _evaluation(),
          "screen_success_delta": 0.0,
          "screen_fall_delta": 0.0,
          "screen_eligible": True,
        }
      )
    collector = {
      "rollout_seed": 1000 + round_index,
      "normal_start_count": 40,
      "failure_start_count": 12,
      "success_start_count": 12,
      "failure_bank_size_after_rollout": 256,
      "success_pool_size_after_rollout": 1024,
      "success_bank_size_after_matching": 256,
      "matched_pair_audit": {
        "pair_count": 12,
        "exact_match_passed": True,
        "maximum_marginal_imbalance": 0,
      },
      "bank_update_transaction": {
        "attempted": True,
        "committed": True,
        "restored_preflight": None,
        "usable_preflight": {"passed": True},
      },
      "cbf_intervention_fraction": 0.1,
      "cbf_correction_mean": 0.01,
    }
    update = {
      "collector_metrics": [collector, {**collector, "rollout_seed": 2000 + round_index}],
      "surrogate": 0.1,
      "value": 0.2,
      "entropy": 1.0,
      "mean_kl": 0.001,
      "maximum_preupdate_minibatch_kl": 0.002,
      "hard_case_transition_fraction": 12 / 64,
      "success_counterexample_transition_fraction": 12 / 64,
    }
    for group in ("normal", "failure", "success"):
      update[f"v19_{group}_advantage_mean_before"] = 0.0
      update[f"v19_{group}_advantage_std_before"] = 1.0
      update[f"v19_{group}_advantage_mean_after_normalization"] = 0.0
      update[f"v19_{group}_advantage_std_after_normalization"] = 1.0
    rounds.append(
      {
        "round": round_index,
        "candidate_screening": {
          "seed": 20_000 * round_index,
          "episodes_per_candidate": 64,
          "old": _evaluation(),
          "variants": variants,
          "best_fraction": 0.5,
        },
        "candidate_confirmation": {
          "seed": 20_000 * round_index + 10_000,
          "old": _evaluation(),
          "candidate": _evaluation(),
          "deltas": {"success_delta": 0.0, "fall_delta": 0.0},
        },
        "selected_candidate_fraction": None,
        "retained_candidate_fraction": None,
        "target_gate_accepted": False,
        "d0_check": {
          "passed": True,
          "candidate": _evaluation(0.9, 0.02),
          "accepted_round_end_actor": _evaluation(0.9, 0.02),
        },
        "d0_rollback": False,
        "policy_changed_at_round_end": False,
        "accepted_update_count": 0,
        "full_update_metrics": update,
        "round_end_adapter": {
          "new_input_column_rms": 0.0,
          "new_input_column_max_abs": 0.0,
          "legacy_input_column_change_from_initial_max_abs": 0.0,
        },
        "round_end_actor_sha256": f"actor-{round_index}",
      }
    )
  return {
    "specialist_mode": "lateral",
    "seed": FORMAL_ADAPTATION_SEEDS[0],
    "initial_actor_sha256": "pi0",
    "baseline_eval": {"DQHMED": baseline_target, "D0": baseline_d0},
    "actor_observation_expansion": {
      "new_first_layer_column_max_abs_before_adaptation": 0.0
    },
    "rounds": rounds,
  }


def test_v20_training_tables_have_exact_aligned_row_counts() -> None:
  summary = _synthetic_summary()
  round_rows = round_metric_rows(summary)
  candidate_rows = candidate_metric_rows(summary)
  replay_rows = replay_metric_rows(summary)
  assert len(round_rows) == 9
  assert [row["round"] for row in round_rows] == list(range(9))
  assert len(candidate_rows) == 24
  assert len(replay_rows) == 16
  assert round_rows[-1]["cumulative_retained_updates"] == 0


def test_v20_d0_curve_reports_restored_actor_not_rejected_candidate() -> None:
  summary = _synthetic_summary()
  summary["rounds"][0]["d0_check"]["candidate"] = _evaluation(0.1, 0.8)
  summary["rounds"][0]["d0_check"]["accepted_round_end_actor"] = _evaluation(
    0.9, 0.02
  )
  summary["rounds"][0]["d0_rollback"] = True
  row = round_metric_rows(summary)[1]
  assert row["d0_checked_candidate_success"] == 0.1
  assert row["d0_success"] == 0.9
  assert row["d0_success_delta"] == pytest.approx(0.0)


def test_v20_mechanism_schema_is_complete_even_without_repairs() -> None:
  assert MECHANISM_FIELDS[:5] == [
    "specialist",
    "adaptation_seed",
    "pair_index",
    "transition_class",
    "policy_role",
  ]
  for field in (
    "centerline_error_rate",
    "heading_error_rate",
    "command_vy",
    "command_wz",
    "support_foot",
    "left_slip_speed",
    "right_slip_speed",
    "contact_phase_mismatch",
    "roll_rad",
    "pitch_rad",
    "angular_velocity_norm",
    "cbf_correction_norm",
  ):
    assert field in MECHANISM_FIELDS


def test_v20_formal_round_loop_has_no_early_break_or_legacy_stop_cli() -> None:
  path = REPO / "experiments/scripts/refine_specialist_v20.py"
  source = path.read_text()
  tree = ast.parse(source)
  loops = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.For)
    and isinstance(node.target, ast.Name)
    and node.target.id == "round_index"
  ]
  assert len(loops) == 1
  assert not any(isinstance(node, ast.Break) for node in ast.walk(loops[0]))
  assert "--minimum-accepted-updates" not in source
  assert "--rejection-patience" not in source


def test_v20_plotting_requires_complete_two_specialist_matrices() -> None:
  modes = ("lateral", "contact_stability")
  round_rows = [
    {"specialist": mode, "adaptation_seed": seed, "round": round_index}
    for mode in modes
    for seed in FORMAL_ADAPTATION_SEEDS
    for round_index in range(9)
  ]
  candidate_rows = [
    {
      "specialist": mode,
      "adaptation_seed": seed,
      "round": round_index,
      "fraction": fraction,
    }
    for mode in modes
    for seed in FORMAL_ADAPTATION_SEEDS
    for round_index in range(1, 9)
    for fraction in (0.5, 1.0, 1.5)
  ]
  replay_rows = [
    {
      "specialist": mode,
      "adaptation_seed": seed,
      "round": round_index,
      "batch": batch,
    }
    for mode in modes
    for seed in FORMAL_ADAPTATION_SEEDS
    for round_index in range(1, 9)
    for batch in (1, 2)
  ]
  _validate_inputs(round_rows, candidate_rows, replay_rows)
  with pytest.raises(RuntimeError, match="row counts differ"):
    _validate_inputs(round_rows[:-1], candidate_rows, replay_rows)
  duplicate = [*round_rows[:-1], round_rows[0]]
  with pytest.raises(RuntimeError, match="matrices are incomplete"):
    _validate_inputs(duplicate, candidate_rows, replay_rows)


def test_v20_queues_are_mode_local_and_protocol_is_precalibration_only() -> None:
  queue = (REPO / "experiments/scripts/run_specialist_queue_v20.sh").read_text()
  assert 'MODE="${1:-${SAFE100_SPECIALIST_MODE:-}}"' in queue
  assert "run_specialist_diagonal_audit_v20.sh" in queue
  assert "for mode in" not in queue.lower()
  assert "SAFE100_V20_TRAINING_REPO" in queue
  assert "SAFE100_V20_AUDIT_REPO" in queue
  assert "SAFE100_V20_AUDIT_COMMIT" in queue
  protocol = json.loads(
    (REPO / "results/online/specialist_v20/protocol_precalibration.json").read_text()
  )
  assert protocol["training"]["fixed_round_budget"] == 8
  assert protocol["training"]["accepted_update_count_is_validity_gate"] is False
  assert protocol["fresh_evidence_boundary"][
    "formal_adaptation_must_follow_a_second_commit_sealing_selected_contexts"
  ] is True


def test_v20_final_protocol_seals_fresh_contexts_before_adaptation() -> None:
  protocol = json.loads(
    (REPO / "results/online/specialist_v20/protocol.json").read_text()
  )
  assert protocol["protocol_revision"] == 1
  assert protocol["status"] == "prospectively_frozen_before_formal_adaptation"
  assert protocol["fresh_evidence_boundary"][
    "formal_adaptation_or_audit_outcomes_seen"
  ] is False
  expected_seeds = {"lateral": 8312, "contact_stability": 8212}
  for mode, seed in expected_seeds.items():
    sealed = protocol["sealed_inputs"]["contexts"][mode]
    path = REPO / sealed["file"]
    assert sealed["selected_calibration_seed"] == seed
    assert all(sealed["validation_checks"].values())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sealed[
      "file_sha256"
    ]
  protocol_commit = subprocess.run(
    [
      "git",
      "log",
      "-1",
      "--format=%H",
      "--",
      "results/online/specialist_v20/protocol.json",
    ],
    cwd=REPO,
    check=True,
    capture_output=True,
    text=True,
  ).stdout.strip()
  for relative, expected_hash in protocol["sealed_inputs"][
    "source_file_sha256"
  ].items():
    frozen_source = subprocess.run(
      ["git", "show", f"{protocol_commit}:{relative}"],
      cwd=REPO,
      check=True,
      capture_output=True,
    ).stdout
    assert hashlib.sha256(frozen_source).hexdigest() == expected_hash


def test_v20_audit_uses_brief_checkpoint_loader_before_any_load() -> None:
  source = (
    REPO / "experiments/scripts/audit_specialist_diagonal_v20.py"
  ).read_text()
  configuration = "alg_cfg.brief_ppo_refinement = True"
  first_load = "warm_start = runner.load_online_checkpoint("
  assert configuration in source
  assert first_load in source
  assert source.index(configuration) < source.index(first_load)
  assert "--audit-commit" in source
  assert "--audit-amendment" in source
  assert "actor_or_checkpoint_tensor_modified" in source


def test_v20_audit_amendment_preserves_training_and_evaluation_seals() -> None:
  amendment = json.loads(
    (
      REPO / "results/online/specialist_v20/audit_amendment.json"
    ).read_text()
  )
  assert amendment["status"] == (
    "prospectively_frozen_before_first_formal_audit_episode_outcome"
  )
  boundary = amendment["fresh_audit_evidence_boundary"]
  assert boundary["formal_audit_episode_outcomes_observed"] is False
  assert boundary["formal_audit_rows_written"] == 0
  assert boundary["training_artifacts_rerun_or_modified"] is False
  assert amendment["training_protocol"]["git_commit"] == (
    "1ded5b84f1c4b8605fd285ef3138c0363db20ee4"
  )
  assert amendment["unchanged_formal_evaluation"]["audit_seed"] == 5_500_000
  assert amendment["unchanged_formal_evaluation"]["bootstrap_seed"] == (
    6_500_000
  )
  for relative, expected_hash in amendment["source_file_sha256"].items():
    assert hashlib.sha256((REPO / relative).read_bytes()).hexdigest() == (
      expected_hash
    )
