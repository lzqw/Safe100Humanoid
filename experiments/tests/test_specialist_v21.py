"""Pure protocol and source-boundary regression tests for specialist v21."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/scripts"))

from specialist_v21_protocol import (
  BETA_GRID,
  CANDIDATE_EVALUATION_EPISODES_PER_ROUND,
  CONFIRMATION_BLOCKS,
  CONFIRMATION_EPISODES_PER_BLOCK,
  CONTEXTS,
  FORMAL_CONTEXTS_BY_MODE,
  FORMAL_D0_EPISODES,
  FORMAL_MONITOR_EPISODES,
  FORMAL_TARGET_EPISODES,
  V21_DEVELOPMENT_CONTEXTS,
  V21_FORMAL_CONTEXTS,
  confirmation_block_gate,
  deployment_mode_gate,
  fixed_budget_status,
  select_development_beta,
)

from src.tasks.stairs_cbf.deployment_context import (
  V21_CONTEXT_KIND,
  V21_CONTEXT_SPECS,
  generate_v21_specialist_context,
  validate_frozen_deployment_context,
)


def test_v21_context_matrix_has_two_excluded_and_ten_formal_families() -> None:
  assert V21_DEVELOPMENT_CONTEXTS == ("L_dev", "C_dev")
  assert V21_FORMAL_CONTEXTS == (
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
  )
  assert set(CONTEXTS) == set(V21_CONTEXT_SPECS)
  assert len({spec["family"] for spec in V21_CONTEXT_SPECS.values()}) == 12
  assert all(
    V21_CONTEXT_SPECS[name]["formal"] is (name in V21_FORMAL_CONTEXTS)
    for name in CONTEXTS
  )


def test_v21_family_candidates_are_deterministic_hashed_and_valid() -> None:
  hashes = set()
  for context_id, spec in V21_CONTEXT_SPECS.items():
    prefix = int(spec["candidate_seed_prefix"])
    first = generate_v21_specialist_context(context_id, prefix * 100 + 8)
    last = generate_v21_specialist_context(context_id, prefix * 100 + 19)
    assert first["kind"] == V21_CONTEXT_KIND
    assert first["candidate_severity"] == 0.0
    assert last["candidate_severity"] == 1.0
    assert validate_frozen_deployment_context(first) == first
    assert validate_frozen_deployment_context(last) == last
    assert validate_frozen_deployment_context(
      json.loads(json.dumps(first))
    ) == first
    assert first["parameters_sha256"] != last["parameters_sha256"]
    hashes.update((first["parameters_sha256"], last["parameters_sha256"]))
  assert len(hashes) == 2 * len(V21_CONTEXT_SPECS)


def test_v21_formal_family_primary_perturbations_are_distinct() -> None:
  contexts = {}
  for context_id in V21_FORMAL_CONTEXTS:
    prefix = int(V21_CONTEXT_SPECS[context_id]["candidate_seed_prefix"])
    contexts[context_id] = generate_v21_specialist_context(
      context_id, prefix * 100 + 19
    )
  assert contexts["L1"]["target"]["command_delay_s"] == 0.4
  assert contexts["L2"]["scenario"]["yaw_command_bias"] != 0.0
  assert contexts["L2"]["scenario"]["lateral_command_bias"] == 0.0
  assert contexts["L3"]["scenario"]["lateral_command_bias"] != 0.0
  assert contexts["L3"]["scenario"]["yaw_command_bias"] == 0.0
  assert contexts["L4"]["scenario"]["centerline_lateral_gain"] == 0.08
  assert contexts["L5"]["target"]["command_delay_s"] > 0.0
  assert contexts["C1"]["scenario"]["foot_friction"] == 0.25
  assert contexts["C2"]["scenario"]["left_response_scale"] != 1.0
  assert contexts["C3"]["target"]["action_gain"] == 0.8
  assert abs(contexts["C3"]["target"]["encoder_bias"]) == 0.03
  assert contexts["C4"]["target"]["command_forward_scale"] == 1.16
  assert contexts["C5"]["scenario"]["contact_observation_delay_steps"] == 2


def test_v21_confirmation_uses_three_independent_positive_block_rule() -> None:
  assert CONFIRMATION_BLOCKS == 3
  assert CONFIRMATION_EPISODES_PER_BLOCK == 64
  accepted, reasons, metrics = confirmation_block_gate(
    (0.05, -0.01, 0.02), (0.0, 0.02, 0.01), mean_kl=0.003, finite=True
  )
  assert accepted is True
  assert not reasons
  assert metrics["positive_success_blocks"] == 2
  rejected, reasons, _ = confirmation_block_gate(
    (0.05, 0.0, -0.01), (0.0, 0.0, 0.0), mean_kl=0.003, finite=True
  )
  assert rejected is False
  assert any("fewer than two" in reason for reason in reasons)


def test_v21_budget_and_evaluation_counts_match_the_frozen_design() -> None:
  assert BETA_GRID == (0.0, 1.0, 4.0, 16.0)
  assert CANDIDATE_EVALUATION_EPISODES_PER_ROUND == 384
  assert FORMAL_TARGET_EPISODES == 1024
  assert FORMAL_D0_EPISODES == 256
  assert FORMAL_MONITOR_EPISODES == 256
  zero = fixed_budget_status(actual_rounds=8, retained_update_count=0)
  assert zero.protocol_valid is True
  assert zero.retained_update_count_is_gate is False
  with pytest.raises(ValueError, match="exactly 8"):
    fixed_budget_status(actual_rounds=7, retained_update_count=0)


def test_v21_formal_gate_uses_five_deployment_contexts_per_mode() -> None:
  assert FORMAL_CONTEXTS_BY_MODE == {
    "lateral": ("L1", "L2", "L3", "L4", "L5"),
    "contact_stability": ("C1", "C2", "C3", "C4", "C5"),
  }
  passed = deployment_mode_gate(
    (0.02, 0.01, 0.03, 0.01, -0.005),
    (0.01, 0.02, 0.0, 0.03, 0.01),
    (-0.01, -0.02, 0.0, 0.01, -0.01),
  )
  assert passed["passed"] is True
  assert passed["positive_context_count"] == 4
  failed = deployment_mode_gate(
    (0.02, 0.01, -0.01, -0.02, 0.03),
    (0.0,) * 5,
    (0.0,) * 5,
  )
  assert failed["passed"] is False


def test_v21_development_beta_selection_uses_both_contexts_and_rr_minus_rg() -> None:
  metrics = {
    beta: {
      "L_dev": {"repair_rate": 0.20, "regression_rate": 0.10},
      "C_dev": {"repair_rate": 0.20, "regression_rate": 0.10},
    }
    for beta in BETA_GRID
  }
  metrics[4.0] = {
    "L_dev": {"repair_rate": 0.30, "regression_rate": 0.05},
    "C_dev": {"repair_rate": 0.25, "regression_rate": 0.04},
  }
  selected = select_development_beta(metrics)
  assert selected["selected_beta"] == 4.0
  assert len(selected["rows"]) == 4


def test_v21_training_saves_pi0_to_pi8_and_has_no_round_early_break() -> None:
  path = REPO / "experiments/scripts/refine_deployment_v21.py"
  tree = ast.parse(path.read_text())
  loops = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.For)
    and isinstance(node.target, ast.Name)
    and node.target.id == "round_index"
    and isinstance(node.iter, ast.Call)
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
  assert 'output_dir / "post_round_000.pt"' in source
  assert 'output_dir / f"post_round_{round_index:03d}.pt"' in source
  assert "--minimum-accepted-updates" not in source
  assert "--rejection-patience" not in source


def test_v21_monitor_and_mechanism_evidence_are_not_selection_replays() -> None:
  monitor = (
    REPO / "experiments/scripts/evaluate_learning_curve_v21.py"
  ).read_text()
  audit = (REPO / "experiments/scripts/audit_deployment_v21.py").read_text()
  evaluator = (REPO / "experiments/scripts/evaluate_online_stairs.py").read_text()
  online = (REPO / "src/tasks/stairs_cbf/online.py").read_text()
  assert "candidate_screen" not in monitor
  assert 'telemetry_env_id=(' in audit
  assert '"post_audit_mechanism_replay_used": False' in audit
  assert "entropy_values[~success_mask].mean()" in online
  for field in (
    "telemetry_success",
    "telemetry_fell",
    "telemetry_failure_type",
    "telemetry_episode_steps",
  ):
    assert field in evaluator


def test_v21_cross_context_analysis_and_figures_are_prospectively_source_bound() -> None:
  freezer = (
    REPO / "experiments/scripts/freeze_specialist_v21_protocol.py"
  ).read_text()
  aggregate = (
    REPO / "experiments/scripts/aggregate_deployment_v21.py"
  ).read_text()
  for relative in (
    "experiments/scripts/aggregate_deployment_v21.py",
    "experiments/scripts/plot_deployment_v21.py",
  ):
    assert f'"{relative}"' in freezer
  assert "deployment_mode_gate(" in aggregate
  assert '"statistical_unit": "deployment_context"' in aggregate
  assert "candidate_screen" not in aggregate
