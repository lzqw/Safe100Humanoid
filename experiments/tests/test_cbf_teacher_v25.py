"""Regression tests for the prospectively frozen v25 teacher path."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/scripts"))

from cbf_teacher_v25_protocol import (
    ADAPTATION_SEED,
    CALIBRATION_EPISODES,
    CALIBRATION_GAINS,
    CALIBRATION_REPEATS,
    CONTEXT_FAMILY,
    CONTEXT_ID,
    FINAL_EPISODES,
    V23_FINAL_SHA256,
    V23_PROTOCOL_SHA256,
    V23_RESULT_GIT_TREE,
    V24_FINAL_SHA256,
    V24_PROTOCOL_SHA256,
    V24_RESULT_GIT_TREE,
    all_v25_fresh_execution_seeds,
    calibration_evaluation_seed,
    calibration_gate,
    canonical_sha256,
    development_gate,
    formal_algorithm_parameters,
    fresh_randomness_report,
    validate_v25_calibrated_context,
)
from plot_cbf_teacher_v25 import FIGURE_CATEGORIES

_TEACHER_MATH_SPEC = importlib.util.spec_from_file_location(
    "v25_teacher_math", REPO / "src/tasks/stairs_cbf/teacher_math.py"
)
assert _TEACHER_MATH_SPEC is not None and _TEACHER_MATH_SPEC.loader is not None
_TEACHER_MATH = importlib.util.module_from_spec(_TEACHER_MATH_SPEC)
_TEACHER_MATH_SPEC.loader.exec_module(_TEACHER_MATH)
actor_coordinate_teacher_action = _TEACHER_MATH.actor_coordinate_teacher_action
successful_teacher_labels = _TEACHER_MATH.successful_teacher_labels
swing_leg_action_scale = _TEACHER_MATH.swing_leg_action_scale
toe_riser_kick_event = _TEACHER_MATH.toe_riser_kick_event
weighted_gaussian_teacher_loss = _TEACHER_MATH.weighted_gaussian_teacher_loss


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualifying_context(index: int = 2) -> dict:
    parameters = {
        "context_id": CONTEXT_ID,
        "family": CONTEXT_FAMILY,
        "selected_candidate_index": index,
        "swing_underresponse_gain": CALIBRATION_GAINS[index],
        "phase_selective": True,
        "affected_joint_suffixes": [
            "hip_pitch_joint",
            "knee_joint",
            "ankle_pitch_joint",
        ],
        "stance_leg_gain": 1.0,
        "other_joint_gain": 1.0,
    }
    attempts = [
        {
            "candidate_index": candidate,
            "swing_underresponse_gain": CALIBRATION_GAINS[candidate],
            "qualifies": candidate == index,
            "evaluation_seeds": [
                calibration_evaluation_seed(candidate, repeat)
                for repeat in range(CALIBRATION_REPEATS)
            ],
        }
        for candidate in range(index + 1)
    ]
    return {
        "schema_version": 1,
        "context_id": CONTEXT_ID,
        "parameters_sha256": canonical_sha256(parameters),
        "shift": {
            "family": CONTEXT_FAMILY,
            "selected_candidate_index": index,
            "swing_underresponse_gain": CALIBRATION_GAINS[index],
            "phase_selective": True,
            "affected_joint_suffixes": parameters["affected_joint_suffixes"],
            "stance_leg_gain": 1.0,
            "other_joint_gain": 1.0,
            "terrain_geometry": "nominal_fixed_DQHMED",
            "friction": "nominal",
            "command": "nominal",
            "controller": "nominal",
            "observation_interface": "original_405D",
            "cbf_geometry": "exact_generated_riser_metadata",
        },
        "calibration": {
            "base_policy_only": True,
            "adapted_policy_evaluations_used": False,
            "attempts": attempts,
        },
    }


def test_v23_and_v24_results_remain_byte_and_tree_frozen() -> None:
    assert (
        _sha(REPO / "results/online/proximal_v23/protocol.json") == V23_PROTOCOL_SHA256
    )
    assert (
        _sha(REPO / "results/online/proximal_v23/final/final_test.json")
        == V23_FINAL_SHA256
    )
    assert (
        _sha(REPO / "results/online/proximal_v24/protocol.json") == V24_PROTOCOL_SHA256
    )
    assert (
        _sha(REPO / "results/online/proximal_v24/final/final_test.json")
        == V24_FINAL_SHA256
    )
    for version, expected in (
        ("v23", V23_RESULT_GIT_TREE),
        ("v24", V24_RESULT_GIT_TREE),
    ):
        actual = subprocess.run(
            ["git", "rev-parse", f"HEAD:results/online/proximal_{version}"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert actual == expected


def test_phase_selective_scale_changes_exactly_three_swing_columns() -> None:
    foot = torch.tensor((0, 1, -1, 0))
    scale = swing_leg_action_scale(
        foot,
        action_dim=12,
        left_joint_indices=(0, 3, 4),
        right_joint_indices=(6, 9, 10),
        gain=0.82,
        dtype=torch.float32,
    )
    assert torch.count_nonzero(scale[0] != 1.0) == 3
    assert torch.count_nonzero(scale[1] != 1.0) == 3
    assert torch.equal(scale[2], torch.ones(12))
    assert scale[0, (0, 3, 4)].tolist() == pytest.approx([0.82] * 3)
    assert scale[1, (6, 9, 10)].tolist() == pytest.approx([0.82] * 3)
    assert torch.equal(scale[3], scale[0])


def test_teacher_inverts_hidden_plant_instead_of_double_scaling() -> None:
    safe = torch.tensor(((0.40, -0.24, 0.10), (0.08, 0.16, -0.32)))
    scale = torch.tensor(((0.8, 1.0, 0.8), (1.0, 0.8, 1.0)))
    teacher, reprojected = actor_coordinate_teacher_action(safe, scale)
    assert teacher[0].tolist() == pytest.approx([0.5, -0.24, 0.125])
    assert teacher[1].tolist() == pytest.approx([0.08, 0.2, -0.32])
    assert torch.allclose(reprojected, safe, atol=1.0e-7, rtol=0.0)
    assert not torch.allclose(scale * safe, safe)


def test_success_labels_require_crossing_and_no_fall_without_episode_leak() -> None:
    intervened = torch.zeros(7, 3, dtype=torch.bool)
    intervened[0, 0] = True
    intervened[2, 1] = True
    intervened[3, 1] = True
    intervened[3, 2] = True
    correction = torch.full((7, 3), 0.025)
    pre = torch.tensor(
        (
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (1, 0, 2),
            (1, 0, 0),
            (1, 1, 1),
            (1, 1, 2),
        )
    )
    post = torch.tensor(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 0, 0),
            (1, 0, 2),
            (1, 0, 0),
            (1, 1, 1),
            (1, 2, 2),
        )
    )
    fell = torch.zeros(7, 3, dtype=torch.bool)
    fell[3, 1] = True
    dones = torch.zeros(7, 3, dtype=torch.bool)
    dones[3, 1] = True
    dones[3, 2] = True
    eligible, weights, diagnostics = successful_teacher_labels(
        intervened,
        correction,
        pre,
        post,
        fell,
        dones,
        horizon=4,
        correction_scale=0.05,
    )
    assert eligible[0, 0]
    assert not eligible[2, 1]  # a fall occurs before/at the horizon boundary
    assert not eligible[3, 2]  # crossing after this done belongs to a new episode
    assert weights[0, 0] == pytest.approx(0.5)
    assert diagnostics["crossed_within_horizon"][0, 0]
    assert not diagnostics["no_fall_within_horizon"][2, 1]


def test_success_label_horizon_is_inclusive_of_exactly_h_transitions() -> None:
    intervened = torch.tensor(((True,), (False,), (False,), (False,), (False,)))
    correction = torch.full((5, 1), 0.05)
    pre = torch.zeros(5, 1, dtype=torch.long)
    post = torch.tensor(((0,), (0,), (0,), (1,), (1,)))
    zeros = torch.zeros(5, 1, dtype=torch.bool)
    eligible_h4, _, _ = successful_teacher_labels(
        intervened,
        correction,
        pre,
        post,
        zeros,
        zeros,
        horizon=4,
        correction_scale=0.05,
    )
    eligible_h3, _, _ = successful_teacher_labels(
        intervened,
        correction,
        pre,
        post,
        zeros,
        zeros,
        horizon=3,
        correction_scale=0.05,
    )
    assert eligible_h4[0, 0]
    assert not eligible_h3[0, 0]


def test_teacher_loss_uses_valid_count_and_empty_minibatch_is_exact_zero() -> None:
    mean = torch.tensor(((1.0, 0.0), (2.0, 0.0)), requires_grad=True)
    std = torch.ones_like(mean)
    target = torch.zeros_like(mean)
    eligible = torch.tensor((True, True))
    weights = torch.tensor((1.0, 0.5))
    loss = weighted_gaussian_teacher_loss(mean, std, target, eligible, weights)
    assert float(loss.detach()) == pytest.approx((0.5 + 1.0) / 2.0)
    loss.backward()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    empty_mean = torch.ones(2, 2, requires_grad=True)
    empty = weighted_gaussian_teacher_loss(
        empty_mean,
        torch.ones_like(empty_mean),
        torch.zeros_like(empty_mean),
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2),
    )
    assert float(empty) == 0.0
    empty.backward()
    assert torch.equal(empty_mean.grad, torch.zeros_like(empty_mean))


def test_kick_events_are_debounced_entries_into_exact_halfspace() -> None:
    previous = torch.tensor((False, True, False))
    event, overlap = toe_riser_kick_event(
        torch.tensor((-0.01, -0.02, 0.01)),
        torch.tensor((True, True, True)),
        previous,
    )
    assert event.tolist() == [True, False, False]
    assert overlap.tolist() == [True, True, False]


def test_calibration_grid_and_gate_have_exact_inclusive_boundaries() -> None:
    assert CALIBRATION_GAINS[0] == 0.98
    assert CALIBRATION_GAINS[-1] == 0.50
    assert all(earlier > later for earlier, later in pairwise(CALIBRATION_GAINS))
    assert CALIBRATION_EPISODES == 512
    lower = calibration_gate(
        off_success_count=205,
        on_success_count=410,
        off_toe_riser_failure_count=246,
        off_failure_count=307,
        rescued_count=185,
    )
    assert lower["qualifies"]
    upper = calibration_gate(
        off_success_count=332,
        on_success_count=486,
        off_toe_riser_failure_count=144,
        off_failure_count=180,
        rescued_count=108,
    )
    assert upper["qualifies"]
    assert not calibration_gate(
        off_success_count=333,
        on_success_count=486,
        off_toe_riser_failure_count=144,
        off_failure_count=179,
        rescued_count=108,
    )["qualifies"]


def test_calibrated_context_is_pure_and_first_qualifier() -> None:
    context = _qualifying_context()
    assert (
        validate_v25_calibrated_context(context)["parameters_sha256"]
        == context["parameters_sha256"]
    )
    not_first = _qualifying_context()
    not_first["calibration"]["attempts"][0]["qualifies"] = True
    with pytest.raises(ValueError, match="first qualifier"):
        validate_v25_calibrated_context(not_first)
    adapted = _qualifying_context()
    adapted["calibration"]["adapted_policy_evaluations_used"] = True
    with pytest.raises(ValueError, match="adapted outcomes"):
        validate_v25_calibrated_context(adapted)


def test_algorithm_contract_extends_only_teacher_fields() -> None:
    actual = formal_algorithm_parameters()
    assert actual["rounds"] == 8
    assert actual["num_envs"] == 64
    assert actual["rollout_steps"] == 1024
    assert actual["moving_kl_beta"] == 0.5
    assert actual["teacher_distillation_weight"] == 0.1
    assert actual["teacher_success_horizon_steps"] == 50
    assert actual["teacher_success_horizon_seconds"] == 1.0
    assert actual["teacher_correction_scale"] == 0.05
    assert actual["empty_teacher_minibatch"] == "exact differentiable zero"


def test_development_gate_exactly_matches_requested_outcomes() -> None:
    passed = development_gate(
        off_success_delta=0.05,
        on_success_delta=0.0,
        base_off_kick_rate=0.5,
        final_off_kick_rate=0.49,
        base_on_intervention_per_riser=2.0,
        final_on_intervention_per_riser=1.6,
    )
    assert passed["passed"]
    assert passed["intervention_per_riser_relative_reduction"] == pytest.approx(0.2)
    failed = development_gate(
        off_success_delta=0.049,
        on_success_delta=-0.001,
        base_off_kick_rate=0.5,
        final_off_kick_rate=0.5,
        base_on_intervention_per_riser=2.0,
        final_on_intervention_per_riser=1.61,
    )
    assert not failed["passed"]
    assert not failed["used_for_training_selection_or_rollback"]


def test_v25_randomness_is_fresh_and_semantic_arm_pairing_is_explicit() -> None:
    seeds = all_v25_fresh_execution_seeds()
    assert ADAPTATION_SEED in seeds
    assert len(seeds) == len(set(seeds))
    report = fresh_randomness_report(REPO)
    assert report["passed"]
    assert report["collisions"] == []


def test_training_ast_has_no_forbidden_selection_or_bank_path() -> None:
    source = (REPO / "experiments/scripts/refine_cbf_teacher_v25.py").read_text()
    tree = ast.parse(source)
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not any("hard_cases" in module for module in imported)
    for forbidden in (
        "failure_precursor_bank",
        "state_restart",
        "candidate_fractions",
        "select_best_so_far",
        "performance_gate",
        "deployable_failure",
    ):
        assert forbidden not in names
    assert "round 8 actor, never best-so-far" in source
    teacher_source = (REPO / "src/tasks/stairs_cbf/teacher.py").read_text()
    assert "self.storage.actions.flatten(0, 1)" in teacher_source
    assert "weighted_gaussian_teacher_loss" in teacher_source
    assert "teacher_distillation_weight * teacher_loss" in teacher_source


def test_final_audit_has_exact_four_conditions_and_512_pair_rows() -> None:
    source = (REPO / "experiments/scripts/audit_cbf_teacher_v25.py").read_text()
    assert FINAL_EPISODES == 512
    for condition in ("pi0_off", "pi0_on", "pi8_on", "pi8_off"):
        assert condition in source
    assert "same_initial_conditions_all_four_arms" in source
    assert FIGURE_CATEGORIES == (
        "round_teacher_curve",
        "four_condition_performance",
        "internalization_and_cbf_dependence",
    )
