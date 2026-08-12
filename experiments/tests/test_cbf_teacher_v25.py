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
    ENVIRONMENT_VARIANT,
    FINAL_EPISODES,
    FINAL_REPEATS,
    PRECALIBRATION_REVISION,
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
    final_evaluation_seed,
    fixed_deployment_audit_contract,
    formal_algorithm_parameters,
    fresh_randomness_report,
    validate_v25_calibrated_context,
)
from freeze_cbf_teacher_v25_precalibration import _committed_protocol_chain
from plot_cbf_teacher_v25 import FIGURE_CATEGORIES
from verify_cbf_teacher_v25 import (
    exact_final_identity_schedule,
    execution_markers_are_valid,
    protocol_input_bindings_are_valid,
    reconstructed_fields_match,
    rollback_reasons_are_protocol_allowed,
    round_actor_hash_chain_is_valid,
    round_status_accounting_is_valid,
    supersession_revision_field_is_valid,
    teacher_signal_accounting_is_valid,
    training_execution_contract_is_valid,
    updated_metric_is_bounded,
    updated_round_dataflow_is_valid,
    updated_round_kl_is_valid,
)

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
        "environment_variant": ENVIRONMENT_VARIANT,
        "actor_observation_corruption": "disabled",
        "encoder_bias": "absent",
        "curriculum": "disabled",
        "fresh_initial_state_reset_events": ["reset_base", "reset_robot_joints"],
    }
    attempts = []
    for candidate in range(index + 1):
        gate = (
            calibration_gate(
                off_success_count=256,
                on_success_count=460,
                off_toe_riser_failure_count=240,
                off_failure_count=256,
                rescued_count=204,
            )
            if candidate == index
            else calibration_gate(
                off_success_count=400,
                on_success_count=480,
                off_toe_riser_failure_count=100,
                off_failure_count=112,
                rescued_count=80,
            )
        )
        attempts.append(
            {
                "candidate_index": candidate,
                "swing_underresponse_gain": CALIBRATION_GAINS[candidate],
                "base_policy_only": True,
                "adapted_policy_evaluations_used": False,
                "evaluation_seeds": [
                    calibration_evaluation_seed(candidate, repeat)
                    for repeat in range(CALIBRATION_REPEATS)
                ],
                "actor_state_sha256": "a" * 64,
                "off_initial_state_signatures": [
                    f"candidate-{candidate}-repeat-{repeat}"
                    for repeat in range(CALIBRATION_REPEATS)
                ],
                "on_initial_state_signatures": [
                    f"candidate-{candidate}-repeat-{repeat}"
                    for repeat in range(CALIBRATION_REPEATS)
                ],
                **gate,
            }
        )
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
            "environment_variant": ENVIRONMENT_VARIANT,
            "actor_observation_corruption": "disabled",
            "encoder_bias": "absent",
            "curriculum": "disabled",
            "fresh_initial_state_reset_events": [
                "reset_base",
                "reset_robot_joints",
            ],
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


def test_v25_zero_episode_protocol_history_is_contiguous_and_committed() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result_root = REPO / "results/online/proximal_v25"
    current = (
        result_root / f"precalibration_protocol_revision{PRECALIBRATION_REVISION}.json"
    )
    latest = (
        current
        if current.is_file()
        else result_root
        / f"precalibration_protocol_revision{PRECALIBRATION_REVISION - 1}.json"
    )
    chain = _committed_protocol_chain(
        REPO,
        commit,
        latest,
        result_root,
    )
    newest = (
        PRECALIBRATION_REVISION if current.is_file() else PRECALIBRATION_REVISION - 1
    )
    assert [item["revision"] for item in chain] == list(range(newest, 0, -1))
    assert Path(chain[0]["file"]).name == latest.name
    assert Path(chain[-1]["file"]).name == "precalibration_protocol.json"
    assert chain[-1]["payload"].get("revision") is None
    assert all(
        item["payload"]["prospective_execution"]["v25_simulator_episode_started"]
        is False
        for item in chain
    )


def test_v25_csv_evidence_has_stable_bytes_across_git_commit() -> None:
    evidence_path = "results/online/proximal_v25/calibration/evidence.csv"
    attributes = subprocess.run(
        ["git", "check-attr", "text", "binary", "--", evidence_path],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert f"{evidence_path}: text: unset" in attributes
    assert f"{evidence_path}: binary: set" in attributes
    for relative in (
        "experiments/scripts/evaluate_cbf_teacher_v25.py",
        "experiments/scripts/calibrate_cbf_teacher_v25.py",
        "experiments/scripts/refine_cbf_teacher_v25.py",
        "experiments/scripts/audit_cbf_teacher_v25.py",
    ):
        assert 'lineterminator="\\n"' in (REPO / relative).read_text()


def test_verifier_accepts_only_the_legacy_revision_one_missing_field() -> None:
    legacy = {
        "file": "results/online/proximal_v25/precalibration_protocol.json"
    }
    assert supersession_revision_field_is_valid({}, legacy, 1)
    assert supersession_revision_field_is_valid(
        {"supersedes_revision": 2}, {"file": "revision2.json"}, 2
    )
    assert not supersession_revision_field_is_valid({}, legacy, 2)
    assert not supersession_revision_field_is_valid(
        {}, {"file": "results/online/proximal_v25/other.json"}, 1
    )


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


def test_rollout_truncation_is_not_mislabeled_as_survival() -> None:
    intervened = torch.tensor(((False,), (False,), (True,)))
    correction = torch.full((3, 1), 0.05)
    pre = torch.zeros(3, 1, dtype=torch.long)
    post = torch.tensor(((0,), (0,), (1,)))
    zeros = torch.zeros(3, 1, dtype=torch.bool)
    eligible, _, diagnostics = successful_teacher_labels(
        intervened,
        correction,
        pre,
        post,
        zeros,
        zeros,
        horizon=3,
        correction_scale=0.05,
    )
    assert diagnostics["crossed_within_horizon"][2, 0]
    assert diagnostics["no_fall_within_horizon"][2, 0]
    assert not diagnostics["horizon_outcome_observed"][2, 0]
    assert not eligible[2, 0]


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
    previous_identity = torch.tensor((-1, 3, 7, 8))
    event, overlap, next_identity = toe_riser_kick_event(
        torch.tensor((-0.01, -0.02, 0.01, -0.03)),
        torch.tensor((True, True, True, True)),
        torch.tensor((3, 3, 7, 9)),
        previous_identity,
    )
    assert event.tolist() == [True, False, False, True]
    assert overlap.tolist() == [True, True, False, True]
    assert next_identity.tolist() == [3, 3, -1, 9]


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
    first_gate = calibration_gate(
        off_success_count=256,
        on_success_count=460,
        off_toe_riser_failure_count=240,
        off_failure_count=256,
        rescued_count=204,
    )
    not_first["calibration"]["attempts"][0].update(first_gate)
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
    zero_base = development_gate(
        off_success_delta=0.05,
        on_success_delta=0.0,
        base_off_kick_rate=0.5,
        final_off_kick_rate=0.49,
        base_on_intervention_per_riser=0.0,
        final_on_intervention_per_riser=0.0,
    )
    assert not zero_base["passed"]
    assert zero_base["intervention_per_riser_relative_reduction"] == 0.0


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
    assert "load_env_cfg(TASK_ID, play=True)" in source
    teacher_source = (REPO / "src/tasks/stairs_cbf/teacher.py").read_text()
    assert "self.storage.actions.flatten(0, 1)" in teacher_source
    assert "weighted_gaussian_teacher_loss" in teacher_source
    assert "teacher_distillation_weight * teacher_loss" in teacher_source
    runner_source = (REPO / "experiments/scripts/run_cbf_teacher_v25.sh").read_text()
    assert (
        f"precalibration_protocol_revision{PRECALIBRATION_REVISION}.json"
        in runner_source
    )


def test_verifier_compares_reconstructed_gate_fields_not_candidate_metadata() -> None:
    gate = calibration_gate(
        off_success_count=256,
        on_success_count=460,
        off_toe_riser_failure_count=240,
        off_failure_count=256,
        rescued_count=204,
    )
    selected_attempt = {
        "candidate_index": 8,
        "swing_underresponse_gain": 0.82,
        "evaluation_seeds": [1, 2, 3, 4],
        **gate,
    }
    assert gate != selected_attempt
    assert reconstructed_fields_match(gate, selected_attempt)
    selected_attempt["rescued_count"] += 1
    assert not reconstructed_fields_match(gate, selected_attempt)


def test_verifier_accepts_valid_all_rollback_and_zero_teacher_outcomes() -> None:
    rollbacks = [
        {
            "round": index,
            "status": "hard_rollback",
            "rollback_reason": "moving forward KL exceeded hard ceiling",
            "round_start_actor_sha256": "a" * 64,
            "round_end_actor_sha256": "a" * 64,
            "round_reference_is_moving_pi_k": True,
            "performance_evaluation_or_gate_used": False,
            "metrics": {
                "hard_rollback": True,
                "hard_rollback_reason": "moving forward KL exceeded hard ceiling",
            },
        }
        for index in range(1, 9)
    ]
    assert updated_round_kl_is_valid(rollbacks)
    assert teacher_signal_accounting_is_valid(rollbacks)
    assert round_status_accounting_is_valid(rollbacks)
    assert rollback_reasons_are_protocol_allowed(rollbacks)
    assert round_actor_hash_chain_is_valid(rollbacks)
    assert updated_metric_is_bounded(
        rollbacks, "policy_storage_max_abs_error", maximum=1.0e-6
    )

    zero_signal_update = {
        "round": 1,
        "status": "updated",
        "rollback_reason": None,
        "round_start_actor_sha256": "a" * 64,
        "round_end_actor_sha256": "b" * 64,
        "round_reference_is_moving_pi_k": True,
        "performance_evaluation_or_gate_used": False,
        "metrics": {
            "moving_forward_kl": 0.002,
            "teacher_transition_count": 0.0,
            "teacher_transition_fraction": 0.0,
            "teacher_loss": 0.0,
            "teacher_minibatches_with_signal": 0,
            "teacher_minibatches_without_signal": 8,
            "actor_epochs_completed": 2,
            "actor_minibatches_completed": 8,
            "teacher_samples_seen_across_epochs": 0,
            "policy_storage_max_abs_error": 0.0,
        },
    }
    assert updated_round_kl_is_valid([zero_signal_update])
    assert teacher_signal_accounting_is_valid([zero_signal_update])
    assert round_status_accounting_is_valid([zero_signal_update])
    assert rollback_reasons_are_protocol_allowed([zero_signal_update])
    assert round_actor_hash_chain_is_valid([zero_signal_update])
    assert updated_metric_is_bounded(
        [zero_signal_update],
        "policy_storage_max_abs_error",
        minimum=0.0,
        maximum=1.0e-6,
    )

    corrupt_zero_signal = {
        **zero_signal_update,
        "metrics": {**zero_signal_update["metrics"], "teacher_loss": 0.1},
    }
    assert not teacher_signal_accounting_is_valid([corrupt_zero_signal])
    excessive_kl = {
        **zero_signal_update,
        "metrics": {**zero_signal_update["metrics"], "moving_forward_kl": 0.02},
    }
    assert not updated_round_kl_is_valid([excessive_kl])
    missing_routing = {
        **zero_signal_update,
        "metrics": {
            key: value
            for key, value in zero_signal_update["metrics"].items()
            if key != "policy_storage_max_abs_error"
        },
    }
    assert not updated_metric_is_bounded(
        [missing_routing], "policy_storage_max_abs_error", maximum=1.0e-6
    )

    malformed = {
        **zero_signal_update,
        "metrics": {
            **zero_signal_update["metrics"],
            "moving_forward_kl": "not-a-number",
            "teacher_transition_count": None,
        },
    }
    assert not updated_round_kl_is_valid([malformed])
    assert not teacher_signal_accounting_is_valid([malformed])

    missing_actor_hash = {
        **zero_signal_update,
        "round_end_actor_sha256": None,
    }
    assert not round_actor_hash_chain_is_valid([missing_actor_hash])

    hidden_performance_gate = {
        **zero_signal_update,
        "performance_evaluation_or_gate_used": True,
    }
    assert not round_status_accounting_is_valid([hidden_performance_gate])
    assert not updated_metric_is_bounded(
        [None], "policy_storage_max_abs_error", maximum=1.0e-6
    )


def test_verifier_rejects_tampered_final_identity_schedule() -> None:
    rows = [
        {
            "pair_index": str(index),
            "evaluation_seed": str(final_evaluation_seed(index // 128)),
            "environment_id": str(index % 128),
        }
        for index in range(FINAL_EPISODES)
    ]
    assert FINAL_REPEATS == 4
    assert exact_final_identity_schedule(rows)
    rows[0]["evaluation_seed"] = "999999999"
    assert not exact_final_identity_schedule(rows)
    rows[0]["evaluation_seed"] = str(final_evaluation_seed(0))
    rows[0]["pair_index"] = "1"
    assert not exact_final_identity_schedule(rows)


def test_verifier_binds_every_formal_protocol_input_hash() -> None:
    digest = "a" * 64
    protocol = {
        "implementation_boundary": {
            "precalibration_protocol": {"sha256": digest},
            "calibrated_context": {"sha256": digest},
            "calibration_execution_started": {"sha256": digest},
            "calibration_summary": {"sha256": digest},
            "calibration_attempts": {"sha256": digest},
            "calibration_paired_episodes": {"sha256": digest},
            "calibration_all_evaluated_paired_episodes": {"sha256": digest},
            "calibration_evidence_verification": {"sha256": digest},
        },
        "calibration_evidence": {
            "sha256": digest,
            "execution_started": {"sha256": digest},
            "attempts": {"sha256": digest},
            "paired_episodes": {"sha256": digest},
            "all_evaluated_paired_episodes": {"sha256": digest},
            "independent_reconstruction": {"sha256": digest},
        },
    }
    kwargs = {
        "precalibration_sha256": digest,
        "context_sha256": digest,
        "calibration_started_sha256": digest,
        "calibration_summary_sha256": digest,
        "calibration_attempts_sha256": digest,
        "calibration_paired_sha256": digest,
        "calibration_all_paired_sha256": digest,
        "calibration_verification_sha256": digest,
    }
    assert protocol_input_bindings_are_valid(protocol, **kwargs)
    protocol["implementation_boundary"]["calibration_summary"]["sha256"] = "b" * 64
    assert not protocol_input_bindings_are_valid(protocol, **kwargs)


def test_verifier_binds_execution_markers_and_rejects_hidden_repeats() -> None:
    digest = "a" * 64
    commit = "b" * 40
    protocol = {"implementation_boundary": {"git_commit": commit}}
    reference = {
        "file": "protocol.json",
        "sha256": digest,
        "implementation_commit": commit,
        "validation": {"protocol_id": True},
    }
    kwargs = {
        "protocol": protocol,
        "precalibration": {
            "revision": PRECALIBRATION_REVISION,
            "prospective_execution": {"v25_simulator_episode_started": False},
        },
        "calibration_started": {
            "protocol_id": "safe100-success-gated-cbf-teacher-v25",
            "precalibration_protocol_sha256": digest,
            "base_policy_only": True,
            "adapted_policy_evaluations_used": False,
            "ordered_first_qualifier_rule": True,
        },
        "training_started": {
            "protocol": reference,
            "adapted_policy_outcomes_observed": False,
            "fresh_adaptation_count": 1,
        },
        "training_completed": {
            "protocol": reference,
            "adapted_policy_outcomes_observed": True,
            "fresh_adaptation_count": 1,
            "final_actor_sha256": digest,
            "training_summary_sha256": digest,
        },
        "final_started": {
            "protocol_id": "safe100-success-gated-cbf-teacher-v25",
            "protocol_sha256": digest,
            "training_summary_sha256": digest,
            "base_checkpoint_sha256": digest,
            "final_checkpoint_sha256": digest,
            "condition_order": ["pi0_off", "pi0_on", "pi8_on", "pi8_off"],
            "fresh_condition_count": FINAL_EPISODES,
        },
        "protocol_sha256": digest,
        "precalibration_sha256": digest,
        "training_sha256": digest,
        "base_checkpoint_sha256": digest,
        "final_checkpoint_sha256": digest,
        "final_actor_sha256": digest,
    }
    assert execution_markers_are_valid(**kwargs)
    kwargs["training_started"] = {
        **kwargs["training_started"],
        "fresh_adaptation_count": 2,
    }
    assert not execution_markers_are_valid(**kwargs)


def test_verifier_rejects_tampered_exclusion_counts() -> None:
    context = _qualifying_context()
    algorithm = formal_algorithm_parameters()
    audit = {
        "algorithm_class": True,
        "action_config_class": True,
        "actor_observation_dim": 405,
        "critic_observation_dim": 838,
        "actor_observation_groups": ["actor"],
        "critic_observation_groups": ["actor", "critic", "online_privileged"],
        "deployable_failure_group_absent": True,
        "one_actor": True,
        "one_privileged_critic": True,
        "auxiliary_critics_absent": True,
        "specialist_reward_absent": True,
        "runtime_filter": True,
        "phase_selective_shift": True,
        "actor_critic_optimizers_disjoint": True,
        "log_std_trainable_parameter_count": 0,
        "actor_learning_rate": algorithm["actor_learning_rate"],
        "critic_learning_rate": algorithm["critic_learning_rate"],
        "ppo_clip": algorithm["ppo_clip"],
        "maximum_actor_epochs": algorithm["maximum_actor_epochs"],
        "critic_epochs": algorithm["critic_epochs"],
        "mini_batches": algorithm["mini_batches"],
        "moving_kl_beta": algorithm["moving_kl_beta"],
        "target_kl": algorithm["target_kl"],
        "hard_kl_ceiling": algorithm["hard_kl_ceiling"],
        "teacher_distillation_weight": algorithm["teacher_distillation_weight"],
        "teacher_success_horizon": algorithm["teacher_success_horizon_steps"],
        "teacher_correction_scale": algorithm["teacher_correction_scale"],
    }
    validation = {
        key: True
        for key in (
            "protocol_id",
            "method",
            "implementation_is_ancestor",
            "randomness_preflight",
            "base_checkpoint",
            "context_file",
            "context_parameters",
            "algorithm",
            "environment",
            "execution_not_started_at_freeze",
            "all_bound_sources_unchanged",
            "protocol_committed_at_head",
            "context_committed_at_head",
        )
    }
    rounds = [{"round_start_actor_sha256": "a" * 64}]
    protocol = {"implementation_boundary": {"git_commit": "c" * 40}}
    training = {
        "schema_version": 1,
        "protocol_id": "safe100-success-gated-cbf-teacher-v25",
        "smoke": False,
        "adaptation_seed": ADAPTATION_SEED,
        "adaptation_seed_count": 1,
        "state_restart_count": 0,
        "failure_or_success_bank_count": 0,
        "context": {
            "selected_candidate_index": context["shift"]["selected_candidate_index"],
            "swing_underresponse_gain": context["shift"]["swing_underresponse_gain"],
            "base_policy_only_first_qualifier": True,
            "reused_without_reselection": True,
            "metadata": {
                "shift": "fixed_phase_selective_swing_leg_underresponse",
                "swing_underresponse_gain": context["shift"][
                    "swing_underresponse_gain"
                ],
                "affected_joints_per_swing_leg": [
                    "hip_pitch_joint",
                    "knee_joint",
                    "ankle_pitch_joint",
                ],
                "stance_leg_scale": 1.0,
                "all_other_action_scales": 1.0,
                "runtime_filter": True,
                "terrain_geometry_changed": False,
                "friction_changed": False,
                "command_changed": False,
                "controller_changed": False,
                "actor_observation_fields_added": 0,
                "cbf_geometry_exact": True,
                "fixed_deployment_environment": fixed_deployment_audit_contract(),
            },
        },
        "warm_start": {
            "actor_observation_dim": 405,
            "critic_observation_dim": 838,
            "actor_layout": "exact-original-interface",
            "critic_layout": "exact-original-privileged-interface",
            "source_optimizer_discarded": True,
            "source_auxiliary_heads_ignored": True,
        },
        "initial_actor_sha256": "a" * 64,
        "protocol": {
            "sha256": "d" * 64,
            "implementation_commit": "c" * 40,
            "validation": validation,
        },
        "structural_audit": audit,
    }
    assert training_execution_contract_is_valid(training, protocol, context, rounds)
    training["context"]["metadata"]["fixed_deployment_environment"][
        "curriculum_disabled"
    ] = False
    assert not training_execution_contract_is_valid(
        training, protocol, context, rounds
    )
    training["context"]["metadata"]["fixed_deployment_environment"] = (
        fixed_deployment_audit_contract()
    )
    training["state_restart_count"] = 1
    assert not training_execution_contract_is_valid(training, protocol, context, rounds)


def test_verifier_binds_updated_behavior_gaussian_metrics() -> None:
    metrics = {
        "behavior_reference_distribution_param_max_abs_error": 0.0,
        "behavior_current_distribution_param_max_abs_error": 0.0,
        "behavior_reference_log_prob_max_abs_error": 0.0,
        "behavior_current_log_prob_max_abs_error": 0.0,
        "moving_kl_beta": 0.5,
        "target_kl": 0.003,
        "hard_kl_ceiling": 0.01,
        "teacher_distillation_weight": 0.1,
        "teacher_success_horizon": 50,
        "teacher_correction_scale": 0.05,
        "freeze_log_std": True,
        "round_reference_index": 1,
    }
    rounds = [{"round": 1, "status": "updated", "metrics": metrics}]
    assert updated_round_dataflow_is_valid(rounds)
    rounds[0]["metrics"]["behavior_current_log_prob_max_abs_error"] = 0.01
    assert not updated_round_dataflow_is_valid(rounds)


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


def test_formal_freeze_binds_calibration_paired_episode_evidence() -> None:
    source = (
        REPO / "experiments/scripts/freeze_cbf_teacher_v25_protocol.py"
    ).read_text()
    assert "--calibration-paired-csv" in source
    assert '"paired_row_count_512"' in source
    assert '"calibration_paired_episodes"' in source
    verifier = (REPO / "experiments/scripts/verify_cbf_teacher_v25.py").read_text()
    assert "--calibration-paired-csv" in verifier
    assert '"calibration_paired_csv_bound"' in verifier
    assert '"calibration_gate_reconstructed"' in verifier


def test_evaluation_pools_interventions_over_all_crossed_risers() -> None:
    evaluator = (REPO / "experiments/scripts/evaluate_cbf_teacher_v25.py").read_text()
    audit = (REPO / "experiments/scripts/audit_cbf_teacher_v25.py").read_text()
    verifier = (REPO / "experiments/scripts/verify_cbf_teacher_v25.py").read_text()
    assert '"total_intervention_count": total_interventions' in evaluator
    assert "max(1, total_reached_risers)" in evaluator
    assert 'aggregate["total_intervention_count"]' in audit
    assert 'int(row[f"{condition}_intervention_count"])' in verifier
