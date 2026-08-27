"""Pure, simulator-free checks for the prospectively fixed v30 semantics."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, REPO / relative)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


MATH = _load("v30_teacher_math", "src/tasks/stairs_cbf/teacher_v30_math.py")
PROTOCOL = _load("v30_protocol", "experiments/scripts/cbf_teacher_v30_protocol.py")


def test_v30_exact_six_arm_matrix_and_contexts_are_json_safe() -> None:
    assert tuple(PROTOCOL.ARMS) == ("A0", "A1", "A2", "A3", "A4", "A5")
    assert tuple(PROTOCOL.TEACHER_ARMS) == ("A1", "A2", "A3", "A4", "A5")
    assert PROTOCOL.ARMS["A2"]["teacher_eta"] == 0.25
    assert PROTOCOL.ARMS["A3"]["teacher_eta"] == 0.5
    assert PROTOCOL.ARMS["A4"]["teacher_eta"] == 1.0
    assert PROTOCOL.ARMS["A5"]["teacher_gate"] == "local_success_50"
    f3 = PROTOCOL.environment_parameters("F3")
    assert f3["riser_profile_m"] == list(PROTOCOL.F3_PROFILE_M)
    json.dumps(
        {
            "arms": PROTOCOL.ARMS,
            "contexts": {
                name: PROTOCOL.environment_parameters(name)
                for name in ("DEV", "F1", "F2", "F3", "D0")
            },
        }
    )


def test_v30_residual_target_uses_round_reference_and_sample_correction() -> None:
    reference = torch.tensor(((1.0, -2.0),), requires_grad=True)
    raw = torch.tensor(((0.2, 0.4),), requires_grad=True)
    safe = torch.tensor(((0.4, 0.0),), requires_grad=True)
    target, correction = MATH.residual_teacher_target(reference, safe, raw, eta=0.5)
    assert torch.allclose(correction, torch.tensor(((0.2, -0.4),)))
    assert torch.allclose(target, torch.tensor(((1.1, -2.2),)))
    assert not target.requires_grad and not correction.requires_grad


def test_v30_intervention_weights_use_actual_intervention_and_clipped_norm() -> None:
    intervened = torch.tensor(((True, False, True), (True, True, False)))
    norm = torch.tensor(((0.0, 0.025, 0.05), (0.10, 0.0125, 1.0)))
    eligible, weights = MATH.intervention_teacher_weights(intervened, norm)
    assert torch.equal(eligible, intervened)
    assert torch.allclose(weights, torch.tensor(((0.0, 0.0, 1.0), (1.0, 0.25, 0.0))))


def test_v30_smooth_l1_is_per_action_mean_and_weight_normalized() -> None:
    mean = torch.tensor(((0.10, 0.0), (0.05, 0.05)), requires_grad=True)
    target = torch.zeros_like(mean, requires_grad=True)
    eligible = torch.tensor((True, True))
    weights = torch.tensor((1.0, 0.5))
    loss = MATH.weighted_smooth_l1_teacher_loss(
        mean, target, eligible, weights, beta=0.05
    )
    # SmoothL1(beta=.05): [0.075, 0] -> mean .0375; [.025, .025] -> .025.
    assert float(loss.detach()) == pytest.approx((0.0375 + 0.5 * 0.025) / 1.5)
    loss.backward()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    assert target.grad is None


def test_v30_empty_teacher_loss_is_exact_differentiable_zero() -> None:
    mean = torch.ones(4, 3, requires_grad=True)
    loss = MATH.weighted_smooth_l1_teacher_loss(
        mean,
        torch.zeros_like(mean),
        torch.zeros(4, dtype=torch.bool),
        torch.zeros(4),
    )
    assert float(loss) == 0.0
    loss.backward()
    assert torch.equal(mean.grad, torch.zeros_like(mean))


def test_v35_masked_actor_mean_preserves_population_scale() -> None:
    values = torch.tensor((2.0, 100.0, 4.0), requires_grad=True)
    selected = MATH.masked_population_mean(
        values, torch.tensor((True, False, True))
    )
    assert float(selected) == pytest.approx(2.0)
    selected.backward()
    assert torch.allclose(values.grad, torch.tensor((1.0 / 3.0, 0.0, 1.0 / 3.0)))

    empty_values = torch.ones(3, requires_grad=True)
    empty = MATH.masked_population_mean(
        empty_values, torch.zeros(3, dtype=torch.bool)
    )
    assert float(empty) == 0.0
    empty.backward()
    assert torch.equal(empty_values.grad, torch.zeros_like(empty_values))


def test_v35_terminal_episode_mask_keeps_environment_identity() -> None:
    episode_ids = torch.tensor(
        ((0, 0), (0, 0), (1, 0), (1, 1)), dtype=torch.long
    )
    terminal = torch.tensor(
        ((False, False), (True, False), (False, True), (False, False))
    )
    mask = MATH.terminal_episode_transition_mask(episode_ids, terminal)
    assert torch.equal(
        mask,
        torch.tensor(
            ((True, True), (True, True), (False, True), (False, False))
        ),
    )


def test_v35_outcome_gate_selects_only_successful_interventions() -> None:
    intervened = torch.tensor(
        ((True, True, False), (True, False, True)), dtype=torch.bool
    )
    failed = torch.tensor(
        ((True, False, False), (True, False, False)), dtype=torch.bool
    )
    successful = torch.tensor(
        ((False, True, True), (False, True, True)), dtype=torch.bool
    )
    eligible = MATH.outcome_gated_interventions(
        intervened, failed, successful, gate="successful"
    )
    assert torch.equal(
        eligible,
        torch.tensor(((False, True, False), (False, False, True))),
    )


def test_v35_joint_top_and_fall_terminal_gets_success_priority() -> None:
    done = torch.tensor((True, True, True, False))
    fell = torch.tensor((True, True, False, True))
    reached_top = torch.tensor((True, False, True, False))
    failed, successful, joint = MATH.disjoint_terminal_outcomes(
        done, fell, reached_top
    )
    assert torch.equal(failed, torch.tensor((False, True, False, False)))
    assert torch.equal(successful, torch.tensor((True, False, True, False)))
    assert torch.equal(joint, torch.tensor((True, False, False, False)))
    assert not bool((failed & successful).any())


def test_v66_allows_success_local_kl_for_shielded_distillation_only() -> None:
    source = (REPO / "experiments/scripts/refine_paper_dual_v35.py").read_text()
    assert "shielded_distillation = (" in source
    assert "args.distill_only_actor" in source
    assert 'args.training_runtime_filter == "on"' in source
    assert 'args.training_filter_schedule == "fixed"' in source
    assert "training_filter_fraction == 1.0" in source
    assert "shielded_distillation or unshielded_ppo" in source


def test_v68_routes_mixed_execution_actor_objectives_before_update() -> None:
    source = (REPO / "src/tasks/stairs_cbf/paper_teacher_v35.py").read_text()
    assert "set_v35_filter_execution_environment_mask" in source
    assert "gated_intervention &= teacher_environment" in source
    assert "return ppo_environment" in source
    script = (REPO / "experiments/scripts/refine_paper_dual_v35.py").read_text()
    assert "runner.alg.set_v35_filter_execution_environment_mask(" in script


def test_v69_routes_actor_backward_through_task_priority_gradient_surgery() -> None:
    source = (REPO / "src/tasks/stairs_cbf/paper_teacher_v35.py").read_text()
    assert "task_priority_project_auxiliary_gradients(" in source
    assert "deployment_loss = (" in source
    assert "parameter.grad = deployment + teacher" in source
    script = (REPO / "experiments/scripts/refine_paper_dual_v35.py").read_text()
    assert '"--task-priority-gradient-surgery"' in script
    assert 'runner_cfg["algorithm"]["v35_task_priority_gradient_surgery"]' in script


def test_v35_filter_dropout_mask_is_balanced_and_rotates() -> None:
    first = MATH.rotating_environment_filter_mask(
        4, 0.5, 1, device="cpu"
    )
    second = MATH.rotating_environment_filter_mask(
        4, 0.5, 2, device="cpu"
    )
    assert torch.equal(first, torch.tensor((True, True, False, False)))
    assert torch.equal(second, torch.tensor((False, False, True, True)))
    assert torch.equal(
        MATH.rotating_environment_filter_mask(4, 1.0, 2, device="cpu"),
        torch.ones(4, dtype=torch.bool),
    )


def test_v56_filter_fraction_schedule_reaches_pure_unshielded_rollout() -> None:
    schedule = MATH.linear_filter_fraction_schedule(4, 1.0, 0.0)
    assert schedule == pytest.approx((1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0))
    assert int(
        MATH.rotating_environment_filter_mask(
            64, schedule[-1], 4, device="cpu"
        ).sum()
    ) == 0
    with pytest.raises(ValueError, match="strictly decrease"):
        MATH.linear_filter_fraction_schedule(4, 0.5, 0.5)


def test_v60_target_terrain_floor_prevents_late_curriculum_retreat() -> None:
    assert MATH.target_terrain_floor_schedule(6, 5, 4) == (0, 0, 0, 4, 4, 4)
    assert MATH.target_terrain_floor_schedule(4, 5, None) == (0, 0, 0, 0)
    with pytest.raises(ValueError, match="within training rounds"):
        MATH.target_terrain_floor_schedule(4, 5, 5)


def test_v36_rescue_gate_uses_matched_on_success_off_failure() -> None:
    on = torch.tensor((True, True, False, False))
    off = torch.tensor((False, True, False, True))
    assert torch.equal(
        MATH.filter_rescued_episode_mask(on, off),
        torch.tensor((True, False, False, False)),
    )


def test_v30_update_has_no_kl_threshold_control_flow() -> None:
    source = (REPO / "src/tasks/stairs_cbf/teacher_v30.py").read_text()
    tree = ast.parse(source)
    update = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "update"
    )
    names = {node.id for node in ast.walk(update) if isinstance(node, ast.Name)}
    attributes = {
        node.attr for node in ast.walk(update) if isinstance(node, ast.Attribute)
    }
    assert "desired_kl" not in names | attributes
    assert "hard_kl_ceiling" not in names | attributes
    assert "rollback" not in names | attributes
    assert "candidate" not in names | attributes
    assert "performance" not in names | attributes
    assert any(
        isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
        for node in ast.walk(update)
    )
