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
