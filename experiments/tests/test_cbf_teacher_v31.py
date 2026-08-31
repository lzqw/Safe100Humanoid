"""Pure checks for the prospectively fixed v31 formal matrix."""

from __future__ import annotations

import ast
import importlib.util
import json
from dataclasses import dataclass
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


MATH = _load("v31_teacher_math", "src/tasks/stairs_cbf/teacher_v30_math.py")
PROTOCOL = _load("v31_protocol", "experiments/scripts/cbf_teacher_v31_protocol.py")


def test_v31_exact_three_by_three_matrix_and_frozen_seeds() -> None:
    assert tuple(PROTOCOL.ARMS) == ("A0", "A1", "A2")
    assert PROTOCOL.METHOD_ARMS == ("A0", "A1", "A2")
    assert PROTOCOL.FORMAL_CONTEXTS == ("F1", "F2", "F3")
    assert PROTOCOL.FORMAL_ADAPTATION_SEEDS == {
        "F1": 181_310_001,
        "F2": 181_320_001,
        "F3": 181_330_001,
    }
    assert PROTOCOL.ARMS["A1"]["teacher_weight"] == 0.1
    assert PROTOCOL.ARMS["A2"]["teacher_eta"] == 0.25
    json.dumps(PROTOCOL.ARMS)


def test_v31_context_patch_counts_include_top_platform() -> None:
    assert PROTOCOL.environment_parameters("F1")["stair_target_patch_slots"] == 10
    assert PROTOCOL.environment_parameters("F2")["stair_target_patch_slots"] == 10
    f3 = PROTOCOL.environment_parameters("F3")
    assert f3["num_risers"] == 11
    assert f3["stair_target_patch_slots"] == 12
    assert f3["riser_profile_m"] == list(PROTOCOL.F3_PROFILE_M)


def test_v31_dynamic_patch_resize_changes_both_allocations() -> None:
    source = (REPO / "src/tasks/stairs_cbf/environment_v31.py").read_text()
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "resize_stair_patch_allocation"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="dataclasses", names=[ast.alias("replace")], level=0),
            helper,
        ],
        type_ignores=[],
    )
    namespace: dict[str, object] = {}
    exec(  # noqa: S102 - execute only the selected local pure helper AST
        compile(ast.fix_missing_locations(module), "environment_v31_helper", "exec"),
        namespace,
    )

    @dataclass
    class Patch:
        num_patches: int

    @dataclass
    class Stairs:
        num_steps: int
        flat_patch_sampling: dict[str, Patch]

    stairs = Stairs(
        num_steps=9,
        flat_patch_sampling={
            "stair_targets": Patch(10),
            "stair_risers": Patch(9),
        },
    )
    resized = namespace["resize_stair_patch_allocation"](stairs, num_risers=11)
    assert resized.num_steps == 11
    assert resized.flat_patch_sampling["stair_targets"].num_patches == 12
    assert resized.flat_patch_sampling["stair_risers"].num_patches == 11


def test_v31_behavior_log_prob_tolerance_is_exactly_1e_3() -> None:
    source = (REPO / "src/tasks/stairs_cbf/online.py").read_text()
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "BEHAVIOR_LOG_PROB_ATOL"
            for target in node.targets
        )
    )
    assert ast.literal_eval(assignment.value) == 1.0e-3
    assert PROTOCOL.BEHAVIOR_LOG_PROB_ATOL == 1.0e-3


def test_v31_residual_target_and_weighted_loss_reuse_fixed_v30_math() -> None:
    reference = torch.tensor(((1.0, -2.0),), requires_grad=True)
    raw = torch.tensor(((0.2, 0.4),), requires_grad=True)
    safe = torch.tensor(((0.4, 0.0),), requires_grad=True)
    target, correction = MATH.residual_teacher_target(reference, safe, raw, eta=0.25)
    assert torch.allclose(correction, torch.tensor(((0.2, -0.4),)))
    assert torch.allclose(target, torch.tensor(((1.05, -2.10),)))
    assert not target.requires_grad and not correction.requires_grad
    mean = torch.zeros_like(target, requires_grad=True)
    loss = MATH.weighted_smooth_l1_teacher_loss(
        mean, target, torch.tensor((True,)), torch.tensor((1.0,)), beta=0.05
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()


def test_v31_empty_teacher_loss_remains_differentiable_zero() -> None:
    mean = torch.ones(4, 3, requires_grad=True)
    loss = MATH.weighted_smooth_l1_teacher_loss(
        mean,
        torch.zeros_like(mean),
        torch.zeros(4, dtype=torch.bool),
        torch.zeros(4),
    )
    assert float(loss) == pytest.approx(0.0)
    loss.backward()
    assert torch.equal(mean.grad, torch.zeros_like(mean))


def test_v31_training_has_no_kl_or_performance_stop_control_flow() -> None:
    source = (REPO / "src/tasks/stairs_cbf/teacher_v30.py").read_text()
    tree = ast.parse(source)
    update = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "update"
    )
    identifiers = {node.id for node in ast.walk(update) if isinstance(node, ast.Name)}
    identifiers |= {
        node.attr for node in ast.walk(update) if isinstance(node, ast.Attribute)
    }
    assert "desired_kl" not in identifiers
    assert "hard_kl_ceiling" not in identifiers
    assert "rollback" not in identifiers
    assert "candidate" not in identifiers
    assert "performance" not in identifiers


def test_v31_freeze_preserves_v30_as_historical_result() -> None:
    freeze = (REPO / "experiments/scripts/freeze_cbf_teacher_v31.py").read_text()
    assert '"v30"' in freeze
    assert "immutable incomplete formal result" in freeze
