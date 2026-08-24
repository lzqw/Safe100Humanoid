"""Focused pure-tensor tests for the v29 teacher semantics."""

from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "v29_teacher_math", REPO / "src/tasks/stairs_cbf/teacher_math.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
labels = MODULE.vectorized_successful_teacher_labels_v29
teacher_loss = MODULE.weighted_gaussian_teacher_loss_v29


def test_v29_labels_use_only_horizon_vectorized_loop() -> None:
    tree = ast.parse(inspect.getsource(labels))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)]
    assert len(loops) == 1
    assert isinstance(loops[0].target, ast.Name)
    assert loops[0].target.id == "offset"


def test_v29_labels_require_crossing_safe_horizon_and_same_episode() -> None:
    shape = (7, 4)
    intervened = torch.zeros(shape, dtype=torch.bool)
    intervened[0, 0] = True  # valid crossing
    intervened[0, 1] = True  # fall before crossing
    intervened[2, 2] = True  # crossing belongs to the next episode
    intervened[5, 3] = True  # rollout tail is incomplete
    correction = torch.full(shape, 0.025)
    pre = torch.zeros(shape, dtype=torch.long)
    post = torch.zeros(shape, dtype=torch.long)
    post[2, 0] = 1
    post[2:, 0] = 1
    post[2, 1] = 1
    post[2:, 1] = 1
    post[4, 2] = 1
    post[4:, 2] = 1
    post[5, 3] = 1
    post[5:, 3] = 1
    episode = torch.zeros(shape, dtype=torch.long)
    episode[3:, 2] = 1
    fell = torch.zeros(shape, dtype=torch.bool)
    fell[1, 1] = True
    recovery = torch.zeros_like(fell)
    emergency = torch.zeros_like(fell)
    dones = torch.zeros_like(fell)
    dones[2, 2] = True
    eligible, weights, diagnostics = labels(
        intervened,
        correction,
        pre,
        post,
        episode,
        fell,
        recovery,
        emergency,
        dones,
        horizon=4,
        correction_scale=0.05,
    )
    assert eligible[0].tolist() == [True, False, False, False]
    assert weights[0, 0] == pytest.approx(0.5)
    assert diagnostics["crossed_within_horizon"][0, 0]
    assert not diagnostics["no_fall_within_horizon"][0, 1]
    assert not diagnostics["horizon_outcome_observed"][5, 3]


@pytest.mark.parametrize("unsafe_kind", ("recovery", "emergency"))
def test_v29_labels_exclude_nonfall_unsafe_termination(unsafe_kind: str) -> None:
    shape = (4, 1)
    intervened = torch.tensor(((True,), (False,), (False,), (False,)))
    correction = torch.full(shape, 0.05)
    pre = torch.zeros(shape, dtype=torch.long)
    post = torch.tensor(((0,), (1,), (1,), (1,)))
    episode = torch.zeros(shape, dtype=torch.long)
    fell = torch.zeros(shape, dtype=torch.bool)
    recovery = torch.zeros_like(fell)
    emergency = torch.zeros_like(fell)
    (recovery if unsafe_kind == "recovery" else emergency)[1, 0] = True
    dones = torch.zeros_like(fell)
    dones[1, 0] = True
    eligible, _, diagnostics = labels(
        intervened,
        correction,
        pre,
        post,
        episode,
        fell,
        recovery,
        emergency,
        dones,
        horizon=4,
        correction_scale=0.05,
    )
    assert not eligible[0, 0]
    assert not diagnostics[
        f"no_{unsafe_kind}_termination_within_horizon"
        if unsafe_kind == "emergency"
        else "no_recovery_takeover_within_horizon"
    ][0, 0]


def test_v29_teacher_loss_normalizes_by_weight_sum_and_detaches_target() -> None:
    mean = torch.tensor(((1.0, 0.0), (2.0, 0.0)), requires_grad=True)
    std = torch.ones_like(mean)
    target = torch.zeros_like(mean, requires_grad=True)
    eligible = torch.tensor((True, True))
    weights = torch.tensor((1.0, 0.5))
    loss = teacher_loss(mean, std, target, eligible, weights)
    assert float(loss.detach()) == pytest.approx((0.5 + 1.0) / 1.5)
    loss.backward()
    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    assert target.grad is None


def test_v29_empty_teacher_minibatch_is_exact_differentiable_zero() -> None:
    mean = torch.ones(3, 2, requires_grad=True)
    loss = teacher_loss(
        mean,
        torch.ones_like(mean),
        torch.zeros_like(mean),
        torch.zeros(3, dtype=torch.bool),
        torch.zeros(3),
    )
    assert float(loss) == 0.0
    loss.backward()
    assert torch.equal(mean.grad, torch.zeros_like(mean))
