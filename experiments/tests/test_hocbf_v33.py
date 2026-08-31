"""Small pure checks for the prospectively fixed v33 HOCBF implementation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "experiments/scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, REPO / relative)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


MATH = _load("hocbf_v33_math", "src/tasks/stairs_cbf/hocbf_math.py")
PROTOCOL = _load("hocbf_v33_protocol_test", "experiments/scripts/hocbf_v33_protocol.py")


def test_v33_sherman_morrison_matches_dense_solve() -> None:
    torch.manual_seed(33)
    vector = torch.randn(7, 12, dtype=torch.float64)
    jac_x = torch.randn_like(vector)
    diagonal = torch.tensor((4, 2, 1, 1, 4, 4, 4, 2, 1, 1, 4, 4), dtype=torch.float64)
    actual = MATH.apply_task_metric_inverse(
        vector,
        diagonal,
        jac_x,
        forward_weight=8.0,
        smoothness_weight=0.1,
    )
    dense = torch.diag_embed(
        (diagonal + 0.1).expand_as(vector)
    ) + 8.0 * jac_x.unsqueeze(-1) * jac_x.unsqueeze(-2)
    expected = torch.linalg.solve(dense, vector.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(actual, expected, atol=1.0e-10, rtol=1.0e-10)


def test_v33_safe_nominal_is_bit_exact_and_violation_is_projected() -> None:
    nominal = torch.tensor(((2.0, -1.0), (-1.0, 0.0)), dtype=torch.float64)
    normal = torch.tensor(((1.0, 0.0), (1.0, 0.0)), dtype=torch.float64)
    rhs = torch.tensor((1.0, 1.0), dtype=torch.float64)
    safe, correction, nominal_margin, projected_margin = (
        MATH.project_task_consistent_hocbf(
            nominal,
            normal,
            rhs,
            torch.zeros_like(nominal),
            torch.tensor((1.0, 4.0), dtype=torch.float64),
            torch.tensor(((0.5, 0.0), (0.5, 0.0)), dtype=torch.float64),
            torch.tensor((True, True)),
            forward_weight=8.0,
            smoothness_weight=0.1,
        )
    )
    assert torch.equal(safe[0], nominal[0])
    assert torch.equal(correction[0], torch.zeros_like(correction[0]))
    assert nominal_margin[1] < 0.0
    assert torch.linalg.vector_norm(correction[1]) > 0.0
    assert projected_margin[1] >= -1.0e-12


def test_v33_discontinuous_identity_resets_drift() -> None:
    hdot, measured, instant, drift = MATH.estimate_hocbf_derivatives(
        torch.tensor((0.2,)),
        torch.tensor(((1.0, 2.0),)),
        torch.tensor(((0.5, -0.25),)),
        torch.tensor((-3.0,)),
        torch.tensor((7.0,)),
        torch.tensor(((9.0, 9.0),)),
        torch.tensor((12.0,)),
        torch.tensor((False,)),
        control_dt=0.02,
    )
    assert torch.allclose(hdot, torch.tensor((0.0,)))
    assert torch.isfinite(measured).all()
    assert torch.equal(instant, torch.zeros_like(instant))
    assert torch.equal(drift, torch.zeros_like(drift))


def test_v33_grid_and_formal_counts_are_exact() -> None:
    grid = PROTOCOL.candidate_grid()
    assert len(grid) == 18
    assert len({row["candidate"] for row in grid}) == 18
    assert {row["omega"] for row in grid} == {4.0, 8.0, 12.0}
    assert {row["lambda_x"] for row in grid} == {0.0, 8.0, 24.0}
    assert {row["lambda_s"] for row in grid} == {0.0, 0.1}
    assert PROTOCOL.FROZEN_POLICY_EPISODES == 512
    assert PROTOCOL.FINAL_TARGET_EPISODES == 512
    assert PROTOCOL.FINAL_D0_EPISODES == 256
    assert PROTOCOL.BOOTSTRAP_SAMPLES == 2_000


def test_v33_action_is_additive_gpu_path_with_fixed_history_rules() -> None:
    source = (REPO / "src/tasks/stairs_cbf/hocbf_action.py").read_text()
    current_source = (REPO / "src/tasks/stairs_cbf/actions.py").read_text()
    assert "class TaskConsistentHocbfAction" in source
    assert "JointPositionAction.process_actions(self, actions)" in source
    assert "project_task_consistent_hocbf" in source
    assert "drift_ema_previous=0.8" in source
    assert "drift_clip=20.0" in source
    assert "current_CBF0_source_unchanged" in source
    assert "task_consistent_acceleration_hocbf" not in current_source
