"""Small pure checks for the prospectively fixed v34 velocity CBF."""

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


MATH = _load("velocity_cbf_v34_math", "src/tasks/stairs_cbf/velocity_cbf_math.py")
PROTOCOL = _load(
    "velocity_cbf_v34_protocol_test",
    "experiments/scripts/velocity_cbf_v34_protocol.py",
)


def test_v34_sherman_morrison_matches_dense_solve() -> None:
    torch.manual_seed(34)
    vector = torch.randn(9, 12, dtype=torch.float64)
    jac_x = torch.randn_like(vector)
    diagonal = torch.rand_like(vector) * 4.0 + 0.25
    actual = MATH.apply_velocity_task_metric_inverse(
        vector,
        diagonal,
        jac_x,
        forward_weight=7.0,
        smoothness_weight=0.2,
    )
    dense = torch.diag_embed(diagonal + 0.2) + 7.0 * (
        jac_x.unsqueeze(-1) * jac_x.unsqueeze(-2)
    )
    expected = torch.linalg.solve(dense, vector.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(actual, expected, atol=1.0e-10, rtol=1.0e-10)


def test_v34_safe_nominal_is_bit_exact_and_violation_is_projected() -> None:
    nominal = torch.tensor(((2.0, -1.0), (-1.0, 0.0)), dtype=torch.float64)
    previous = torch.tensor(((99.0, 99.0), (0.5, -0.5)), dtype=torch.float64)
    normal = torch.tensor(((1.0, 0.0), (1.0, 0.0)), dtype=torch.float64)
    rhs = torch.tensor((1.0, 1.0), dtype=torch.float64)
    safe, correction, nominal_margin, projected_margin = (
        MATH.project_task_metric_velocity_cbf(
            nominal,
            previous,
            normal,
            rhs,
            torch.tensor(((1.0, 4.0), (1.0, 4.0)), dtype=torch.float64),
            torch.tensor(((0.5, 0.25), (0.5, 0.25)), dtype=torch.float64),
            torch.tensor((True, True)),
            torch.tensor((True, True)),
            forward_weight=8.0,
            smoothness_weight=0.4,
        )
    )
    assert torch.equal(safe[0], nominal[0])
    assert torch.equal(correction[0], torch.zeros_like(correction[0]))
    assert nominal_margin[1] < 0.0
    assert torch.linalg.vector_norm(correction[1]) > 0.0
    assert projected_margin[1] >= -1.0e-12


def test_v34_candidate_budget_ranges_and_symmetry_are_fixed() -> None:
    candidates = PROTOCOL.candidate_grid()
    assert len(candidates) == 60
    assert len({row["candidate"] for row in candidates}) == 60
    assert candidates[0]["mode"] == PROTOCOL.CURRENT_CBF_MODE
    for candidate in candidates[1:]:
        assert candidate["mode"] == PROTOCOL.OPTIMIZED_CBF_MODE
        for name, (lower, upper) in PROTOCOL.PARAMETER_RANGES.items():
            assert lower <= candidate[name] <= upper
    assert PROTOCOL.STAGE1_EPISODES == 64
    assert PROTOCOL.STAGE2_TOP_K == 8
    assert PROTOCOL.STAGE2_EPISODES == 256
    assert PROTOCOL.TRAIN_TOP_K == 2
    assert PROTOCOL.TRAINED_DEVELOPMENT_EPISODES == 256


def test_v34_action_path_is_velocity_only_and_gpu_vectorized() -> None:
    source = (REPO / "src/tasks/stairs_cbf/velocity_cbf_action.py").read_text()
    hot_path = source[
        source.index("class TaskMetricVelocityCbfAction") : source.index(
            "def _quat_roll_pitch"
        )
    ]
    assert "JointPositionAction.process_actions(self, actions)" in hot_path
    assert "project_task_metric_velocity_cbf" in hot_path
    assert "projected_target = torch.where" in hot_path
    assert "current_CBF0_source_unchanged" in source
    for forbidden in (".cpu(", ".numpy(", ".item(", ".tolist(", "ddot", "drift"):
        assert forbidden not in hot_path


def test_v34_final_identity_seeds_are_absent_from_search_protocol() -> None:
    frozen = PROTOCOL.frozen_search_specification()
    assert frozen["final_seeds_created"] is False
    assert frozen["final_identities_accessible_during_development"] is False
    source = (SCRIPTS / "velocity_cbf_v34_protocol.py").read_text()
    assert "final_target_seed" not in source
    assert "final_d0_seed" not in source
