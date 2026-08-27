"""Small pure checks for the prospectively fixed v34 velocity CBF."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import pytest

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
CBF_MATH = _load("stair_cbf_math", "src/tasks/stairs_cbf/cbf_math.py")
PROTOCOL = _load(
    "velocity_cbf_v34_protocol_test",
    "experiments/scripts/velocity_cbf_v34_protocol.py",
)
ADAPTER = _load(
    "observable_cbf_adapter_v49_test",
    "experiments/scripts/refine_observable_cbf_adapter_v49.py",
)
OBSERVABLE_PPO = _load(
    "observable_cbf_ppo_v51_test",
    "experiments/scripts/refine_observable_cbf_ppo_v51.py",
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


def test_v92_geometry_adapter_supports_direction_preserving_sgd() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = ADAPTER._build_adapter_optimizer(
        [parameter], optimizer_name="sgd", learning_rate=0.1
    )
    assert isinstance(optimizer, torch.optim.SGD)
    assert ADAPTER.FULL_BATCH_SGD_METHOD_ID.endswith("adapter-v92")
    with pytest.raises(ValueError, match="unknown v49-v93 adapter optimizer"):
        ADAPTER._build_adapter_optimizer(
            [parameter], optimizer_name="invalid", learning_rate=0.1
        )


def test_v93_conditional_geometry_separates_side_and_barrier_phase() -> None:
    base = torch.tensor(
        (
            (0.2, 0.4, -0.3, -1.0, 1.0),
            (0.1, 0.5, 0.2, 1.0, 1.0),
            (0.3, 0.2, -0.1, 1.0, 0.0),
        )
    )
    conditional = CBF_MATH.conditional_deployable_cbf_geometry(base)
    assert conditional.shape == (3, 16)
    assert torch.equal(conditional[0, :4], torch.tensor((0.2, 0.4, -0.3, 1.0)))
    assert torch.count_nonzero(conditional[0, 4:]) == 0
    assert torch.equal(conditional[1, 12:], torch.tensor((0.1, 0.5, 0.2, 1.0)))
    assert torch.count_nonzero(conditional[1, :12]) == 0
    assert torch.count_nonzero(conditional[2]) == 0
    assert torch.equal(
        ADAPTER._geometry_active(torch.cat((torch.zeros(3, 405), conditional), dim=1)),
        torch.tensor((True, True, False)),
    )


def test_v94_persistent_geometry_is_visible_before_toe_off() -> None:
    persistent = CBF_MATH.persistent_next_riser_geometry(
        torch.tensor((0.4, 2.1)),
        torch.tensor(
            (
                ((0.30, 0.10), (0.35, 0.10)),
                ((2.05, 0.40), (2.00, 0.40)),
            )
        ),
        torch.tensor(((True, True), (True, False))),
        torch.tensor(((1.0, 2.0), (1.0, 2.0))),
        torch.tensor(((0.2, 0.4), (0.2, 0.4))),
        toe_margin=0.0,
        top_clearance=0.0,
        barrier_slope=1.0,
        lookahead_distance=1.0,
        horizontal_scale=1.0,
        vertical_scale=1.0,
    )
    assert persistent.shape == (2, 10)
    assert torch.allclose(
        persistent[0],
        torch.tensor((0.70, -0.10, 0.60, 1.0, 1.0, 0.65, -0.10, 0.55, 1.0, 1.0)),
    )
    assert torch.count_nonzero(persistent[1]) == 0
    assert torch.equal(
        ADAPTER._geometry_active(torch.cat((torch.zeros(2, 405), persistent), dim=1)),
        torch.tensor((True, False)),
    )


def test_v95_critic_expansion_accepts_ten_persistent_features() -> None:
    source = {"mlp.0.weight": torch.randn(3, 7)}
    target = {"mlp.0.weight": torch.randn(3, 17)}
    expanded, metadata = OBSERVABLE_PPO._expand_critic_state(source, target)
    assert metadata["new_feature_count"] == 10
    assert torch.equal(expanded["mlp.0.weight"][:, :7], source["mlp.0.weight"])
    assert torch.count_nonzero(expanded["mlp.0.weight"][:, 7:]) == 0
    assert OBSERVABLE_PPO.PERSISTENT_METHOD_ID.endswith("v95")
