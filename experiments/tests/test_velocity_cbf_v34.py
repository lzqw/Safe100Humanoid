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
PAIRED_RESCUE = _load(
    "paired_rescue_v109_math",
    "experiments/scripts/paired_rescue_v109_math.py",
)
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
GATED_RESIDUAL = _load(
    "paired_gated_residual_v113_test",
    "experiments/scripts/train_paired_gated_residual_v113.py",
)
ADAPTER_CALIBRATION = _load(
    "observable_adapter_calibration_v114_test",
    "experiments/scripts/calibrate_observable_adapter_v114.py",
)
CAUSAL_GATE = _load(
    "causal_gated_residual_v115_test",
    "experiments/scripts/train_causal_gated_residual_v115.py",
)
FILTER_OFF_PPO = _load(
    "filter_off_residual_ppo_v117_test",
    "experiments/scripts/train_filter_off_residual_ppo_v117.py",
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


def test_v109_paired_rescue_traces_first_correction_into_approach() -> None:
    off_nominal = torch.zeros(6, 2)
    on_nominal = torch.zeros_like(off_nominal)
    on_safe = torch.zeros_like(off_nominal)
    on_safe[3] = torch.tensor((1.0, 0.0))
    on_safe[4] = torch.tensor((0.0, 2.0))
    intervened = torch.tensor((False, False, False, True, False, False))
    trace = PAIRED_RESCUE.paired_rescue_action_trace(
        off_nominal,
        on_nominal,
        on_safe,
        intervened,
        pre_horizon=2,
        post_horizon=2,
        pre_decay=0.5,
    )
    assert torch.equal(trace["indices"], torch.tensor((1, 2, 3, 4)))
    assert torch.allclose(
        trace["corrections"],
        torch.tensor(((0.25, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 2.0))),
    )
    assert trace["first_intervention_step"] == 3
    assert trace["pre_transition_count"] == 2
    assert trace["post_transition_count"] == 2
    assert torch.isclose(trace["weights"].sum(), torch.tensor(1.0))


def test_v109_dataset_uses_off_states_and_on_trajectory_targets() -> None:
    nominal = torch.zeros(6, 2)
    on_safe = torch.zeros_like(nominal)
    on_safe[3] = torch.tensor((1.0, 0.0))
    on_safe[4] = torch.tensor((0.0, 2.0))
    common = {
        "observations": torch.arange(6 * 415, dtype=torch.float32).reshape(6, 415),
        "nominal_actions": nominal,
        "safe_actions": nominal.clone(),
        "would_intervene": torch.zeros(6, dtype=torch.bool),
        "environment_ids": torch.zeros(6, dtype=torch.long),
        "episode_steps": torch.arange(6),
    }
    off = {"dataset": common}
    on_data = {key: value.clone() for key, value in common.items()}
    on_data["safe_actions"] = on_safe
    on_data["would_intervene"][3] = True
    on = {"dataset": on_data}
    dataset, weights, summary = ADAPTER._paired_trajectory_rescue_dataset(
        off,
        on,
        torch.tensor((True,)),
        environment_offset=7,
        pre_horizon=2,
        post_horizon=2,
        pre_decay=0.5,
    )
    assert len(dataset["observations"]) == 4
    assert torch.equal(dataset["observations"], common["observations"][[1, 2, 3, 4]])
    assert torch.equal(dataset["environment_ids"], torch.full((4,), 7))
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert summary["episode_count"] == 1
    assert summary["pre_transition_count"] == 2


def test_v110_dataset_keeps_deployment_states_and_cbf_targets_paired() -> None:
    nominal = torch.zeros(6, 2)
    off_safe = nominal.clone()
    off_safe[3] = torch.tensor((1.0, 0.0))
    off_safe[4] = torch.tensor((0.0, 2.0))
    off_data = {
        "observations": torch.arange(6 * 415, dtype=torch.float32).reshape(6, 415),
        "nominal_actions": nominal,
        "safe_actions": off_safe,
        "would_intervene": torch.tensor((False, False, False, True, False, False)),
        "environment_ids": torch.zeros(6, dtype=torch.long),
        "episode_steps": torch.arange(6),
    }
    on_data = {key: value.clone() for key, value in off_data.items()}
    on_data["observations"] += 10000.0
    on_data["safe_actions"].zero_()
    on_data["safe_actions"][5] = torch.tensor((9.0, 9.0))
    on_data["would_intervene"] = torch.tensor(
        (False, False, False, False, False, True)
    )
    dataset, weights, summary = ADAPTER._paired_trajectory_rescue_dataset(
        {"dataset": off_data},
        {"dataset": on_data},
        torch.tensor((True,)),
        environment_offset=11,
        pre_horizon=2,
        post_horizon=2,
        pre_decay=0.5,
        target_mode="deployment-counterfactual",
    )
    assert torch.equal(dataset["observations"], off_data["observations"][[1, 2, 3, 4]])
    assert torch.allclose(
        dataset["safe_actions"] - dataset["nominal_actions"],
        torch.tensor(((0.25, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 2.0))),
    )
    assert torch.equal(dataset["environment_ids"], torch.full((4,), 11))
    assert summary["target_mode"] == "deployment-counterfactual"
    assert summary["episode_count"] == 1
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_v111_contrasts_rescued_and_harmed_terminal_pairs() -> None:
    nominal = torch.zeros(10, 2)
    off_safe = nominal.clone()
    off_safe[2] = torch.tensor((1.0, 0.0))
    off_safe[7] = torch.tensor((0.0, 2.0))
    environment_ids = torch.tensor((0, 0, 0, 0, 0, 1, 1, 1, 1, 1))
    episode_steps = torch.tensor((0, 1, 2, 3, 4, 0, 1, 2, 3, 4))
    intervened = torch.zeros(10, dtype=torch.bool)
    intervened[[2, 7]] = True
    off_data = {
        "observations": torch.arange(10 * 415, dtype=torch.float32).reshape(10, 415),
        "nominal_actions": nominal,
        "safe_actions": off_safe,
        "would_intervene": intervened,
        "environment_ids": environment_ids,
        "episode_steps": episode_steps,
    }
    on_data = {key: value.clone() for key, value in off_data.items()}
    dataset, weights, summary = ADAPTER._paired_trajectory_rescue_dataset(
        {"dataset": off_data},
        {"dataset": on_data},
        torch.tensor((True, True)),
        environment_offset=20,
        pre_horizon=1,
        post_horizon=1,
        pre_decay=0.5,
        target_mode="paired-outcome-contrast",
        episode_signs=torch.tensor((1, -1), dtype=torch.int8),
    )
    assert torch.allclose(
        dataset["safe_actions"] - dataset["nominal_actions"],
        torch.tensor(((0.5, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, -2.0))),
    )
    assert torch.equal(dataset["environment_ids"], torch.tensor((20, 20, 21, 21)))
    assert summary["positive_episode_count"] == 1
    assert summary["negative_episode_count"] == 1
    assert torch.isclose(weights.sum(), torch.tensor(2.0))


def test_v112_full_first_layer_retains_state_conditioning_gradients() -> None:
    gradient = torch.ones(3, 415)
    geometry_only = ADAPTER._apply_first_layer_update_scope(
        gradient.clone(), "geometry-columns"
    )
    state_conditioned = ADAPTER._apply_first_layer_update_scope(
        gradient.clone(), "full-first-layer"
    )
    assert torch.count_nonzero(geometry_only[:, :405]) == 0
    assert torch.equal(geometry_only[:, 405:], torch.ones(3, 10))
    assert torch.equal(state_conditioned, gradient)
    with pytest.raises(ValueError, match="unsupported"):
        ADAPTER._apply_first_layer_update_scope(gradient.clone(), "all-layers")


def test_v113_gate_balances_classes_and_suppresses_low_confidence_states() -> None:
    labels = torch.tensor((True, True, False, False, False, False))
    environment_ids = torch.tensor((0, 0, 1, 2, 2, 2))
    weights = GATED_RESIDUAL.balanced_episode_weights(labels, environment_ids)
    assert torch.isclose(weights[labels].sum(), torch.tensor(0.5))
    assert torch.isclose(weights[~labels].sum(), torch.tensor(0.5))
    assert torch.isclose(weights[environment_ids == 1].sum(), torch.tensor(0.25))
    assert torch.isclose(weights[environment_ids == 2].sum(), torch.tensor(0.25))
    gate = GATED_RESIDUAL.deployable_gate(
        torch.tensor((0.4, 0.6, 0.9)),
        torch.tensor((True, True, False)),
        threshold=0.5,
    )
    assert torch.allclose(gate, torch.tensor((0.0, 0.2, 0.0)))


def test_v114_interpolation_preserves_endpoints_and_allows_bounded_extrapolation() -> None:
    base = {
        "weight": torch.tensor((1.0, 2.0)),
        "count": torch.tensor(3, dtype=torch.long),
    }
    adapter = {
        "weight": torch.tensor((3.0, 0.0)),
        "count": torch.tensor(3, dtype=torch.long),
    }
    zero = ADAPTER_CALIBRATION.interpolate_actor_state(base, adapter, 0.0)
    one = ADAPTER_CALIBRATION.interpolate_actor_state(base, adapter, 1.0)
    extrapolated = ADAPTER_CALIBRATION.interpolate_actor_state(base, adapter, 1.5)
    assert torch.equal(zero["weight"], base["weight"])
    assert torch.equal(one["weight"], adapter["weight"])
    assert torch.allclose(extrapolated["weight"], torch.tensor((4.0, -1.0)))
    assert extrapolated["count"] == 3


def test_v115_gate_is_causal_and_balances_discordant_episodes() -> None:
    torch.manual_seed(115)
    model = CAUSAL_GATE.CausalGatedResidual(max_residual=0.25)
    prefix = torch.randn(1, 3, CAUSAL_GATE.FEATURE_DIM)
    first = torch.cat((prefix, torch.zeros(1, 2, CAUSAL_GATE.FEATURE_DIM)), dim=1)
    second = torch.cat((prefix, torch.randn(1, 2, CAUSAL_GATE.FEATURE_DIM)), dim=1)
    first_logits = model.gate_logits_sequence(first, torch.tensor((5,)))
    second_logits = model.gate_logits_sequence(second, torch.tensor((5,)))
    assert torch.allclose(first_logits[:, :3], second_logits[:, :3])
    labels = torch.tensor((True, False, False))
    masks = [
        torch.tensor((False, True, True)),
        torch.tensor((True,)),
        torch.tensor((False, True, True, True)),
    ]
    weights = CAUSAL_GATE.balanced_sequence_weights(labels, masks)
    assert torch.isclose(weights[0].sum(), torch.tensor(0.5))
    assert torch.isclose(weights[1].sum(), torch.tensor(0.25))
    assert torch.isclose(weights[2].sum(), torch.tensor(0.25))


def test_v116_routes_residual_to_actual_successful_filtered_trajectory() -> None:
    method_id, trace_mode = CAUSAL_GATE.residual_teacher_configuration(
        "successful-filtered-trajectory"
    )
    assert method_id.endswith("v116")
    assert trace_mode == "paired-trajectory"
    with pytest.raises(ValueError, match="unsupported"):
        CAUSAL_GATE.residual_teacher_configuration("unsigned-local-filter")


def test_v117_outcome_credit_is_balanced_and_episode_equal() -> None:
    environment_ids = torch.tensor((0, 0, 1, 2, 2, 2), dtype=torch.long)
    success = torch.tensor((True, False, False))
    weights, advantages = FILTER_OFF_PPO.balanced_outcome_weights(
        environment_ids, success
    )
    assert torch.isclose(weights[environment_ids == 0].sum(), torch.tensor(0.5))
    assert torch.isclose(weights[environment_ids == 1].sum(), torch.tensor(0.25))
    assert torch.isclose(weights[environment_ids == 2].sum(), torch.tensor(0.25))
    assert torch.equal(advantages, torch.tensor((1.0, 1.0, -1.0, -1.0, -1.0, -1.0)))
    actions = torch.tensor(((0.1, -0.1),))
    log_prob = FILTER_OFF_PPO._normal_log_prob(actions, torch.zeros_like(actions), 0.2)
    expected = torch.distributions.Normal(0.0, 0.2).log_prob(actions).sum(dim=-1)
    assert torch.allclose(log_prob, expected)


def test_v118_uses_direction_preserving_sgd_without_adam_state() -> None:
    residual = FILTER_OFF_PPO.LearnedCbfResidual(max_residual=0.1)
    optimizer = FILTER_OFF_PPO._build_optimizer(
        residual, name="sgd", learning_rate=0.01
    )
    assert isinstance(optimizer, torch.optim.SGD)
    assert not optimizer.state
    assert FILTER_OFF_PPO.FULL_BATCH_METHOD_ID.endswith("v118")
    with pytest.raises(ValueError, match="unsupported"):
        FILTER_OFF_PPO._build_optimizer(
            residual, name="sign-adam", learning_rate=0.01
        )


def test_v95_critic_expansion_accepts_ten_persistent_features() -> None:
    source = {"mlp.0.weight": torch.randn(3, 7)}
    target = {"mlp.0.weight": torch.randn(3, 17)}
    expanded, metadata = OBSERVABLE_PPO._expand_critic_state(source, target)
    assert metadata["new_feature_count"] == 10
    assert torch.equal(expanded["mlp.0.weight"][:, :7], source["mlp.0.weight"])
    assert torch.count_nonzero(expanded["mlp.0.weight"][:, 7:]) == 0
    assert OBSERVABLE_PPO.PERSISTENT_METHOD_ID.endswith("v95")
