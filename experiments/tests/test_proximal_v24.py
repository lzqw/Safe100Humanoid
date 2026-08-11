"""Regression tests for independent v24 Contact Completion."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/scripts"))

from plot_proximal_v24 import FIGURE_CATEGORIES
from proximal_v23_protocol import (
    formal_algorithm_parameters as v23_formal_algorithm_parameters,
)
from proximal_v24_protocol import (
    ADAPTATION_SEED,
    CALIBRATION_CANDIDATE_PARAMETER_SEEDS,
    CALIBRATION_EPISODES,
    CALIBRATION_FRICTIONS,
    FINAL_D0_SEED,
    FINAL_TARGET_SEED,
    V23_FINAL_TEST_SHA256,
    V23_FROZEN_RESULT,
    V23_PROTOCOL_SHA256,
    V23_RESULT_GIT_TREE,
    all_v24_fresh_execution_seeds,
    calibration_gate,
    development_gate,
    formal_algorithm_parameters,
    fresh_randomness_report,
    pure_contact_context_audit,
    validate_v24_calibrated_context,
)

from src.tasks.stairs_cbf.config import g1_online_stairs_env_cfg
from src.tasks.stairs_cbf.deployment_context import (
    generate_v22_specialist_context,
    validate_frozen_deployment_context,
)
from src.tasks.stairs_cbf.proximal_context import (
    apply_cbf_proximal_context,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualifying_context(candidate_index: int = 0) -> dict:
    payload = validate_frozen_deployment_context(
        generate_v22_specialist_context(
            "C_effect", CALIBRATION_CANDIDATE_PARAMETER_SEEDS[candidate_index]
        )
    )
    attempt = {
        "candidate_index": candidate_index,
        "candidate_parameter_seed": CALIBRATION_CANDIDATE_PARAMETER_SEEDS[
            candidate_index
        ],
        "candidate_foot_friction": CALIBRATION_FRICTIONS[candidate_index],
        "parameters_sha256": payload["parameters_sha256"],
        "base_policy_only": True,
        "num_episodes": CALIBRATION_EPISODES,
        "success_count": 358,
        "fall_count": 154,
        "contact_fall_count": 140,
        "non_success_count": 154,
        "qualifies": True,
    }
    payload["calibration"] = {
        "kind": "base_policy_pure_contact_first_qualifying_v24",
        "adapted_policy_evaluations_used": False,
        "candidate_parameter_seeds": list(CALIBRATION_CANDIDATE_PARAMETER_SEEDS),
        "attempts": [attempt],
    }
    return payload


def test_v23_sources_results_and_negative_values_are_unchanged() -> None:
    protocol_path = REPO / "results/online/proximal_v23/protocol.json"
    final_path = REPO / "results/online/proximal_v23/final/final_test.json"
    assert _sha256(protocol_path) == V23_PROTOCOL_SHA256
    assert _sha256(final_path) == V23_FINAL_TEST_SHA256
    protocol = json.loads(protocol_path.read_text())
    for relative, expected in protocol["implementation_boundary"][
        "source_files"
    ].items():
        assert _sha256(REPO / relative) == expected
    result_tree = subprocess.run(
        ["git", "rev-parse", "HEAD:results/online/proximal_v23"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert result_tree == V23_RESULT_GIT_TREE
    final = json.loads(final_path.read_text())
    actual = {
        "target_base_success": final["target"]["success"]["baseline_mean"],
        "target_final_success": final["target"]["success"]["final_mean"],
        "target_success_delta": final["target"]["success"]["delta"],
        "target_fall_delta": final["target"]["fall"]["delta"],
        "target_repairs": final["target"]["repairs_regressions"]["repair_count"],
        "target_regressions": final["target"]["repairs_regressions"][
            "regression_count"
        ],
    }
    assert actual == V23_FROZEN_RESULT


def test_v24_algorithm_contract_is_exactly_v23() -> None:
    assert formal_algorithm_parameters() == v23_formal_algorithm_parameters()
    assert formal_algorithm_parameters() == {
        "rounds": 8,
        "num_envs": 64,
        "rollout_steps": 1024,
        "actor_learning_rate": 5.0e-6,
        "critic_learning_rate": 1.0e-4,
        "ppo_clip": 0.05,
        "maximum_actor_epochs": 2,
        "critic_epochs": 2,
        "mini_batches": 4,
        "moving_kl_beta": 0.5,
        "target_kl": 0.003,
        "hard_kl_ceiling": 0.01,
        "maximum_gradient_norm": 0.5,
        "freeze_log_std": True,
        "std_scale_from_base": 0.35,
        "minimum_std": 0.05,
        "maximum_std": 0.25,
        "entropy_coefficient": 0.0,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "whole_batch_advantage_normalization": True,
    }


def test_v24_candidate_grid_changes_only_foot_friction() -> None:
    contexts = []
    for index, (seed, friction) in enumerate(
        zip(
            CALIBRATION_CANDIDATE_PARAMETER_SEEDS,
            CALIBRATION_FRICTIONS,
            strict=True,
        )
    ):
        context = validate_frozen_deployment_context(
            generate_v22_specialist_context("C_effect", seed)
        )
        audit = pure_contact_context_audit(context, candidate_index=index)
        assert audit["passed"]
        assert audit["foot_friction"] == pytest.approx(friction, abs=5.0e-7)
        contexts.append(context)
    baseline = contexts[0]
    for context in contexts[1:]:
        assert context["target"] == baseline["target"]
        scenario = dict(context["scenario"])
        baseline_scenario = dict(baseline["scenario"])
        scenario.pop("foot_friction")
        baseline_scenario.pop("foot_friction")
        assert scenario == baseline_scenario
    assert [context["scenario"]["foot_friction"] for context in contexts] == list(
        CALIBRATION_FRICTIONS
    )


def test_v24_context_keeps_original_interface_reward_and_runtime_cbf() -> None:
    context = _qualifying_context()
    cfg = g1_online_stairs_env_cfg("DQHMED")
    metadata = apply_cbf_proximal_context(cfg, context, role="target")
    assert "deployable_failure" not in cfg.observations
    assert "specialist_failure_signal" not in cfg.rewards
    assert cfg.rewards["cbf_dual"].weight == 1.0
    assert cfg.rewards["fall_termination"].weight == -200.0
    assert cfg.actions["joint_pos"].enabled is True
    friction = cfg.events["specialist_foot_friction"].params["ranges"]
    assert friction == (CALIBRATION_FRICTIONS[0], CALIBRATION_FRICTIONS[0])
    assert metadata["actor_context_fields_added"] == 0
    assert metadata["cbf_proximal_interface"]["original_observation_interface"]


def test_v24_calibration_gate_has_exact_boundaries_and_fall_requirement() -> None:
    lower = calibration_gate(
        success_count=333,
        fall_count=179,
        contact_fall_count=153,
        non_success_count=179,
    )
    assert lower["qualifies"]
    upper = calibration_gate(
        success_count=384,
        fall_count=128,
        contact_fall_count=109,
        non_success_count=128,
    )
    assert upper["qualifies"]
    too_few_falls = calibration_gate(
        success_count=358,
        fall_count=99,
        contact_fall_count=99,
        non_success_count=154,
    )
    assert not too_few_falls["qualifies"]
    timeout_dilution = calibration_gate(
        success_count=358,
        fall_count=130,
        contact_fall_count=120,
        non_success_count=154,
    )
    assert timeout_dilution["contact_purity_over_falls"] >= 0.85
    assert timeout_dilution["contact_purity_over_all_non_success"] < 0.85
    assert not timeout_dilution["qualifies"]


def test_v24_calibrated_context_is_first_base_only_qualifier() -> None:
    context = _qualifying_context()
    assert (
        validate_v24_calibrated_context(context)["parameters_sha256"]
        == context["parameters_sha256"]
    )
    adapted = _qualifying_context()
    adapted["calibration"]["adapted_policy_evaluations_used"] = True
    with pytest.raises(ValueError, match="adapted policy"):
        validate_v24_calibrated_context(adapted)


def test_v24_randomness_is_unique_and_fresh() -> None:
    seeds = all_v24_fresh_execution_seeds()
    assert ADAPTATION_SEED in seeds
    assert FINAL_TARGET_SEED in seeds
    assert FINAL_D0_SEED in seeds
    assert len(seeds) == len(set(seeds))
    report = fresh_randomness_report(REPO)
    assert report["passed"]
    assert report["collisions"] == []


def test_v24_gate_is_report_only_and_plots_have_three_categories() -> None:
    assert development_gate(
        target_success_delta=0.03,
        target_fall_delta=0.01,
        d0_success_delta=-0.05,
    )["passed"]
    result = development_gate(
        target_success_delta=0.029,
        target_fall_delta=0.011,
        d0_success_delta=-0.051,
    )
    assert not result["passed"]
    assert result["confidence_intervals_are_gates"] is False
    assert FIGURE_CATEGORIES == (
        "round_curve",
        "base_vs_final",
        "repairs_vs_regressions",
    )


def test_v24_training_has_no_forbidden_algorithm_path_or_selection() -> None:
    source = (REPO / "experiments/scripts/refine_proximal_v24.py").read_text()
    tree = ast.parse(source)
    imported_from = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "refine_proximal_v23" in imported_from
    assert not any("hard_cases" in name for name in imported_from)
    for forbidden in (
        "failure_precursor_bank",
        "state_restart",
        "candidate_fractions",
        "select_best_so_far",
        "specialist_reward",
        "performance_gate",
    ):
        assert forbidden not in referenced_names
    assert "round 8 actor, never best-so-far" in source
