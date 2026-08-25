"""Pure checks for the prospectively fixed v32 long-horizon experiment."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "experiments/scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, REPO / relative)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


PROTOCOL = _load("v32_protocol", "experiments/scripts/cbf_teacher_v32_protocol.py")


def test_v32_exact_six_continuations_plus_one_mixed_run() -> None:
    runs = PROTOCOL.run_matrix()
    continuation = [run for run in runs if run["kind"] == "continuation"]
    mixed = [run for run in runs if run["kind"] == "mixed"]
    assert len(continuation) == 6
    assert len(mixed) == 1
    assert [(run["context"], run["schedule"]) for run in continuation] == [
        ("F1", "LongConstant"),
        ("F1", "LongDecay"),
        ("F2", "LongConstant"),
        ("F2", "LongDecay"),
        ("F3", "LongConstant"),
        ("F3", "LongDecay"),
    ]
    assert all(run["start_round"] == 8 for run in continuation)
    assert all(run["final_round"] == 24 for run in runs)
    assert all(run["additional_rounds"] == 16 for run in continuation)
    assert mixed[0]["additional_rounds"] == 24
    assert len({run["seed"] for run in runs}) == 7
    json.dumps(runs)


def test_v32_continuation_learning_rate_schedules_are_exact() -> None:
    full = (5.0e-6, 1.0e-4)
    half = (2.5e-6, 5.0e-5)
    low = (1.0e-6, 2.5e-5)
    assert all(
        PROTOCOL.learning_rates("continuation", "LongConstant", round_index) == full
        for round_index in range(9, 25)
    )
    assert all(
        PROTOCOL.learning_rates("continuation", "LongDecay", round_index) == full
        for round_index in range(9, 17)
    )
    assert all(
        PROTOCOL.learning_rates("continuation", "LongDecay", round_index) == half
        for round_index in range(17, 21)
    )
    assert all(
        PROTOCOL.learning_rates("continuation", "LongDecay", round_index) == low
        for round_index in range(21, 25)
    )


def test_v32_mixed_schedule_and_allocation_are_exactly_balanced() -> None:
    full = (5.0e-6, 1.0e-4)
    half = (2.5e-6, 5.0e-5)
    low = (1.0e-6, 2.5e-5)
    assert all(
        PROTOCOL.learning_rates("mixed", "LongDecay", round_index) == full
        for round_index in range(1, 13)
    )
    assert all(
        PROTOCOL.learning_rates("mixed", "LongDecay", round_index) == half
        for round_index in range(13, 19)
    )
    assert all(
        PROTOCOL.learning_rates("mixed", "LongDecay", round_index) == low
        for round_index in range(19, 25)
    )
    counts = [PROTOCOL.mixed_context_env_counts(index) for index in range(1, 25)]
    assert all(sorted(item.values()) == [21, 21, 22] for item in counts)
    assert all(sum(item.values()) == 64 for item in counts)
    totals = {
        context: sum(item[context] for item in counts)
        for context in PROTOCOL.FORMAL_CONTEXTS
    }
    assert totals == {"F1": 512, "F2": 512, "F3": 512}


def test_v32_reuses_v31_A2_and_forbids_selection_gates() -> None:
    algorithm = PROTOCOL.common_algorithm_parameters()
    assert algorithm["teacher"] == {
        "name": "residual_eta_025_all_interventions",
        "teacher_mode": "residual",
        "teacher_gate": "all_interventions",
        "teacher_eta": 0.25,
        "teacher_loss": "weighted_smooth_l1_per_action_mean",
        "teacher_weight": 1.0,
    }
    assert algorithm["target_kl_early_stopping"] is False
    assert algorithm["hard_kl_rollback"] is False
    assert algorithm["performance_gate"] is False
    assert algorithm["candidate_line_search"] is False
    assert algorithm["best_checkpoint_selection"] is False
    assert algorithm["final_policy"] == "unconditional round 24"


def test_v32_fixed_monitor_and_formal_episode_counts() -> None:
    assert PROTOCOL.MONITOR_ROUNDS == (8, 16, 24)
    assert PROTOCOL.MONITOR_EPISODES == 128
    assert PROTOCOL.FORMAL_TARGET_EPISODES == 512
    assert PROTOCOL.FORMAL_D0_EPISODES == 256
    assert len(PROTOCOL.PREFLIGHT_CASES) == 2


def test_v32_freeze_binds_v31_and_prior_result_trees() -> None:
    source = (REPO / "experiments/scripts/freeze_cbf_teacher_v32.py").read_text()
    assert "range(25, 32)" in source
    assert "V31_A2_ROUND8_SHA256" in source
    assert "v31_result_recomputed_or_modified" in source


def test_v32_queue_order_and_single_gpu_execution_contract() -> None:
    source = (REPO / "experiments/scripts/run_cbf_teacher_v32.sh").read_text()
    assert "for context in F1 F2 F3" in source
    assert 'run_continuation "$context" LongConstant' in source
    assert 'run_continuation "$context" LongDecay' in source
    assert "run_mixed" in source
    assert "cuda:0" in source
