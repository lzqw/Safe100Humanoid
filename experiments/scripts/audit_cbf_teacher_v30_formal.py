"""Run the frozen three-context v30 teacher-vs-control formal audit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from cbf_teacher_v30_eval_io import (
    assert_paired,
    atomic_json,
    evaluate_condition,
    paired_ci,
    paired_repairs_regressions,
    paired_wide_rows,
    write_csv,
)
from cbf_teacher_v30_protocol import (
    BASE_CHECKPOINT_SHA256,
    BOOTSTRAP_SEED_BASE,
    FORMAL_BOOTSTRAP_SAMPLES,
    FORMAL_CONTEXTS,
    FORMAL_D0_EPISODES,
    FORMAL_D0_SEED_BASES,
    FORMAL_TARGET_EPISODES,
    FORMAL_TARGET_SEED_BASES,
    PREFERRED_EVAL_BATCH_SIZE,
    PROTOCOL_ID,
    TEACHER_ARMS,
    arm_parameters,
)
from proximal_v23_io import file_sha256

TARGET_CONDITIONS = (
    "base_off",
    "base_on",
    "control_on",
    "control_off",
    "teacher_on",
    "teacher_off",
)
D0_CONDITIONS = ("base_on", "control_on", "teacher_on")
PUBLISHED_METRICS = (
    "success_rate",
    "fall_rate",
    "mean_return",
    "mean_reached_riser",
    "mean_completion_time_s",
    "mean_success_completion_time_s",
    "intervention_steps_per_riser",
    "counterfactual_would_intervene_fraction",
    "mean_correction_norm",
    "mean_counterfactual_correction_norm",
    "nominal_barrier_violation_steps_per_riser",
    "toe_riser_kick_events_per_riser",
    "unsafe_overlap_steps_per_riser",
    "mean_episode_minimum_nominal_barrier_margin",
    "global_minimum_nominal_barrier_margin",
    "kick_episode_rate",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _training_checkpoint(
    root: Path, context: str, arm: str, protocol_hash: str
) -> tuple[Path, dict[str, Any]]:
    directory = root / context / arm
    checkpoint = directory / "round_08.pt"
    summary_path = directory / "training_summary.json"
    if not checkpoint.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"formal v30 training output missing: {context}/{arm}")
    summary = json.loads(summary_path.read_text())
    checks = {
        "protocol": summary.get("protocol_id") == PROTOCOL_ID,
        "protocol_hash": summary.get("protocol", {}).get("sha256") == protocol_hash,
        "phase": summary.get("phase") == "formal",
        "context": summary.get("context") == context,
        "arm": summary.get("arm") == arm,
        "rounds": summary.get("rounds_completed") == 8,
        "checkpoint": summary.get("final_checkpoint_sha256") == file_sha256(checkpoint),
        "unconditional_final": summary.get("final_policy_rule")
        == "unconditional round 8 actor",
        "no_gates": summary.get("kl_stop_count") == 0
        and summary.get("kl_rollback_count") == 0
        and summary.get("performance_selection_count") == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid formal v30 run {context}/{arm}: {checks}")
    return checkpoint, summary


def _aggregate_row(
    context: str, domain: str, condition: str, value: dict[str, Any]
) -> dict[str, Any]:
    return {
        "context": context,
        "domain": domain,
        "condition": condition,
        "runtime_filter": value["runtime_filter"],
        "episodes": value["num_episodes"],
        **{field: value.get(field) for field in PUBLISHED_METRICS},
        "checkpoint_sha256": value["checkpoint_sha256"],
        "actor_state_sha256": value["actor_state_sha256"],
    }


def _context_effects(
    context: str,
    target: dict[str, dict[str, Any]],
    d0: dict[str, dict[str, Any]],
    target_rows: dict[str, list[dict[str, str]]],
    d0_rows: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    success = {key: value["success_rate"] for key, value in target.items()}
    deltas = {
        "teacher_on_vs_base": success["teacher_on"] - success["base_on"],
        "control_on_vs_base": success["control_on"] - success["base_on"],
        "teacher_on_vs_control": success["teacher_on"] - success["control_on"],
        "teacher_off_vs_base": success["teacher_off"] - success["base_off"],
        "teacher_off_vs_control": success["teacher_off"] - success["control_off"],
        "D0_teacher_on_vs_base": d0["teacher_on"]["success_rate"]
        - d0["base_on"]["success_rate"],
        "D0_control_on_vs_base": d0["control_on"]["success_rate"]
        - d0["base_on"]["success_rate"],
        "D0_teacher_on_vs_control": d0["teacher_on"]["success_rate"]
        - d0["control_on"]["success_rate"],
    }
    comparisons = {
        "teacher_on_vs_base": ("base_on", "teacher_on", target_rows),
        "control_on_vs_base": ("base_on", "control_on", target_rows),
        "teacher_on_vs_control": ("control_on", "teacher_on", target_rows),
        "teacher_off_vs_base": ("base_off", "teacher_off", target_rows),
        "teacher_off_vs_control": ("control_off", "teacher_off", target_rows),
        "D0_teacher_on_vs_base": ("base_on", "teacher_on", d0_rows),
        "D0_control_on_vs_base": ("base_on", "control_on", d0_rows),
        "D0_teacher_on_vs_control": ("control_on", "teacher_on", d0_rows),
    }
    intervals = {}
    repairs = {}
    context_offset = FORMAL_CONTEXTS.index(context) * 100
    for index, (name, (before, after, rows)) in enumerate(comparisons.items()):
        intervals[name] = paired_ci(
            rows[before],
            rows[after],
            field="success",
            seed=BOOTSTRAP_SEED_BASE + context_offset + index,
            samples=FORMAL_BOOTSTRAP_SAMPLES,
        )
        repairs[name] = paired_repairs_regressions(rows[before], rows[after])
    internalization = {
        "off_success_teacher_minus_control": target["teacher_off"]["success_rate"]
        - target["control_off"]["success_rate"],
        "off_would_intervene_teacher_minus_control": target["teacher_off"][
            "counterfactual_would_intervene_fraction"
        ]
        - target["control_off"]["counterfactual_would_intervene_fraction"],
        "off_correction_teacher_minus_control": target["teacher_off"][
            "mean_counterfactual_correction_norm"
        ]
        - target["control_off"]["mean_counterfactual_correction_norm"],
        "off_nominal_violation_teacher_minus_control": target["teacher_off"][
            "nominal_barrier_violation_steps_per_riser"
        ]
        - target["control_off"]["nominal_barrier_violation_steps_per_riser"],
        "off_kick_events_teacher_minus_control": target["teacher_off"][
            "toe_riser_kick_events_per_riser"
        ]
        - target["control_off"]["toe_riser_kick_events_per_riser"],
        "off_overlap_teacher_minus_control": target["teacher_off"][
            "unsafe_overlap_steps_per_riser"
        ]
        - target["control_off"]["unsafe_overlap_steps_per_riser"],
        "off_margin_teacher_minus_control": target["teacher_off"][
            "mean_episode_minimum_nominal_barrier_margin"
        ]
        - target["control_off"]["mean_episode_minimum_nominal_barrier_margin"],
    }
    return {
        "context": context,
        "success_deltas": deltas,
        "paired_success_95_CI": intervals,
        "repairs_and_regressions": repairs,
        "internalization_teacher_vs_control": internalization,
    }


def _paper_value(contexts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    effects = [contexts[name]["effects"] for name in FORMAL_CONTEXTS]
    mean_teacher_gain = float(
        np.mean([item["success_deltas"]["teacher_on_vs_base"] for item in effects])
    )
    mean_teacher_control = float(
        np.mean([item["success_deltas"]["teacher_on_vs_control"] for item in effects])
    )
    contexts_won = sum(
        item["success_deltas"]["teacher_on_vs_control"] > 0.0 for item in effects
    )
    mean_d0 = float(
        np.mean([item["success_deltas"]["D0_teacher_on_vs_base"] for item in effects])
    )
    directions = {
        "off_success": ("off_success_teacher_minus_control", 1.0),
        "would_intervene": ("off_would_intervene_teacher_minus_control", -1.0),
        "correction_norm": ("off_correction_teacher_minus_control", -1.0),
        "nominal_violations": (
            "off_nominal_violation_teacher_minus_control",
            -1.0,
        ),
        "kick_events": ("off_kick_events_teacher_minus_control", -1.0),
        "unsafe_overlap": ("off_overlap_teacher_minus_control", -1.0),
        "barrier_margin": ("off_margin_teacher_minus_control", 1.0),
    }
    consistent = []
    for label, (field, direction) in directions.items():
        values = [
            item["internalization_teacher_vs_control"][field] * direction
            for item in effects
        ]
        if all(value >= 0.0 for value in values) and any(
            value > 0.0 for value in values
        ):
            consistent.append(label)
    criteria = {
        "mean_teacher_on_gain_at_least_3pp": mean_teacher_gain >= 0.03,
        "teacher_beats_control_in_at_least_2_contexts": contexts_won >= 2,
        "mean_teacher_vs_control_on_at_least_2pp": mean_teacher_control >= 0.02,
        "mean_D0_drop_no_worse_than_minus_5pp": mean_d0 >= -0.05,
        "at_least_one_internalization_metric_improves_consistently": bool(consistent),
    }
    valuable = all(criteria.values())
    return {
        "paper_value": valuable,
        "criteria": criteria,
        "mean_teacher_on_gain_over_base": mean_teacher_gain,
        "teacher_control_contexts_won": contexts_won,
        "mean_teacher_on_gain_over_control": mean_teacher_control,
        "mean_D0_teacher_on_gain_over_base": mean_d0,
        "consistently_improved_internalization_metrics": consistent,
        "decision": (
            "retain selected formulation for fixed real-stair follow-up"
            if valuable
            else "end the current CBF action-teacher route without further search"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--eval-batch-size", type=int, default=PREFERRED_EVAL_BATCH_SIZE
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    base = args.base_checkpoint.resolve()
    training_root = args.training_root.resolve()
    output = args.output_dir.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v30 formal audit requires a clean worktree")
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen_before_v30_formal":
        raise RuntimeError("v30 formal audit requires the formal freeze")
    if file_sha256(base) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v30 formal audit base checkpoint differs")
    selected_arm = protocol["formal"]["selected_teacher"]["arm"]
    if selected_arm not in TEACHER_ARMS or protocol["formal"]["selected_teacher"][
        "configuration"
    ] != arm_parameters(selected_arm):
        raise RuntimeError("v30 formal selected teacher differs")
    if (output / "combined_results.json").exists() and not args.resume:
        raise RuntimeError("v30 formal audit output already exists")
    output.mkdir(parents=True, exist_ok=True)
    protocol_hash = file_sha256(protocol_path)
    all_rows = []
    context_payloads = {}
    for context in FORMAL_CONTEXTS:
        control, control_training = _training_checkpoint(
            training_root, context, "A0", protocol_hash
        )
        teacher, teacher_training = _training_checkpoint(
            training_root, context, selected_arm, protocol_hash
        )
        checkpoint_by_condition = {
            "base_off": base,
            "base_on": base,
            "control_on": control,
            "control_off": control,
            "teacher_on": teacher,
            "teacher_off": teacher,
        }
        target_aggregates = {}
        target_rows = {}
        for condition in TARGET_CONDITIONS:
            mode = "off" if condition.endswith("off") else "on"
            target_aggregates[condition], target_rows[condition] = evaluate_condition(
                repo=repo,
                protocol=protocol_path,
                checkpoint=checkpoint_by_condition[condition],
                context=context,
                condition=condition,
                runtime_filter=mode,
                episodes=FORMAL_TARGET_EPISODES,
                batch_size=min(args.eval_batch_size, FORMAL_TARGET_EPISODES),
                seed_base=FORMAL_TARGET_SEED_BASES[context],
                output_root=output,
                device=args.device,
                resume=args.resume,
            )
        assert_paired(target_aggregates, target_rows, label=f"formal/{context}/target")
        d0_checkpoint = {
            "base_on": base,
            "control_on": control,
            "teacher_on": teacher,
        }
        d0_aggregates = {}
        d0_rows = {}
        for condition in D0_CONDITIONS:
            d0_aggregates[condition], d0_rows[condition] = evaluate_condition(
                repo=repo,
                protocol=protocol_path,
                checkpoint=d0_checkpoint[condition],
                context="D0",
                condition=f"{context}_{condition}",
                runtime_filter="on",
                episodes=FORMAL_D0_EPISODES,
                batch_size=min(args.eval_batch_size, FORMAL_D0_EPISODES),
                seed_base=FORMAL_D0_SEED_BASES[context],
                output_root=output,
                device=args.device,
                resume=args.resume,
            )
        assert_paired(d0_aggregates, d0_rows, label=f"formal/{context}/D0")
        effects = _context_effects(
            context, target_aggregates, d0_aggregates, target_rows, d0_rows
        )
        context_payload = {
            "context": context,
            "selected_teacher_arm": selected_arm,
            "target": target_aggregates,
            "D0": d0_aggregates,
            "effects": effects,
            "control_training_summary_sha256": file_sha256(
                training_root / context / "A0" / "training_summary.json"
            ),
            "teacher_training_summary_sha256": file_sha256(
                training_root / context / selected_arm / "training_summary.json"
            ),
            "control_final_checkpoint_sha256": control_training[
                "final_checkpoint_sha256"
            ],
            "teacher_final_checkpoint_sha256": teacher_training[
                "final_checkpoint_sha256"
            ],
        }
        context_payloads[context] = context_payload
        context_dir = output / context
        atomic_json(context_dir / "context_results.json", context_payload)
        condition_rows = [
            _aggregate_row(context, "target", name, target_aggregates[name])
            for name in TARGET_CONDITIONS
        ] + [
            _aggregate_row(context, "D0", name, d0_aggregates[name])
            for name in D0_CONDITIONS
        ]
        write_csv(context_dir / "condition_results.csv", condition_rows)
        paired_rows = paired_wide_rows(
            domain="target", conditions=target_rows
        ) + paired_wide_rows(domain="D0", conditions=d0_rows)
        write_csv(context_dir / "paired_episode_metrics.csv", paired_rows)
        all_rows.extend(condition_rows)
    paper_value = _paper_value(context_payloads)
    combined = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "protocol_sha256": protocol_hash,
        "selected_teacher": protocol["formal"]["selected_teacher"],
        "contexts": context_payloads,
        "paper_value_assessment": paper_value,
        "all_formal_runs_and_audits_published_without_rerun": True,
        "paired_CI_used_as_gate": False,
        "final_audit_used_for_policy_selection_or_modification": False,
    }
    atomic_json(output / "combined_results.json", combined)
    write_csv(output / "combined_results.csv", all_rows)
    atomic_json(
        output / "formal_audit_complete.json",
        {
            "protocol_id": PROTOCOL_ID,
            "combined_results_sha256": file_sha256(output / "combined_results.json"),
            "combined_results_csv_sha256": file_sha256(output / "combined_results.csv"),
        },
    )
    print(json.dumps(paper_value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
