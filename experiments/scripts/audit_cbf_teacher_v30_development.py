"""Evaluate all six v30 development arms and apply the frozen selection rule."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from cbf_teacher_v30_eval_io import (
    assert_paired,
    atomic_json,
    evaluate_condition,
    paired_repairs_regressions,
    write_csv,
)
from cbf_teacher_v30_protocol import (
    ARMS,
    BASE_CHECKPOINT_SHA256,
    DEVELOPMENT_D0_EPISODES,
    DEVELOPMENT_D0_SEED_BASE,
    DEVELOPMENT_TARGET_EPISODES,
    DEVELOPMENT_TARGET_SEED_BASE,
    PREFERRED_EVAL_BATCH_SIZE,
    PROTOCOL_ID,
    TEACHER_ARMS,
    arm_parameters,
)
from proximal_v23_io import file_sha256


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _training_inputs(root: Path, arm: str) -> tuple[Path, dict[str, Any]]:
    directory = root / arm
    summary_path = directory / "training_summary.json"
    checkpoint = directory / "round_08.pt"
    if not summary_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"v30 development output missing for {arm}")
    summary = json.loads(summary_path.read_text())
    checks = {
        "protocol": summary.get("protocol_id") == PROTOCOL_ID,
        "phase": summary.get("phase") == "development",
        "arm": summary.get("arm") == arm,
        "context": summary.get("context") == "DEV",
        "configuration": summary.get("arm_configuration") == arm_parameters(arm),
        "rounds": summary.get("rounds_completed") == 8,
        "final_rule": summary.get("final_policy_rule") == "unconditional round 8 actor",
        "checkpoint": summary.get("final_checkpoint_sha256") == file_sha256(checkpoint),
        "no_kl_stop": summary.get("kl_stop_count") == 0
        and summary.get("kl_rollback_count") == 0,
        "no_performance_selection": summary.get("performance_selection_count") == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid v30 development run {arm}: {checks}")
    return checkpoint, summary


def _summary_row(
    arm: str,
    training: dict[str, Any],
    target: dict[str, dict[str, Any]],
    d0: dict[str, dict[str, Any]],
    repairs: dict[str, int],
) -> dict[str, Any]:
    base_off = target["base_off"]
    base_on = target["base_on"]
    final_on = target["arm_on"]
    final_off = target["arm_off"]
    return {
        "arm": arm,
        "teacher_name": training["arm_configuration"]["name"],
        "teacher_mode": training["arm_configuration"]["teacher_mode"],
        "teacher_gate": training["arm_configuration"]["teacher_gate"],
        "teacher_eta": training["arm_configuration"]["teacher_eta"],
        "teacher_weight": training["arm_configuration"]["teacher_weight"],
        "target_base_on_success": base_on["success_rate"],
        "target_round8_on_success": final_on["success_rate"],
        "target_on_gain_over_base": final_on["success_rate"] - base_on["success_rate"],
        "target_base_off_success": base_off["success_rate"],
        "target_round8_off_success": final_off["success_rate"],
        "target_off_gain_over_base": final_off["success_rate"]
        - base_off["success_rate"],
        "target_round8_on_fall": final_on["fall_rate"],
        "target_round8_off_fall": final_off["fall_rate"],
        "target_round8_on_return": final_on["mean_return"],
        "target_round8_on_mean_riser": final_on["mean_reached_riser"],
        "target_round8_on_completion_time_s": final_on["mean_completion_time_s"],
        "target_round8_on_intervention_steps_per_riser": final_on[
            "intervention_steps_per_riser"
        ],
        "target_round8_off_would_intervene_fraction": final_off[
            "counterfactual_would_intervene_fraction"
        ],
        "target_round8_off_mean_correction_norm": final_off[
            "mean_counterfactual_correction_norm"
        ],
        "target_round8_off_nominal_violation_steps_per_riser": final_off[
            "nominal_barrier_violation_steps_per_riser"
        ],
        "target_round8_off_kick_events_per_riser": final_off[
            "toe_riser_kick_events_per_riser"
        ],
        "target_round8_off_unsafe_overlap_steps_per_riser": final_off[
            "unsafe_overlap_steps_per_riser"
        ],
        "target_round8_off_minimum_nominal_barrier_margin": final_off[
            "mean_episode_minimum_nominal_barrier_margin"
        ],
        "target_repairs_vs_base_on": repairs["repairs"],
        "target_regressions_vs_base_on": repairs["regressions"],
        "D0_base_on_success": d0["base_on"]["success_rate"],
        "D0_round8_on_success": d0["arm_on"]["success_rate"],
        "D0_on_gain_over_base": d0["arm_on"]["success_rate"]
        - d0["base_on"]["success_rate"],
        "moving_KL_round8": training["rounds"][-1]["metrics"]["moving_forward_kl"],
        "final_checkpoint_sha256": training["final_checkpoint_sha256"],
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
    checkpoint = args.base_checkpoint.resolve()
    training_root = args.training_root.resolve()
    output = args.output_dir.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v30 development audit requires a clean worktree")
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen_before_v30_development":
        raise RuntimeError("v30 development audit requires development freeze")
    if file_sha256(checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v30 development audit base checkpoint differs")
    committed = subprocess.run(
        ["git", "show", f"HEAD:{protocol_path.relative_to(repo)}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    if committed != protocol_path.read_bytes():
        raise RuntimeError("v30 development protocol is not committed")
    if (output / "arm_summary.json").exists() and not args.resume:
        raise RuntimeError("v30 development audit output already exists")
    output.mkdir(parents=True, exist_ok=True)

    training: dict[str, dict[str, Any]] = {}
    checkpoints: dict[str, Path] = {}
    for arm in ARMS:
        checkpoints[arm], training[arm] = _training_inputs(training_root, arm)

    target_aggregates: dict[str, dict[str, Any]] = {}
    target_rows: dict[str, list[dict[str, str]]] = {}
    d0_aggregates: dict[str, dict[str, Any]] = {}
    d0_rows: dict[str, list[dict[str, str]]] = {}
    for label, mode, context, episodes, seed_base, aggregates, rows in (
        (
            "base_off",
            "off",
            "DEV",
            DEVELOPMENT_TARGET_EPISODES,
            DEVELOPMENT_TARGET_SEED_BASE,
            target_aggregates,
            target_rows,
        ),
        (
            "base_on",
            "on",
            "DEV",
            DEVELOPMENT_TARGET_EPISODES,
            DEVELOPMENT_TARGET_SEED_BASE,
            target_aggregates,
            target_rows,
        ),
        (
            "base_on",
            "on",
            "D0",
            DEVELOPMENT_D0_EPISODES,
            DEVELOPMENT_D0_SEED_BASE,
            d0_aggregates,
            d0_rows,
        ),
    ):
        aggregates[label], rows[label] = evaluate_condition(
            repo=repo,
            protocol=protocol_path,
            checkpoint=checkpoint,
            context=context,
            condition=label,
            runtime_filter=mode,
            episodes=episodes,
            batch_size=min(args.eval_batch_size, episodes),
            seed_base=seed_base,
            output_root=output,
            device=args.device,
            resume=args.resume,
        )

    summary_rows = []
    arm_payloads = {}
    for arm in ARMS:
        arm_target_aggregates = {
            "base_off": target_aggregates["base_off"],
            "base_on": target_aggregates["base_on"],
        }
        arm_target_rows = {
            "base_off": target_rows["base_off"],
            "base_on": target_rows["base_on"],
        }
        for mode in ("on", "off"):
            condition = f"{arm}_{mode}"
            aggregate, rows = evaluate_condition(
                repo=repo,
                protocol=protocol_path,
                checkpoint=checkpoints[arm],
                context="DEV",
                condition=condition,
                runtime_filter=mode,
                episodes=DEVELOPMENT_TARGET_EPISODES,
                batch_size=min(args.eval_batch_size, DEVELOPMENT_TARGET_EPISODES),
                seed_base=DEVELOPMENT_TARGET_SEED_BASE,
                output_root=output,
                device=args.device,
                resume=args.resume,
            )
            arm_target_aggregates[f"arm_{mode}"] = aggregate
            arm_target_rows[f"arm_{mode}"] = rows
        arm_d0_aggregate, arm_d0_rows = evaluate_condition(
            repo=repo,
            protocol=protocol_path,
            checkpoint=checkpoints[arm],
            context="D0",
            condition=f"{arm}_on",
            runtime_filter="on",
            episodes=DEVELOPMENT_D0_EPISODES,
            batch_size=min(args.eval_batch_size, DEVELOPMENT_D0_EPISODES),
            seed_base=DEVELOPMENT_D0_SEED_BASE,
            output_root=output,
            device=args.device,
            resume=args.resume,
        )
        arm_d0_aggregates = {
            "base_on": d0_aggregates["base_on"],
            "arm_on": arm_d0_aggregate,
        }
        assert_paired(
            arm_target_aggregates,
            arm_target_rows,
            label=f"development/{arm}/target",
        )
        assert_paired(
            arm_d0_aggregates,
            {"base_on": d0_rows["base_on"], "arm_on": arm_d0_rows},
            label=f"development/{arm}/D0",
        )
        repairs = paired_repairs_regressions(
            arm_target_rows["base_on"], arm_target_rows["arm_on"]
        )
        row = _summary_row(
            arm,
            training[arm],
            arm_target_aggregates,
            arm_d0_aggregates,
            repairs,
        )
        summary_rows.append(row)
        arm_payloads[arm] = {
            "training_summary_sha256": file_sha256(
                training_root / arm / "training_summary.json"
            ),
            "target": arm_target_aggregates,
            "D0": arm_d0_aggregates,
            "repairs_and_regressions_vs_base_on": repairs,
            "summary": row,
        }

    teacher_rows = [row for row in summary_rows if row["arm"] in TEACHER_ARMS]
    selected_row = min(
        teacher_rows,
        key=lambda row: (
            -float(row["target_round8_on_success"]),
            -float(row["target_round8_off_success"]),
            float(row["target_round8_on_intervention_steps_per_riser"]),
        ),
    )
    selected_arm = str(selected_row["arm"])
    selected = {
        "protocol_id": PROTOCOL_ID,
        "arm": selected_arm,
        "configuration": arm_parameters(selected_arm),
        "rule_applied_in_order": protocol["development"]["selection_rule"],
        "A0_excluded": True,
        "KL_used": False,
        "intermediate_checkpoint_used": False,
        "selected_metrics": selected_row,
    }
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "protocol_sha256": file_sha256(protocol_path),
        "base_checkpoint_sha256": file_sha256(checkpoint),
        "development_arm_order": list(ARMS),
        "all_runs_retained_without_rerun": True,
        "base_conditions_evaluated_once_and_shared": True,
        "evaluation_batch_size": args.eval_batch_size,
        "arms": arm_payloads,
        "selected_teacher": selected,
    }
    atomic_json(output / "arm_summary.json", payload)
    write_csv(output / "arm_summary.csv", summary_rows)
    atomic_json(output / "selected_teacher.json", selected)
    atomic_json(
        output / "development_audit_complete.json",
        {
            "protocol_id": PROTOCOL_ID,
            "arm_summary_sha256": file_sha256(output / "arm_summary.json"),
            "arm_summary_csv_sha256": file_sha256(output / "arm_summary.csv"),
            "selected_teacher_sha256": file_sha256(output / "selected_teacher.json"),
        },
    )
    print(json.dumps(selected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
