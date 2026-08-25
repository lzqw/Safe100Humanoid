"""Read-only v32 monitors at the prospectively fixed rounds 8, 16, and 24."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from cbf_teacher_v32_eval_io import assert_paired, evaluate_condition, write_csv
from cbf_teacher_v32_protocol import (
    CONTINUATION_SCHEDULES,
    FORMAL_CONTEXTS,
    MIXED_SCHEDULE,
    MONITOR_EPISODES,
    MONITOR_ROUNDS,
    MONITOR_SEED_BASES,
    PROTOCOL_ID,
    V31_A2_ROUND8_SHA256,
)
from proximal_v23_io import file_sha256


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _training_checkpoint(
    training_root: Path,
    *,
    kind: str,
    context: str,
    schedule: str,
    round_index: int,
    protocol_hash: str,
) -> tuple[Path, dict[str, Any]]:
    directory = (
        training_root / "continuation" / context / schedule
        if kind == "continuation"
        else training_root / "mixed" / schedule
    )
    checkpoint = directory / f"round_{round_index:02d}.pt"
    summary_path = directory / "training_summary.json"
    if not checkpoint.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"missing v32 {kind} checkpoint {checkpoint}")
    summary = json.loads(summary_path.read_text())
    expected_rounds = 16 if kind == "continuation" else 24
    checks = {
        "protocol": summary.get("protocol_id") == PROTOCOL_ID,
        "protocol_hash": summary.get("protocol", {}).get("sha256") == protocol_hash,
        "phase": summary.get("phase") == "formal",
        "kind": summary.get("kind") == kind,
        "context": summary.get("context") == context,
        "schedule": summary.get("schedule") == schedule,
        "rounds": summary.get("rounds_completed") == expected_rounds,
        "fixed_final": summary.get("final_policy_rule")
        == "unconditional round 24 actor",
        "no_gates": summary.get("performance_selection_count") == 0
        and summary.get("KL_stop_count") == 0
        and summary.get("KL_rollback_count") == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid v32 training summary: {checks}")
    return checkpoint, summary


def _row(
    *,
    family: str,
    context: str,
    schedule: str,
    round_index: int,
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "family": family,
        "context": context,
        "schedule": schedule,
        "round": round_index,
        "success_rate": aggregate["success_rate"],
        "fall_rate": aggregate["fall_rate"],
        "mean_return": aggregate["mean_return"],
        "mean_reached_riser": aggregate["mean_reached_riser"],
        "intervention_steps_per_riser": aggregate["intervention_steps_per_riser"],
        "mean_correction_norm": aggregate["mean_correction_norm"],
        "checkpoint_sha256": aggregate["checkpoint_sha256"],
        "actor_state_sha256": aggregate["actor_state_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v31-formal-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    v31_root = args.v31_formal_root.resolve()
    training_root = args.training_root.resolve()
    output = args.output_dir.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v32 monitor requires a clean committed worktree")
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status") != "frozen_before_v32_preflight_and_formal"
    ):
        raise RuntimeError("v32 monitor protocol differs")
    if (output / "monitor_summary.json").exists() and not args.resume:
        raise RuntimeError("v32 monitor is already complete")
    output.mkdir(parents=True, exist_ok=True)
    protocol_hash = file_sha256(protocol_path)

    continuation_rows = []
    raw_rows: dict[str, dict[str, list[dict[str, str]]]] = {}
    checkpoint_hashes_before = {}
    for context in FORMAL_CONTEXTS:
        raw_rows[context] = {}
        checkpoints: list[tuple[str, int, Path]] = [
            ("v31_A2", 8, v31_root / context / "A2" / "round_08.pt")
        ]
        if file_sha256(checkpoints[0][2]) != V31_A2_ROUND8_SHA256[context]:
            raise RuntimeError(f"v32 monitor v31 input differs for {context}")
        for schedule in CONTINUATION_SCHEDULES:
            for round_index in (16, 24):
                checkpoint, _ = _training_checkpoint(
                    training_root,
                    kind="continuation",
                    context=context,
                    schedule=schedule,
                    round_index=round_index,
                    protocol_hash=protocol_hash,
                )
                checkpoints.append((schedule, round_index, checkpoint))
        aggregates = {}
        for schedule, round_index, checkpoint in checkpoints:
            condition = f"continuation_{context}_{schedule}_round_{round_index:02d}"
            checkpoint_hashes_before[str(checkpoint)] = file_sha256(checkpoint)
            aggregate, rows = evaluate_condition(
                repo=repo,
                protocol=protocol_path,
                checkpoint=checkpoint,
                context=context,
                condition=condition,
                runtime_filter="on",
                episodes=MONITOR_EPISODES,
                batch_size=MONITOR_EPISODES,
                seed_base=MONITOR_SEED_BASES[context],
                output_root=output,
                device=args.device,
                resume=args.resume,
            )
            aggregates[condition] = aggregate
            raw_rows[context][condition] = rows
            continuation_rows.append(
                _row(
                    family="continuation",
                    context=context,
                    schedule=schedule,
                    round_index=round_index,
                    aggregate=aggregate,
                )
            )
        assert_paired(aggregates, raw_rows[context], label=f"v32 monitor/{context}")

    mixed_rows = []
    mixed_aggregates: dict[str, dict[str, dict[str, Any]]] = {
        context: {} for context in FORMAL_CONTEXTS
    }
    mixed_raw_rows: dict[str, dict[str, list[dict[str, str]]]] = {
        context: {} for context in FORMAL_CONTEXTS
    }
    for round_index in MONITOR_ROUNDS:
        checkpoint, _ = _training_checkpoint(
            training_root,
            kind="mixed",
            context="mixed",
            schedule=MIXED_SCHEDULE,
            round_index=round_index,
            protocol_hash=protocol_hash,
        )
        checkpoint_hashes_before[str(checkpoint)] = file_sha256(checkpoint)
        for context in FORMAL_CONTEXTS:
            condition = f"mixed_{context}_round_{round_index:02d}"
            aggregate, rows = evaluate_condition(
                repo=repo,
                protocol=protocol_path,
                checkpoint=checkpoint,
                context=context,
                condition=condition,
                runtime_filter="on",
                episodes=MONITOR_EPISODES,
                batch_size=MONITOR_EPISODES,
                seed_base=MONITOR_SEED_BASES[context],
                output_root=output,
                device=args.device,
                resume=args.resume,
            )
            mixed_rows.append(
                _row(
                    family="mixed",
                    context=context,
                    schedule=MIXED_SCHEDULE,
                    round_index=round_index,
                    aggregate=aggregate,
                )
            )
            mixed_aggregates[context][condition] = aggregate
            mixed_raw_rows[context][condition] = rows
    for context in FORMAL_CONTEXTS:
        assert_paired(
            mixed_aggregates[context],
            mixed_raw_rows[context],
            label=f"v32 mixed monitor/{context}",
        )

    checkpoint_hashes_after = {
        path: file_sha256(Path(path)) for path in checkpoint_hashes_before
    }
    if checkpoint_hashes_after != checkpoint_hashes_before:
        raise RuntimeError("v32 read-only monitor changed a checkpoint")
    write_csv(output / "continuation_monitor.csv", continuation_rows)
    write_csv(output / "mixed_monitor.csv", mixed_rows)
    write_csv(output / "monitor_results.csv", continuation_rows + mixed_rows)

    def success(family: str, context: str, schedule: str, round_index: int) -> float:
        rows = continuation_rows if family == "continuation" else mixed_rows
        return float(
            next(
                row["success_rate"]
                for row in rows
                if row["context"] == context
                and row["schedule"] == schedule
                and row["round"] == round_index
            )
        )

    saturation = {
        context: {
            schedule: {
                "round8_to_round16": success("continuation", context, schedule, 16)
                - success("continuation", context, "v31_A2", 8),
                "round16_to_round24": success("continuation", context, schedule, 24)
                - success("continuation", context, schedule, 16),
            }
            for schedule in CONTINUATION_SCHEDULES
        }
        for context in FORMAL_CONTEXTS
    }
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "fixed_rounds": list(MONITOR_ROUNDS),
        "episodes_per_condition": MONITOR_EPISODES,
        "runtime_CBF": "on",
        "deterministic_policy_mean": True,
        "continuation_rows": len(continuation_rows),
        "mixed_rows": len(mixed_rows),
        "read_only": True,
        "checkpoint_hashes_unchanged": True,
        "used_for_selection_stopping_or_training": False,
        "saturation_diagnostics": saturation,
        "results_csv_sha256": file_sha256(output / "monitor_results.csv"),
    }
    (output / "monitor_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
