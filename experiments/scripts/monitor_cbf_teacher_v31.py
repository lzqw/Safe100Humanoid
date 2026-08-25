"""Read-only F1 round-0-through-8 monitor after all formal v31 work."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from cbf_teacher_v31_eval_io import (
    assert_paired,
    atomic_json,
    evaluate_condition,
    write_csv,
)
from cbf_teacher_v31_protocol import (
    METHOD_ARMS,
    MONITOR_EPISODES,
    MONITOR_SEED,
    PROTOCOL_ID,
)
from proximal_v23_io import file_sha256


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--formal-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    training_root = args.training_root.resolve()
    formal_audit_path = args.formal_audit.resolve()
    output = args.output_dir.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v31 monitor requires a clean worktree")
    protocol = json.loads(protocol_path.read_text())
    formal_audit = json.loads(formal_audit_path.read_text())
    if (
        protocol.get("status") != "frozen_before_v31_preflight_and_formal"
        or formal_audit.get("protocol_id") != PROTOCOL_ID
        or not formal_audit.get("complete")
    ):
        raise RuntimeError("v31 monitor can run only after the formal audit")
    if (output / "F1_checkpoint_curve.csv").exists() and not args.resume:
        raise RuntimeError("v31 monitor output already exists")
    output.mkdir(parents=True, exist_ok=True)
    method_arms = {arm: arm for arm in METHOD_ARMS}
    checkpoint_hashes_before = {}
    training_summaries = {}
    for method, arm in method_arms.items():
        directory = training_root / "F1" / arm
        summary_path = directory / "training_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        training_summaries[method] = json.loads(summary_path.read_text())
        for round_index in range(9):
            checkpoint = directory / f"round_{round_index:02d}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoint_hashes_before[str(checkpoint)] = file_sha256(checkpoint)

    curve_rows = []
    for method, arm in method_arms.items():
        summary = training_summaries[method]
        for round_index in range(9):
            checkpoint = training_root / "F1" / arm / f"round_{round_index:02d}.pt"
            aggregates = {}
            episode_rows = {}
            for mode in ("on", "off"):
                condition = f"{method}_round_{round_index:02d}_{mode}"
                aggregates[mode], episode_rows[mode] = evaluate_condition(
                    repo=repo,
                    protocol=protocol_path,
                    checkpoint=checkpoint,
                    context="F1",
                    condition=condition,
                    runtime_filter=mode,
                    episodes=MONITOR_EPISODES,
                    batch_size=MONITOR_EPISODES,
                    seed_base=MONITOR_SEED,
                    output_root=output,
                    device=args.device,
                    resume=args.resume,
                )
            assert_paired(
                aggregates,
                episode_rows,
                label=f"monitor/F1/{method}/round_{round_index:02d}",
            )
            training_metrics = (
                {}
                if round_index == 0
                else summary["rounds"][round_index - 1]["metrics"]
            )
            curve_rows.append(
                {
                    "context": "F1",
                    "method": method,
                    "arm": arm,
                    "round": round_index,
                    "CBF_on_success": aggregates["on"]["success_rate"],
                    "CBF_off_success": aggregates["off"]["success_rate"],
                    "CBF_on_fall": aggregates["on"]["fall_rate"],
                    "CBF_off_fall": aggregates["off"]["fall_rate"],
                    "CBF_on_intervention_steps_per_riser": aggregates["on"][
                        "intervention_steps_per_riser"
                    ],
                    "CBF_off_would_intervene_fraction": aggregates["off"][
                        "counterfactual_would_intervene_fraction"
                    ],
                    "policy_to_teacher_correction_gap": aggregates["off"][
                        "mean_counterfactual_correction_norm"
                    ],
                    "moving_KL": training_metrics.get("moving_forward_kl", 0.0),
                    "training_policy_target_distance_after": training_metrics.get(
                        "mean_policy_to_target_distance_after_update"
                    ),
                    "checkpoint_sha256": file_sha256(checkpoint),
                    "actor_state_sha256": aggregates["on"]["actor_state_sha256"],
                }
            )
    checkpoint_hashes_after = {
        path: file_sha256(Path(path)) for path in checkpoint_hashes_before
    }
    if checkpoint_hashes_after != checkpoint_hashes_before:
        raise RuntimeError("v31 read-only monitor changed a checkpoint")
    write_csv(output / "F1_checkpoint_curve.csv", curve_rows)
    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "context": "F1",
        "fixed_identities": MONITOR_EPISODES,
        "seed": MONITOR_SEED,
        "methods": method_arms,
        "rounds": list(range(9)),
        "filter_modes": ["on", "off"],
        "read_only": True,
        "checkpoint_hashes_unchanged": True,
        "used_for_selection_or_training": False,
        "curve_csv_sha256": file_sha256(output / "F1_checkpoint_curve.csv"),
    }
    atomic_json(output / "monitor_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
