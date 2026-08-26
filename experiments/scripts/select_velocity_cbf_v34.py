"""Evaluate the six trained v34 runs on development identities and select one CBF."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from proximal_v23_io import file_sha256
from velocity_cbf_v34_protocol import (
    FORMAL_CONTEXTS,
    OPTIMIZED_CBF_MODE,
    PARAMETER_RANGES,
    PROTOCOL_ID,
    TRAINED_DEVELOPMENT_EPISODES,
    trained_development_seed,
)

PARAMETER_NAMES = tuple(PARAMETER_RANGES)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--top2", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _evaluate(
    *,
    repo: Path,
    config: Path,
    candidate: dict[str, Any],
    checkpoint: Path,
    context: str,
    run_dir: Path,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    episodes_path = run_dir / "episodes.csv"
    seed = trained_development_seed(context)
    valid = False
    if resume and summary_path.is_file() and episodes_path.is_file():
        prior = json.loads(summary_path.read_text())
        valid = (
            prior.get("protocol_id") == PROTOCOL_ID
            and prior.get("seed") == seed
            and prior.get("num_episodes") == TRAINED_DEVELOPMENT_EPISODES
            and prior.get("checkpoint_sha256") == file_sha256(checkpoint)
            and prior.get("candidate") == candidate["candidate"]
        )
    if not valid:
        parameters = {name: float(candidate[name]) for name in PARAMETER_NAMES}
        command = [
            sys.executable,
            str(repo / "experiments/scripts/evaluate_velocity_cbf_v34.py"),
            "--repo",
            str(repo),
            "--search-config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--context",
            context,
            "--cbf-mode",
            OPTIMIZED_CBF_MODE,
            "--parameters-json",
            json.dumps(parameters),
            "--runtime-filter",
            "on",
            "--num-envs",
            str(TRAINED_DEVELOPMENT_EPISODES),
            "--num-episodes",
            str(TRAINED_DEVELOPMENT_EPISODES),
            "--seed",
            str(seed),
            "--policy-label",
            "new_A2_round_8",
            "--candidate",
            candidate["candidate"],
            "--device",
            device,
            "--output-json",
            str(summary_path),
            "--output-csv",
            str(episodes_path),
        ]
        completed = subprocess.run(
            command, cwd=repo, capture_output=True, text=True, check=False
        )
        if completed.returncode:
            diagnostic = "\n".join(
                (completed.stdout + completed.stderr).splitlines()[-120:]
            )
            raise RuntimeError(
                f"v34 trained development evaluation failed for "
                f"{candidate['candidate']}/{context}:\n{diagnostic}"
            )
    return json.loads(summary_path.read_text())


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    config_path = args.search_config.resolve()
    top2_path = args.top2.resolve()
    config = json.loads(config_path.read_text())
    top2 = json.loads(top2_path.read_text())
    if (
        config.get("protocol_id") != PROTOCOL_ID
        or top2.get("protocol_id") != PROTOCOL_ID
    ):
        raise RuntimeError("v34 trained selection inputs differ")
    output = args.output_root.resolve()
    if output.exists() and not args.resume:
        raise RuntimeError("v34 trained development output exists")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    checkpoint_records: dict[str, dict[str, Any]] = {}
    for candidate in top2["top2"]:
        successes = []
        checkpoint_records[candidate["candidate"]] = {}
        row: dict[str, Any] = {
            "candidate": candidate["candidate"],
            "candidate_index": int(candidate["candidate_index"]),
        }
        for context in FORMAL_CONTEXTS:
            training_dir = (
                args.training_root.resolve() / candidate["candidate"] / context
            )
            checkpoint = training_dir / "round_08.pt"
            training = json.loads((training_dir / "training_summary.json").read_text())
            if (
                training.get("protocol_id") != PROTOCOL_ID
                or training.get("rounds_completed") != 8
                or training.get("candidate") != candidate["candidate"]
                or training.get("context") != context
                or training.get("final_checkpoint_sha256") != file_sha256(checkpoint)
            ):
                raise RuntimeError(
                    f"v34 training boundary differs for {candidate['candidate']}/{context}"
                )
            summary = _evaluate(
                repo=repo,
                config=config_path,
                candidate=candidate,
                checkpoint=checkpoint,
                context=context,
                run_dir=output / "raw" / candidate["candidate"] / context,
                device=args.device,
                resume=args.resume,
            )
            success = float(summary["success_rate"])
            successes.append(success)
            row[f"{context}_success"] = success
            row[f"{context}_fall_rate"] = float(summary["fall_rate"])
            row[f"{context}_mean_return"] = float(summary["mean_return"])
            checkpoint_records[candidate["candidate"]][context] = {
                "external_path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
                "actor_state_sha256": training["final_actor_sha256"],
            }
            print(
                json.dumps(
                    {
                        "phase": "trained_development",
                        "candidate": candidate["candidate"],
                        "context": context,
                        "success": success,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        row["mean_success"] = sum(successes) / len(successes)
        rows.append(row)
    rows.sort(
        key=lambda row: (-float(row["mean_success"]), int(row["candidate_index"]))
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    _write_csv(output / "trained_top2_results.csv", rows)
    selected_name = rows[0]["candidate"]
    selected = next(row for row in top2["top2"] if row["candidate"] == selected_name)
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "development_selected_parameters_and_round8_policies",
        "selection_objective": "trained mean F1/F2/F3 CBF-on success only",
        "candidate": selected_name,
        "candidate_index": int(selected["candidate_index"]),
        "parameters": {name: float(selected[name]) for name in PARAMETER_NAMES},
        "trained_development_results": rows,
        "trained_development_mean_success": float(rows[0]["mean_success"]),
        "trained_checkpoints": checkpoint_records[selected_name],
        "all_top2_checkpoint_records": checkpoint_records,
        "elapsed_seconds": time.monotonic() - started,
        "final_seeds_created": False,
        "final_test_started": False,
    }
    _atomic_json(output / "development_selection.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
