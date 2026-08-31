"""Batched v32 evaluation I/O using v31's unchanged aggregation formulas."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from cbf_teacher_v31_eval_io import (
    aggregate_condition,
    assert_paired,
    atomic_json,
    identity,
    paired_ci,
    paired_repairs_regressions,
    paired_wide_rows,
    write_csv,
)
from cbf_teacher_v32_protocol import PROTOCOL_ID
from proximal_v23_io import file_sha256

__all__ = [
    "assert_paired",
    "atomic_json",
    "evaluate_condition",
    "paired_ci",
    "paired_repairs_regressions",
    "paired_wide_rows",
    "write_csv",
]


def evaluate_condition(
    *,
    repo: Path,
    protocol: Path,
    checkpoint: Path,
    context: str,
    condition: str,
    runtime_filter: str,
    episodes: int,
    batch_size: int,
    seed_base: int,
    output_root: Path,
    device: str,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if episodes < 1 or episodes % batch_size:
        raise ValueError("v32 episodes must divide exactly by evaluation batch")
    summaries = []
    rows: list[dict[str, str]] = []
    checkpoint_hash = file_sha256(checkpoint)
    protocol_hash = file_sha256(protocol)
    for repeat in range(episodes // batch_size):
        seed = seed_base + repeat
        run_dir = output_root / "raw" / context / condition / f"seed_{seed}"
        output_json = run_dir / "summary.json"
        output_csv = run_dir / "episodes.csv"
        summary = None
        if resume and output_json.is_file() and output_csv.is_file():
            candidate = json.loads(output_json.read_text())
            if (
                candidate.get("context") == context
                and candidate.get("seed") == seed
                and candidate.get("num_envs") == batch_size
                and candidate.get("num_episodes") == batch_size
                and candidate.get("runtime_filter") == (runtime_filter == "on")
                and candidate.get("checkpoint_sha256") == checkpoint_hash
                and candidate.get("v32_protocol_id") == PROTOCOL_ID
                and candidate.get("v32_protocol_sha256") == protocol_hash
            ):
                summary = candidate
        if summary is None:
            command = [
                sys.executable,
                str(repo / "experiments/scripts/evaluate_cbf_teacher_v32.py"),
                "--repo",
                str(repo),
                "--protocol",
                str(protocol),
                "--checkpoint",
                str(checkpoint),
                "--context",
                context,
                "--runtime-filter",
                runtime_filter,
                "--num-envs",
                str(batch_size),
                "--num-episodes",
                str(batch_size),
                "--seed",
                str(seed),
                "--device",
                device,
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
            ]
            completed = subprocess.run(
                command, cwd=repo, capture_output=True, text=True, check=False
            )
            if completed.returncode:
                diagnostic = "\n".join(
                    (completed.stdout + "\n" + completed.stderr).splitlines()[-180:]
                )
                raise RuntimeError(
                    f"v32 evaluation failed for {context}/{condition}:\n{diagnostic}"
                )
            summary = json.loads(output_json.read_text())
        with output_csv.open(newline="") as handle:
            batch_rows = list(csv.DictReader(handle))
        if len(batch_rows) != batch_size:
            raise RuntimeError(f"incomplete v32 rows for {context}/{condition}")
        summaries.append(summary)
        rows.extend(batch_rows)
    rows.sort(key=identity)
    if len({identity(row) for row in rows}) != len(rows):
        raise RuntimeError(f"duplicate v32 identity in {context}/{condition}")
    aggregate = aggregate_condition(
        context=context,
        condition=condition,
        runtime_filter=runtime_filter == "on",
        summaries=summaries,
        rows=rows,
    )
    aggregate["v32_protocol_id"] = PROTOCOL_ID
    aggregate["v32_protocol_sha256"] = protocol_hash
    return aggregate, rows
