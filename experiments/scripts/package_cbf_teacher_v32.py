"""Build v32's compact bilingual evidence package and figures."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cbf_teacher_v32_protocol import (
    CONTINUATION_SCHEDULES,
    FORMAL_CONTEXTS,
    PROTOCOL_ID,
)
from proximal_v23_io import file_sha256

METHODS = ("v31_A2", "LongConstant", "LongDecay", "Mixed")
COLORS = {
    "v31_A2": "#718096",
    "LongConstant": "#dd6b20",
    "LongDecay": "#2b6cb0",
    "Mixed": "#805ad5",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_union_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("v32 package cannot write an empty CSV")
    leading = [
        field
        for field in ("kind", "context", "schedule", "absolute_round")
        if any(field in row for row in rows)
    ]
    fields = leading + sorted({key for row in rows for key in row} - set(leading))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=180)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _success_figure(formal: dict[str, Any], base: Path) -> None:
    x = np.arange(len(FORMAL_CONTEXTS))
    width = 0.19
    fig, axis = plt.subplots(figsize=(10, 4.7))
    for index, method in enumerate(METHODS):
        values = [
            formal["contexts"][context]["target"][f"{method}_on"]["success_rate"]
            for context in FORMAL_CONTEXTS
        ]
        axis.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=method,
            color=COLORS[method],
        )
    axis.set(
        title="v32 CBF-on success by context",
        ylabel="success rate",
        xticks=x,
        xticklabels=FORMAL_CONTEXTS,
        ylim=(0.0, 1.0),
    )
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    _save(fig, base)


def _delta_figure(formal: dict[str, Any], base: Path) -> None:
    x = np.arange(len(FORMAL_CONTEXTS))
    width = 0.25
    fig, axis = plt.subplots(figsize=(9, 4.5))
    for index, method in enumerate(("LongConstant", "LongDecay", "Mixed")):
        values = formal["three_context_summary"]["direction_consistency"][method][
            "per_context_CBF_on_delta_vs_v31_A2"
        ]
        axis.bar(
            x + (index - 1) * width,
            [values[context] for context in FORMAL_CONTEXTS],
            width,
            label=method,
            color=COLORS[method],
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title="CBF-on success change versus v31 A2 round 8",
        ylabel="success-rate difference",
        xticks=x,
        xticklabels=FORMAL_CONTEXTS,
    )
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    _save(fig, base)


def _continuation_monitor_figure(rows: list[dict[str, str]], base: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for axis, context in zip(axes, FORMAL_CONTEXTS, strict=True):
        baseline = next(
            float(row["success_rate"])
            for row in rows
            if row["context"] == context and row["schedule"] == "v31_A2"
        )
        for schedule in CONTINUATION_SCHEDULES:
            subset = [
                row
                for row in rows
                if row["context"] == context and row["schedule"] == schedule
            ]
            points = [(8, baseline)] + [
                (int(row["round"]), float(row["success_rate"])) for row in subset
            ]
            axis.plot(
                [item[0] for item in points],
                [item[1] for item in points],
                marker="o",
                label=schedule,
                color=COLORS[schedule],
            )
        axis.set(title=context, xlabel="absolute round", xticks=(8, 16, 24))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("CBF-on success")
    axes[0].legend(fontsize=8)
    fig.suptitle("Continuation saturation monitor (128 fixed identities)")
    _save(fig, base)


def _mixed_monitor_figure(rows: list[dict[str, str]], base: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.3))
    for context in FORMAL_CONTEXTS:
        subset = [row for row in rows if row["context"] == context]
        axis.plot(
            [int(row["round"]) for row in subset],
            [float(row["success_rate"]) for row in subset],
            marker="o",
            label=context,
        )
    axis.set(
        title="Mixed-policy fixed-node monitor",
        xlabel="round",
        ylabel="CBF-on success",
        xticks=(8, 16, 24),
        ylim=(0.0, 1.0),
    )
    axis.legend()
    axis.grid(alpha=0.25)
    _save(fig, base)


def _retention_dependence_figure(formal: dict[str, Any], base: Path) -> None:
    means = formal["three_context_summary"]["method_means"]
    x = np.arange(len(METHODS))
    fields = (
        ("D0_success", "D0 success"),
        ("CBF_on_interventions_per_riser", "CBF-on interventions/riser"),
        ("CBF_off_would_intervene_fraction", "CBF-off would-intervene"),
        ("CBF_off_nominal_violation_steps_per_riser", "CBF-off violations/riser"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for axis, (field, title) in zip(axes.flat, fields, strict=True):
        axis.bar(
            x,
            [means[method][field] for method in METHODS],
            color=[COLORS[m] for m in METHODS],
        )
        axis.set(title=title, xticks=x, xticklabels=METHODS)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("D0 retention and CBF dependence")
    _save(fig, base)


def _external_manifest(
    training_root: Path, monitor_root: Path, audit_root: Path
) -> dict[str, Any]:
    groups = {}
    for label, root in (
        ("training", training_root),
        ("monitor", monitor_root),
        ("formal_audit", audit_root),
    ):
        files = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            include = (
                path.suffix == ".pt"
                or "raw" in path.relative_to(root).parts
                or "paired_episode_metrics" in path.name
            )
            if include:
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
        groups[label] = {"file_count": len(files), "files": files}
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "logical_roots": groups,
        "absolute_host_paths_not_published": True,
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _readme(protocol: dict[str, Any], formal: dict[str, Any]) -> str:
    summary = formal["three_context_summary"]
    means = summary["method_means"]
    highest = summary["highest_mean_CBF_on_success_method"]
    rows = []
    for method in METHODS:
        values = means[method]
        rows.append(
            f"| {method} | {_percent(values['CBF_on_success'])} | "
            f"{_percent(values['CBF_off_success'])} | {_percent(values['D0_success'])} | "
            f"{values['CBF_on_interventions_per_riser']:.3f} |"
        )
    direction = summary["direction_consistency"]
    return f"""# v32 Long-Horizon CBF-Protected Continual Refinement

## 中文摘要

v32 保持 v31 A2 的 Actor/Critic、CBF、reward、raw-action PPO、soft moving-KL 与 CBF-guided correction 不变。三个 per-context 策略从各自 v31 A2 round 8 继续到固定 round 24，并比较恒定学习率与后期衰减；另一个 mixed 策略从共同 base 开始，用每轮轮换的 22/21/21 个 F1/F2/F3 环境训练到固定 round 24。没有 candidate、性能/KL gate、best checkpoint 或结果依赖重跑。

| method | mean CBF-on success | mean CBF-off success | D0 success | interventions/riser |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

平均 CBF-on success 最高的是 **{highest}**（{_percent(means[highest]['CBF_on_success'])}）。Long-Constant、Long-Decay、Mixed 相对 v31 A2 round 8 的平均变化分别为 {direction['LongConstant']['mean_delta']:+.4f}、{direction['LongDecay']['mean_delta']:+.4f}、{direction['Mixed']['mean_delta']:+.4f}；正向 context 数分别为 {direction['LongConstant']['positive_contexts']}/3、{direction['LongDecay']['positive_contexts']}/3、{direction['Mixed']['positive_contexts']}/3。

本结果只回答长期仿真在线交互是否继续提高成功率、是否方向一致、D0 是否保持以及 CBF 依赖是否变化。所有 paired 95% CI 都是描述性结果，不是 gate。真机仍应保留 base+CBF 安全基线。

## English summary

v32 leaves the v31 A2 networks, CBF, reward, raw-action PPO, soft moving-KL, and corrective objective unchanged. Six per-context continuations compare constant and decayed learning rates from v31 round 8 through the unconditional round-24 policy. One base-initialized mixed policy trains for 24 rounds with an exactly balanced rotating F1/F2/F3 allocation. No candidate search, performance/KL gate, best-checkpoint selection, or outcome-dependent rerun is used.

The highest three-context mean CBF-on success belongs to **{highest}** at {_percent(means[highest]['CBF_on_success'])}. Context-wise changes, D0 retention, CBF-off behavior, intervention rate, correction norm, would-intervene fraction, nominal violations, returns, falls, and reached-riser metrics are published in the formal JSON files.

Protocol source commit: `{protocol['source_boundary']['git_commit']}`. `SHA256SUMS` binds every compact publication file; `external_artifacts_manifest.json` binds the external checkpoints and raw telemetry.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--monitor-dir", type=Path, required=True)
    parser.add_argument("--formal-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    preflight_dir = args.preflight_dir.resolve()
    training_root = args.training_root.resolve()
    monitor_dir = args.monitor_dir.resolve()
    formal_dir = args.formal_audit_dir.resolve()
    output = args.output_dir.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v32 packaging requires a clean committed worktree")
    protocol = _json(protocol_path)
    preflight = _json(preflight_dir / "preflight_summary.json")
    monitor = _json(monitor_dir / "monitor_summary.json")
    formal = _json(formal_dir / "combined_results.json")
    if protocol.get("status") != "frozen_before_v32_preflight_and_formal":
        raise RuntimeError("v32 package requires frozen protocol")
    if (
        not preflight.get("passed")
        or not monitor.get("complete")
        or not formal.get("complete")
    ):
        raise RuntimeError("v32 package input is incomplete")
    if any(
        item.get("protocol_id") != PROTOCOL_ID
        for item in (protocol, preflight, monitor, formal)
    ):
        raise RuntimeError("v32 package protocol ids differ")
    commit = _git(repo, "rev-parse", "HEAD")
    prior = protocol["prior_results_immutable"]["git_trees"]
    actual = {
        version: _git(repo, "rev-parse", f"{commit}:results/online/proximal_{version}")
        for version in prior
    }
    if actual != prior:
        raise RuntimeError("v25-v31 result trees changed after v32 freeze")
    committed_protocol = subprocess.run(
        ["git", "show", f"HEAD:{protocol_path.relative_to(repo)}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    if committed_protocol != protocol_path.read_bytes():
        raise RuntimeError("v32 protocol is not committed")
    existing = (
        [] if not output.exists() else sorted(path.name for path in output.iterdir())
    )
    if existing not in ([], ["protocol.json"]):
        raise RuntimeError(f"v32 package output is not empty: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    if protocol_path != output / "protocol.json":
        _copy(protocol_path, output / "protocol.json")
    _copy(preflight_dir / "preflight_summary.json", output / "preflight_summary.json")
    _copy(monitor_dir / "monitor_results.csv", output / "monitor/monitor_results.csv")
    _copy(monitor_dir / "monitor_summary.json", output / "monitor/monitor_summary.json")
    for context in FORMAL_CONTEXTS:
        for name in ("context_results.json", "condition_results.csv"):
            _copy(formal_dir / context / name, output / "formal" / context / name)
    for name in (
        "combined_results.json",
        "combined_results.csv",
        "mixed_D0_results.json",
    ):
        _copy(formal_dir / name, output / "formal" / name)

    training_rows = []
    for context in FORMAL_CONTEXTS:
        for schedule in CONTINUATION_SCHEDULES:
            rows = _read_csv(
                training_root
                / "continuation"
                / context
                / schedule
                / "round_metrics.csv"
            )
            if len(rows) != 16:
                raise RuntimeError(
                    f"v32 {context}/{schedule} lacks 16 continuation rounds"
                )
            training_rows.extend(
                {
                    "kind": "continuation",
                    "context": context,
                    "schedule": schedule,
                    **row,
                }
                for row in rows
            )
    mixed_training = _read_csv(
        training_root / "mixed" / "LongDecay" / "round_metrics.csv"
    )
    if len(mixed_training) != 24:
        raise RuntimeError("v32 mixed run lacks 24 rounds")
    training_rows.extend(
        {"kind": "mixed", "context": "mixed", "schedule": "LongDecay", **row}
        for row in mixed_training
    )
    _write_union_csv(output / "training/round_metrics.csv", training_rows)

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    continuation_monitor = _read_csv(monitor_dir / "continuation_monitor.csv")
    mixed_monitor = _read_csv(monitor_dir / "mixed_monitor.csv")
    _success_figure(formal, figures / "success_by_context")
    _delta_figure(formal, figures / "success_delta_vs_v31")
    _continuation_monitor_figure(
        continuation_monitor, figures / "continuation_saturation"
    )
    _mixed_monitor_figure(mixed_monitor, figures / "mixed_fixed_nodes")
    _retention_dependence_figure(formal, figures / "D0_and_CBF_dependence")
    (output / "README.md").write_text(_readme(protocol, formal))
    (output / "external_artifacts_manifest.json").write_text(
        json.dumps(
            _external_manifest(training_root, monitor_dir, formal_dir),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    checksum = output / "SHA256SUMS"
    files = sorted(
        path.relative_to(output)
        for path in output.rglob("*")
        if path.is_file() and path != checksum
    )
    checksum.write_text(
        "".join(
            f"{file_sha256(output / relative)}  {relative.as_posix()}\n"
            for relative in files
        )
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "files": len(files) + 1,
                "training_round_rows": len(training_rows),
                "highest_mean_CBF_on_success_method": formal["three_context_summary"][
                    "highest_mean_CBF_on_success_method"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
