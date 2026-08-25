"""Build v31's aggregate-only bilingual evidence package and five figures."""

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
from cbf_teacher_v31_protocol import FORMAL_CONTEXTS, METHOD_ARMS, PROTOCOL_ID
from proximal_v23_io import file_sha256


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
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_union_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("v31 package cannot write an empty CSV")
    leading = [
        field
        for field in ("phase", "context", "method", "arm", "round")
        if any(field in row for row in rows)
    ]
    fields = leading + sorted({key for row in rows for key in row} - set(leading))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=180)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _success_figure(formal: dict[str, Any], base: Path) -> None:
    x = np.arange(len(FORMAL_CONTEXTS))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9, 4.5))
    for index, arm in enumerate(METHOD_ARMS):
        values = [
            formal["contexts"][context]["target"][f"{arm}_on"]["success_rate"]
            for context in FORMAL_CONTEXTS
        ]
        axis.bar(x + (index - 1) * width, values, width, label=arm)
    axis.set(
        title="v31 CBF-on success by deployment context",
        ylabel="success rate",
        xticks=x,
        xticklabels=FORMAL_CONTEXTS,
        ylim=(0.0, 1.0),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, base)


def _teacher_control_figure(formal: dict[str, Any], base: Path) -> None:
    x = np.arange(len(FORMAL_CONTEXTS))
    width = 0.19
    series = []
    for arm in ("A1", "A2"):
        for mode in ("on", "off"):
            series.append(
                (
                    f"{arm}−A0 {mode}",
                    [
                        formal["contexts"][context]["target"][f"{arm}_{mode}"][
                            "success_rate"
                        ]
                        - formal["contexts"][context]["target"][f"A0_{mode}"][
                            "success_rate"
                        ]
                        for context in FORMAL_CONTEXTS
                    ],
                )
            )
    fig, axis = plt.subplots(figsize=(10, 4.6))
    for index, (label, values) in enumerate(series):
        axis.bar(x + (index - 1.5) * width, values, width, label=label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        title="Teacher success difference versus A0",
        ylabel="success-rate difference",
        xticks=x,
        xticklabels=FORMAL_CONTEXTS,
    )
    axis.legend(ncols=2, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, base)


def _internalization_figure(formal: dict[str, Any], base: Path) -> None:
    x = np.arange(len(FORMAL_CONTEXTS))
    fields = (
        ("off_success_vs_base", "off success", 1.0),
        ("off_would_intervene_vs_base", "would intervene", -1.0),
        ("off_counterfactual_correction_vs_base", "correction norm", -1.0),
        ("off_nominal_violation_vs_base", "barrier violations", -1.0),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    width = 0.36
    for axis, (field, title, direction) in zip(axes.flat, fields, strict=True):
        for index, arm in enumerate(("A1", "A2")):
            values = [
                formal["contexts"][context]["effects"]["nominal_internalization"][arm][
                    field
                ]
                * direction
                for context in FORMAL_CONTEXTS
            ]
            axis.bar(x + (index - 0.5) * width, values, width, label=arm)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(title=f"improvement: {title}", xticks=x, xticklabels=FORMAL_CONTEXTS)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Nominal-policy internalization versus base (positive is better)")
    _save_figure(fig, base)


def _risk_figure(formal: dict[str, Any], base: Path) -> None:
    summary = formal["three_context_summary"]["method_means"]
    x = np.arange(len(METHOD_ARMS))
    definitions = (
        ("CBF_on_interventions_per_riser", "CBF-on interventions / riser"),
        ("CBF_off_would_intervene_fraction", "CBF-off would-intervene fraction"),
        ("CBF_off_kick_events_per_riser", "CBF-off kick events / riser"),
        ("CBF_off_unsafe_overlap_steps_per_riser", "CBF-off overlap / riser"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4))
    for axis, (field, title) in zip(axes.flat, definitions, strict=True):
        axis.bar(x, [summary[arm][field] for arm in METHOD_ARMS])
        axis.set(title=title, xticks=x, xticklabels=METHOD_ARMS)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Three-context mean CBF dependence and toe-riser risk")
    _save_figure(fig, base)


def _float(value: str | None) -> float:
    return float("nan") if value in (None, "", "None") else float(value)


def _learning_curve_figure(rows: list[dict[str, str]], base: Path) -> None:
    fields = (
        ("CBF_on_success", "CBF-on success"),
        ("CBF_off_success", "CBF-off success"),
        ("CBF_on_intervention_steps_per_riser", "interventions/riser"),
        ("training_policy_target_distance_after", "teacher target distance"),
        ("moving_KL", "moving KL"),
    )
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    colors = {"A0": "#718096", "A1": "#805ad5", "A2": "#2b6cb0"}
    for arm in METHOD_ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        rounds = [int(row["round"]) for row in subset]
        for axis, (field, title) in zip(axes, fields, strict=True):
            axis.plot(
                rounds,
                [_float(row[field]) for row in subset],
                marker="o",
                label=arm,
                color=colors[arm],
            )
            axis.set(title=title, xlabel="round", xticks=rounds)
            axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("F1 read-only checkpoint learning curve")
    _save_figure(fig, base)


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _interpretation(summary: dict[str, Any]) -> dict[str, Any]:
    means = summary["method_means"]
    highest = summary["highest_mean_CBF_on_success_method"]
    internalization = {
        arm: means[arm]["CBF_off_success"] - means["A0"]["CBF_off_success"]
        for arm in ("A1", "A2")
    }
    dependence = {
        arm: means[arm]["CBF_on_interventions_per_riser"]
        - means["A0"]["CBF_on_interventions_per_riser"]
        for arm in ("A1", "A2")
    }
    d0 = summary["D0_success_change_vs_base"]
    return {
        "highest": highest,
        "internalization": internalization,
        "dependence": dependence,
        "D0": d0,
        "real_stair_recommendation": (
            f"Use {highest} as the adapted candidate only if its shielded gain, D0 "
            "retention, and risk profile are acceptable beside the base+CBF baseline."
        ),
    }


def _readme(protocol: dict[str, Any], formal: dict[str, Any]) -> str:
    summary = formal["three_context_summary"]
    means = summary["method_means"]
    interpretation = _interpretation(summary)
    context_lines = []
    for context in FORMAL_CONTEXTS:
        target = formal["contexts"][context]["target"]
        d0 = formal["contexts"][context]["D0"]
        context_lines.append(
            "| "
            + context
            + " | "
            + " | ".join(
                f"{_percent(target[f'{arm}_on']['success_rate'])} / "
                f"{_percent(target[f'{arm}_off']['success_rate'])} / "
                f"{_percent(d0[f'{arm}_on']['success_rate'])}"
                for arm in METHOD_ARMS
            )
            + " |"
        )
    mean_lines = [
        f"| {arm} | {_percent(means[arm]['CBF_on_success'])} | "
        f"{_percent(means[arm]['CBF_off_success'])} | "
        f"{_percent(means[arm]['D0_success'])} | "
        f"{means[arm]['CBF_on_interventions_per_riser']:.3f} | "
        f"{means[arm]['CBF_off_kick_events_per_riser']:.3f} |"
        for arm in METHOD_ARMS
    ]
    highest = interpretation["highest"]
    return f"""# v31 CBF-Teacher Formal Matrix

## 中文摘要

v30 因 F2 behavior-log-prob float32 reduction 容差过紧以及 F3 固定 10-slot terrain patch allocation 而未完成；其结果保持原样且未补写。v31 是独立冻结的新实验：将 log-prob 容差预先改为 `1e-3`（raw action、Gaussian 参数与 safe-action 路由检查仍保持严格），并按 `num_risers + 1` 动态分配 stair-target patches，使 F3 的 11 个 riser 正确包含 12 个 target patches。

三个方法固定为 A0（PPO + moving KL，无 teacher）、A1（完整 safe-action、50-step local-success gate、Gaussian NLL、weight 0.1）和 A2（residual `eta=0.25`、全部 intervention、weighted Smooth-L1、weight 1.0）。每个 context/method 只适配一次，固定使用 round 8；KL、训练回报和最终评价均未用于停止或选择 checkpoint。

| context | A0 on / off / D0 | A1 on / off / D0 | A2 on / off / D0 |
|---|---:|---:|---:|
{chr(10).join(context_lines)}

| method | mean CBF-on success | mean CBF-off success | mean D0 success | interventions/riser | off kick events/riser |
|---|---:|---:|---:|---:|---:|
{chr(10).join(mean_lines)}

三场景平均 shielded success 最高的方法是 **{highest}**（{_percent(means[highest]['CBF_on_success'])}）。A1/A2 相对 A0 的平均 CBF-on success 差异分别为 {summary['teacher_minus_A0_means']['A1']['CBF_on_success']:+.4f} 和 {summary['teacher_minus_A0_means']['A2']['CBF_on_success']:+.4f}；CBF-off 差异分别为 {interpretation['internalization']['A1']:+.4f} 和 {interpretation['internalization']['A2']:+.4f}。相对 A0 的 interventions/riser 差异分别为 {interpretation['dependence']['A1']:+.4f} 和 {interpretation['dependence']['A2']:+.4f}。D0 相对 base 的变化为 A0 {interpretation['D0']['A0']:+.4f}、A1 {interpretation['D0']['A1']:+.4f}、A2 {interpretation['D0']['A2']:+.4f}。

实机建议：{interpretation['real_stair_recommendation']} 真机仍应以 base+CBF 为安全基线，并同时审查 nominal would-intervene、correction norm、toe-riser risk 和 D0 retention，而不是只看单一 success 指标。

## English summary

v31 is a separately frozen rerun; v30 remains an immutable incomplete result. It prospectively raises only the float32 behavior-log-probability reduction tolerance to `1e-3`, while retaining strict raw-action, Gaussian-parameter, and safe-action routing audits. It also allocates stair patches dynamically, including all 12 target patches for F3's 11-riser profile.

All nine A0/A1/A2 × F1/F2/F3 adaptations use their predeclared seeds and unconditional round-8 policies. The highest mean CBF-on success belongs to **{highest}** at {_percent(means[highest]['CBF_on_success'])}. Paired 95% bootstrap intervals, repairs/regressions, nominal internalization, CBF dependence, toe-riser risk, and D0 retention are reported in each context JSON. No result was used as a gate.

For a real-stair follow-up, keep base+CBF as the safety baseline and consider the highest-success adapted method only together with its D0 retention and risk/dependence measurements.

Protocol source commit: `{protocol['source_boundary']['git_commit']}`. Protocol SHA-256 and every published aggregate are bound by `SHA256SUMS`; external checkpoints and raw telemetry are bound by `external_artifacts_manifest.json`.
"""


def _external_manifest(
    training_root: Path, formal_audit: Path, monitor: Path
) -> dict[str, Any]:
    roots = {
        "training": training_root,
        "formal_audit": formal_audit,
        "monitor": monitor,
    }
    groups = {}
    for label, root in roots.items():
        files = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            include = (
                path.suffix == ".pt"
                or "raw" in path.relative_to(root).parts
                or path.name == "paired_episode_metrics.csv"
            )
            if include:
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
        groups[label] = {"files": files, "file_count": len(files)}
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "logical_roots": groups,
        "absolute_host_paths_not_published": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--formal-training-root", type=Path, required=True)
    parser.add_argument("--formal-audit-dir", type=Path, required=True)
    parser.add_argument("--monitor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    preflight_dir = args.preflight_dir.resolve()
    training_root = args.formal_training_root.resolve()
    formal_dir = args.formal_audit_dir.resolve()
    monitor_dir = args.monitor_dir.resolve()
    output = args.output_dir.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v31 packaging requires a clean committed worktree")
    protocol = _json(protocol_path)
    preflight = _json(preflight_dir / "preflight_summary.json")
    formal = _json(formal_dir / "combined_results.json")
    monitor = _json(monitor_dir / "monitor_summary.json")
    if protocol.get("status") != "frozen_before_v31_preflight_and_formal":
        raise RuntimeError("v31 package requires the frozen protocol")
    if (
        not preflight.get("passed")
        or not formal.get("complete")
        or not monitor.get("complete")
    ):
        raise RuntimeError("v31 package input is incomplete")
    if any(
        payload.get("protocol_id") != PROTOCOL_ID
        for payload in (protocol, preflight, formal, monitor)
    ):
        raise RuntimeError("v31 package input protocol id differs")
    commit = _git(repo, "rev-parse", "HEAD")
    actual_trees = {
        version: _git(repo, "rev-parse", f"{commit}:results/online/proximal_{version}")
        for version in protocol["prior_results_immutable"]["git_trees"]
    }
    if actual_trees != protocol["prior_results_immutable"]["git_trees"]:
        raise RuntimeError("v25-v30 result trees changed after v31 freeze")
    committed_protocol = subprocess.run(
        ["git", "show", f"HEAD:{protocol_path.relative_to(repo)}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    if committed_protocol != protocol_path.read_bytes():
        raise RuntimeError("v31 protocol is not committed")
    existing = [] if not output.exists() else [path.name for path in output.iterdir()]
    if sorted(existing) not in ([], ["protocol.json"]):
        raise RuntimeError(f"v31 output already contains package files: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    if protocol_path != output / "protocol.json":
        _copy(protocol_path, output / "protocol.json")
    _copy(preflight_dir / "preflight_summary.json", output / "preflight_summary.json")
    for context in FORMAL_CONTEXTS:
        for name in ("context_results.json", "condition_results.csv"):
            _copy(formal_dir / context / name, output / "formal" / context / name)
    for name in ("combined_results.csv", "combined_results.json"):
        _copy(formal_dir / name, output / "formal" / name)
    _copy(
        monitor_dir / "F1_checkpoint_curve.csv",
        output / "monitor" / "F1_learning_curve.csv",
    )

    training_rows = []
    for context in FORMAL_CONTEXTS:
        for arm in METHOD_ARMS:
            rows = _read_csv(training_root / context / arm / "round_metrics.csv")
            if len(rows) != 8:
                raise RuntimeError(f"v31 formal {context}/{arm} lacks eight rounds")
            training_rows.extend(
                {
                    "phase": "formal",
                    "context": context,
                    "method": arm,
                    "arm": arm,
                    **row,
                }
                for row in rows
            )
    _write_union_csv(output / "training" / "round_metrics.csv", training_rows)

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _success_figure(formal, figures / "success_by_context")
    _teacher_control_figure(formal, figures / "teacher_vs_control")
    _internalization_figure(formal, figures / "internalization")
    _risk_figure(formal, figures / "cbf_dependence_and_risk")
    monitor_rows = _read_csv(monitor_dir / "F1_checkpoint_curve.csv")
    _learning_curve_figure(monitor_rows, figures / "F1_learning_curve")
    (output / "README.md").write_text(_readme(protocol, formal))
    (output / "external_artifacts_manifest.json").write_text(
        json.dumps(
            _external_manifest(training_root, formal_dir, monitor_dir),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    checksum = output / "SHA256SUMS"
    relative_files = sorted(
        path.relative_to(output)
        for path in output.rglob("*")
        if path.is_file() and path != checksum
    )
    checksum.write_text(
        "".join(
            f"{file_sha256(output / relative)}  {relative.as_posix()}\n"
            for relative in relative_files
        )
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "files": len(relative_files) + 1,
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
