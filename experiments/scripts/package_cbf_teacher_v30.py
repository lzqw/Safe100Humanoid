"""Build v30's aggregate-only bilingual Git evidence package and five figures."""

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
from cbf_teacher_v30_protocol import ARMS, FORMAL_CONTEXTS, PROTOCOL_ID
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
        raise ValueError("v30 package cannot write empty training metrics")
    leading = [
        name
        for name in ("phase", "context", "method", "arm")
        if any(name in row for row in rows)
    ]
    fields = leading + sorted({key for row in rows for key in row} - set(leading))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float:
    return float("nan") if value in (None, "", "None") else float(value)


def _save_figure(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=180)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def _development_figure(development: dict[str, Any], base: Path) -> None:
    arms = list(ARMS)
    rows = [development["arms"][arm]["summary"] for arm in arms]
    on = [row["target_round8_on_success"] for row in rows]
    off = [row["target_round8_off_success"] for row in rows]
    d0 = [row["D0_round8_on_success"] for row in rows]
    x = np.arange(len(arms))
    width = 0.25
    fig, axis = plt.subplots(figsize=(9, 4.4))
    axis.bar(x - width, on, width, label="target CBF on")
    axis.bar(x, off, width, label="target CBF off")
    axis.bar(x + width, d0, width, label="D0 CBF on")
    axis.set(
        title="v30 development teacher matrix (round 8)",
        ylabel="success rate",
        xticks=x,
        xticklabels=arms,
        ylim=(0.0, 1.0),
    )
    selected = development["selected_teacher"]["arm"]
    axis.get_xticklabels()[arms.index(selected)].set_weight("bold")
    axis.legend(ncols=3, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, base)


def _formal_success_figure(formal: dict[str, Any], base: Path) -> None:
    contexts = list(FORMAL_CONTEXTS)
    x = np.arange(len(contexts))
    width = 0.17
    series = (
        ("base_on", "base on"),
        ("control_on", "control on"),
        ("teacher_on", "teacher on"),
        ("teacher_off", "teacher off"),
    )
    fig, axis = plt.subplots(figsize=(9, 4.5))
    for index, (condition, label) in enumerate(series):
        values = [
            formal["contexts"][context]["target"][condition]["success_rate"]
            for context in contexts
        ]
        axis.bar(x + (index - 1.5) * width, values, width, label=label)
    axis.set(
        title="v30 formal success by deployment context",
        ylabel="success rate",
        xticks=x,
        xticklabels=contexts,
        ylim=(0.0, 1.0),
    )
    axis.legend(ncols=4, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, base)


def _internalization_figure(formal: dict[str, Any], base: Path) -> None:
    contexts = list(FORMAL_CONTEXTS)
    x = np.arange(len(contexts))
    fields = (
        ("off_success_teacher_minus_control", "off success", 1.0),
        ("off_would_intervene_teacher_minus_control", "would intervene", -1.0),
        ("off_correction_teacher_minus_control", "correction norm", -1.0),
        ("off_nominal_violation_teacher_minus_control", "barrier violations", -1.0),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for axis, (field, title, direction) in zip(axes.flat, fields, strict=True):
        values = [
            formal["contexts"][context]["effects"][
                "internalization_teacher_vs_control"
            ][field]
            * direction
            for context in contexts
        ]
        axis.bar(
            x,
            values,
            color=["#2b6cb0" if value >= 0 else "#c53030" for value in values],
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(title=f"improvement: {title}", xticks=x, xticklabels=contexts)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Teacher minus control nominal internalization (positive is better)")
    _save_figure(fig, base)


def _risk_figure(formal: dict[str, Any], base: Path) -> None:
    contexts = list(FORMAL_CONTEXTS)
    x = np.arange(len(contexts))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    definitions = (
        ("intervention_steps_per_riser", "CBF-on interventions / riser", "on"),
        ("toe_riser_kick_events_per_riser", "CBF-off kick events / riser", "off"),
        ("unsafe_overlap_steps_per_riser", "CBF-off overlap steps / riser", "off"),
    )
    for axis, (field, title, mode) in zip(axes, definitions, strict=True):
        control = [
            formal["contexts"][context]["target"][f"control_{mode}"][field]
            for context in contexts
        ]
        teacher = [
            formal["contexts"][context]["target"][f"teacher_{mode}"][field]
            for context in contexts
        ]
        axis.bar(x - width / 2, control, width, label="control")
        axis.bar(x + width / 2, teacher, width, label="teacher")
        axis.set(title=title, xticks=x, xticklabels=contexts)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("v30 CBF dependence and toe-riser risk")
    _save_figure(fig, base)


def _monitor_figure(rows: list[dict[str, str]], base: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    fields = (
        ("CBF_on_success", "CBF-on success"),
        ("CBF_off_success", "CBF-off success"),
        ("CBF_on_intervention_steps_per_riser", "interventions/riser"),
        ("policy_to_teacher_correction_gap", "correction gap"),
        ("moving_KL", "moving KL"),
    )
    for method, color in (("control", "#718096"), ("teacher", "#2b6cb0")):
        subset = [row for row in rows if row["method"] == method]
        rounds = [int(row["round"]) for row in subset]
        for axis, (field, title) in zip(axes, fields, strict=True):
            axis.plot(
                rounds,
                [_float(row[field]) for row in subset],
                marker="o",
                label=method,
                color=color,
            )
            axis.set(title=title, xlabel="round", xticks=rounds)
            axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("F1 read-only checkpoint monitor")
    _save_figure(fig, base)


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _readme(
    protocol: dict[str, Any],
    development: dict[str, Any],
    formal: dict[str, Any],
) -> str:
    selected = development["selected_teacher"]
    assessment = formal["paper_value_assessment"]
    dev_lines = []
    for arm in ARMS:
        row = development["arms"][arm]["summary"]
        dev_lines.append(
            f"| {arm} | {row['teacher_mode']} | {row['teacher_eta']} | "
            f"{row['teacher_gate']} | {_percent(row['target_round8_on_success'])} | "
            f"{_percent(row['target_round8_off_success'])} |"
        )
    formal_lines = []
    for context in FORMAL_CONTEXTS:
        item = formal["contexts"][context]
        target = item["target"]
        effects = item["effects"]["success_deltas"]
        formal_lines.append(
            f"| {context} | {_percent(target['base_on']['success_rate'])} | "
            f"{_percent(target['control_on']['success_rate'])} | "
            f"{_percent(target['teacher_on']['success_rate'])} | "
            f"{_percent(target['teacher_off']['success_rate'])} | "
            f"{effects['teacher_on_vs_control']:+.3f} | "
            f"{effects['D0_teacher_on_vs_base']:+.3f} |"
        )
    verdict_cn = "具有预先定义的论文价值" if assessment["paper_value"] else "未达到预先定义的论文价值标准"
    verdict_en = (
        "meets the predeclared paper-value criteria"
        if assessment["paper_value"]
        else "does not meet the predeclared paper-value criteria"
    )
    return f"""# v30 Overnight Paper-Grade CBF-Teacher Experiment

## 中文摘要

本实验在 v29 已验证但因 hard-KL 全部回滚的 teacher 数据流基础上，将 moving KL 仅作为软正则，完整执行六个 development arm，并在 F1/F2/F3 三个预先固定场景中比较选中 teacher 与无 teacher 的 A0。CBF 固定为 18 cm 主场景、clearance slope 0.8、recovery 0.15 m、alpha 10；每个训练 run 均无条件发布 round 8，不按 KL、性能或中间 checkpoint 选择。

Development 选中 **{selected['arm']}**：`{json.dumps(selected['configuration'], sort_keys=True)}`。正式结论：**{verdict_cn}**。Teacher 的三场景平均 CBF-on 相对 base 增益为 {assessment['mean_teacher_on_gain_over_base']:+.3f}，相对 control 为 {assessment['mean_teacher_on_gain_over_control']:+.3f}，胜出 {assessment['teacher_control_contexts_won']}/3 个场景；D0 平均变化 {assessment['mean_D0_teacher_on_gain_over_base']:+.3f}。一致改善的 nominal internalization 指标：{', '.join(assessment['consistently_improved_internalization_metrics']) or '无'}。因此决策为：`{assessment['decision']}`。

## English summary

v30 removes v29's KL acceptance gate while retaining moving KL as a soft penalty. It runs the complete six-arm development matrix, freezes one teacher formulation, and compares it with the no-teacher A0 control in three predeclared deployment contexts. Every run publishes the unconditional round-8 policy; no result-driven reruns, performance rollback, or checkpoint selection were used.

The selected formulation is **{selected['arm']}**, and the formal result **{verdict_en}**. The mean teacher CBF-on gain is {assessment['mean_teacher_on_gain_over_base']:+.3f} over base and {assessment['mean_teacher_on_gain_over_control']:+.3f} over control. Paired 95% bootstrap intervals (2,000 samples maximum) are reported in each formal context JSON for uncertainty only.

## Development arms

| arm | teacher mode | eta | gate | target on success | target off success |
|---|---|---:|---|---:|---:|
{chr(10).join(dev_lines)}

## Formal results

| context | base on | control on | teacher on | teacher off | teacher-control on | D0 teacher-base |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(formal_lines)}

The context JSON files report success, falls, return, reached riser, completion time, repairs/regressions, CBF dependence, nominal violations, kick events, unsafe overlap, and nominal barrier margins. Episode-level kick rate is appendix-only. Round 0–8 checkpoint monitoring is read-only and was performed only after formal training/auditing.

Protocol SHA-256: `{file_sha256(Path(protocol['_package_protocol_path'])) if '_package_protocol_path' in protocol else 'see SHA256SUMS'}`.
Checkpoints and per-step/per-episode telemetry remain outside Git; this directory contains aggregate CSV/JSON evidence and figures only.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--development-training-root", type=Path, required=True)
    parser.add_argument("--development-audit-dir", type=Path, required=True)
    parser.add_argument("--formal-training-root", type=Path, required=True)
    parser.add_argument("--formal-audit-dir", type=Path, required=True)
    parser.add_argument("--monitor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    output = args.output_dir.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v30 packaging requires a clean committed worktree")
    protocol = _json(protocol_path)
    smoke = _json(args.smoke_dir.resolve() / "smoke_summary.json")
    development = _json(args.development_audit_dir.resolve() / "arm_summary.json")
    formal = _json(args.formal_audit_dir.resolve() / "combined_results.json")
    monitor = _json(args.monitor_dir.resolve() / "monitor_summary.json")
    if protocol.get("status") != "frozen_before_v30_formal":
        raise RuntimeError("v30 package requires formal protocol")
    if (
        not smoke.get("passed")
        or not development.get("complete")
        or not formal.get("complete")
        or not monitor.get("complete")
    ):
        raise RuntimeError("v30 package input is incomplete")
    if any(
        payload.get("protocol_id") != PROTOCOL_ID
        for payload in (protocol, smoke, development, formal, monitor)
    ):
        raise RuntimeError("v30 package input protocol id differs")
    selected = development["selected_teacher"]["arm"]
    if selected != protocol["formal"]["selected_teacher"]["arm"]:
        raise RuntimeError("v30 package selected teacher differs")
    commit = _git(repo, "rev-parse", "HEAD")
    actual_trees = {
        version: _git(repo, "rev-parse", f"{commit}:results/online/proximal_{version}")
        for version in protocol["prior_results_immutable"]["git_trees"]
    }
    if actual_trees != protocol["prior_results_immutable"]["git_trees"]:
        raise RuntimeError("v25-v29 result trees changed after v30 freeze")
    committed_protocol = subprocess.run(
        ["git", "show", f"HEAD:{protocol_path.relative_to(repo)}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    if committed_protocol != protocol_path.read_bytes():
        raise RuntimeError("v30 formal protocol is not committed")
    existing = [] if not output.exists() else [path.name for path in output.iterdir()]
    if sorted(existing) not in ([], ["protocol.json"]):
        raise RuntimeError(f"v30 output already contains package files: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    if protocol_path != output / "protocol.json":
        _copy(protocol_path, output / "protocol.json")

    development_dir = args.development_audit_dir.resolve()
    for name in ("arm_summary.csv", "arm_summary.json", "selected_teacher.json"):
        _copy(development_dir / name, output / "development" / name)
    formal_dir = args.formal_audit_dir.resolve()
    for context in FORMAL_CONTEXTS:
        for name in ("context_results.json", "condition_results.csv"):
            _copy(formal_dir / context / name, output / "formal" / context / name)
    for name in ("combined_results.csv", "combined_results.json"):
        _copy(formal_dir / name, output / "formal" / name)
    _copy(
        args.monitor_dir.resolve() / "F1_checkpoint_curve.csv",
        output / "monitor" / "F1_checkpoint_curve.csv",
    )

    development_rounds = []
    for arm in ARMS:
        rows = _read_csv(
            args.development_training_root.resolve() / arm / "round_metrics.csv"
        )
        if len(rows) != 8:
            raise RuntimeError(f"v30 development {arm} does not have eight rounds")
        development_rounds.extend(
            {"phase": "development", "context": "DEV", "arm": arm, **row}
            for row in rows
        )
    _write_union_csv(
        output / "training" / "development_round_metrics.csv",
        development_rounds,
    )
    formal_rounds = []
    for context in FORMAL_CONTEXTS:
        for method, arm in (("control", "A0"), ("teacher", selected)):
            rows = _read_csv(
                args.formal_training_root.resolve()
                / context
                / arm
                / "round_metrics.csv"
            )
            if len(rows) != 8:
                raise RuntimeError(f"v30 formal {context}/{arm} lacks eight rounds")
            formal_rounds.extend(
                {
                    "phase": "formal",
                    "context": context,
                    "method": method,
                    "arm": arm,
                    **row,
                }
                for row in rows
            )
    _write_union_csv(output / "training" / "formal_round_metrics.csv", formal_rounds)

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _development_figure(development, figures / "development_teacher_matrix")
    _formal_success_figure(formal, figures / "formal_success_by_context")
    _internalization_figure(formal, figures / "teacher_vs_control_internalization")
    _risk_figure(formal, figures / "intervention_and_kick_reduction")
    monitor_rows = _read_csv(args.monitor_dir.resolve() / "F1_checkpoint_curve.csv")
    _monitor_figure(monitor_rows, figures / "F1_learning_curve")
    protocol_for_readme = dict(protocol)
    protocol_for_readme["_package_protocol_path"] = str(output / "protocol.json")
    (output / "README.md").write_text(_readme(protocol_for_readme, development, formal))
    checksum_path = output / "SHA256SUMS"
    relative_files = sorted(
        path.relative_to(output)
        for path in output.rglob("*")
        if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(
            f"{file_sha256(output / relative)}  {relative}\n"
            for relative in relative_files
        )
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "selected_teacher": selected,
                "paper_value": formal["paper_value_assessment"]["paper_value"],
                "files": len(relative_files) + 1,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
