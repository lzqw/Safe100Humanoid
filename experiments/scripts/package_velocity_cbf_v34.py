"""Build the aggregate-only bilingual v34 Git evidence package and figures."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from proximal_v23_io import file_sha256
from velocity_cbf_v34_protocol import (
    CURRENT_CBF_MODE,
    FORMAL_CONTEXTS,
    PARAMETER_RANGES,
    PROTOCOL_ID,
)


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _copy(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("v34 package CSV cannot be empty")
    fields = sorted({field for row in rows for field in row})
    leading = [field for field in ("candidate", "context", "round") if field in fields]
    fields = leading + [field for field in fields if field not in leading]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _search_figure(
    stage1: list[dict[str, str]],
    stage2: list[dict[str, str]],
    selected: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    completed = [row for row in stage1 if row["status"] == "completed"]
    axes[0].scatter(
        [int(row["candidate_index"]) for row in completed],
        [float(row["mean_success"]) for row in completed],
        s=24,
        alpha=0.8,
    )
    current = next(row for row in completed if row["candidate"] == "c000_current")
    axes[0].axhline(
        float(current["mean_success"]),
        color="#d62728",
        linestyle="--",
        label="current CBF",
    )
    axes[0].set(
        title="Stage 1: 60 candidates × 64/context",
        xlabel="candidate index",
        ylabel="mean F1/F2/F3 success",
        ylim=(0, 1),
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    ordered = sorted(stage2, key=lambda row: int(row["rank"]) if row["rank"] else 999)
    colors = [
        "#d62728" if row["candidate"] == selected else "#4c78a8" for row in ordered
    ]
    axes[1].bar(
        np.arange(len(ordered)),
        [float(row["mean_success"]) for row in ordered],
        color=colors,
    )
    axes[1].set(
        title="Stage 2: top 8 × 256/context",
        ylabel="mean F1/F2/F3 success",
        xticks=np.arange(len(ordered)),
        xticklabels=[row["candidate"] for row in ordered],
        ylim=(0, 1),
    )
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    _save(fig, path)


def _success_by_context(final: dict[str, Any], selected_mode: str, path: Path) -> None:
    conditions = (
        "v31_A2_current",
        "v31_A2_optimized",
        "trained_A2_optimized",
    )
    labels = (
        (
            "v31 A2 + current",
            "v31 A2 + selected (current repeat)",
            "new A2 + selected",
        )
        if selected_mode == CURRENT_CBF_MODE
        else ("v31 A2 + current", "v31 A2 + optimized", "new A2 + optimized")
    )
    x = np.arange(len(FORMAL_CONTEXTS))
    width = 0.25
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    for index, (condition, label) in enumerate(zip(conditions, labels, strict=True)):
        values = [
            final["contexts"][context]["target"]["conditions"][condition][
                "success_rate"
            ]
            for context in FORMAL_CONTEXTS
        ]
        axis.bar(x + (index - 1) * width, values, width, label=label)
    axis.set(
        title="Held-out CBF-on success by high-step context",
        ylabel="success rate",
        xticks=x,
        xticklabels=FORMAL_CONTEXTS,
        ylim=(0, 1),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    _save(fig, path)


def _current_optimized(final: dict[str, Any], selected_mode: str, path: Path) -> None:
    means = final["three_context_target_means"]
    conditions = (
        "base_current",
        "v31_A2_current",
        "v31_A2_optimized",
        "trained_A2_optimized",
    )
    labels = (
        (
            "base\ncurrent",
            "v31 A2\ncurrent",
            "v31 A2\nselected repeat",
            "new A2\nselected current",
        )
        if selected_mode == CURRENT_CBF_MODE
        else (
            "base\ncurrent",
            "v31 A2\ncurrent",
            "v31 A2\noptimized",
            "new A2\noptimized",
        )
    )
    values = [means[condition]["success_rate"] for condition in conditions]
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    bars = axis.bar(
        np.arange(len(values)),
        values,
        color=("#9c9c9c", "#4c78a8", "#f2cf5b", "#e45756"),
    )
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{100 * value:.1f}%",
            ha="center",
        )
    axis.set(
        title="Current versus outcome-optimized velocity CBF",
        ylabel="three-context mean success",
        xticks=np.arange(len(values)),
        xticklabels=labels,
        ylim=(0, 1),
    )
    axis.grid(axis="y", alpha=0.25)
    _save(fig, path)


def _safety_metrics(final: dict[str, Any], path: Path) -> None:
    means = final["three_context_target_means"]
    conditions = ("v31_A2_current", "v31_A2_optimized", "trained_A2_optimized")
    labels = ("v31 current", "v31 optimized", "new optimized")
    fields = (
        ("fall_rate", "fall rate"),
        ("intervention_steps_per_riser", "intervention steps / riser"),
        ("mean_velocity_correction_norm", "velocity correction norm"),
        ("mean_velocity_correction_jerk", "correction jerk"),
        ("mean_toe_riser_contact_impulse", "toe-riser impulse"),
        ("unsafe_overlap_steps_per_riser", "unsafe overlap / riser"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    for axis, (field, title) in zip(axes.flat, fields, strict=True):
        axis.bar(np.arange(3), [means[condition][field] for condition in conditions])
        axis.set(title=title, xticks=np.arange(3), xticklabels=labels)
        axis.tick_params(axis="x", rotation=20, labelsize=8)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Held-out auxiliary safety and correction metrics")
    _save(fig, path)


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _readme(
    *,
    config: dict[str, Any],
    selected: dict[str, Any],
    smoke: dict[str, Any],
    final: dict[str, Any],
    runtime: dict[str, float],
) -> str:
    main = final["main_table"]
    headline = final["headline"]
    target = final["three_context_target_means"]
    d0 = final["three_source_context_D0_means"]
    parameters = ", ".join(
        f"{name}={float(selected['parameters'][name]):.6g}" for name in PARAMETER_RANGES
    )
    table = "\n".join(
        f"| {row['method']} | {_percent(row['F1'])} | {_percent(row['F2'])} | "
        f"{_percent(row['F3'])} | {_percent(row['mean_success'])} |"
        for row in main
    )
    d0_rows = "\n".join(
        f"| {label} | {_percent(d0[key]['success_rate'])} |"
        for label, key in (
            ("v31 A2 + current CBF", "v31_A2_current"),
            ("v31 A2 + optimized CBF", "v31_A2_optimized"),
            ("new A2 + optimized CBF", "trained_A2_optimized"),
            ("v31 A2 CBF-off", "v31_A2_off"),
            ("new A2 CBF-off", "trained_A2_off"),
        )
    )
    achieved = "是" if headline["development_target_met"] else "否"
    achieved_en = "yes" if headline["development_target_met"] else "no"
    selected_is_control = selected.get("mode") == CURRENT_CBF_MODE
    direct_zh = (
        f"自动选择退化为 current control，因此表中的 v31 A2 + selected 条件不是新 CBF；"
        f"其独立重复成功率为 {_percent(headline['direct_optimized_mean_success'])}，"
        f"与基线相差 {headline['direct_minus_v31_current_pp']:+.2f} pp，只能视为 GPU 仿真运行波动，"
        "不能解释为 CBF 改进。"
        if selected_is_control
        else f"直接替换 optimized CBF 的平均成功率为 "
        f"{_percent(headline['direct_optimized_mean_success'])}（相对基线 "
        f"{headline['direct_minus_v31_current_pp']:+.2f} pp）。"
    )
    direct_en = (
        f"The outcome-only selector fell back to the current control. The v31 A2 + "
        f"selected row is therefore an independent repeat of the same method "
        f"({_percent(headline['direct_optimized_mean_success'])}, "
        f"{headline['direct_minus_v31_current_pp']:+.2f} pp versus the baseline run), "
        "which is run-to-run GPU simulation variation rather than a CBF gain."
        if selected_is_control
        else f"Direct replacement reaches "
        f"{_percent(headline['direct_optimized_mean_success'])} "
        f"({headline['direct_minus_v31_current_pp']:+.2f} pp versus baseline)."
    )
    return f"""# v34 Outcome-Optimized Task-Metric Velocity CBF

## 中文摘要

v33 的伪加速度 HOCBF 与现有“关节位置目标 → 速度级安全投影”接口不匹配：直接替换和重新训练都降低成功率。因此 v34 保留有效的 sloped toe-clearance 速度约束，只把欧氏投影改成纯 Torch、单约束闭式 task-metric 投影。安全 nominal target 逐位原样通过；没有 CPU QP、HOCBF、MPC、控制器切换或新训练 gate。

自动开发严格按冻结协议完成：60 个 candidate 的 64 episodes/context 初筛、前 8 个的 256 episodes/context 精评、前 2 个各自在 F1/F2/F3 从共同 base 完成固定 8 轮 v31 A2（共 6 次训练），再按训练后 development 平均成功率选择统一参数。最终选择 **{selected["candidate"]}**：`{parameters}`。

| 方法 | F1 success | F2 success | F3 success | Mean success |
|---|---:|---:|---:|---:|
{table}

最终 held-out 主结果：new A2 + selected CBF 为 **{_percent(headline["trained_optimized_mean_success"])}**，v31 A2 + current CBF 基线为 **{_percent(headline["v31_current_mean_success"])}**，差值 **{headline["trained_minus_v31_current_pp"]:+.2f} pp**。达到预设 +3 pp 开发目标：**{achieved}**。{direct_zh} 无论正负，以上均为参数冻结后只运行一次的最终测试结果。

| D0 条件（F1/F2/F3 来源策略平均） | success |
|---|---:|
{d0_rows}

辅助 fall/return/riser/time、intervention、correction norm/jerk、toe impulse、unsafe overlap、support-foot slip、post-intervention fall、CBF-off 和逐 context paired effect 均在 `final/combined_results.json`。单次 8×256 smoke 通过：optimized/current 吞吐比 `{smoke["throughput_ratio"]:.3f}`；最终 optimized 条件平均 CBF action 计算时间 `{target["trained_A2_optimized"]["mean_cbf_compute_time_ms"]:.4f} ms/step`。记录到的开发、训练、选择和最终审计总 wall time 约 `{runtime["total_hours"]:.2f} h`。

真机部署仍使用当前关节位置目标接口，投影在 GPU 端向量化执行。实际控制器应复核控制周期预算、关节速度/位置限制、传感延迟、接触检测和仿真到实机的动力学偏差。

## English summary

v33's pseudo-acceleration HOCBF was mismatched to the existing joint-position-target interface and reduced success both as a direct replacement and after retraining. v34 returns to the validated velocity-level sloped toe-clearance constraint and changes only its projection metric. The implementation is a pure-Torch, vectorized, closed-form single-constraint solve; an already-safe nominal target is returned exactly.

The frozen automated search evaluated 60 candidates, refined the top 8, trained the top 2 in all three contexts with the unchanged eight-round v31 A2 procedure, and selected one global parameter set solely by trained development success. Final identities were created only after that selection was committed, and the held-out audit ran once.

New A2 + selected CBF reaches **{_percent(headline["trained_optimized_mean_success"])}** mean held-out CBF-on success versus **{_percent(headline["v31_current_mean_success"])}** for v31 A2 + current CBF, a **{headline["trained_minus_v31_current_pp"]:+.2f} pp** change. The +3 pp development target was met: **{achieved_en}**. {direct_en} See the aggregate JSON for all paired target/D0, CBF-off, safety, correction, and compute-time metrics.

Source boundary: `{config["source_boundary"]["git_commit"]}`. `SHA256SUMS` binds every published aggregate and figure; raw episode traces and checkpoints remain in the external artifact directory.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    config = _json(args.search_config.resolve())
    selected = _json(args.selected.resolve())
    smoke = _json(args.smoke.resolve())
    final = _json(args.final_root.resolve() / "combined_results.json")
    if any(
        payload.get("protocol_id") != PROTOCOL_ID
        for payload in (config, selected, smoke, final)
    ):
        raise RuntimeError("v34 package protocol id differs")
    output.mkdir(parents=True, exist_ok=True)
    _copy(args.search_config.resolve(), output / "search_config.json")
    _copy(args.selected.resolve(), output / "search" / "selected_cbf.json")
    _copy(
        args.search_root.resolve() / "all_candidates.csv",
        output / "search" / "all_candidates.csv",
    )
    _copy(
        args.search_root.resolve() / "top8_results.csv",
        output / "search" / "top8_results.csv",
    )
    _copy(args.smoke.resolve(), output / "smoke_summary.json")
    _copy(
        args.selection_root.resolve() / "trained_top2_results.csv",
        output / "search" / "trained_top2_results.csv",
    )
    training_rows: list[dict[str, Any]] = []
    training_seconds = 0.0
    top2 = _json(args.search_root.resolve() / "top2_candidates.json")
    for candidate in top2["top2"]:
        for context in FORMAL_CONTEXTS:
            directory = args.training_root.resolve() / candidate["candidate"] / context
            rows = _read_csv(directory / "round_metrics.csv")
            if len(rows) != 8:
                raise RuntimeError(
                    f"v34 training lacks eight rounds for {candidate['candidate']}/{context}"
                )
            training_rows.extend(
                {"candidate": candidate["candidate"], "context": context, **row}
                for row in rows
            )
            training_seconds += float(
                _json(directory / "training_summary.json")["elapsed_seconds"]
            )
    _write_csv(output / "training" / "round_metrics.csv", training_rows)
    for context in FORMAL_CONTEXTS:
        _copy(
            args.final_root.resolve() / context / "target_results.json",
            output / "final" / context / "target_results.json",
        )
        _copy(
            args.final_root.resolve() / context / "D0_results.json",
            output / "final" / context / "D0_results.json",
        )
    _copy(
        args.final_root.resolve() / "combined_results.json",
        output / "final" / "combined_results.json",
    )
    stage1 = _read_csv(output / "search" / "all_candidates.csv")
    stage2 = _read_csv(output / "search" / "top8_results.csv")
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _search_figure(
        stage1, stage2, selected["candidate"], figures / "search_progress.png"
    )
    _success_by_context(
        final, str(selected["mode"]), figures / "success_by_context.png"
    )
    _current_optimized(
        final, str(selected["mode"]), figures / "current_vs_optimized_cbf.png"
    )
    _safety_metrics(final, figures / "safety_metrics.png")
    selection = _json(args.selection_root.resolve() / "development_selection.json")
    completed = _json(args.final_root.resolve() / "execution_completed.json")
    started = _json(args.final_root.resolve() / "execution_started.json")
    runtime = {
        "search_seconds": float(top2["search_elapsed_seconds"]),
        "training_seconds": training_seconds,
        "trained_selection_seconds": float(selection["elapsed_seconds"]),
        "final_seconds": float(completed["completed_unix_seconds"])
        - float(started["started_unix_seconds"]),
    }
    runtime["total_seconds"] = sum(runtime.values())
    runtime["total_hours"] = runtime["total_seconds"] / 3600.0
    _atomic_runtime = output / "runtime_summary.json"
    _atomic_runtime.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text(
        _readme(
            config=config,
            selected=selected,
            smoke=smoke,
            final=final,
            runtime=runtime,
        )
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
                "protocol_id": PROTOCOL_ID,
                "output": str(output),
                "published_files": len(files) + 1,
                "selected": selected["candidate"],
                "headline": final["headline"],
                "runtime": runtime,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
