"""Build the aggregate-only bilingual v33 Git evidence package and figures."""

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
from hocbf_v33_protocol import FORMAL_CONTEXTS, PROTOCOL_ID
from proximal_v23_io import file_sha256


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
        raise ValueError("v33 package CSV cannot be empty")
    fields = sorted({key for row in rows for key in row})
    leading = [key for key in ("context", "round") if key in fields]
    fields = leading + [key for key in fields if key not in leading]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _parameter_figure(
    rows: list[dict[str, str]], selected: dict[str, Any], path: Path
) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    x = np.asarray([float(row["A1_mean_interference"]) for row in rows])
    y = np.asarray([float(row["A2_mean_on_success"]) for row in rows])
    sizes = 25.0 + 18.0 * np.asarray([float(row["omega"]) for row in rows])
    colors = np.asarray([float(row["lambda_x"]) for row in rows])
    scatter = axis.scatter(x, y, s=sizes, c=colors, cmap="viridis", alpha=0.8)
    chosen = next(row for row in rows if row["candidate"] == selected["candidate"])
    axis.scatter(
        [float(chosen["A1_mean_interference"])],
        [float(chosen["A2_mean_on_success"])],
        marker="*",
        s=320,
        color="#d62728",
        edgecolor="black",
        label="selected",
    )
    axis.set(
        title="v33 HOCBF parameter screening",
        xlabel="A1 paired interference rate (lower is better)",
        ylabel="A2 mean HOCBF-on success",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.colorbar(scatter, ax=axis, label=r"forward weight $\lambda_x$")
    _save(fig, path)


def _success_figure(final: dict[str, Any], path: Path) -> None:
    conditions = (
        "base_current",
        "v31_A2_current",
        "v31_A2_new_hocbf",
        "trained_A2_new_hocbf",
    )
    labels = ("base + current", "v31 A2 + current", "v31 A2 + HOCBF", "new A2 + HOCBF")
    x = np.arange(len(FORMAL_CONTEXTS))
    width = 0.19
    fig, axis = plt.subplots(figsize=(10, 5.0))
    for index, (condition, label) in enumerate(zip(conditions, labels, strict=True)):
        values = [
            final["contexts"][context]["target"]["conditions"][condition][
                "success_rate"
            ]
            for context in FORMAL_CONTEXTS
        ]
        axis.bar(x + (index - 1.5) * width, values, width, label=label)
    axis.set(
        title="Current velocity CBF versus acceleration HOCBF",
        ylabel="success rate",
        xticks=x,
        xticklabels=FORMAL_CONTEXTS,
        ylim=(0, 1),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    _save(fig, path)


def _rescue_figure(a1: dict[str, Any], a2: dict[str, Any], path: Path) -> None:
    x = np.arange(len(FORMAL_CONTEXTS))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for axis, payload, policy in zip(axes, (a1, a2), ("A1", "A2"), strict=True):
        for index, mode in enumerate(("current_vs_off", "new_hocbf_vs_off")):
            rescue = [
                payload["contexts"][context]["effects"][mode]["rescue_rate"]
                for context in FORMAL_CONTEXTS
            ]
            interference = [
                -payload["contexts"][context]["effects"][mode]["interference_rate"]
                for context in FORMAL_CONTEXTS
            ]
            offset = (index - 0.5) * 0.34
            axis.bar(
                x + offset,
                rescue,
                0.32,
                label=("current rescue" if index == 0 else "HOCBF rescue"),
            )
            axis.bar(
                x + offset,
                interference,
                0.32,
                alpha=0.45,
                label=("current interference" if index == 0 else "HOCBF interference"),
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=policy, xticks=x, xticklabels=FORMAL_CONTEXTS)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("paired episode rate (+ rescue / − interference)")
    axes[1].legend(fontsize=7, ncols=2)
    fig.suptitle("Frozen-policy rescue and interference")
    _save(fig, path)


def _correction_figure(final: dict[str, Any], path: Path) -> None:
    means = final["three_context_target_means"]
    conditions = ("v31_A2_current", "v31_A2_new_hocbf", "trained_A2_new_hocbf")
    labels = ("v31 current", "v31 HOCBF", "trained HOCBF")
    fields = (
        ("mean_qddot_correction_norm", "qddot correction norm"),
        ("mean_qddot_correction_jerk", "correction jerk"),
        ("post_intervention_fall_rate", "post-intervention fall"),
        ("unsafe_overlap_steps_per_riser", "unsafe overlap / riser"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.3))
    for axis, (field, title) in zip(axes.flat, fields, strict=True):
        axis.bar(np.arange(3), [means[condition][field] for condition in conditions])
        axis.set(title=title, xticks=np.arange(3), xticklabels=labels)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Correction and post-intervention balance")
    _save(fig, path)


def _training_figure(
    rows: list[dict[str, str]], final: dict[str, Any], path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for context in FORMAL_CONTEXTS:
        subset = [row for row in rows if row["context"] == context]
        axes[0].plot(
            [int(row["round"]) for row in subset],
            [float(row["rollout_success_rate"]) for row in subset],
            marker="o",
            label=context,
        )
    axes[0].set(
        title="Eight-round online refinement", xlabel="round", ylabel="rollout success"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    means = final["three_context_target_means"]
    labels = ("v31 current", "v31 HOCBF", "trained HOCBF")
    keys = ("v31_A2_current", "v31_A2_new_hocbf", "trained_A2_new_hocbf")
    axes[1].bar(np.arange(3), [means[key]["success_rate"] for key in keys])
    axes[1].set(
        title="Final three-context mean",
        ylabel="success rate",
        xticks=np.arange(3),
        xticklabels=labels,
        ylim=(0, 1),
    )
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].grid(axis="y", alpha=0.25)
    _save(fig, path)


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _readme(
    config: dict[str, Any],
    selected: dict[str, Any],
    smoke: dict[str, Any],
    final: dict[str, Any],
) -> str:
    means = final["three_context_target_means"]
    headline = final["headline"]
    lines = []
    for context in FORMAL_CONTEXTS:
        conditions = final["contexts"][context]["target"]["conditions"]
        lines.append(
            f"| {context} | {_percent(conditions['base_current']['success_rate'])} | "
            f"{_percent(conditions['v31_A2_current']['success_rate'])} | "
            f"{_percent(conditions['v31_A2_new_hocbf']['success_rate'])} | "
            f"{_percent(conditions['trained_A2_new_hocbf']['success_rate'])} |"
        )
    source_commit = config["source_boundary"]["git_commit"]
    selected_text = f"ω={selected['omega']:g}, λx={selected['lambda_x']:g}, λs={selected['lambda_s']:g}"
    return f"""# v33 Task-Consistent Acceleration HOCBF-QP

## 中文摘要

v33 保留未修改的当前速度级 CBF（CBF0）、sloped toe-clearance 几何、405-D Actor、838-D privileged Critic 和 v31 A2 在线目标。CBF0 直接约束关节速度，容易通过改变摆动脚前向运动来满足半空间；新方法改为二阶约束

`ḧ + 2ζω ḣ + ω²h ≥ 0`，其中 `ζ=1`，并用 `D + λs I + λx JxᵀJx` 的纯 Torch Sherman–Morrison 闭式投影惩罚前向任务偏差和修正抖动。若 nominal margin 已安全，safe target 与 nominal target 精确相同。

18 个预定参数只在冻结的 v31 A1/A2 策略上开发；最终全局参数为 **{selected_text}**，没有 per-context 参数、额外搜索或训练 gate。单次 smoke 的 HOCBF/CBF0 吞吐比为 `{smoke["throughput_ratio"]:.3f}`，平均 HOCBF action 计算时间为 `{smoke["new_HOCBF"]["mean_cbf_compute_time_ms"]:.4f} ms/step`。

| context | base + current | v31 A2 + current | v31 A2 + HOCBF | newly trained A2 + HOCBF |
|---|---:|---:|---:|---:|
{chr(10).join(lines)}

三场景平均成功率：v31 A2 + current 为 **{_percent(means["v31_A2_current"]["success_rate"])}**，直接替换 HOCBF 后为 **{_percent(means["v31_A2_new_hocbf"]["success_rate"])}**（{headline["v31_A2_new_minus_current"]:+.4f}），在新 HOCBF 下从共同 base 完成固定 8 轮 A2 微调后为 **{_percent(means["trained_A2_new_hocbf"]["success_rate"])}**（相对 v31 current {headline["trained_new_minus_v31_current"]:+.4f}）。最高条件为 `{headline["highest_mean_success_condition"]}`；是否突破 72% 平台：**{headline["breaks_72_percent_plateau"]}**。

A1 current 的 on−off gap 为 {headline["A1_current_on_minus_off"]:+.4f}，新 HOCBF 的 on−off gap 为 {headline["A1_new_on_minus_off"]:+.4f}。逐 context 的 rescue/interference、post-intervention fall、correction norm/jerk、toe impulse、overlap、root roll/pitch、support slip、D0 和 paired bootstrap CI 均在 JSON/CSV 中。逐步 telemetry 与 checkpoint 只保存在外部 artifact，不提交 Git。

实机含义：该实现不引入 CPU QP solver，投影保持 GPU 向量化；控制周期还需为上表 smoke 实测计算时间、仿真到实机动力学偏差与传感延迟留出裕量。

## English summary

v33 leaves CBF0 and all v31/v32 artifacts unchanged. It replaces the velocity-level Euclidean projection only in a new mode with an acceleration HOCBF and a task-consistent weighted one-constraint QP. The pure-Torch Sherman–Morrison solve penalizes forward-foot acceleration changes and temporal correction changes, while an already-safe nominal target remains exactly unchanged.

The globally selected parameters are **{selected_text}**. Frozen-policy A1/A2 comparisons, three unconditional eight-round A2 runs from the common base, target/D0 paired audits, and at most 2,000 bootstrap samples follow the prospectively fixed protocol without outcome-dependent gates or checkpoint selection.

Mean target success changes from {_percent(means["v31_A2_current"]["success_rate"])} for v31 A2 + current CBF to {_percent(means["v31_A2_new_hocbf"]["success_rate"])} after the direct HOCBF replacement and {_percent(means["trained_A2_new_hocbf"]["success_rate"])} after new-HOCBF refinement. The measured smoke throughput ratio is {smoke["throughput_ratio"]:.3f}; inspect the aggregate files for safety, balance, D0, and compute-time tradeoffs before real-hardware use.

Source boundary: `{source_commit}`. `SHA256SUMS` binds every published aggregate and figure.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    config = _json(args.config.resolve())
    smoke = _json(args.smoke.resolve())
    selected = _json(args.development_root.resolve() / "selected_hocbf.json")
    a1 = _json(args.frozen_root.resolve() / "A1_results.json")
    a2 = _json(args.frozen_root.resolve() / "A2_results.json")
    final = _json(args.final_root.resolve() / "combined_results.json")
    if any(
        payload.get("protocol_id") != PROTOCOL_ID
        for payload in (config, smoke, selected, a1, a2, final)
    ):
        raise RuntimeError("v33 package protocol id differs")
    output.mkdir(parents=True, exist_ok=True)
    _copy(args.config.resolve(), output / "config.json")
    _copy(args.smoke.resolve(), output / "smoke_summary.json")
    development = args.development_root.resolve()
    for name in ("screening.csv", "top4_confirmation.csv", "selected_hocbf.json"):
        _copy(development / name, output / "development" / name)
    frozen = args.frozen_root.resolve()
    _copy(
        frozen / "A1_results.json", output / "frozen_policy_audit" / "A1_results.json"
    )
    _copy(
        frozen / "A2_results.json", output / "frozen_policy_audit" / "A2_results.json"
    )
    training_rows: list[dict[str, Any]] = []
    for context in FORMAL_CONTEXTS:
        rows = _read_csv(args.training_root.resolve() / context / "round_metrics.csv")
        if len(rows) != 8:
            raise RuntimeError(f"v33 {context} training lacks eight rounds")
        training_rows.extend({"context": context, **row} for row in rows)
    _write_csv(output / "training" / "round_metrics.csv", training_rows)
    final_root = args.final_root.resolve()
    for context in FORMAL_CONTEXTS:
        _copy(
            final_root / context / "target_results.json",
            output / "final" / context / "target_results.json",
        )
        _copy(
            final_root / context / "D0_results.json",
            output / "final" / context / "D0_results.json",
        )
    _copy(
        final_root / "combined_results.json", output / "final" / "combined_results.json"
    )
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    screening = _read_csv(development / "screening.csv")
    _parameter_figure(screening, selected, figures / "hocbf_parameter_screen.png")
    _success_figure(final, figures / "current_vs_hocbf_success.png")
    _rescue_figure(a1, a2, figures / "rescue_vs_interference.png")
    _correction_figure(final, figures / "correction_and_balance.png")
    _training_figure(training_rows, final, figures / "online_refinement_results.png")
    (output / "README.md").write_text(_readme(config, selected, smoke, final))
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
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
