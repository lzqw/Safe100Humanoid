"""Create only v29's minimal Git evidence package and three fixed figures."""

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

from cbf_teacher_v29_protocol import PROTOCOL_ID
from proximal_v23_io import file_sha256


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _number(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def _copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _training_figure(rows: list[dict[str, str]], output: Path) -> None:
    rounds = [int(row["round"]) for row in rows]
    success = [_number(row["rollout_success_rate"]) for row in rows]
    interventions = [_number(row["cbf_intervention_per_riser"]) for row in rows]
    teacher_gap = [
        _number(row["mean_weighted_policy_to_teacher_action_distance"])
        for row in rows
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    series = (
        (success, "Rollout success", "rate"),
        (interventions, "CBF interventions / riser", "count"),
        (teacher_gap, "Weighted policy-teacher gap", "action norm"),
    )
    for axis, (values, title, ylabel) in zip(axes, series, strict=True):
        axis.plot(rounds, values, marker="o", linewidth=2)
        axis.set(title=title, xlabel="round", ylabel=ylabel)
        axis.set_xticks(rounds)
        axis.grid(alpha=0.25)
    fig.suptitle("v29 fixed eight-round adaptation")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _four_condition_figure(final: dict[str, Any], output: Path) -> None:
    names = ("pi0_off", "pi0_on", "pi8_on", "pi8_off")
    target = final["conditions"]["target_18cm"]
    success = [target[name]["success_rate"] for name in names]
    fall = [target[name]["fall_rate"] for name in names]
    x = np.arange(len(names))
    width = 0.36
    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.bar(x - width / 2, success, width, label="success")
    axis.bar(x + width / 2, fall, width, label="fall")
    axis.set(
        title="18 cm paired final audit",
        ylabel="episode rate",
        xticks=x,
        xticklabels=names,
        ylim=(0.0, 1.0),
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _internalization_figure(final: dict[str, Any], output: Path) -> None:
    target = final["conditions"]["target_18cm"]
    labels = ("off kick rate", "on interventions/riser", "on policy-safe gap")
    base = (
        target["pi0_off"]["kick_rate"],
        target["pi0_on"]["intervention_per_riser"],
        target["pi0_on"]["mean_counterfactual_correction_norm"],
    )
    adapted = (
        target["pi8_off"]["kick_rate"],
        target["pi8_on"]["intervention_per_riser"],
        target["pi8_on"]["mean_counterfactual_correction_norm"],
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for axis, label, before, after in zip(
        axes, labels, base, adapted, strict=True
    ):
        axis.bar((0, 1), (before, after), color=("#718096", "#2b6cb0"))
        axis.set(title=label, xticks=(0, 1), xticklabels=("pi0", "pi8"))
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("v29 teacher internalization indicators (lower is better)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _readme(
    final: dict[str, Any], training: dict[str, Any], rows: list[dict[str, str]]
) -> str:
    target = final["conditions"]["target_18cm"]
    d0 = final["conditions"]["D0_13cm"]
    primary = final["primary_outcomes"]
    eligible = sum(_number(row["teacher_eligible_count"]) for row in rows)
    weighted = sum(_number(row["teacher_weighted_count"]) for row in rows)
    interventions = sum(_number(row["cbf_intervention_count"]) for row in rows)
    table_rows = []
    for name in ("pi0_off", "pi0_on", "pi8_on", "pi8_off"):
        item = target[name]
        table_rows.append(
            f"| {name} | {_percent(item['success_rate'])} | "
            f"{_percent(item['fall_rate'])} | {_percent(item['kick_rate'])} | "
            f"{item['intervention_per_riser']:.4f} |"
        )
    return f"""# v29 CBF-Teacher Proximal Online Fine-Tuning

## 中文摘要

固定场景为 18 cm 等高台阶、sloped-clearance slope 0.8、recovery window 0.15 m、CBF alpha 10。训练使用原始 405D actor / 838D privileged critic、运行时 CBF 始终开启，并进行固定 8 轮 raw-action PPO + moving-KL + 成功门控 CBF teacher 更新。最终策略无条件取 round 8，没有按性能选 checkpoint。

Teacher 数据量：共 {eligible:.0f} 个 eligible intervention transitions，权重和 {weighted:.2f}，训练 rollout 中 CBF intervention 约 {interventions:.0f} 次。最终结论为 **{final['interpretation']}**；18 cm CBF-on success 变化 {primary['target_on_success_delta']:+.3f}，CBF-off success 变化 {primary['target_off_success_delta']:+.3f}，D0 CBF-on success 变化 {primary['D0_on_success_delta']:+.3f}。后续决策：`{final['followup_decision']}`。

## English summary

The fixed target uses uniform 18 cm risers, clearance slope 0.8, a 0.15 m recovery window, and CBF alpha 10. The original 405D actor and 838D privileged critic were updated for exactly eight rounds with raw-action PPO, moving KL, and locally successful CBF-action teaching while the runtime filter remained enabled. The published policy is the unconditional round-8 actor.

The final interpretation is **{final['interpretation']}**. Paired bootstrap 95% intervals are reported in `final/final_test.json` for uncertainty only and were not used as a training or conclusion gate.

## 18 cm four-condition audit

| condition | success | fall | toe-riser kick | interventions/riser |
|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

D0 (13 cm) CBF-on success: pi0 {_percent(d0['pi0_on']['success_rate'])}, pi8 {_percent(d0['pi8_on']['success_rate'])}. Final checkpoint SHA-256: `{training['final_checkpoint_sha256']}`.

Only the requested aggregate evidence and figures are tracked here. Checkpoints and per-step telemetry remain outside Git.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    smoke_dir = args.smoke_dir.resolve()
    training_dir = args.training_dir.resolve()
    final_dir = args.final_dir.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo / "results/online/proximal_v29"
    )
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v29 packaging requires a clean committed worktree")
    config_path = output / "config.json"
    config = _load_json(config_path)
    smoke = _load_json(smoke_dir / "smoke_summary.json")
    training = _load_json(training_dir / "training_summary.json")
    final = _load_json(final_dir / "final_test.json")
    actor_hash = _load_json(final_dir / "final_actor_hash.json")
    for payload in (config, smoke, training, final):
        if payload.get("protocol_id") != PROTOCOL_ID:
            raise RuntimeError("v29 packaging input has a wrong protocol id")
    if not smoke.get("passed"):
        raise RuntimeError("cannot package a failed v29 smoke")
    if training.get("final_checkpoint_sha256") != actor_hash.get(
        "checkpoint_sha256"
    ):
        raise RuntimeError("v29 final actor and training checkpoint differ")
    if final.get("training_summary_sha256") != file_sha256(
        training_dir / "training_summary.json"
    ):
        raise RuntimeError("v29 final audit does not bind the training summary")
    if final.get("paired_episode_metrics_sha256") != file_sha256(
        final_dir / "paired_episode_metrics.csv"
    ):
        raise RuntimeError("v29 paired episode table hash differs from final audit")
    commit = _git(repo, "rev-parse", "HEAD")
    expected_trees = config["prior_results_immutable"]["git_trees"]
    actual_trees = {
        version: _git(
            repo, "rev-parse", f"{commit}:results/online/proximal_{version}"
        )
        for version in expected_trees
    }
    if actual_trees != expected_trees:
        raise RuntimeError("v25-v28 result trees changed after the v29 freeze")

    destinations = {
        smoke_dir / "smoke_summary.json": output / "smoke_summary.json",
        training_dir / "round_metrics.csv": output / "training/round_metrics.csv",
        training_dir / "training_summary.json": output
        / "training/training_summary.json",
        final_dir / "final_test.json": output / "final/final_test.json",
        final_dir / "paired_episode_metrics.csv": output
        / "final/paired_episode_metrics.csv",
        final_dir / "final_actor_hash.json": output
        / "final/final_actor_hash.json",
    }
    for source, target in destinations.items():
        _copy(source, target)
    with (output / "training/round_metrics.csv").open(newline="") as handle:
        round_rows = list(csv.DictReader(handle))
    if len(round_rows) != 8 or [int(row["round"]) for row in round_rows] != list(
        range(1, 9)
    ):
        raise RuntimeError("v29 package requires exactly rounds 1 through 8")
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _training_figure(round_rows, figures / "training_curves.png")
    _four_condition_figure(final, figures / "four_condition_success.png")
    _internalization_figure(final, figures / "teacher_internalization.png")
    (output / "README.md").write_text(_readme(final, training, round_rows))
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
                "interpretation": final["interpretation"],
                "files": [str(path) for path in relative_files]
                + ["SHA256SUMS"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
