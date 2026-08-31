"""Create the three minimal v25 evidence figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from proximal_v23_io import file_sha256

FIGURE_CATEGORIES = (
    "round_teacher_curve",
    "four_condition_performance",
    "internalization_and_cbf_dependence",
)


def _save(fig, root: Path, stem: str) -> dict[str, str]:
    png = root / f"{stem}.png"
    pdf = root / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {
        "png": str(png),
        "png_sha256": file_sha256(png),
        "pdf": str(pdf),
        "pdf_sha256": file_sha256(pdf),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--final-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    training = json.loads(args.training_summary.read_text())
    final = json.loads(args.final_test.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"categories": list(FIGURE_CATEGORIES), "figures": {}}

    rounds = training["rounds"]
    x = [row["round"] for row in rounds]
    success = [row["metrics"].get("rollout_success_rate", 0.0) for row in rounds]
    teachers = [
        row["metrics"].get("teacher_transition_fraction", 0.0) for row in rounds
    ]
    fig, axis = plt.subplots(figsize=(6.4, 3.8))
    axis.plot(x, success, marker="o", label="rollout success")
    axis.plot(x, teachers, marker="s", label="teacher fraction")
    axis.set(xlabel="fixed round", ylabel="fraction", ylim=(0.0, 1.0))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    manifest["figures"]["round_teacher_curve"] = _save(
        fig, args.output_dir, "round_teacher_curve"
    )

    order = ("pi0_off", "pi0_on", "pi8_on", "pi8_off")
    conditions = final["conditions"]
    fig, axis = plt.subplots(figsize=(6.4, 3.8))
    positions = range(len(order))
    axis.bar(
        [value - 0.18 for value in positions],
        [conditions[name]["success_rate"] for name in order],
        width=0.36,
        label="success",
    )
    axis.bar(
        [value + 0.18 for value in positions],
        [conditions[name]["kick_rate"] for name in order],
        width=0.36,
        label="toe/riser kick",
    )
    axis.set_xticks(list(positions), order)
    axis.set(ylabel="episode rate", ylim=(0.0, 1.0))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    manifest["figures"]["four_condition_performance"] = _save(
        fig, args.output_dir, "four_condition_performance"
    )

    primary = final["primary_outcomes"]
    values = (
        primary["internalization_delta"],
        primary["shielded_task_delta"],
        primary["on_intervention_per_riser_relative_reduction"],
    )
    labels = ("off success Δ", "on success Δ", "CBF dependence ↓")
    fig, axis = plt.subplots(figsize=(6.4, 3.8))
    colors = ["#2a9d8f" if value >= 0.0 else "#e76f51" for value in values]
    axis.bar(labels, values, color=colors)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(ylabel="absolute / relative change")
    axis.grid(axis="y", alpha=0.25)
    manifest["figures"]["internalization_and_cbf_dependence"] = _save(
        fig, args.output_dir, "internalization_and_cbf_dependence"
    )

    manifest_path = args.output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
