"""Generate deterministic v21 figures from compact reconstructed evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from specialist_v21_protocol import (
  FORMAL_CONTEXTS_BY_MODE,
  FORMAL_ROUNDS,
  PROTOCOL_ID,
  SPECIALIST_MODES,
)

COLORS = {"base": "0.25", "control": "#D55E00", "v21": "#0072B2"}
ROLE_LABELS = {"base": "Base $\\pi_0$", "control": "v20 control", "v21": "v21"}
POLICY_ROLES = ("base", "control", "v21")


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
  with path.open(newline="") as handle:
    return list(csv.DictReader(handle))


def _save(fig: plt.Figure, base: Path) -> list[Path]:
  base.parent.mkdir(parents=True, exist_ok=True)
  outputs = [base.with_suffix(".png"), base.with_suffix(".pdf")]
  fig.savefig(
    outputs[0],
    dpi=220,
    bbox_inches="tight",
    metadata={"Software": "Safe100 v21"},
  )
  fig.savefig(
    outputs[1],
    bbox_inches="tight",
    metadata={"Creator": "Safe100 v21", "CreationDate": None, "ModDate": None},
  )
  plt.close(fig)
  return outputs


def _formal_forest(rows: list[dict[str, str]]) -> plt.Figure:
  fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True)
  for axis, mode in zip(axes, SPECIALIST_MODES, strict=True):
    contexts = FORMAL_CONTEXTS_BY_MODE[mode]
    y = np.arange(len(contexts))
    for offset, (comparison, role) in zip(
      (-0.13, 0.13),
      (("control_minus_base", "control"), ("v21_minus_base", "v21")),
      strict=True,
    ):
      selected = {
        row["context_id"]: row
        for row in rows
        if row["specialist_mode"] == mode
        and row["evaluation_role"] == "target"
        and row["comparison"] == comparison
      }
      estimates = np.array([100.0 * float(selected[name]["success_delta"]) for name in contexts])
      lower = np.array([100.0 * float(selected[name]["success_delta_lcb95"]) for name in contexts])
      upper = np.array([100.0 * float(selected[name]["success_delta_ucb95"]) for name in contexts])
      axis.errorbar(
        estimates,
        y + offset,
        xerr=np.vstack((estimates - lower, upper - estimates)),
        fmt="o",
        capsize=3,
        color=COLORS[role],
        label=ROLE_LABELS[role],
      )
    axis.axvline(0.0, color="0.35", linewidth=0.9)
    axis.set(
      yticks=y,
      yticklabels=contexts,
      xlabel="Paired target $\\Delta$SR (percentage points)",
      title="Lateral deployments" if mode == "lateral" else "Contact deployments",
    )
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.25)
    axis.legend(fontsize=8)
  fig.suptitle("One adaptation per frozen deployment context")
  return fig


def _selectivity(results: dict[str, Any]) -> plt.Figure:
  fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), sharey=True)
  for axis, mode in zip(axes, SPECIALIST_MODES, strict=True):
    values = results["base_referenced_control_vs_v21_selectivity"][mode]
    rr = values["target_repair_rate"]
    rg = values["target_regression_rate"]
    x = np.arange(2)
    width = 0.34
    control = 100.0 * np.array([rr["control_mean"], rg["control_mean"]])
    v21 = 100.0 * np.array([rr["v21_mean"], rg["v21_mean"]])
    axis.bar(x - width / 2, control, width, color=COLORS["control"], label=ROLE_LABELS["control"])
    axis.bar(x + width / 2, v21, width, color=COLORS["v21"], label=ROLE_LABELS["v21"])
    axis.set(
      xticks=x,
      xticklabels=("Repair rate", "Regression rate"),
      ylabel="Mean across contexts (%)",
      title="Lateral" if mode == "lateral" else "Contact stability",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
  fig.suptitle("Base-referenced selectivity: repairs retained, regressions suppressed")
  return fig


def _monitor(rows: list[dict[str, str]]) -> plt.Figure:
  fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
  for axis, mode in zip(axes, SPECIALIST_MODES, strict=True):
    for role in ("control", "v21"):
      selected = sorted(
        (
          row
          for row in rows
          if row["specialist_mode"] == mode
          and row["method_role"] == role
          and row["metric"] == "success_delta_from_pi0"
        ),
        key=lambda row: int(row["round"]),
      )
      if [int(row["round"]) for row in selected] != list(range(FORMAL_ROUNDS + 1)):
        raise RuntimeError("v21 monitor plot received incomplete rounds")
      rounds = np.arange(FORMAL_ROUNDS + 1)
      mean = 100.0 * np.array([float(row["mean"]) for row in selected])
      lower = 100.0 * np.array([float(row["lcb95"]) for row in selected])
      upper = 100.0 * np.array([float(row["ucb95"]) for row in selected])
      axis.fill_between(rounds, lower, upper, color=COLORS[role], alpha=0.14)
      axis.plot(rounds, mean, marker="o", color=COLORS[role], label=ROLE_LABELS[role])
    axis.axhline(0.0, color="0.35", linewidth=0.8)
    axis.set(
      xlabel="Online round",
      ylabel="Unseen-monitor $\\Delta$SR (pp)",
      title="Lateral" if mode == "lateral" else "Contact stability",
      xticks=np.arange(FORMAL_ROUNDS + 1),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
  fig.suptitle("Post-training evaluation of $\\pi_0,\\ldots,\\pi_8$ on frozen $E_{curve}$")
  return fig


def _mechanism(rows: list[dict[str, str]], mode: str) -> plt.Figure:
  configurations = {
    "lateral": (
      ("abs_centerline_error", "$|e_y|$ (m)"),
      ("abs_heading_error", "$|e_\\psi|$ (rad)"),
      ("abs_centerline_error_rate", "$|\\dot e_y|$"),
      ("abs_heading_error_rate", "$|\\dot e_\\psi|$"),
    ),
    "contact_stability": (
      ("maximum_slip_speed", "Max foot slip speed"),
      ("contact_phase_mismatch", "Contact/phase mismatch"),
      ("abs_roll_rad", "$|roll|$ (rad)"),
      ("angular_velocity_norm", "$|\\omega|$"),
    ),
  }
  fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
  for axis, (metric, label) in zip(axes.flat, configurations[mode], strict=True):
    for role in POLICY_ROLES:
      selected = sorted(
        (
          row
          for row in rows
          if row["specialist_mode"] == mode
          and row["policy_role"] == role
          and row["metric"] == metric
        ),
        key=lambda row: int(row["phase_bin"]),
      )
      phase = np.array([float(row["normalized_episode_phase"]) for row in selected])
      mean = np.array([float(row["mean"]) for row in selected])
      lower = np.array([float(row["q25"]) for row in selected])
      upper = np.array([float(row["q75"]) for row in selected])
      axis.fill_between(phase, lower, upper, color=COLORS[role], alpha=0.10)
      axis.plot(phase, mean, color=COLORS[role], label=ROLE_LABELS[role])
    axis.set(ylabel=label)
    axis.grid(alpha=0.25)
  for axis in axes[-1]:
    axis.set_xlabel("Normalized same-rollout episode phase")
  axes[0, 0].legend(fontsize=8)
  fig.suptitle(
    "Inline lateral mechanism telemetry"
    if mode == "lateral"
    else "Inline contact-stability mechanism telemetry"
  )
  return fig


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--formal-results", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  results_path = args.formal_results.resolve()
  results = json.loads(results_path.read_text())
  if results.get("protocol_id") != PROTOCOL_ID:
    raise RuntimeError("unexpected v21 formal results")
  tables = results["tables"]
  paths = {
    name: (
      Path(record["path"])
      if Path(record["path"]).is_absolute()
      else repo / record["path"]
    )
    for name, record in tables.items()
  }
  for name, path in paths.items():
    if _sha256(path) != tables[name]["sha256"]:
      raise RuntimeError(f"v21 compact table hash differs: {name}")
  context_rows = _read_csv(paths["formal_context_metrics"])
  monitor_rows = _read_csv(paths["unseen_monitor_aggregate"])
  mechanism_rows = _read_csv(paths["mechanism_normalized_curves"])
  output_dir = args.output_dir.resolve()
  figure_paths: list[Path] = []
  figure_paths.extend(_save(_formal_forest(context_rows), output_dir / "formal_context_forest"))
  figure_paths.extend(_save(_selectivity(results), output_dir / "repair_regression_selectivity"))
  figure_paths.extend(_save(_monitor(monitor_rows), output_dir / "unseen_monitor_learning_curve"))
  for mode in SPECIALIST_MODES:
    figure_paths.extend(_save(_mechanism(mechanism_rows, mode), output_dir / f"mechanism_{mode}"))
  manifest = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "formal_results": {"path": str(results_path.relative_to(repo)), "sha256": _sha256(results_path)},
    "figures": [
      {"path": str(path.relative_to(repo)), "bytes": path.stat().st_size, "sha256": _sha256(path)}
      for path in figure_paths
    ],
  }
  manifest_path = output_dir / "figure_manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
