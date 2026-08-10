"""Generate exactly four compact v22 effect-first result figures."""

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

from specialist_v22_protocol import PROTOCOL_ID

MODE_LABELS = {
  "L_effect": "Lateral",
  "C_effect": "Contact",
}
POLICY_ROLES = ("base", "best")
POLICY_COLORS = {"base": "#64748b", "best": "#2563eb"}


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--lateral-training-dir", type=Path, required=True)
  parser.add_argument("--lateral-test-dir", type=Path, required=True)
  parser.add_argument("--contact-training-dir", type=Path)
  parser.add_argument("--contact-test-dir", type=Path)
  parser.add_argument("--output-dir", type=Path, required=True)
  return parser.parse_args()


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bundle(training_dir: Path, test_dir: Path) -> dict[str, Any]:
  training = json.loads((training_dir / "specialist_summary.json").read_text())
  result = json.loads((test_dir / "final_test.json").read_text())
  if (
    training.get("protocol_id") != PROTOCOL_ID
    or result.get("protocol_id") != PROTOCOL_ID
    or training.get("context_id") != result.get("context_id")
  ):
    raise RuntimeError("v22 plot inputs do not share one valid context")
  with (test_dir / "mechanism_telemetry.csv").open(newline="") as handle:
    telemetry = list(csv.DictReader(handle))
  return {"training": training, "result": result, "telemetry": telemetry}


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[dict[str, Any]]:
  records = []
  for suffix in ("png", "pdf"):
    path = output_dir / f"{stem}.{suffix}"
    fig.savefig(path, dpi=180 if suffix == "png" else None, bbox_inches="tight")
    records.append(
      {
        "file": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
      }
    )
  plt.close(fig)
  return records


def _validation_figure(bundles: dict[str, dict[str, Any]]) -> plt.Figure:
  fig, ax = plt.subplots(figsize=(7.2, 4.4))
  colors = {"L_effect": "#2563eb", "C_effect": "#ea580c"}
  for context_id, bundle in bundles.items():
    rows = bundle["training"]["validation_monitor"]["rows"]
    x = [int(row["round"]) for row in rows]
    y = [100.0 * float(row["success_rate"]) for row in rows]
    best_round = int(bundle["training"]["validation_monitor"]["best_so_far"]["round"])
    ax.plot(
      x,
      y,
      marker="o",
      linewidth=2,
      color=colors[context_id],
      label=f"{MODE_LABELS[context_id]} accepted checkpoints",
    )
    best_index = x.index(best_round)
    ax.scatter(
      [best_round],
      [y[best_index]],
      s=100,
      marker="*",
      color=colors[context_id],
      edgecolor="black",
      linewidth=0.6,
      zorder=4,
    )
    ax.axhline(y[0], color=colors[context_id], linestyle=":", alpha=0.5)
  ax.set_xlabel("Accepted checkpoint round")
  ax.set_ylabel("Fixed validation success (%)")
  ax.set_title("Best-so-far validation monitor")
  ax.set_xticks(range(0, 9))
  ax.grid(alpha=0.25)
  ax.legend(frameon=False)
  return fig


def _base_best_figure(bundles: dict[str, dict[str, Any]]) -> plt.Figure:
  fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2))
  contexts = list(bundles)
  labels = [MODE_LABELS[context_id] for context_id in contexts]
  x = np.arange(len(contexts), dtype=float)
  width = 0.34
  for axis, metric, title in (
    (axes[0], "success_rate", "Final target success"),
    (axes[1], "fall_rate", "Final target falls"),
  ):
    base = [
      100.0 * float(bundles[c]["result"]["comparisons"]["target"][f"old_{metric}"])
      for c in contexts
    ]
    best = [
      100.0 * float(bundles[c]["result"]["comparisons"]["target"][f"new_{metric}"])
      for c in contexts
    ]
    axis.bar(x - width / 2, base, width, label="Base", color=POLICY_COLORS["base"])
    axis.bar(x + width / 2, best, width, label="Best", color=POLICY_COLORS["best"])
    axis.set_xticks(x, labels)
    axis.set_ylabel("Episodes (%)")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    for index, value in enumerate(base):
      axis.text(index - width / 2, value + 0.6, f"{value:.1f}", ha="center", fontsize=8)
    for index, value in enumerate(best):
      axis.text(index + width / 2, value + 0.6, f"{value:.1f}", ha="center", fontsize=8)
  axes[0].legend(frameon=False)
  fig.suptitle("Fresh 512-pair base vs best test")
  fig.tight_layout()
  return fig


def _repair_regression_figure(bundles: dict[str, dict[str, Any]]) -> plt.Figure:
  fig, ax = plt.subplots(figsize=(7.2, 4.4))
  contexts = list(bundles)
  x = np.arange(len(contexts), dtype=float)
  width = 0.34
  rates = [
    bundles[c]["result"]["comparisons"]["target"]["repairs_regressions"]
    for c in contexts
  ]
  repair = [100.0 * float(row["repair_rate"]) for row in rates]
  regression = [100.0 * float(row["regression_rate"]) for row in rates]
  ax.bar(x - width / 2, repair, width, label="Failure → success", color="#16a34a")
  ax.bar(x + width / 2, regression, width, label="Success → failure", color="#dc2626")
  ax.set_xticks(x, [MODE_LABELS[c] for c in contexts])
  ax.set_ylabel("Conditional transition rate (%)")
  ax.set_title("Repair versus regression on fresh paired episodes")
  ax.grid(axis="y", alpha=0.25)
  ax.legend(frameon=False)
  fig.tight_layout()
  return fig


def _normalized_trace_means(
  rows: list[dict[str, str]], field: str
) -> dict[str, np.ndarray]:
  phase = np.linspace(0.0, 1.0, 101)
  traces: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
  for row in rows:
    role = row["policy_role"]
    key = (role, row["evaluation_seed"], row["environment_id"])
    traces.setdefault(key, []).append((int(row["step"]), abs(float(row[field]))))
  by_role: dict[str, list[np.ndarray]] = {"base": [], "best": []}
  for (role, _, _), values in traces.items():
    ordered = sorted(values)
    y = np.asarray([value for _, value in ordered], dtype=float)
    x = np.linspace(0.0, 1.0, len(y))
    by_role[role].append(np.interp(phase, x, y))
  return {
    role: np.mean(np.stack(values), axis=0)
    for role, values in by_role.items()
    if values
  }


def _telemetry_figure(bundles: dict[str, dict[str, Any]]) -> plt.Figure:
  panels: list[tuple[str, str, str, str]] = []
  if "L_effect" in bundles:
    panels.extend(
      (
        ("L_effect", "centerline_error", "|e_y|", "Centerline error"),
        ("L_effect", "heading_error", "|e_psi|", "Heading error"),
      )
    )
  if "C_effect" in bundles:
    panels.append(("C_effect", "maximum_slip", "v_slip", "Maximum foot slip"))
  fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.9))
  if len(panels) == 1:
    axes = [axes]
  phase = np.linspace(0.0, 1.0, 101)
  for axis, (context_id, field, ylabel, title) in zip(axes, panels, strict=True):
    rows = bundles[context_id]["telemetry"]
    if field == "maximum_slip":
      enriched = []
      for row in rows:
        copy = dict(row)
        copy[field] = max(float(row["left_slip_speed"]), float(row["right_slip_speed"]))
        enriched.append(copy)
      rows = enriched
    curves = _normalized_trace_means(rows, field)
    for role in POLICY_ROLES:
      axis.plot(
        phase,
        curves[role],
        linewidth=2,
        color=POLICY_COLORS[role],
        label=role.capitalize(),
      )
    axis.set_xlabel("Normalized episode phase")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
  axes[0].legend(frameon=False)
  fig.suptitle("Failure-specific same-rollout telemetry")
  fig.tight_layout()
  return fig


def main() -> None:
  args = _parse_args()
  if (args.contact_training_dir is None) != (args.contact_test_dir is None):
    raise ValueError("contact training and test directories must be supplied together")
  bundles = {
    "L_effect": _load_bundle(
      args.lateral_training_dir.resolve(), args.lateral_test_dir.resolve()
    )
  }
  if args.contact_training_dir is not None and args.contact_test_dir is not None:
    bundles["C_effect"] = _load_bundle(
      args.contact_training_dir.resolve(), args.contact_test_dir.resolve()
    )
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  figures = []
  for stem, figure in (
    ("validation_learning_curve", _validation_figure(bundles)),
    ("base_vs_best_final", _base_best_figure(bundles)),
    ("repair_vs_regression", _repair_regression_figure(bundles)),
    ("failure_specific_telemetry", _telemetry_figure(bundles)),
  ):
    figures.extend(_save(figure, output_dir, stem))
  manifest = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "context_ids": list(bundles),
    "figure_categories": 4,
    "files": figures,
  }
  output = output_dir / "figure_manifest.json"
  output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
