"""Generate deterministic paper figures from published v20 CSV/JSON only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from specialist_v20_protocol import (
  FORMAL_ADAPTATION_SEEDS,
  FORMAL_BOOTSTRAP_SEED,
  FORMAL_ROUNDS,
  SPECIALIST_MODES,
)

COLORS = plt.get_cmap("tab10")


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
  with path.open(newline="") as handle:
    return list(csv.DictReader(handle))


def _number(row: dict[str, str], field: str) -> float:
  value = row.get(field, "")
  return float("nan") if value == "" else float(value)


def _mode_rows(
  rows: list[dict[str, str]], mode: str
) -> list[dict[str, str]]:
  return [row for row in rows if row["specialist"] == mode]


def _seed_series(
  rows: list[dict[str, str]], field: str
) -> dict[int, np.ndarray]:
  output: dict[int, np.ndarray] = {}
  for seed in FORMAL_ADAPTATION_SEEDS:
    selected = sorted(
      (row for row in rows if int(row["adaptation_seed"]) == seed),
      key=lambda row: int(row["round"]),
    )
    if [int(row["round"]) for row in selected] != list(
      range(FORMAL_ROUNDS + 1)
    ):
      raise RuntimeError(f"v20 curve rounds differ for seed {seed}")
    output[seed] = np.array([_number(row, field) for row in selected])
  return output


def _bootstrap_band(
  matrix: np.ndarray, *, seed: int, samples: int = 10_000
) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  indices = rng.integers(0, matrix.shape[0], size=(samples, matrix.shape[0]))
  means = matrix[indices].mean(axis=1)
  return np.quantile(means, 0.025, axis=0), np.quantile(
    means, 0.975, axis=0
  )


def _save(fig: plt.Figure, base: Path) -> list[Path]:
  base.parent.mkdir(parents=True, exist_ok=True)
  outputs = [base.with_suffix(".png"), base.with_suffix(".pdf")]
  for path in outputs:
    fig.savefig(path, dpi=220, bbox_inches="tight", metadata={"Creator": "Safe100 v20"})
  plt.close(fig)
  return outputs


def _learning_curve(
  rows: list[dict[str, str]],
  *,
  mode: str,
  field: str,
  ylabel: str,
  title: str,
  bootstrap_offset: int,
  scale: float = 100.0,
) -> plt.Figure:
  series = _seed_series(_mode_rows(rows, mode), field)
  matrix = np.stack(list(series.values()))
  rounds = np.arange(FORMAL_ROUNDS + 1)
  fig, ax = plt.subplots(figsize=(7.2, 4.5))
  for index, (seed, values) in enumerate(series.items()):
    ax.plot(
      rounds,
      scale * values,
      color=COLORS(index),
      linewidth=1.0,
      alpha=0.55,
      label=f"seed {seed}",
    )
  lower, upper = _bootstrap_band(
    matrix,
    seed=FORMAL_BOOTSTRAP_SEED + bootstrap_offset,
  )
  mean = matrix.mean(axis=0)
  ax.fill_between(
    rounds, scale * lower, scale * upper, color="black", alpha=0.12
  )
  ax.plot(rounds, scale * mean, color="black", linewidth=2.4, label="mean")
  ax.set(xlabel="Online round", ylabel=ylabel, title=title)
  ax.set_xticks(rounds)
  ax.grid(alpha=0.25)
  ax.legend(ncol=3, fontsize=8)
  ax.text(
    0.01,
    0.01,
    "Training-gate diagnostics; final claim uses the fresh paired audit.",
    transform=ax.transAxes,
    fontsize=7,
    color="0.35",
  )
  return fig


def _d0_curve(rows: list[dict[str, str]], mode: str) -> plt.Figure:
  selected = _mode_rows(rows, mode)
  series = _seed_series(selected, "d0_success")
  baselines = {
    seed: _seed_series(selected, "d0_baseline_success")[seed]
    for seed in FORMAL_ADAPTATION_SEEDS
  }
  rounds = np.arange(FORMAL_ROUNDS + 1)
  matrix = np.stack(list(series.values()))
  fig, ax = plt.subplots(figsize=(7.2, 4.5))
  for index, (seed, values) in enumerate(series.items()):
    ax.plot(rounds, 100.0 * values, color=COLORS(index), alpha=0.5)
  ax.plot(rounds, 100.0 * matrix.mean(axis=0), color="black", linewidth=2.4)
  threshold = np.mean([values[0] for values in baselines.values()]) - 0.05
  ax.axhline(
    100.0 * threshold,
    linestyle="--",
    color="crimson",
    label="mean baseline − 5 pp",
  )
  ax.set(
    xlabel="Online round",
    ylabel="D0 success rate (%)",
    title=f"{mode}: D0 retention",
  )
  ax.set_xticks(rounds)
  ax.grid(alpha=0.25)
  ax.legend()
  return fig


def _candidate_trajectory(
  candidate_rows: list[dict[str, str]], mode: str
) -> plt.Figure:
  rows = _mode_rows(candidate_rows, mode)
  rounds = np.arange(1, FORMAL_ROUNDS + 1)
  fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True)
  for index, fraction in enumerate((0.5, 1.0, 1.5)):
    means = []
    for round_index in rounds:
      values = [
        _number(row, "screen_success_delta")
        for row in rows
        if int(row["round"]) == round_index
        and float(row["fraction"]) == fraction
      ]
      means.append(100.0 * np.mean(values))
    top.plot(
      rounds,
      means,
      marker="o",
      color=COLORS(index),
      label=f"fraction {fraction:g}",
    )
  top.axhline(0.0, color="0.25", linewidth=0.8)
  confirmation_means = []
  for round_index in rounds:
    per_run = {
      (row["specialist"], row["adaptation_seed"], row["round"]): _number(
        row, "confirmation_success_delta"
      )
      for row in rows
      if int(row["round"]) == round_index
    }
    finite = [value for value in per_run.values() if np.isfinite(value)]
    confirmation_means.append(
      float("nan") if not finite else 100.0 * np.mean(finite)
    )
  top.plot(
    rounds,
    confirmation_means,
    color="black",
    linestyle="--",
    marker="s",
    label="confirmation ΔSR",
  )
  top.set(ylabel="Mean screen ΔSR (pp)", title=f"{mode}: candidate backtracking")
  top.grid(alpha=0.25)
  top.legend()
  for seed_index, seed in enumerate(FORMAL_ADAPTATION_SEEDS):
    seed_rows = [
      row
      for row in rows
      if int(row["adaptation_seed"]) == seed and row["retained"] == "True"
    ]
    bottom.scatter(
      [int(row["round"]) + 0.035 * (seed_index - 2) for row in seed_rows],
      [float(row["fraction"]) for row in seed_rows],
      color=COLORS(seed_index),
      label=f"seed {seed}",
      s=28,
    )
    retained_rounds = {int(row["round"]) for row in seed_rows}
    rejected_rounds = [
      round_index
      for round_index in rounds
      if round_index not in retained_rounds
    ]
    bottom.scatter(
      [round_index + 0.035 * (seed_index - 2) for round_index in rejected_rounds],
      [0.0] * len(rejected_rounds),
      color=COLORS(seed_index),
      marker="x",
      s=24,
    )
  bottom.set(
    xlabel="Online round",
    ylabel="Retained fraction",
    yticks=(0.0, 0.5, 1.0, 1.5),
    yticklabels=("rejected", "0.5", "1.0", "1.5"),
  )
  bottom.grid(alpha=0.25)
  bottom.legend(ncol=3, fontsize=8)
  return fig


def _kl_curve(round_rows: list[dict[str, str]], mode: str) -> plt.Figure:
  rows = [row for row in _mode_rows(round_rows, mode) if int(row["round"]) > 0]
  rounds = np.arange(1, FORMAL_ROUNDS + 1)
  fig, ax = plt.subplots(figsize=(7.2, 4.5))
  for index, seed in enumerate(FORMAL_ADAPTATION_SEEDS):
    values = [
      _number(row, "ppo_mean_kl")
      for row in sorted(
        (item for item in rows if int(item["adaptation_seed"]) == seed),
        key=lambda item: int(item["round"]),
      )
    ]
    ax.plot(rounds, values, color=COLORS(index), alpha=0.65, label=f"seed {seed}")
  ax.axhline(0.003, color="darkorange", linestyle="--", label="target KL 0.003")
  ax.axhline(0.01, color="crimson", linestyle="--", label="hard ceiling 0.01")
  ax.set(xlabel="Online round", ylabel="Mean KL", title=f"{mode}: PPO update magnitude")
  ax.set_xticks(rounds)
  ax.grid(alpha=0.25)
  ax.legend(ncol=3, fontsize=8)
  return fig


def _replay_figure(replay_rows: list[dict[str, str]], mode: str) -> plt.Figure:
  rows = _mode_rows(replay_rows, mode)
  rounds = np.arange(1, FORMAL_ROUNDS + 1)
  fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex=True)
  metrics = [
    ("failure_bank_size", "Failure bank size"),
    ("matched_success_bank_size", "Matched success bank size"),
    ("exact_pair_count", "Exact pair count"),
    ("maximum_marginal_imbalance", "Max marginal imbalance"),
  ]
  for ax, (field, label) in zip(axes.flat, metrics, strict=True):
    means = [
      np.mean(
        [_number(row, field) for row in rows if int(row["round"]) == round_index]
      )
      for round_index in rounds
    ]
    ax.plot(rounds, means, color="black", marker="o")
    ax.set_ylabel(label)
    ax.grid(alpha=0.25)
  restored = [
    sum(
      row["bank_transaction_restored"] == "True"
      for row in rows
      if int(row["round"]) == round_index
    )
    for round_index in rounds
  ]
  axes[0, 0].bar(rounds, restored, alpha=0.25, color="crimson", label="restores")
  axes[0, 0].legend(fontsize=8)
  for ax in axes[-1]:
    ax.set_xlabel("Online round")
    ax.set_xticks(rounds)
  fig.suptitle(f"{mode}: replay-bank integrity")
  return fig


def _adapter_figure(round_rows: list[dict[str, str]], mode: str) -> plt.Figure:
  rows = _mode_rows(round_rows, mode)
  rms = _seed_series(rows, "new_input_column_rms")
  legacy = _seed_series(rows, "legacy_input_column_max_drift")
  rounds = np.arange(FORMAL_ROUNDS + 1)
  fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
  for index, seed in enumerate(FORMAL_ADAPTATION_SEEDS):
    top.plot(rounds, rms[seed], color=COLORS(index), alpha=0.65, label=f"seed {seed}")
    bottom.plot(rounds, legacy[seed], color=COLORS(index), alpha=0.65)
  top.set(ylabel="New-column weight RMS", title=f"{mode}: observable adapter learning")
  top.grid(alpha=0.25)
  top.legend(ncol=3, fontsize=8)
  bottom.set(xlabel="Online round", ylabel="Legacy-column max drift")
  bottom.set_xticks(rounds)
  bottom.grid(alpha=0.25)
  return fig


def _forest(
  paired_rows: list[dict[str, str]], summary: dict[str, Any], mode: str
) -> plt.Figure:
  target = [
    row
    for row in paired_rows
    if row["specialist_mode"] == mode
    and row["evaluation_role"] == "target_diagonal_primary"
  ]
  estimates = []
  intervals = []
  rng = np.random.default_rng(FORMAL_BOOTSTRAP_SEED + 700)
  for seed in FORMAL_ADAPTATION_SEEDS:
    rows = [row for row in target if int(row["adaptation_seed"]) == seed]
    delta = np.array(
      [int(row["final_success"]) - int(row["baseline_success"]) for row in rows]
    )
    indices = rng.integers(0, len(delta), size=(10_000, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    estimates.append(delta.mean())
    intervals.append(np.quantile(bootstrap, (0.025, 0.975)))
  aggregate = summary["independent_claim"]["target"][
    "paired_success_delta_mean_lcb95_ucb95"
  ]
  estimates.append(float(aggregate[0]))
  intervals.append(np.array([aggregate[1], aggregate[2]]))
  labels = [f"seed {seed}" for seed in FORMAL_ADAPTATION_SEEDS] + ["aggregate"]
  y = np.arange(len(labels))
  fig, ax = plt.subplots(figsize=(7.2, 4.7))
  for index, (estimate, interval) in enumerate(zip(estimates, intervals, strict=True)):
    color = "black" if index == len(labels) - 1 else COLORS(index)
    ax.errorbar(
      100.0 * estimate,
      y[index],
      xerr=np.array(
        [[100.0 * (estimate - interval[0])], [100.0 * (interval[1] - estimate)]]
      ),
      fmt="o",
      color=color,
      capsize=3,
    )
  ax.axvline(0.0, color="0.35", linewidth=0.9)
  ax.set(yticks=y, yticklabels=labels, xlabel="Paired target ΔSR (pp)", title=f"{mode}: final paired improvement")
  ax.invert_yaxis()
  ax.grid(axis="x", alpha=0.25)
  return fig


def _repairs(
  paired_rows: list[dict[str, str]], mode: str
) -> plt.Figure:
  target = [
    row
    for row in paired_rows
    if row["specialist_mode"] == mode
    and row["evaluation_role"] == "target_diagonal_primary"
  ]
  repairs = sum(row["transition_class"] == "failure_to_success" for row in target)
  regressions = sum(row["transition_class"] == "success_to_failure" for row in target)
  values = [repairs, regressions, repairs - regressions]
  fig, ax = plt.subplots(figsize=(6.4, 4.3))
  bars = ax.bar(
    ["Repairs", "Regressions", "Net"],
    values,
    color=("seagreen", "indianred", "steelblue"),
  )
  ax.bar_label(bars)
  ax.axhline(0.0, color="0.25", linewidth=0.8)
  ax.set(ylabel="Paired episode count", title=f"{mode}: repairs vs regressions")
  ax.grid(axis="y", alpha=0.25)
  return fig


def _mechanism(
  mechanism_rows: list[dict[str, str]], mode: str
) -> plt.Figure:
  rows = [row for row in mechanism_rows if row["specialist"] == mode]
  if mode == "lateral":
    fields = (
      ("centerline_error", "|centerline error| (m)"),
      ("heading_error", "|heading error| (rad)"),
    )
    transform: Callable[[np.ndarray], np.ndarray] = np.abs
  else:
    fields = (
      ("left_slip_speed", "max foot slip speed (m/s)"),
      ("contact_phase_mismatch", "contact-phase mismatch"),
    )
    transform = lambda values: values
  grid = np.linspace(0.0, 1.0, 101)
  fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
  for ax, (field, ylabel) in zip(axes, fields, strict=True):
    role_curves: dict[str, list[np.ndarray]] = {"baseline": [], "final": []}
    for seed in FORMAL_ADAPTATION_SEEDS:
      for role in ("baseline", "final"):
        selected = sorted(
          (
            row
            for row in rows
            if int(row["adaptation_seed"]) == seed
            and row["policy_role"] == role
          ),
          key=lambda row: int(row["step"]),
        )
        if not selected:
          continue
        values = np.array([_number(row, field) for row in selected])
        if mode == "contact_stability" and field == "left_slip_speed":
          right = np.array(
            [_number(row, "right_slip_speed") for row in selected]
          )
          values = np.maximum(values, right)
        values = transform(values)
        progress = np.linspace(0.0, 1.0, len(values))
        curve = np.interp(grid, progress, values)
        role_curves[role].append(curve)
        ax.plot(
          grid,
          curve,
          color="0.55" if role == "baseline" else "lightskyblue",
          alpha=0.25,
          linewidth=0.8,
        )
    for role, color in (("baseline", "black"), ("final", "tab:blue")):
      if role_curves[role]:
        ax.plot(
          grid,
          np.mean(role_curves[role], axis=0),
          color=color,
          linewidth=2.2,
          label=role,
        )
    ax.set(xlabel="Normalized episode progress", ylabel=ylabel)
    ax.grid(alpha=0.25)
    if any(role_curves.values()):
      ax.legend()
    else:
      ax.text(
        0.5,
        0.5,
        "No failure→success repair in the formal paired audit",
        transform=ax.transAxes,
        ha="center",
        va="center",
      )
  fig.suptitle(
    f"{mode}: deterministic lowest-ID repaired trajectories (no visual selection)"
  )
  return fig


def _validate_inputs(
  round_rows: list[dict[str, str]],
  candidate_rows: list[dict[str, str]],
  replay_rows: list[dict[str, str]],
) -> None:
  expected = {
    "round": len(SPECIALIST_MODES) * len(FORMAL_ADAPTATION_SEEDS) * 9,
    "candidate": len(SPECIALIST_MODES)
    * len(FORMAL_ADAPTATION_SEEDS)
    * FORMAL_ROUNDS
    * 3,
    "replay": len(SPECIALIST_MODES)
    * len(FORMAL_ADAPTATION_SEEDS)
    * FORMAL_ROUNDS
    * 2,
  }
  actual = {
    "round": len(round_rows),
    "candidate": len(candidate_rows),
    "replay": len(replay_rows),
  }
  if actual != expected:
    raise RuntimeError(f"v20 plotting input row counts differ: {actual} != {expected}")
  try:
    round_keys = Counter(
      (
        row["specialist"],
        int(row["adaptation_seed"]),
        int(row["round"]),
      )
      for row in round_rows
    )
    candidate_keys = Counter(
      (
        row["specialist"],
        int(row["adaptation_seed"]),
        int(row["round"]),
        float(row["fraction"]),
      )
      for row in candidate_rows
    )
    replay_keys = Counter(
      (
        row["specialist"],
        int(row["adaptation_seed"]),
        int(row["round"]),
        int(row["batch"]),
      )
      for row in replay_rows
    )
  except (KeyError, TypeError, ValueError) as exc:
    raise RuntimeError("v20 plotting input matrix schema differs") from exc
  expected_round_keys = {
    (mode, seed, round_index)
    for mode in SPECIALIST_MODES
    for seed in FORMAL_ADAPTATION_SEEDS
    for round_index in range(FORMAL_ROUNDS + 1)
  }
  expected_candidate_keys = {
    (mode, seed, round_index, fraction)
    for mode in SPECIALIST_MODES
    for seed in FORMAL_ADAPTATION_SEEDS
    for round_index in range(1, FORMAL_ROUNDS + 1)
    for fraction in (0.5, 1.0, 1.5)
  }
  expected_replay_keys = {
    (mode, seed, round_index, batch)
    for mode in SPECIALIST_MODES
    for seed in FORMAL_ADAPTATION_SEEDS
    for round_index in range(1, FORMAL_ROUNDS + 1)
    for batch in (1, 2)
  }
  matrices = {
    "round": (set(round_keys), expected_round_keys, round_keys),
    "candidate": (
      set(candidate_keys),
      expected_candidate_keys,
      candidate_keys,
    ),
    "replay": (set(replay_keys), expected_replay_keys, replay_keys),
  }
  invalid = {
    name: {
      "missing": len(expected_keys - actual_keys),
      "unexpected": len(actual_keys - expected_keys),
      "duplicates": sum(count - 1 for count in counts.values()),
    }
    for name, (actual_keys, expected_keys, counts) in matrices.items()
    if actual_keys != expected_keys or any(count != 1 for count in counts.values())
  }
  if invalid:
    raise RuntimeError(f"v20 plotting matrices are incomplete: {invalid}")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--evidence-root", type=Path, required=True)
  parser.add_argument("--audit-root", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  evidence_root = args.evidence_root.resolve()
  audit_root = args.audit_root.resolve()
  output_dir = args.output_dir.resolve()
  round_path = evidence_root / "curves/round_metrics.csv"
  candidate_path = evidence_root / "curves/candidate_metrics.csv"
  replay_path = evidence_root / "curves/replay_metrics.csv"
  mechanism_path = evidence_root / "curves/mechanism_metrics.csv"
  round_rows = _read_csv(round_path)
  candidate_rows = _read_csv(candidate_path)
  replay_rows = _read_csv(replay_path)
  mechanism_rows = _read_csv(mechanism_path)
  _validate_inputs(round_rows, candidate_rows, replay_rows)
  generated: list[Path] = []
  for mode_index, mode in enumerate(SPECIALIST_MODES):
    paired_path = audit_root / mode / "paired_episode_metrics.csv"
    summary_path = audit_root / mode / "final_audit_summary.json"
    verification_path = audit_root / mode / "verification.json"
    paired_rows = _read_csv(paired_path)
    summary = json.loads(summary_path.read_text())
    verification = json.loads(verification_path.read_text())
    if verification.get("verified") is not True:
      raise RuntimeError(f"v20 {mode} audit is not independently verified")
    figures = [
      ("main", "target_success_vs_round", _learning_curve(round_rows, mode=mode, field="accepted_target_success", ylabel="Target success rate (%)", title=f"{mode}: online success", bootstrap_offset=800 + mode_index)),
      ("main", "target_fall_vs_round", _learning_curve(round_rows, mode=mode, field="accepted_target_fall", ylabel="Target fall rate (%)", title=f"{mode}: online fall rate", bootstrap_offset=810 + mode_index)),
      ("appendix", "d0_retention_vs_round", _d0_curve(round_rows, mode)),
      ("appendix", "candidate_trajectory", _candidate_trajectory(candidate_rows, mode)),
      ("appendix", "ppo_kl_vs_round", _kl_curve(round_rows, mode)),
      ("appendix", "accepted_updates", _learning_curve(round_rows, mode=mode, field="cumulative_retained_updates", ylabel="Cumulative retained updates", title=f"{mode}: retained-update trajectory", bootstrap_offset=820 + mode_index, scale=1.0)),
      ("appendix", "replay_integrity", _replay_figure(replay_rows, mode)),
      ("appendix", "adapter_learning", _adapter_figure(round_rows, mode)),
      ("main", "paired_delta_forest", _forest(paired_rows, summary, mode)),
      ("main", "repairs_vs_regressions", _repairs(paired_rows, mode)),
      ("main", "repaired_mechanism", _mechanism(mechanism_rows, mode)),
    ]
    for section, name, figure in figures:
      generated.extend(
        _save(figure, output_dir / section / f"{mode}_{name}")
      )
  sources = [round_path, candidate_path, replay_path, mechanism_path]
  manifest = {
    "schema_version": 1,
    "deterministic": True,
    "source_files": {
      str(path): _sha256(path) for path in sources
    },
    "figures": [
      {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
      }
      for path in generated
    ],
    "png_count": sum(path.suffix == ".png" for path in generated),
    "pdf_count": sum(path.suffix == ".pdf" for path in generated),
  }
  path = output_dir / "figure_manifest.json"
  path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
