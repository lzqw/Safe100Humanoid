"""Reconstruct and aggregate v21 evidence across deployment contexts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from specialist_v21_protocol import (
  FORMAL_BOOTSTRAP_SAMPLES,
  FORMAL_BOOTSTRAP_SEED,
  FORMAL_CONTEXTS_BY_MODE,
  FORMAL_D0_EPISODES,
  FORMAL_EVAL_BATCH_SIZE,
  FORMAL_MONITOR_EPISODES,
  FORMAL_ROUNDS,
  FORMAL_TARGET_EPISODES,
  POLICY_METHOD,
  PROTOCOL_ID,
  SPECIALIST_MODES,
  V21_FORMAL_CONTEXTS,
  deployment_mode_gate,
  repair_regression_rates,
)

POLICY_ROLES = ("base", "control", "v21")
COMPARISONS = (
  ("control_minus_base", "base", "control"),
  ("v21_minus_base", "base", "v21"),
  ("v21_minus_control", "control", "v21"),
)
EVALUATION_ROLES = ("target", "D0")
TRACE_METRICS = {
  "lateral": (
    "abs_centerline_error",
    "abs_heading_error",
    "abs_centerline_error_rate",
    "abs_heading_error_rate",
    "command_vy",
    "command_wz",
    "cbf_correction_norm",
  ),
  "contact_stability": (
    "maximum_slip_speed",
    "contact_phase_mismatch",
    "abs_roll_rad",
    "abs_pitch_rad",
    "angular_velocity_norm",
    "cbf_correction_norm",
  ),
}
NORMALIZED_PHASE_BINS = 101


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _git_output(repo: Path, *args: str) -> str:
  return subprocess.run(
    ["git", *args], cwd=repo, check=True, capture_output=True, text=True
  ).stdout.strip()


def _portable_path(repo: Path, path: Path) -> str:
  try:
    return str(path.resolve().relative_to(repo))
  except ValueError:
    return str(path.resolve())


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
  with path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    return list(reader.fieldnames or []), list(reader)


def _write_csv(
  path: Path, fields: list[str], rows: list[dict[str, Any]]
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.tmp")
  with temporary.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
  temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
  rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
  temporary = path.with_name(f".{path.name}.tmp")
  temporary.write_text(rendered)
  temporary.replace(path)


def _parse_binary(value: str, *, field: str) -> int:
  if value in ("1", "True", "true"):
    return 1
  if value in ("0", "False", "false"):
    return 0
  raise ValueError(f"invalid binary {field}: {value!r}")


def _paired_interval(
  values: np.ndarray, *, samples: int, seed: int
) -> list[float]:
  values = np.asarray(values, dtype=np.float64)
  if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
    raise ValueError("paired v21 interval requires a finite non-empty vector")
  rng = np.random.default_rng(seed)
  means = np.empty(samples, dtype=np.float64)
  chunk = 1000
  for start in range(0, samples, chunk):
    stop = min(start + chunk, samples)
    indices = rng.integers(0, len(values), size=(stop - start, len(values)))
    means[start:stop] = values[indices].mean(axis=1)
  return [
    float(values.mean()),
    float(np.quantile(means, 0.025)),
    float(np.quantile(means, 0.975)),
  ]


def _comparison_metrics(
  rows: list[dict[str, str]],
  *,
  old_role: str,
  new_role: str,
  samples: int,
  seed: int,
) -> dict[str, Any]:
  old_success = np.array(
    [_parse_binary(row[f"{old_role}_success"], field="success") for row in rows],
    dtype=np.float64,
  )
  new_success = np.array(
    [_parse_binary(row[f"{new_role}_success"], field="success") for row in rows],
    dtype=np.float64,
  )
  old_fall = np.array(
    [_parse_binary(row[f"{old_role}_fell"], field="fell") for row in rows],
    dtype=np.float64,
  )
  new_fall = np.array(
    [_parse_binary(row[f"{new_role}_fell"], field="fell") for row in rows],
    dtype=np.float64,
  )
  rates = repair_regression_rates(
    [bool(value) for value in old_success],
    [bool(value) for value in new_success],
  )
  return {
    "paired_episode_count": len(rows),
    "old_success_rate": float(old_success.mean()),
    "new_success_rate": float(new_success.mean()),
    "success_delta_mean_lcb95_ucb95": _paired_interval(
      new_success - old_success, samples=samples, seed=seed
    ),
    "old_fall_rate": float(old_fall.mean()),
    "new_fall_rate": float(new_fall.mean()),
    "fall_delta_mean_lcb95_ucb95": _paired_interval(
      new_fall - old_fall, samples=samples, seed=seed + 1
    ),
    "repairs_regressions": rates,
  }


def _assert_metric_equal(actual: Any, expected: Any, label: str) -> None:
  if isinstance(actual, dict) and isinstance(expected, dict):
    if set(actual) != set(expected):
      raise RuntimeError(f"v21 reconstructed keys differ for {label}")
    for key in actual:
      _assert_metric_equal(actual[key], expected[key], f"{label}.{key}")
    return
  if isinstance(actual, list) and isinstance(expected, list):
    if len(actual) != len(expected):
      raise RuntimeError(f"v21 reconstructed list length differs for {label}")
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
      _assert_metric_equal(left, right, f"{label}[{index}]")
    return
  if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1.0e-12):
      raise RuntimeError(f"v21 reconstructed value differs for {label}")
    return
  if actual != expected:
    raise RuntimeError(f"v21 reconstructed value differs for {label}")


def _audit_context(
  audit_root: Path,
  *,
  context_id: str,
  protocol_commit: str,
  protocol_sha256: str,
  samples: int,
) -> tuple[
  dict[str, Any],
  list[dict[str, Any]],
  list[dict[str, str]],
  list[dict[str, Any]],
]:
  context_root = audit_root / context_id
  summary_path = context_root / "formal_audit_summary.json"
  paired_path = context_root / "paired_episode_metrics.csv"
  telemetry_path = context_root / "inline_mechanism_telemetry.csv"
  summary = json.loads(summary_path.read_text())
  mode = "lateral" if context_id.startswith("L") else "contact_stability"
  expected = {
    "protocol_id": summary.get("protocol_id") == PROTOCOL_ID,
    "context_id": summary.get("context_id") == context_id,
    "mode": summary.get("specialist_mode") == mode,
    "protocol_commit": summary.get("protocol_file", {}).get("git_commit")
    == protocol_commit,
    "protocol_sha256": summary.get("protocol_file", {}).get("sha256")
    == protocol_sha256,
    "paired_sha256": summary.get("paired_episode_metrics", {}).get("sha256")
    == _sha256(paired_path),
    "telemetry_sha256": summary.get("inline_mechanism_telemetry", {}).get(
      "sha256"
    )
    == _sha256(telemetry_path),
    "same_rollout": summary.get("inline_mechanism_telemetry", {}).get(
      "same_rollout_outcomes_embedded"
    )
    is True,
    "no_replay": summary.get("evaluation_protocol", {}).get(
      "post_audit_mechanism_replay_used"
    )
    is False,
  }
  failed = [name for name, passed in expected.items() if not passed]
  if failed:
    raise RuntimeError(f"invalid v21 audit boundary for {context_id}: {failed}")

  fields, raw_rows = _read_csv(paired_path)
  required = {
    "context_id",
    "specialist_mode",
    "evaluation_role",
    "evaluation_seed",
    "environment_id",
    *(
      f"{role}_{field}"
      for role in POLICY_ROLES
      for field in ("success", "fell")
    ),
  }
  if not required.issubset(fields):
    raise RuntimeError(f"v21 paired CSV schema is incomplete for {context_id}")
  ordering = lambda row: (
    row["evaluation_role"],
    int(row["evaluation_seed"]),
    int(row["environment_id"]),
  )
  raw_rows = sorted(raw_rows, key=ordering)
  keys = [ordering(row) for row in raw_rows]
  if len(keys) != len(set(keys)):
    raise RuntimeError(f"duplicate paired v21 rows for {context_id}")
  grouped = {
    role: [row for row in raw_rows if row["evaluation_role"] == role]
    for role in EVALUATION_ROLES
  }
  if len(grouped["target"]) != FORMAL_TARGET_EPISODES or len(
    grouped["D0"]
  ) != FORMAL_D0_EPISODES:
    raise RuntimeError(f"formal v21 paired row count differs for {context_id}")

  context_index = list(V21_FORMAL_CONTEXTS).index(context_id)
  reconstructed: dict[str, dict[str, Any]] = {}
  context_rows: list[dict[str, Any]] = []
  for evaluation_index, evaluation_role in enumerate(EVALUATION_ROLES):
    reconstructed[evaluation_role] = {}
    for comparison_index, (name, old_role, new_role) in enumerate(COMPARISONS):
      metrics = _comparison_metrics(
        grouped[evaluation_role],
        old_role=old_role,
        new_role=new_role,
        samples=samples,
        seed=(
          FORMAL_BOOTSTRAP_SEED
          + 100 * context_index
          + 10 * evaluation_index
          + 2 * comparison_index
        ),
      )
      _assert_metric_equal(
        metrics,
        summary["comparisons"][evaluation_role][name],
        f"{context_id}.{evaluation_role}.{name}",
      )
      reconstructed[evaluation_role][name] = metrics
      interval = metrics["success_delta_mean_lcb95_ucb95"]
      fall_interval = metrics["fall_delta_mean_lcb95_ucb95"]
      rates = metrics["repairs_regressions"]
      context_rows.append(
        {
          "context_id": context_id,
          "specialist_mode": mode,
          "evaluation_role": evaluation_role,
          "comparison": name,
          "old_role": old_role,
          "new_role": new_role,
          "paired_episodes": metrics["paired_episode_count"],
          "old_success_rate": metrics["old_success_rate"],
          "new_success_rate": metrics["new_success_rate"],
          "success_delta": interval[0],
          "success_delta_lcb95": interval[1],
          "success_delta_ucb95": interval[2],
          "old_fall_rate": metrics["old_fall_rate"],
          "new_fall_rate": metrics["new_fall_rate"],
          "fall_delta": fall_interval[0],
          "fall_delta_lcb95": fall_interval[1],
          "fall_delta_ucb95": fall_interval[2],
          "base_failure_count": rates["base_failure_count"],
          "base_success_count": rates["base_success_count"],
          "repair_count": rates["repair_count"],
          "regression_count": rates["regression_count"],
          "repair_rate": rates["repair_rate"],
          "regression_rate": rates["regression_rate"],
          "repair_minus_regression": rates["selection_score"],
        }
      )

  telemetry_fields, telemetry_rows = _read_csv(telemetry_path)
  required_telemetry = {
    "context_id",
    "specialist_mode",
    "policy_role",
    "evaluation_seed",
    "environment_id",
    "step",
    "telemetry_success",
    "telemetry_fell",
    "telemetry_failure_type",
    "telemetry_episode_steps",
  }
  if not required_telemetry.issubset(telemetry_fields):
    raise RuntimeError(f"v21 inline telemetry schema differs for {context_id}")
  if len(telemetry_rows) != summary["inline_mechanism_telemetry"]["row_count"]:
    raise RuntimeError(f"v21 inline telemetry row count differs for {context_id}")
  manifest = [
    {"scope": "formal_audit_summary", "path": str(summary_path), "sha256": _sha256(summary_path)},
    {"scope": "formal_paired_episodes", "path": str(paired_path), "sha256": _sha256(paired_path)},
    {"scope": "formal_inline_telemetry", "path": str(telemetry_path), "sha256": _sha256(telemetry_path)},
  ]
  for role in ("control", "v21"):
    training_summary_path = Path(summary["training"][role]["summary"])
    training_checkpoint_path = training_summary_path.parent / "accepted_final.pt"
    if not training_summary_path.is_file() or not training_checkpoint_path.is_file():
      raise FileNotFoundError(f"missing v21 {context_id}/{role} training artifact")
    manifest.extend(
      [
        {"scope": "formal_training_summary", "path": str(training_summary_path), "sha256": _sha256(training_summary_path)},
        {"scope": "formal_final_checkpoint", "path": str(training_checkpoint_path), "sha256": _sha256(training_checkpoint_path)},
      ]
    )
  return summary, context_rows, telemetry_rows, manifest


def _context_interval(
  values: list[float], *, seed: int, samples: int
) -> list[float]:
  if len(values) != 5:
    raise ValueError("v21 context bootstrap requires five deployment contexts")
  return _paired_interval(np.array(values), samples=samples, seed=seed)


def _aggregate_modes(
  context_rows: list[dict[str, Any]], *, samples: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  output: dict[str, Any] = {}
  flat_rows: list[dict[str, Any]] = []
  metric_fields = (
    ("target_success_delta", "target", "success_delta"),
    ("target_fall_delta", "target", "fall_delta"),
    ("d0_success_delta", "D0", "success_delta"),
    ("d0_fall_delta", "D0", "fall_delta"),
    ("target_repair_rate", "target", "repair_rate"),
    ("target_regression_rate", "target", "regression_rate"),
    ("target_repair_minus_regression", "target", "repair_minus_regression"),
  )
  for mode_index, mode in enumerate(SPECIALIST_MODES):
    output[mode] = {"contexts": list(FORMAL_CONTEXTS_BY_MODE[mode])}
    for comparison_index, (comparison, _, _) in enumerate(COMPARISONS):
      selected = [
        row
        for row in context_rows
        if row["specialist_mode"] == mode and row["comparison"] == comparison
      ]
      result: dict[str, Any] = {}
      for metric_index, (metric, evaluation_role, field) in enumerate(metric_fields):
        matching = [
          row for row in selected if row["evaluation_role"] == evaluation_role
        ]
        by_context = {row["context_id"]: float(row[field]) for row in matching}
        if tuple(by_context) != FORMAL_CONTEXTS_BY_MODE[mode]:
          raise RuntimeError(f"v21 aggregate context order differs for {mode}")
        interval = _context_interval(
          list(by_context.values()),
          samples=samples,
          seed=(
            FORMAL_BOOTSTRAP_SEED
            + 10_000
            + 1_000 * mode_index
            + 100 * comparison_index
            + 2 * metric_index
          ),
        )
        result[metric] = {
          "mean_lcb95_ucb95": interval,
          "per_context": by_context,
          "statistical_unit": "deployment_context",
        }
        flat_rows.append(
          {
            "specialist_mode": mode,
            "comparison": comparison,
            "metric": metric,
            "context_count": 5,
            "mean": interval[0],
            "lcb95": interval[1],
            "ucb95": interval[2],
          }
        )
      result["formal_gate"] = deployment_mode_gate(
        list(result["target_success_delta"]["per_context"].values()),
        list(result["target_fall_delta"]["per_context"].values()),
        list(result["d0_success_delta"]["per_context"].values()),
      )
      result["target_success_lcb95_positive"] = (
        result["target_success_delta"]["mean_lcb95_ucb95"][1] > 0.0
      )
      output[mode][comparison] = result
  return output, flat_rows


def _selectivity_comparison(
  aggregate: dict[str, Any], *, samples: int
) -> dict[str, Any]:
  output: dict[str, Any] = {}
  for mode_index, mode in enumerate(SPECIALIST_MODES):
    control = aggregate[mode]["control_minus_base"]
    v21 = aggregate[mode]["v21_minus_base"]
    metrics: dict[str, Any] = {}
    for metric_index, metric in enumerate(
      ("target_repair_rate", "target_regression_rate", "target_repair_minus_regression")
    ):
      control_values = control[metric]["per_context"]
      v21_values = v21[metric]["per_context"]
      deltas = {
        context_id: v21_values[context_id] - control_values[context_id]
        for context_id in FORMAL_CONTEXTS_BY_MODE[mode]
      }
      metrics[metric] = {
        "control_mean": float(np.mean(list(control_values.values()))),
        "v21_mean": float(np.mean(list(v21_values.values()))),
        "v21_minus_control_mean_lcb95_ucb95": _context_interval(
          list(deltas.values()),
          samples=samples,
          seed=FORMAL_BOOTSTRAP_SEED + 20_000 + 100 * mode_index + metric_index,
        ),
        "v21_minus_control_per_context": deltas,
      }
    metrics["interpretation"] = (
      "Preservation is more selective when base-referenced RR is retained while "
      "base-referenced RG decreases; no post-hoc threshold is used."
    )
    output[mode] = metrics
  return output


def _monitor_tables(
  monitor_root: Path, *, samples: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
  combined: list[dict[str, Any]] = []
  manifest: list[dict[str, Any]] = []
  for context_id in V21_FORMAL_CONTEXTS:
    mode = "lateral" if context_id.startswith("L") else "contact_stability"
    for role in ("control", "v21"):
      root = monitor_root / context_id / role
      json_path = root / "unseen_monitor_curve.json"
      csv_path = root / "unseen_monitor_curve.csv"
      summary = json.loads(json_path.read_text())
      fields, rows = _read_csv(csv_path)
      if (
        summary.get("protocol_id") != PROTOCOL_ID
        or summary.get("context_id") != context_id
        or summary.get("method_role") != role
        or summary.get("paired_conditions") != FORMAL_MONITOR_EPISODES
        or summary.get("training_selection_diagnostics_used") is not False
        or summary.get("monitor_set_accessed_only_after_all_checkpoints_were_saved")
        is not True
        or summary.get("checkpoint_count") != FORMAL_ROUNDS + 1
        or summary.get("curve_csv", {}).get("sha256") != _sha256(csv_path)
        or len(rows) != FORMAL_ROUNDS + 1
        or [int(row["round"]) for row in rows] != list(range(FORMAL_ROUNDS + 1))
      ):
        raise RuntimeError(f"invalid unseen monitor evidence for {context_id}/{role}")
      required = {
        "context_id",
        "method_role",
        "beta",
        "round",
        "actor_sha256",
        "success_rate",
        "fall_rate",
        "success_delta_from_pi0",
        "fall_delta_from_pi0",
        "repair_rate_from_pi0",
        "regression_rate_from_pi0",
      }
      if not required.issubset(fields):
        raise RuntimeError("v21 unseen monitor CSV schema differs")
      combined.extend({"specialist_mode": mode, **row} for row in rows)
      manifest.extend(
        [
          {"scope": "unseen_monitor_summary", "path": str(json_path), "sha256": _sha256(json_path)},
          {"scope": "unseen_monitor_curve", "path": str(csv_path), "sha256": _sha256(csv_path)},
        ]
      )

  aggregate_rows: list[dict[str, Any]] = []
  metric_fields = (
    "success_rate",
    "fall_rate",
    "success_delta_from_pi0",
    "fall_delta_from_pi0",
    "repair_rate_from_pi0",
    "regression_rate_from_pi0",
  )
  for mode_index, mode in enumerate(SPECIALIST_MODES):
    for role_index, role in enumerate(("control", "v21")):
      for round_index in range(FORMAL_ROUNDS + 1):
        selected = [
          row
          for row in combined
          if row["specialist_mode"] == mode
          and row["method_role"] == role
          and int(row["round"]) == round_index
        ]
        if len(selected) != 5:
          raise RuntimeError("v21 unseen monitor aggregate requires five contexts")
        for metric_index, metric in enumerate(metric_fields):
          interval = _context_interval(
            [float(row[metric]) for row in selected],
            samples=samples,
            seed=(
              FORMAL_BOOTSTRAP_SEED
              + 30_000
              + 2_000 * mode_index
              + 1_000 * role_index
              + 20 * round_index
              + metric_index
            ),
          )
          aggregate_rows.append(
            {
              "specialist_mode": mode,
              "method_role": role,
              "round": round_index,
              "metric": metric,
              "context_count": 5,
              "mean": interval[0],
              "lcb95": interval[1],
              "ucb95": interval[2],
              "evidence_role": "post-training unseen frozen monitor only",
            }
          )
  return combined, aggregate_rows, manifest


def _trace_series(
  mode: str, rows: list[dict[str, str]]
) -> dict[str, np.ndarray]:
  number = lambda field: np.array([float(row[field]) for row in rows])
  if mode == "lateral":
    return {
      "abs_centerline_error": np.abs(number("centerline_error")),
      "abs_heading_error": np.abs(number("heading_error")),
      "abs_centerline_error_rate": np.abs(number("centerline_error_rate")),
      "abs_heading_error_rate": np.abs(number("heading_error_rate")),
      "command_vy": number("command_vy"),
      "command_wz": number("command_wz"),
      "cbf_correction_norm": number("cbf_correction_norm"),
    }
  return {
    "maximum_slip_speed": np.maximum(
      number("left_slip_speed"), number("right_slip_speed")
    ),
    "contact_phase_mismatch": number("contact_phase_mismatch"),
    "abs_roll_rad": np.abs(number("roll_rad")),
    "abs_pitch_rad": np.abs(number("pitch_rad")),
    "angular_velocity_norm": number("angular_velocity_norm"),
    "cbf_correction_norm": number("cbf_correction_norm"),
  }


def _mechanism_tables(
  telemetry_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  grouped: dict[tuple[str, str, str, int, int], list[dict[str, str]]] = defaultdict(list)
  for row in telemetry_rows:
    key = (
      row["context_id"],
      row["specialist_mode"],
      row["policy_role"],
      int(row["evaluation_seed"]),
      int(row["environment_id"]),
    )
    grouped[key].append(row)
  expected_trace_count = len(V21_FORMAL_CONTEXTS) * len(POLICY_ROLES) * (
    FORMAL_TARGET_EPISODES // FORMAL_EVAL_BATCH_SIZE
  )
  if len(grouped) != expected_trace_count:
    raise RuntimeError("v21 mechanism telemetry trace count differs")

  grid = np.linspace(0.0, 1.0, NORMALIZED_PHASE_BINS)
  trace_rows: list[dict[str, Any]] = []
  interpolated: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
  success_by_group: dict[tuple[str, str], list[int]] = defaultdict(list)
  for (context_id, mode, role, evaluation_seed, environment_id), rows in sorted(
    grouped.items()
  ):
    rows = sorted(rows, key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in rows]
    if len(steps) != len(set(steps)) or any(
      right <= left for left, right in pairwise(steps)
    ):
      raise RuntimeError("v21 inline telemetry steps are not unique and ordered")
    outcome_fields = (
      "telemetry_success",
      "telemetry_fell",
      "telemetry_failure_type",
      "telemetry_episode_steps",
    )
    outcomes = {field: {row[field] for row in rows} for field in outcome_fields}
    if any(len(values) != 1 for values in outcomes.values()):
      raise RuntimeError("v21 trace rows do not share one same-rollout outcome")
    episode_steps = int(next(iter(outcomes["telemetry_episode_steps"])))
    if episode_steps < steps[-1]:
      raise RuntimeError("v21 telemetry extends beyond its bound episode outcome")
    success = _parse_binary(
      next(iter(outcomes["telemetry_success"])), field="telemetry_success"
    )
    fell = _parse_binary(next(iter(outcomes["telemetry_fell"])), field="telemetry_fell")
    series = _trace_series(mode, rows)
    phase = np.asarray(steps, dtype=np.float64) / max(1, episode_steps)
    phase = np.clip(phase, 0.0, 1.0)
    trace = {
      "context_id": context_id,
      "specialist_mode": mode,
      "policy_role": role,
      "evaluation_seed": evaluation_seed,
      "environment_id": environment_id,
      "telemetry_rows": len(rows),
      "episode_steps": episode_steps,
      "success": success,
      "fell": fell,
      "failure_type": next(iter(outcomes["telemetry_failure_type"])),
      "same_rollout_outcome_bound": True,
    }
    for metric, values in series.items():
      if not np.isfinite(values).all():
        raise RuntimeError("v21 mechanism trace contains a non-finite value")
      trace[f"{metric}_mean"] = float(values.mean())
      trace[f"{metric}_max"] = float(values.max())
      interpolated[(mode, role, metric)].append(np.interp(grid, phase, values))
    trace_rows.append(trace)
    success_by_group[(mode, role)].append(success)

  curve_rows: list[dict[str, Any]] = []
  for mode in SPECIALIST_MODES:
    for role in POLICY_ROLES:
      successes = success_by_group[(mode, role)]
      if len(successes) != 5 * (
        FORMAL_TARGET_EPISODES // FORMAL_EVAL_BATCH_SIZE
      ):
        raise RuntimeError("v21 mechanism policy/mode trace count differs")
      for metric in TRACE_METRICS[mode]:
        matrix = np.stack(interpolated[(mode, role, metric)])
        means = matrix.mean(axis=0)
        lower = np.quantile(matrix, 0.25, axis=0)
        upper = np.quantile(matrix, 0.75, axis=0)
        for phase_bin, phase in enumerate(grid):
          curve_rows.append(
            {
              "specialist_mode": mode,
              "policy_role": role,
              "metric": metric,
              "phase_bin": phase_bin,
              "normalized_episode_phase": phase,
              "trace_count": len(matrix),
              "same_rollout_success_trace_count": sum(successes),
              "mean": float(means[phase_bin]),
              "q25": float(lower[phase_bin]),
              "q75": float(upper[phase_bin]),
              "post_audit_replay_used": False,
            }
          )
  return trace_rows, curve_rows


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--protocol-file", type=Path, required=True)
  parser.add_argument("--protocol-commit", required=True)
  parser.add_argument("--audit-root", type=Path, required=True)
  parser.add_argument("--monitor-root", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
    "--bootstrap-samples", type=int, default=FORMAL_BOOTSTRAP_SAMPLES
  )
  return parser.parse_args()


def main() -> None:
  args = _parse_args()
  repo = args.repo.resolve()
  if args.bootstrap_samples != FORMAL_BOOTSTRAP_SAMPLES:
    raise ValueError("v21 aggregate bootstrap size differs from its formal freeze")
  current_commit = _git_output(repo, "rev-parse", "HEAD")
  if current_commit != args.protocol_commit:
    raise RuntimeError("v21 aggregation HEAD differs from the formal freeze")
  if _git_output(repo, "status", "--porcelain", "--untracked-files=no"):
    raise RuntimeError("formal v21 aggregation requires clean tracked files")
  protocol_path = args.protocol_file.resolve()
  protocol = json.loads(protocol_path.read_text())
  protocol_sha256 = _sha256(protocol_path)
  relative_protocol = protocol_path.relative_to(repo)
  frozen_protocol = subprocess.run(
    ["git", "show", f"{current_commit}:{relative_protocol}"],
    cwd=repo,
    check=True,
    capture_output=True,
  ).stdout
  if (
    hashlib.sha256(frozen_protocol).hexdigest() != protocol_sha256
    or protocol.get("protocol_id") != PROTOCOL_ID
    or protocol.get("protocol_revision") != 2
    or protocol.get("status") != "prospectively_frozen_before_formal_adaptation"
  ):
    raise RuntimeError("unexpected v21 formal protocol for aggregation")

  all_context_rows: list[dict[str, Any]] = []
  all_telemetry_rows: list[dict[str, str]] = []
  input_manifest: list[dict[str, Any]] = []
  audit_bindings: dict[str, Any] = {}
  for context_id in V21_FORMAL_CONTEXTS:
    summary, context_rows, telemetry_rows, manifest = _audit_context(
      args.audit_root.resolve(),
      context_id=context_id,
      protocol_commit=current_commit,
      protocol_sha256=protocol_sha256,
      samples=args.bootstrap_samples,
    )
    audit_bindings[context_id] = {
      "summary_sha256": manifest[0]["sha256"],
      "paired_episode_sha256": manifest[1]["sha256"],
      "inline_telemetry_sha256": manifest[2]["sha256"],
      "selected_beta": summary["selected_beta"],
      "adaptation_seed": summary["adaptation_seed"],
      "context_parameters_sha256": summary["context"]["parameters_sha256"],
    }
    if float(summary["selected_beta"]) != float(protocol["formal"]["selected_beta"]):
      raise RuntimeError(f"v21 selected beta differs in audit {context_id}")
    all_context_rows.extend(context_rows)
    all_telemetry_rows.extend(telemetry_rows)
    input_manifest.extend(manifest)

  mode_results, mode_rows = _aggregate_modes(
    all_context_rows, samples=args.bootstrap_samples
  )
  selectivity = _selectivity_comparison(
    mode_results, samples=args.bootstrap_samples
  )
  monitor_rows, monitor_aggregate_rows, monitor_manifest = _monitor_tables(
    args.monitor_root.resolve(), samples=args.bootstrap_samples
  )
  input_manifest.extend(monitor_manifest)
  trace_rows, mechanism_curve_rows = _mechanism_tables(all_telemetry_rows)

  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  context_csv = output_dir / "formal_context_metrics.csv"
  mode_csv = output_dir / "formal_mode_metrics.csv"
  monitor_csv = output_dir / "unseen_monitor_curves.csv"
  monitor_aggregate_csv = output_dir / "unseen_monitor_aggregate.csv"
  trace_csv = output_dir / "mechanism_trace_metrics.csv"
  mechanism_curve_csv = output_dir / "mechanism_normalized_curves.csv"
  _write_csv(context_csv, list(all_context_rows[0]), all_context_rows)
  _write_csv(mode_csv, list(mode_rows[0]), mode_rows)
  _write_csv(monitor_csv, list(monitor_rows[0]), monitor_rows)
  _write_csv(
    monitor_aggregate_csv,
    list(monitor_aggregate_rows[0]),
    monitor_aggregate_rows,
  )
  trace_fields = [
    "context_id",
    "specialist_mode",
    "policy_role",
    "evaluation_seed",
    "environment_id",
    "telemetry_rows",
    "episode_steps",
    "success",
    "fell",
    "failure_type",
    "same_rollout_outcome_bound",
  ]
  for metric in dict.fromkeys(
    metric for metrics in TRACE_METRICS.values() for metric in metrics
  ):
    trace_fields.extend((f"{metric}_mean", f"{metric}_max"))
  _write_csv(trace_csv, trace_fields, trace_rows)
  _write_csv(
    mechanism_curve_csv,
    list(mechanism_curve_rows[0]),
    mechanism_curve_rows,
  )

  claims = {
    mode: {
      "formal_gate": mode_results[mode]["v21_minus_base"]["formal_gate"],
      "strong_evidence_lcb95_positive": mode_results[mode][
        "v21_minus_base"
      ]["target_success_lcb95_positive"],
      "target_success_delta_mean_lcb95_ucb95": mode_results[mode][
        "v21_minus_base"
      ]["target_success_delta"]["mean_lcb95_ucb95"],
    }
    for mode in SPECIALIST_MODES
  }
  result = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "policy_method": POLICY_METHOD,
    "evidence_role": "fresh formal cross-deployment tri-policy reconstruction",
    "protocol": {
      "path": _portable_path(repo, protocol_path),
      "git_commit": current_commit,
      "sha256": protocol_sha256,
    },
    "experimental_unit": "one fixed deployment context and one adaptation",
    "statistical_unit": "deployment_context",
    "formal_context_count": len(V21_FORMAL_CONTEXTS),
    "formal_adaptation_count": 20,
    "paired_target_episodes_per_policy_per_context": FORMAL_TARGET_EPISODES,
    "paired_d0_episodes_per_policy_per_context": FORMAL_D0_EPISODES,
    "bootstrap_samples": args.bootstrap_samples,
    "audit_bindings": audit_bindings,
    "mode_results": mode_results,
    "base_referenced_control_vs_v21_selectivity": selectivity,
    "formal_v21_claims": claims,
    "all_mode_gates_passed": all(
      claim["formal_gate"]["passed"] for claim in claims.values()
    ),
    "tables": {
      "formal_context_metrics": {"path": _portable_path(repo, context_csv), "sha256": _sha256(context_csv), "rows": len(all_context_rows)},
      "formal_mode_metrics": {"path": _portable_path(repo, mode_csv), "sha256": _sha256(mode_csv), "rows": len(mode_rows)},
      "unseen_monitor_curves": {"path": _portable_path(repo, monitor_csv), "sha256": _sha256(monitor_csv), "rows": len(monitor_rows)},
      "unseen_monitor_aggregate": {"path": _portable_path(repo, monitor_aggregate_csv), "sha256": _sha256(monitor_aggregate_csv), "rows": len(monitor_aggregate_rows)},
      "mechanism_trace_metrics": {"path": _portable_path(repo, trace_csv), "sha256": _sha256(trace_csv), "rows": len(trace_rows), "same_rollout_outcome_bound": True},
      "mechanism_normalized_curves": {"path": _portable_path(repo, mechanism_curve_csv), "sha256": _sha256(mechanism_curve_csv), "rows": len(mechanism_curve_rows), "post_audit_replay_used": False},
    },
  }
  result_path = output_dir / "formal_results.json"
  _write_json(result_path, result)
  output_manifest = [
    {
      "scope": "published_compact_evidence",
      "path": _portable_path(repo, path),
      "bytes": path.stat().st_size,
      "sha256": _sha256(path),
    }
    for path in (
      result_path,
      context_csv,
      mode_csv,
      monitor_csv,
      monitor_aggregate_csv,
      trace_csv,
      mechanism_curve_csv,
    )
  ]
  artifact_manifest = {
    "schema_version": 1,
    "protocol_id": PROTOCOL_ID,
    "input_artifacts": input_manifest,
    "published_compact_artifacts": output_manifest,
  }
  manifest_path = output_dir / "artifact_manifest.json"
  _write_json(manifest_path, artifact_manifest)
  print(
    json.dumps(
      {
        "formal_results": str(result_path),
        "artifact_manifest": str(manifest_path),
        "formal_v21_claims": claims,
      },
      indent=2,
      sort_keys=True,
    )
  )


if __name__ == "__main__":
  main()
