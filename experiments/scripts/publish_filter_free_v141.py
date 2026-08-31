"""Publish the compact v141 formal tables, plots, and final checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from filter_free_v141_protocol import (
    DEVELOPMENT_THRESHOLDS,
    FORMAL_EVALUATION_EPISODES,
    FORMAL_EVALUATION_SEED,
    FORMAL_TRAINING_SEEDS,
    METHOD_ID,
    PROTOCOL_ID,
    RETENTION_CONTEXT,
    SPECIALISTS,
)


METHODS = (
    "frozen_v139",
    "v140_dual_safe_ft",
    "v141_intervention_aware",
)
V140_TRAINING_SEEDS = (201_357_000, 201_357_001, 201_357_002)
METHOD_LABELS = {
    "frozen_v139": "Frozen v139",
    "v140_dual_safe_ft": "v140 Dual Safe-FT",
    "v141_intervention_aware": "v141 IA-CBF Distill",
}
COLORS = {
    "frozen_v139": "#6b7280",
    "v140_dual_safe_ft": "#d97706",
    "v141_intervention_aware": "#047857",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commit-and-push", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"cannot publish empty CSV {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": record["method"],
        "method_label": METHOD_LABELS[record["method"]],
        "specialist": record["specialist"],
        "training_seed": (
            "frozen" if record["training_seed"] is None else record["training_seed"]
        ),
        "context": record["context"],
        "context_role": record["context_role"],
        "runtime_filter": record["runtime_filter"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "initial_state_signature": record["initial_state_signature"],
        "deterministic_policy_mean": record["deterministic_policy_mean"],
        "evaluation_seed": record["evaluation_seed"],
        "evaluation_episodes": record["evaluation_episodes"],
        **record["metrics"],
    }


def _validate_raw(raw: dict[str, Any]) -> None:
    expected_metadata = {
        "formal_training_seeds": list(FORMAL_TRAINING_SEEDS),
        "formal_evaluation_seed": FORMAL_EVALUATION_SEED,
        "formal_evaluation_episodes": FORMAL_EVALUATION_EPISODES,
        "paired_initial_conditions": True,
        "fixed_final_round_checkpoint": True,
        "best_so_far_selection": False,
    }
    mismatches = {
        key: (raw.get(key), value)
        for key, value in expected_metadata.items()
        if raw.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"v141 formal metadata differs: {mismatches}")

    expected_training = {
        (specialist, seed)
        for specialist in SPECIALISTS
        for seed in FORMAL_TRAINING_SEEDS
    }
    actual_training = {
        (item.get("specialist"), item.get("training_seed"))
        for item in raw.get("training_summaries", [])
        if item.get("method") == "v141_intervention_aware"
    }
    if actual_training != expected_training or len(raw.get("training_summaries", [])) != 6:
        raise RuntimeError("v141 formal training matrix is incomplete or duplicated")

    expected_evaluations: set[tuple[Any, ...]] = set()
    for specialist in SPECIALISTS:
        for context, role in (
            (specialist, "target"),
            (RETENTION_CONTEXT, "retention"),
        ):
            for condition in ("off", "on"):
                expected_evaluations.add(
                    ("frozen_v139", specialist, None, context, role, condition)
                )
                for seed in V140_TRAINING_SEEDS:
                    expected_evaluations.add(
                        (
                            "v140_dual_safe_ft",
                            specialist,
                            seed,
                            context,
                            role,
                            condition,
                        )
                    )
                for seed in FORMAL_TRAINING_SEEDS:
                    expected_evaluations.add(
                        (
                            "v141_intervention_aware",
                            specialist,
                            seed,
                            context,
                            role,
                            condition,
                        )
                    )
    records = raw.get("evaluation_records", [])
    actual_evaluations = {
        (
            item.get("method"),
            item.get("specialist"),
            item.get("training_seed"),
            item.get("context"),
            item.get("context_role"),
            item.get("runtime_filter"),
        )
        for item in records
    }
    if actual_evaluations != expected_evaluations or len(records) != len(expected_evaluations):
        raise RuntimeError("v141 formal evaluation matrix is incomplete or duplicated")
    signatures: dict[str, set[str]] = {
        context: set() for context in (RETENTION_CONTEXT, *SPECIALISTS)
    }
    for item in records:
        if (
            item.get("deterministic_policy_mean") is not True
            or item.get("evaluation_seed") != FORMAL_EVALUATION_SEED
            or item.get("evaluation_episodes") != FORMAL_EVALUATION_EPISODES
        ):
            raise RuntimeError("v141 formal evaluation identity differs")
        signature = item.get("initial_state_signature")
        if not isinstance(signature, str) or len(signature) != 64:
            raise RuntimeError("v141 formal initial-state signature is invalid")
        signatures[item["context"]].add(signature)
        for name, value in item.get("metrics", {}).items():
            if not math.isfinite(float(value)):
                raise RuntimeError(f"non-finite formal metric {name}")
    expected_signatures = {
        context: next(iter(values))
        for context, values in signatures.items()
        if len(values) == 1
    }
    if (
        len(expected_signatures) != len(signatures)
        or raw.get("paired_initial_state_signatures") != expected_signatures
    ):
        raise RuntimeError("v141 formal paired initial-state identities differ")


def _select(
    records: list[dict[str, Any]],
    *,
    method: str,
    specialist: str,
    role: str,
    condition: str,
) -> list[dict[str, Any]]:
    selected = [
        item
        for item in records
        if item["method"] == method
        and item["specialist"] == specialist
        and item["context_role"] == role
        and item["runtime_filter"] == condition
    ]
    if not selected:
        raise RuntimeError(
            f"formal result group is empty: {method}/{specialist}/{role}/{condition}"
        )
    return selected


def _metric_mean(records: list[dict[str, Any]], metric: str) -> float:
    return mean(float(item["metrics"][metric]) for item in records)


def _main_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for specialist in SPECIALISTS:
        for method in METHODS:
            target_off = _select(
                records,
                method=method,
                specialist=specialist,
                role="target",
                condition="off",
            )
            target_on = _select(
                records,
                method=method,
                specialist=specialist,
                role="target",
                condition="on",
            )
            retention_off = _select(
                records,
                method=method,
                specialist=specialist,
                role="retention",
                condition="off",
            )
            retention_on = _select(
                records,
                method=method,
                specialist=specialist,
                role="retention",
                condition="on",
            )
            target_off_success = _metric_mean(target_off, "success_rate")
            target_on_success = _metric_mean(target_on, "success_rate")
            off_values = [float(item["metrics"]["success_rate"]) for item in target_off]
            rows.append(
                {
                    "specialist": specialist,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "policy_count": len(target_off),
                    "target_off_success": target_off_success,
                    "target_off_success_std": pstdev(off_values) if len(off_values) > 1 else 0.0,
                    "target_on_success": target_on_success,
                    "f1_retention_off_success": _metric_mean(
                        retention_off, "success_rate"
                    ),
                    "f1_retention_on_success": _metric_mean(
                        retention_on, "success_rate"
                    ),
                    "shield_gap": target_on_success - target_off_success,
                    "would_intervene_fraction_off": _metric_mean(
                        target_off, "counterfactual_would_intervene_fraction"
                    ),
                    "counterfactual_correction_norm_off": _metric_mean(
                        target_off, "mean_counterfactual_correction_norm"
                    ),
                    "nominal_violation_steps_per_riser_off": _metric_mean(
                        target_off, "nominal_barrier_violation_steps_per_riser"
                    ),
                    "fall_rate_off": _metric_mean(target_off, "fall_rate"),
                    "mean_reached_riser_off": _metric_mean(
                        target_off, "mean_reached_riser"
                    ),
                }
            )
    return rows


def _formal_checks(main_rows: list[dict[str, Any]]) -> dict[str, Any]:
    specialists: dict[str, Any] = {}
    for specialist in SPECIALISTS:
        by_method = {
            row["method"]: row
            for row in main_rows
            if row["specialist"] == specialist
        }
        frozen = by_method["frozen_v139"]
        candidate = by_method["v141_intervention_aware"]
        base_gap = abs(float(frozen["shield_gap"]))
        candidate_gap = abs(float(candidate["shield_gap"]))
        checks = {
            "target_off_improves_2pp": float(candidate["target_off_success"])
            - float(frozen["target_off_success"])
            >= DEVELOPMENT_THRESHOLDS["target_off_improvement_pp"] / 100.0,
            "shield_gap_target": candidate_gap
            <= DEVELOPMENT_THRESHOLDS["shield_gap_pp"] / 100.0
            or candidate_gap
            <= base_gap
            * (1.0 - DEVELOPMENT_THRESHOLDS["shield_gap_reduction_fraction"]),
            "would_intervene_reduced_25pct": float(
                candidate["would_intervene_fraction_off"]
            )
            <= float(frozen["would_intervene_fraction_off"])
            * (1.0 - DEVELOPMENT_THRESHOLDS["would_intervene_reduction_fraction"]),
            "f1_retention_within_1p5pp": float(
                candidate["f1_retention_off_success"]
            )
            >= float(frozen["f1_retention_off_success"])
            - DEVELOPMENT_THRESHOLDS["f1_off_retention_loss_pp"] / 100.0,
        }
        specialists[specialist] = {
            "checks": checks,
            "passed": all(checks.values()),
            "target_off_improvement": float(candidate["target_off_success"])
            - float(frozen["target_off_success"]),
            "shield_gap": float(candidate["shield_gap"]),
            "frozen_shield_gap": float(frozen["shield_gap"]),
            "would_intervene_reduction_fraction": 1.0
            - float(candidate["would_intervene_fraction_off"])
            / max(1.0e-12, float(frozen["would_intervene_fraction_off"])),
            "f1_retention_delta": float(candidate["f1_retention_off_success"])
            - float(frozen["f1_retention_off_success"]),
        }
    return {
        "specialists": specialists,
        "both_specialists_pass": all(
            item["passed"] for item in specialists.values()
        ),
    }


def _learning_curves(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in raw["training_summaries"]:
        summary = json.loads(Path(item["summary"]).read_text())
        for record in summary["round_metrics"]:
            metrics = record["metrics"]
            target_falls = int(metrics.get("rollout_target_fall_count", 0))
            retention_falls = int(
                metrics.get("rollout_retention_f1_fall_count", 0)
            )
            rows.append(
                {
                    "method": item["method"],
                    "specialist": item["specialist"],
                    "training_seed": item["training_seed"],
                    "round": record["round"],
                    "target_rollout_success_rate": metrics.get(
                        "rollout_target_success_rate"
                    ),
                    "retention_rollout_success_rate": metrics.get(
                        "rollout_retention_f1_success_rate"
                    ),
                    "target_mean_return": metrics.get(
                        "rollout_target_mean_return"
                    ),
                    "retention_mean_return": metrics.get(
                        "rollout_retention_f1_mean_return"
                    ),
                    "training_nominal_violation_fraction": metrics.get(
                        "rollout_nominal_violation_fraction"
                    ),
                    "training_executed_violation_fraction": metrics.get(
                        "rollout_executed_violation_fraction"
                    ),
                    "training_intervention_count": metrics.get(
                        "v141_intervention_count"
                    ),
                    "training_fall_count": target_falls + retention_falls,
                    "correction_loss": metrics.get("teacher_loss"),
                    "ppo_loss": metrics.get("surrogate"),
                    "actor_moving_forward_kl": metrics.get("moving_forward_kl"),
                    "actor_learning_rate": summary["hyperparameters"].get(
                        "actor_learning_rate"
                    ),
                }
            )
    return rows


def _save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plots(output: Path, rows: list[dict[str, Any]]) -> None:
    x = np.arange(len(SPECIALISTS), dtype=float)
    width = 0.24

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for index, method in enumerate(METHODS):
        values = [
            next(
                row["target_off_success"]
                for row in rows
                if row["method"] == method and row["specialist"] == specialist
            )
            for specialist in SPECIALISTS
        ]
        errors = [
            next(
                row["target_off_success_std"]
                for row in rows
                if row["method"] == method and row["specialist"] == specialist
            )
            for specialist in SPECIALISTS
        ]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=METHOD_LABELS[method],
            color=COLORS[method],
        )
    ax.set_xticks(x, SPECIALISTS)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("CBF-off success rate")
    ax.set_title("Filter-free target success")
    ax.legend(frameon=False, fontsize=8)
    _save_figure(fig, output, "target_off_success")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for index, method in enumerate(METHODS):
        values = [
            next(
                row["shield_gap"]
                for row in rows
                if row["method"] == method and row["specialist"] == specialist
            )
            for specialist in SPECIALISTS
        ]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(0.02, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.axhline(-0.02, color="#9ca3af", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, SPECIALISTS)
    ax.set_ylabel("CBF-on minus CBF-off success")
    ax.set_title("Runtime-shield dependence")
    ax.legend(frameon=False, fontsize=8)
    _save_figure(fig, output, "shield_dependence")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for index, method in enumerate(METHODS):
        intervention = [
            next(
                row["would_intervene_fraction_off"]
                for row in rows
                if row["method"] == method and row["specialist"] == specialist
            )
            for specialist in SPECIALISTS
        ]
        correction = [
            next(
                row["counterfactual_correction_norm_off"]
                for row in rows
                if row["method"] == method and row["specialist"] == specialist
            )
            for specialist in SPECIALISTS
        ]
        position = x + (index - 1) * width
        axes[0].bar(position, intervention, width, color=COLORS[method], label=METHOD_LABELS[method])
        axes[1].bar(position, correction, width, color=COLORS[method])
    for axis in axes:
        axis.set_xticks(x, SPECIALISTS)
    axes[0].set_ylabel("Would-intervene fraction")
    axes[0].set_title("Intervention demand")
    axes[1].set_ylabel("Mean correction norm")
    axes[1].set_title("Counterfactual correction")
    axes[0].legend(frameon=False, fontsize=8)
    _save_figure(fig, output, "intervention_internalization")


def _copy_checkpoints(
    output: Path, raw: dict[str, Any]
) -> list[dict[str, Any]]:
    destination = output / "checkpoints"
    destination.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for item in raw["training_summaries"]:
        source = Path(item["checkpoint"])
        source_sha256 = _sha256(source)
        if source_sha256 != item["checkpoint_sha256"]:
            raise RuntimeError(f"formal checkpoint hash differs: {source}")
        name = (
            f"v141_{item['specialist']}_seed_{item['training_seed']}_"
            f"{source.name}"
        )
        target = destination / name
        if not target.is_file() or _sha256(target) != source_sha256:
            shutil.copy2(source, target)
        manifest.append(
            {
                "specialist": item["specialist"],
                "training_seed": item["training_seed"],
                "relative_path": str(target.relative_to(output)),
                "sha256": source_sha256,
                "bytes": target.stat().st_size,
                "fixed_final_round": True,
            }
        )
    return manifest


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    raw_path = args.raw_results.resolve()
    frozen_path = args.frozen_config.resolve()
    output = args.output_dir.resolve()
    raw = json.loads(raw_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    if (
        raw.get("protocol_id") != PROTOCOL_ID
        or raw.get("method_id") != METHOD_ID
        or frozen.get("frozen_before_formal") is not True
    ):
        raise RuntimeError("v141 formal publication inputs differ")
    _validate_raw(raw)
    output.mkdir(parents=True, exist_ok=True)
    records = raw["evaluation_records"]
    per_seed = [_flatten_record(record) for record in records]
    main_rows = _main_table(records)
    checks = _formal_checks(main_rows)
    curves = _learning_curves(raw)
    manifest = _copy_checkpoints(output, raw)
    _write_csv(output / "per_seed_results.csv", per_seed)
    _write_csv(output / "main_table.csv", main_rows)
    _write_csv(output / "learning_curves.csv", curves)
    _plots(output, main_rows)
    _atomic_json(
        output / "checkpoint_sha256_manifest.json",
        {"schema_version": 1, "checkpoints": manifest},
    )
    formal = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "formal_success": checks["both_specialists_pass"],
        "required_filter_free_goal_satisfied": checks[
            "both_specialists_pass"
        ],
        "status": (
            "success"
            if checks["both_specialists_pass"]
            else "failed_return_to_development"
        ),
        "frozen_configuration_sha256": _sha256(frozen_path),
        "formal_training_seeds": raw["formal_training_seeds"],
        "formal_evaluation_seed": raw["formal_evaluation_seed"],
        "formal_evaluation_episodes": raw["formal_evaluation_episodes"],
        "paired_initial_conditions": raw["paired_initial_conditions"],
        "paired_initial_state_signatures": raw[
            "paired_initial_state_signatures"
        ],
        "fixed_final_round_checkpoint": raw["fixed_final_round_checkpoint"],
        "best_so_far_selection": raw["best_so_far_selection"],
        "formal_checks": checks,
        "main_table": main_rows,
        "checkpoint_manifest": manifest,
    }
    _atomic_json(output / "formal_results.json", formal)
    outcome = "PASS" if formal["formal_success"] else "FAIL — return to development"
    table_lines = [
        "| Specialist | Method | Target off | Target on | F1 off | Shield gap | Would intervene |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        table_lines.append(
            "| {specialist} | {method_label} | {target_off_success:.4f} | "
            "{target_on_success:.4f} | {f1_retention_off_success:.4f} | "
            "{shield_gap:+.4f} | {would_intervene_fraction_off:.4f} |".format(
                **row
            )
        )
    readme = "\n".join(
        [
            "# v141 Intervention-Aware CBF Distillation PPO",
            "",
            f"Formal outcome: **{outcome}**.",
            "",
            "All policies use the original 405-D actor. New specialists were retrained from frozen v139 with three fresh seeds and evaluated on fixed final-round checkpoints using 512 paired deterministic episodes with runtime CBF off/on.",
            "",
            *table_lines,
            "",
            "Published files are intentionally limited to the requested tables, learning curves, three figures, six final PT checkpoints, and their SHA-256 manifest. No ONNX, hardware proxy, or real-robot artifact was generated.",
            "",
        ]
    )
    (output / "README.md").write_text(readme)
    if args.commit_and_push:
        if not formal["formal_success"]:
            raise RuntimeError(
                "formal v141 failed its fixed gates; refusing GitHub publication"
            )
        relative = output.relative_to(repo)
        subprocess.run(["git", "add", str(relative)], cwd=repo, check=True)
        if subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=repo, check=False
        ).returncode != 0:
            subprocess.run(
                ["git", "commit", "-m", "Publish v141 formal filter-free results"],
                cwd=repo,
                check=True,
            )
        subprocess.run(
            ["git", "push", "origin", "feature/online-safe-refinement"],
            cwd=repo,
            check=True,
        )
    print(json.dumps(formal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
