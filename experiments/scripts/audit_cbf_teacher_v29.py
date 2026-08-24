"""Run v29's sequential four-condition target audit and paired D0 audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from cbf_teacher_v29_protocol import (
    BASE_CHECKPOINT_SHA256,
    CLEARANCE_BARRIER_SLOPE,
    CONDITIONS,
    D0_CONDITIONS,
    D0_EPISODES,
    D0_RISER_HEIGHT_M,
    D0_SEED_BASE,
    ENVIRONMENT_VARIANT,
    FILTER_ALPHA,
    FINAL_EPISODES,
    FINAL_SEED_BASE,
    POLICY_METHOD,
    PREFERRED_EVAL_BATCH_SIZE,
    PROTOCOL_ID,
    RECOVERY_DISTANCE_M,
    RISER_HEIGHT_M,
    SOURCE_FILES,
    TASK_ID,
)
from proximal_v23_io import file_sha256


EPISODE_FIELDS = (
    "success",
    "fell",
    "timed_out",
    "failure_type",
    "toe_riser_kick",
    "toe_riser_kick_count",
    "return",
    "max_riser",
    "intervention_count",
    "intervention_per_riser",
    "would_intervene_count",
    "would_intervene_per_riser",
    "mean_correction_norm",
    "mean_counterfactual_correction_norm",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--eval-batch-size", type=int, default=PREFERRED_EVAL_BATCH_SIZE
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _bool(value: str) -> bool:
    normalized = value.lower()
    if normalized not in ("true", "false"):
        raise ValueError(f"invalid CSV boolean: {value!r}")
    return normalized == "true"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _condition_seed(domain: str, repeat: int) -> int:
    return (FINAL_SEED_BASE if domain == "target_18cm" else D0_SEED_BASE) + repeat


def _evaluate_condition(
    *,
    repo: Path,
    checkpoint: Path,
    condition: str,
    filter_mode: str,
    domain: str,
    height: float,
    episodes: int,
    batch_size: int,
    output_dir: Path,
    device: str,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if episodes % batch_size:
        raise ValueError("v29 episode count must divide by eval batch size")
    summaries = []
    rows: list[dict[str, str]] = []
    for repeat in range(episodes // batch_size):
        seed = _condition_seed(domain, repeat)
        run_dir = output_dir / "raw" / domain / condition / f"seed_{seed}"
        output_json = run_dir / "summary.json"
        output_csv = run_dir / "episodes.csv"
        summary = None
        if resume and output_json.is_file() and output_csv.is_file():
            candidate = json.loads(output_json.read_text())
            if (
                candidate.get("seed") == seed
                and candidate.get("num_envs") == batch_size
                and candidate.get("num_episodes") == batch_size
                and candidate.get("runtime_filter") == (filter_mode == "on")
                and candidate.get("riser_height_m") == height
                and candidate.get("clearance_barrier_slope")
                == CLEARANCE_BARRIER_SLOPE
                and candidate.get("recovery_distance_m") == RECOVERY_DISTANCE_M
                and candidate.get("filter_alpha") == FILTER_ALPHA
                and candidate.get("checkpoint_sha256") == file_sha256(checkpoint)
            ):
                summary = candidate
        if summary is None:
            command = [
                sys.executable,
                str(repo / "experiments/scripts/evaluate_cbf_teacher_v29.py"),
                "--repo",
                str(repo),
                "--checkpoint",
                str(checkpoint),
                "--riser-height",
                str(height),
                "--clearance-slope",
                str(CLEARANCE_BARRIER_SLOPE),
                "--recovery-distance",
                str(RECOVERY_DISTANCE_M),
                "--filter-alpha",
                str(FILTER_ALPHA),
                "--runtime-filter",
                filter_mode,
                "--num-envs",
                str(batch_size),
                "--num-episodes",
                str(batch_size),
                "--seed",
                str(seed),
                "--device",
                device,
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
            ]
            completed = subprocess.run(
                command, cwd=repo, capture_output=True, text=True
            )
            if completed.returncode:
                diagnostic = "\n".join(
                    (completed.stdout + "\n" + completed.stderr).splitlines()[-160:]
                )
                raise RuntimeError(
                    f"v29 final condition {domain}/{condition} failed:\n{diagnostic}"
                )
            summary = json.loads(output_json.read_text())
        with output_csv.open(newline="") as handle:
            batch_rows = list(csv.DictReader(handle))
        if len(batch_rows) != batch_size:
            raise RuntimeError(f"incomplete v29 rows for {domain}/{condition}")
        summaries.append(summary)
        rows.extend(batch_rows)

    success = [_bool(row["success"]) for row in rows]
    fell = [_bool(row["fell"]) for row in rows]
    kick = [_bool(row["toe_riser_kick"]) for row in rows]
    total_risers = sum(int(row["max_riser"]) for row in rows)
    total_interventions = sum(int(row["intervention_count"]) for row in rows)
    aggregate = {
        "domain": domain,
        "condition": condition,
        "runtime_filter": filter_mode == "on",
        "num_episodes": len(rows),
        "seeds": [_condition_seed(domain, i) for i in range(len(summaries))],
        "initial_state_signatures": [
            item["initial_state_signature"] for item in summaries
        ],
        "checkpoint_sha256": summaries[0]["checkpoint_sha256"],
        "actor_state_sha256": summaries[0]["actor_state_sha256"],
        "deterministic_policy_mean": all(
            item["deterministic_policy_mean"] for item in summaries
        ),
        "original_actor_observation_interface": all(
            item["original_observation_interface"] for item in summaries
        ),
        "actor_observation_dim": summaries[0]["actor_observation_dim"],
        "success_count": sum(success),
        "success_rate": sum(success) / len(rows),
        "fall_count": sum(fell),
        "fall_rate": sum(fell) / len(rows),
        "kick_episode_count": sum(kick),
        "kick_rate": sum(kick) / len(rows),
        "mean_return": float(np.mean([float(row["return"]) for row in rows])),
        "mean_reached_riser": float(
            np.mean([float(row["max_riser"]) for row in rows])
        ),
        "total_reached_risers": total_risers,
        "total_intervention_count": total_interventions,
        "intervention_per_riser": total_interventions / max(1, total_risers),
        "mean_correction_norm": float(
            np.mean([float(row["mean_correction_norm"]) for row in rows])
        ),
        "mean_counterfactual_correction_norm": float(
            np.mean(
                [float(row["mean_counterfactual_correction_norm"]) for row in rows]
            )
        ),
        "teacher_reprojection_max_abs_error": max(
            float(item["teacher_reprojection_max_abs_error"]) for item in summaries
        ),
        "swing_selection_mismatch_count": sum(
            int(item["swing_selection_mismatch_count"]) for item in summaries
        ),
    }
    if any(
        item["actor_state_sha256"] != aggregate["actor_state_sha256"]
        or item["checkpoint_sha256"] != aggregate["checkpoint_sha256"]
        for item in summaries
    ):
        raise RuntimeError(f"v29 actor changed within {domain}/{condition}")
    return aggregate, rows


def _identity(row: dict[str, str]) -> tuple[int, int]:
    return int(row["evaluation_seed"]), int(row["environment_id"])


def _paired_ci(
    baseline: list[float], final: list[float], *, seed: int, samples: int = 20_000
) -> dict[str, Any]:
    if not baseline or len(baseline) != len(final):
        raise ValueError("paired CI vectors must be non-empty and equal")
    delta = np.asarray(final, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(0, samples, 1000):
        count = min(1000, samples - len(means) * 1000)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means.append(delta[indices].mean(axis=1))
    bootstrap = np.concatenate(means)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "paired_count": len(delta),
        "point_estimate": float(delta.mean()),
        "paired_bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "used_as_training_or_conclusion_gate": False,
    }


def _column(rows: list[dict[str, str]], name: str, *, boolean: bool) -> list[float]:
    if boolean:
        return [float(_bool(row[name])) for row in rows]
    return [float(row[name]) for row in rows]


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    base_checkpoint = args.base_checkpoint.resolve()
    final_checkpoint = args.final_checkpoint.resolve()
    training_path = args.training_summary.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True
    ).stdout:
        raise RuntimeError("v29 final audit requires a clean committed worktree")
    for path in (base_checkpoint, final_checkpoint, training_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if file_sha256(base_checkpoint) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v29 final audit base checkpoint differs from pi0")
    if args.eval_batch_size not in (128, PREFERRED_EVAL_BATCH_SIZE):
        raise ValueError("v29 final audit batch size must be preferred 256 or fallback 128")
    training = json.loads(training_path.read_text())
    config = json.loads(config_path.read_text())
    if training.get("protocol_id") != PROTOCOL_ID or config.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("v29 final audit has wrong protocol inputs")
    if training.get("final_checkpoint_sha256") != file_sha256(final_checkpoint):
        raise RuntimeError("v29 final checkpoint differs from unconditional round 8")
    if training.get("final_policy_rule") != "round 8 actor, never best checkpoint":
        raise RuntimeError("v29 final audit was not given the fixed round-8 actor")
    if training.get("config", {}).get("sha256") != file_sha256(config_path):
        raise RuntimeError("v29 final config differs from training")
    sources = config.get("implementation_boundary", {}).get("source_files", {})
    if set(sources) != set(SOURCE_FILES) or any(
        file_sha256(repo / relative) != sources.get(relative)
        for relative in SOURCE_FILES
    ):
        raise RuntimeError("v29 final source differs from the fixed implementation")

    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "final_evaluation_started.json"
    if marker.exists() and not args.resume:
        raise RuntimeError("v29 final audit already started; use --resume after a crash")
    if not marker.exists():
        _atomic_json(
            marker,
            {
                "protocol_id": PROTOCOL_ID,
                "config_sha256": file_sha256(config_path),
                "training_summary_sha256": file_sha256(training_path),
                "base_checkpoint_sha256": file_sha256(base_checkpoint),
                "final_checkpoint_sha256": file_sha256(final_checkpoint),
                "condition_order": [item[0] for item in CONDITIONS],
                "d0_condition_order": [item[0] for item in D0_CONDITIONS],
                "eval_batch_size": args.eval_batch_size,
                "performance_selection": False,
            },
        )

    checkpoints = {"base": base_checkpoint, "final": final_checkpoint}
    aggregates: dict[str, dict[str, dict[str, Any]]] = {
        "target_18cm": {},
        "D0_13cm": {},
    }
    rows: dict[str, dict[str, list[dict[str, str]]]] = {
        "target_18cm": {},
        "D0_13cm": {},
    }
    for condition, policy, filter_mode in CONDITIONS:
        aggregate, episode_rows = _evaluate_condition(
            repo=repo,
            checkpoint=checkpoints[policy],
            condition=condition,
            filter_mode=filter_mode,
            domain="target_18cm",
            height=RISER_HEIGHT_M,
            episodes=FINAL_EPISODES,
            batch_size=args.eval_batch_size,
            output_dir=output_dir,
            device=args.device,
            resume=args.resume,
        )
        aggregates["target_18cm"][condition] = aggregate
        rows["target_18cm"][condition] = episode_rows
        print(json.dumps(aggregate, sort_keys=True), flush=True)
    for condition, policy, filter_mode in D0_CONDITIONS:
        aggregate, episode_rows = _evaluate_condition(
            repo=repo,
            checkpoint=checkpoints[policy],
            condition=condition,
            filter_mode=filter_mode,
            domain="D0_13cm",
            height=D0_RISER_HEIGHT_M,
            episodes=D0_EPISODES,
            batch_size=args.eval_batch_size,
            output_dir=output_dir,
            device=args.device,
            resume=args.resume,
        )
        aggregates["D0_13cm"][condition] = aggregate
        rows["D0_13cm"][condition] = episode_rows
        print(json.dumps(aggregate, sort_keys=True), flush=True)

    for domain, conditions in rows.items():
        for condition_rows in conditions.values():
            condition_rows.sort(key=_identity)
        identity_lists = [
            [_identity(row) for row in value] for value in conditions.values()
        ]
        if any(value != identity_lists[0] for value in identity_lists[1:]):
            raise RuntimeError(f"v29 {domain} paired identities differ")
        signatures = [
            item["initial_state_signatures"]
            for item in aggregates[domain].values()
        ]
        if any(value != signatures[0] for value in signatures[1:]):
            raise RuntimeError(f"v29 {domain} initial state signatures differ")

    paired_rows: list[dict[str, Any]] = []
    all_condition_names = [item[0] for item in CONDITIONS]
    for domain in ("target_18cm", "D0_13cm"):
        domain_conditions = rows[domain]
        first = next(iter(domain_conditions.values()))
        for index, source in enumerate(first):
            output: dict[str, Any] = {
                "domain": domain,
                "pair_index": index,
                "evaluation_seed": source["evaluation_seed"],
                "environment_id": source["environment_id"],
            }
            for condition in all_condition_names:
                episode = (
                    domain_conditions.get(condition, [])[index]
                    if condition in domain_conditions
                    else None
                )
                for field in EPISODE_FIELDS:
                    output[f"{condition}_{field}"] = (
                        "" if episode is None else episode[field]
                    )
            paired_rows.append(output)
    paired_path = output_dir / "paired_episode_metrics.csv"
    with paired_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(paired_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(paired_rows)

    target = rows["target_18cm"]
    d0 = rows["D0_13cm"]
    intervals = {
        "target_on_success_delta": _paired_ci(
            _column(target["pi0_on"], "success", boolean=True),
            _column(target["pi8_on"], "success", boolean=True),
            seed=162_290_001,
        ),
        "target_off_success_delta": _paired_ci(
            _column(target["pi0_off"], "success", boolean=True),
            _column(target["pi8_off"], "success", boolean=True),
            seed=162_290_002,
        ),
        "target_on_fall_delta": _paired_ci(
            _column(target["pi0_on"], "fell", boolean=True),
            _column(target["pi8_on"], "fell", boolean=True),
            seed=162_290_007,
        ),
        "target_off_fall_delta": _paired_ci(
            _column(target["pi0_off"], "fell", boolean=True),
            _column(target["pi8_off"], "fell", boolean=True),
            seed=162_290_008,
        ),
        "target_off_kick_rate_delta": _paired_ci(
            _column(target["pi0_off"], "toe_riser_kick", boolean=True),
            _column(target["pi8_off"], "toe_riser_kick", boolean=True),
            seed=162_290_003,
        ),
        "target_on_intervention_per_riser_delta": _paired_ci(
            _column(target["pi0_on"], "intervention_per_riser", boolean=False),
            _column(target["pi8_on"], "intervention_per_riser", boolean=False),
            seed=162_290_004,
        ),
        "target_on_policy_safe_gap_delta": _paired_ci(
            _column(
                target["pi0_on"], "mean_counterfactual_correction_norm", boolean=False
            ),
            _column(
                target["pi8_on"], "mean_counterfactual_correction_norm", boolean=False
            ),
            seed=162_290_005,
        ),
        "D0_on_success_delta": _paired_ci(
            _column(d0["pi0_on"], "success", boolean=True),
            _column(d0["pi8_on"], "success", boolean=True),
            seed=162_290_006,
        ),
        "D0_on_fall_delta": _paired_ci(
            _column(d0["pi0_on"], "fell", boolean=True),
            _column(d0["pi8_on"], "fell", boolean=True),
            seed=162_290_009,
        ),
    }
    primary = {
        key: value["point_estimate"] for key, value in intervals.items()
    }
    internalization = {
        "off_success_improved": primary["target_off_success_delta"] > 0.0,
        "off_kick_rate_decreased": primary["target_off_kick_rate_delta"] < 0.0,
        "on_intervention_per_riser_decreased": primary[
            "target_on_intervention_per_riser_delta"
        ]
        < 0.0,
        "policy_safe_gap_decreased": primary[
            "target_on_policy_safe_gap_delta"
        ]
        < 0.0,
    }
    internalization_count = sum(internalization.values())
    clear = (
        primary["target_on_success_delta"] >= 0.03
        and primary["target_off_success_delta"] >= 0.05
        and internalization["off_kick_rate_decreased"]
        and internalization["on_intervention_per_riser_decreased"]
        and primary["D0_on_success_delta"] >= -0.05
    )
    partial = (
        not clear
        and primary["target_on_success_delta"] >= 0.0
        and internalization_count >= 2
    )
    ineffective = (
        primary["target_on_success_delta"] < 0.0
        and internalization_count == 0
    )
    interpretation = (
        "clearly_effective"
        if clear
        else "partially_effective"
        if partial
        else "ineffective"
        if ineffective
        else "mixed_or_insufficient_effect"
    )
    final = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "policy_method": POLICY_METHOD,
        "checkpoints": {
            "pi0_sha256": file_sha256(base_checkpoint),
            "pi8_sha256": file_sha256(final_checkpoint),
            "pi0_actor_sha256": aggregates["target_18cm"]["pi0_off"][
                "actor_state_sha256"
            ],
            "pi8_actor_sha256": aggregates["target_18cm"]["pi8_off"][
                "actor_state_sha256"
            ],
        },
        "evaluation": {
            "eval_batch_size": args.eval_batch_size,
            "target_paired_identities": FINAL_EPISODES,
            "D0_paired_identities": D0_EPISODES,
            "same_target_identities_all_four_conditions": True,
            "same_D0_identities_pi0_pi8": True,
            "deterministic_mean_action": True,
            "condition_order": [item[0] for item in CONDITIONS],
            "gpu_jobs_sequential": True,
        },
        "conditions": aggregates,
        "primary_outcomes": primary,
        "paired_95_intervals": intervals,
        "internalization_metrics": internalization,
        "internalization_metric_count_improved": internalization_count,
        "interpretation": interpretation,
        "interpretation_conditions": {
            "clearly_effective": clear,
            "partially_effective": partial,
            "ineffective": ineffective,
            "confidence_intervals_are_not_gates": True,
        },
        "followup_decision": (
            "run_fixed_teacher_weight_zero_control"
            if clear
            else "stop_without_no_teacher_control"
        ),
        "final_policy_rule": "unconditional round 8 actor",
        "training_summary_sha256": file_sha256(training_path),
        "config_sha256": file_sha256(config_path),
        "paired_episode_metrics_sha256": file_sha256(paired_path),
    }
    _atomic_json(output_dir / "final_test.json", final)
    _atomic_json(
        output_dir / "final_actor_hash.json",
        {
            "checkpoint_sha256": file_sha256(final_checkpoint),
            "actor_state_sha256": final["checkpoints"]["pi8_actor_sha256"],
            "round": 8,
            "selection": "unconditional final round",
        },
    )
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
