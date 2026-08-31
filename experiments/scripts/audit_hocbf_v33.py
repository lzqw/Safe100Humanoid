"""Run v33 frozen-policy and final paired audits with aggregate-only outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from hocbf_v33_protocol import (
    BOOTSTRAP_SAMPLES,
    CURRENT_CBF_MODE,
    FINAL_D0_EPISODES,
    FINAL_TARGET_EPISODES,
    FORMAL_CONTEXTS,
    FROZEN_POLICY_EPISODES,
    HOCBF_MODE,
    PROTOCOL_ID,
    V31_CHECKPOINT_SHA256,
    bootstrap_seed,
    final_d0_seed,
    final_target_seed,
    frozen_audit_seed,
)
from proximal_v23_io import file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--v31-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("frozen", "final"), required=True)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("v33 audit CSV cannot be empty")
    fields = sorted({key for row in rows for key in row})
    leading = [
        key
        for key in (
            "phase",
            "source_context",
            "evaluation_context",
            "policy",
            "condition",
        )
        if key in fields
    ]
    fields = leading + [key for key in fields if key not in leading]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str | bool) -> bool:
    return value if isinstance(value, bool) else value.lower() == "true"


def _identity(row: dict[str, str]) -> tuple[int, int]:
    return int(row["evaluation_seed"]), int(row["environment_id"])


def _v31_checkpoint(root: Path, context: str, policy: str) -> Path:
    checkpoint = root / context / policy / "round_08.pt"
    if (
        not checkpoint.is_file()
        or file_sha256(checkpoint) != V31_CHECKPOINT_SHA256[context][policy]
    ):
        raise RuntimeError(f"v33 v31 checkpoint differs for {context}/{policy}")
    return checkpoint


def _aggregate(
    *,
    condition: str,
    summaries: list[dict[str, Any]],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    total_steps = sum(int(row["steps"]) for row in rows)
    total_risers = sum(int(row["max_riser"]) for row in rows)
    ever = [row for row in rows if _bool(row["ever_intervened"])]
    successful_times = [
        float(row["completion_time_s"]) for row in rows if _bool(row["success"])
    ]

    def step_mean(field: str) -> float:
        return sum(float(row[field]) * int(row["steps"]) for row in rows) / max(
            1, total_steps
        )

    def episode_mean(field: str, source: list[dict[str, str]] | None = None) -> float:
        values = rows if source is None else source
        return float(np.mean([float(row[field]) for row in values])) if values else 0.0

    return {
        "condition": condition,
        "num_episodes": len(rows),
        "checkpoint_sha256": summaries[0]["checkpoint_sha256"],
        "actor_state_sha256": summaries[0]["actor_state_sha256"],
        "seeds": [summary["seed"] for summary in summaries],
        "initial_state_signatures": [
            summary["initial_state_signature"] for summary in summaries
        ],
        "success_count": sum(_bool(row["success"]) for row in rows),
        "success_rate": sum(_bool(row["success"]) for row in rows) / len(rows),
        "fall_rate": sum(_bool(row["fell"]) for row in rows) / len(rows),
        "post_intervention_fall_count": sum(
            _bool(row["post_intervention_fall"]) for row in rows
        ),
        "post_intervention_episode_count": len(ever),
        "post_intervention_fall_rate": sum(
            _bool(row["post_intervention_fall"]) for row in rows
        )
        / max(1, len(ever)),
        "intervention_steps_per_riser": sum(
            int(row["intervention_count"]) for row in rows
        )
        / max(1, total_risers),
        "intervention_events_per_riser": sum(
            int(row["intervention_event_count"]) for row in rows
        )
        / max(1, total_risers),
        "mean_qddot_correction_norm": step_mean("mean_qddot_correction_norm"),
        "mean_qddot_correction_jerk": step_mean("mean_qddot_correction_jerk"),
        "mean_foot_forward_acceleration_deviation": step_mean(
            "mean_foot_forward_acceleration_deviation"
        ),
        "mean_foot_vertical_acceleration_change": step_mean(
            "mean_foot_vertical_acceleration_change"
        ),
        "mean_toe_riser_contact_impulse": episode_mean("toe_riser_contact_impulse"),
        "toe_riser_contact_force_peak": max(
            float(row["toe_riser_contact_force_peak"]) for row in rows
        ),
        "unsafe_overlap_steps_per_riser": sum(
            int(row["unsafe_overlap_steps"]) for row in rows
        )
        / max(1, total_risers),
        "post_intervention_mean_abs_root_roll": episode_mean(
            "post_intervention_mean_abs_root_roll", ever
        ),
        "post_intervention_mean_abs_root_pitch": episode_mean(
            "post_intervention_mean_abs_root_pitch", ever
        ),
        "post_intervention_mean_base_angular_velocity": episode_mean(
            "post_intervention_mean_base_angular_velocity", ever
        ),
        "post_intervention_mean_support_foot_slip": episode_mean(
            "post_intervention_mean_support_foot_slip", ever
        ),
        "mean_reached_riser": episode_mean("max_riser"),
        "mean_completion_time_s": episode_mean("completion_time_s"),
        "mean_success_completion_time_s": float(np.mean(successful_times))
        if successful_times
        else None,
        "mean_cbf_compute_time_ms": float(
            np.mean([summary["mean_cbf_compute_time_ms"] for summary in summaries])
        ),
        "maximum_nominal_safe_target_error": max(
            float(row["maximum_nominal_safe_target_error"]) for row in rows
        ),
        "minimum_filtered_hocbf_margin": min(
            float(row["minimum_filtered_hocbf_margin"]) for row in rows
        ),
        "all_hocbf_telemetry_finite": all(
            _bool(row["all_hocbf_telemetry_finite"]) for row in rows
        ),
        "total_steps": total_steps,
        "total_reached_risers": total_risers,
    }


def _run_condition(
    *,
    repo: Path,
    config: Path,
    selected: dict[str, Any],
    checkpoint: Path,
    context: str,
    condition: str,
    mode: str,
    runtime_filter: str,
    episodes: int,
    batch_size: int,
    seed_base: int,
    output: Path,
    device: str,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if episodes % batch_size:
        raise ValueError("v33 evaluation episodes must divide by batch size")
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    for repeat in range(episodes // batch_size):
        seed = seed_base + repeat
        run_dir = output / "raw" / context / condition / f"seed_{seed}"
        summary_path = run_dir / "summary.json"
        rows_path = run_dir / "episodes.csv"
        valid = False
        if resume and summary_path.is_file() and rows_path.is_file():
            prior = json.loads(summary_path.read_text())
            valid = (
                prior.get("seed") == seed
                and prior.get("num_episodes") == batch_size
                and prior.get("checkpoint_sha256") == file_sha256(checkpoint)
                and prior.get("runtime_filter") == (runtime_filter == "on")
                and prior.get("cbf", {}).get("mode") == mode
            )
        if not valid:
            command = [
                sys.executable,
                str(repo / "experiments/scripts/evaluate_hocbf_v33.py"),
                "--repo",
                str(repo),
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--context",
                context,
                "--policy-label",
                condition,
                "--cbf-mode",
                mode,
                "--runtime-filter",
                runtime_filter,
                "--num-envs",
                str(batch_size),
                "--num-episodes",
                str(batch_size),
                "--seed",
                str(seed),
                "--device",
                device,
                "--output-json",
                str(summary_path),
                "--output-csv",
                str(rows_path),
            ]
            if mode == HOCBF_MODE:
                command.extend(
                    [
                        "--omega",
                        str(selected["omega"]),
                        "--lambda-x",
                        str(selected["lambda_x"]),
                        "--lambda-s",
                        str(selected["lambda_s"]),
                    ]
                )
            completed = subprocess.run(
                command, cwd=repo, capture_output=True, text=True, check=False
            )
            if completed.returncode:
                diagnostic = "\n".join(
                    (completed.stdout + completed.stderr).splitlines()[-140:]
                )
                raise RuntimeError(
                    f"v33 audit evaluation failed for {context}/{condition}:\n{diagnostic}"
                )
        summaries.append(json.loads(summary_path.read_text()))
        batch_rows = _read_csv(rows_path)
        if len(batch_rows) != batch_size:
            raise RuntimeError("v33 audit batch is incomplete")
        rows.extend(batch_rows)
        print(
            json.dumps(
                {
                    "condition": condition,
                    "context": context,
                    "batch": f"{repeat + 1}/{episodes // batch_size}",
                    "success": summaries[-1]["success_rate"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    rows.sort(key=_identity)
    return _aggregate(condition=condition, summaries=summaries, rows=rows), rows


def _assert_paired(
    conditions: dict[str, dict[str, Any]], rows: dict[str, list[dict[str, str]]]
) -> None:
    identities = [[_identity(row) for row in value] for value in rows.values()]
    signatures = [value["initial_state_signatures"] for value in conditions.values()]
    if any(value != identities[0] for value in identities[1:]) or any(
        value != signatures[0] for value in signatures[1:]
    ):
        raise RuntimeError("v33 audit conditions are not paired")


def _paired_effect(
    off: list[dict[str, str]], on: list[dict[str, str]]
) -> dict[str, Any]:
    if [_identity(row) for row in off] != [_identity(row) for row in on]:
        raise RuntimeError("v33 paired effect identities differ")
    off_success = [_bool(row["success"]) for row in off]
    on_success = [_bool(row["success"]) for row in on]
    rescue = sum(
        (not before) and after
        for before, after in zip(off_success, on_success, strict=True)
    )
    interference = sum(
        before and (not after)
        for before, after in zip(off_success, on_success, strict=True)
    )
    return {
        "paired_episodes": len(off),
        "rescue_count": rescue,
        "rescue_rate": rescue / len(off),
        "interference_count": interference,
        "interference_rate": interference / len(off),
        "net_success_change": (rescue - interference) / len(off),
    }


def _paired_ci(
    baseline: list[dict[str, str]],
    final: list[dict[str, str]],
    *,
    seed: int,
) -> dict[str, Any]:
    delta = np.asarray(
        [
            float(_bool(after["success"])) - float(_bool(before["success"]))
            for before, after in zip(baseline, final, strict=True)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sample = delta[
        rng.integers(0, len(delta), size=(BOOTSTRAP_SAMPLES, len(delta)))
    ].mean(axis=1)
    lower, upper = np.quantile(sample, (0.025, 0.975))
    return {
        "paired_count": len(delta),
        "point_estimate": float(delta.mean()),
        "paired_bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": seed,
        "used_as_gate": False,
    }


def _flat_condition_row(
    *,
    phase: str,
    source_context: str,
    evaluation_context: str,
    policy: str,
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": phase,
        "source_context": source_context,
        "evaluation_context": evaluation_context,
        "policy": policy,
        **aggregate,
    }


def _frozen(
    args: argparse.Namespace,
    repo: Path,
    config: Path,
    selected: dict[str, Any],
    v31_root: Path,
    output: Path,
) -> None:
    csv_rows: list[dict[str, Any]] = []
    for policy in ("A1", "A2"):
        result: dict[str, Any] = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "policy": f"v31_{policy}_round_8",
            "episodes_per_condition_context": FROZEN_POLICY_EPISODES,
            "contexts": {},
        }
        for context in FORMAL_CONTEXTS:
            checkpoint = _v31_checkpoint(v31_root, context, policy)
            conditions: dict[str, dict[str, Any]] = {}
            rows: dict[str, list[dict[str, str]]] = {}
            seed = frozen_audit_seed(policy, context)
            definitions = (
                ("off", HOCBF_MODE, "off"),
                ("current", CURRENT_CBF_MODE, "on"),
                ("new_hocbf", HOCBF_MODE, "on"),
            )
            for condition, mode, runtime in definitions:
                conditions[condition], rows[condition] = _run_condition(
                    repo=repo,
                    config=config,
                    selected=selected,
                    checkpoint=checkpoint,
                    context=context,
                    condition=f"{policy}_{condition}",
                    mode=mode,
                    runtime_filter=runtime,
                    episodes=FROZEN_POLICY_EPISODES,
                    batch_size=args.eval_batch_size,
                    seed_base=seed,
                    output=output / policy,
                    device=args.device,
                    resume=args.resume,
                )
                csv_rows.append(
                    _flat_condition_row(
                        phase="frozen_policy_audit",
                        source_context=context,
                        evaluation_context=context,
                        policy=policy,
                        aggregate=conditions[condition],
                    )
                )
            _assert_paired(conditions, rows)
            result["contexts"][context] = {
                "conditions": conditions,
                "effects": {
                    "current_vs_off": _paired_effect(rows["off"], rows["current"]),
                    "new_hocbf_vs_off": _paired_effect(rows["off"], rows["new_hocbf"]),
                    "new_hocbf_vs_current_success": _paired_ci(
                        rows["current"],
                        rows["new_hocbf"],
                        seed=bootstrap_seed(context, 10 if policy == "A1" else 20),
                    ),
                },
            }
        result["three_context_means"] = {
            condition: {
                field: float(
                    np.mean(
                        [
                            result["contexts"][context]["conditions"][condition][field]
                            for context in FORMAL_CONTEXTS
                        ]
                    )
                )
                for field in (
                    "success_rate",
                    "post_intervention_fall_rate",
                    "intervention_events_per_riser",
                    "mean_qddot_correction_norm",
                    "mean_qddot_correction_jerk",
                    "unsafe_overlap_steps_per_riser",
                    "mean_cbf_compute_time_ms",
                )
            }
            for condition in ("off", "current", "new_hocbf")
        }
        _atomic_json(output / f"{policy}_results.json", result)
    _write_csv(output / "condition_results.csv", csv_rows)


def _final(
    args: argparse.Namespace,
    repo: Path,
    config: Path,
    selected: dict[str, Any],
    v31_root: Path,
    output: Path,
) -> None:
    if args.base_checkpoint is None or args.training_root is None:
        raise ValueError("v33 final audit requires base checkpoint and training root")
    base = args.base_checkpoint.resolve()
    training = args.training_root.resolve()
    combined: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "selected_hocbf": selected,
        "target_episodes_per_condition": FINAL_TARGET_EPISODES,
        "D0_episodes_per_condition": FINAL_D0_EPISODES,
        "contexts": {},
        "selection_or_gate": False,
    }
    csv_rows: list[dict[str, Any]] = []
    for context in FORMAL_CONTEXTS:
        trained = training / context / "round_08.pt"
        if not trained.is_file():
            raise FileNotFoundError(trained)
        checkpoints = {
            "base_current": base,
            "v31_A2_current": _v31_checkpoint(v31_root, context, "A2"),
            "v31_A2_new_hocbf": _v31_checkpoint(v31_root, context, "A2"),
            "trained_A2_new_hocbf": trained,
            "A1_off": _v31_checkpoint(v31_root, context, "A1"),
            "A1_current": _v31_checkpoint(v31_root, context, "A1"),
            "A1_new_hocbf": _v31_checkpoint(v31_root, context, "A1"),
        }
        definitions = {
            "base_current": (CURRENT_CBF_MODE, "on"),
            "v31_A2_current": (CURRENT_CBF_MODE, "on"),
            "v31_A2_new_hocbf": (HOCBF_MODE, "on"),
            "trained_A2_new_hocbf": (HOCBF_MODE, "on"),
            "A1_off": (HOCBF_MODE, "off"),
            "A1_current": (CURRENT_CBF_MODE, "on"),
            "A1_new_hocbf": (HOCBF_MODE, "on"),
        }
        domains: dict[str, Any] = {}
        domain_rows: dict[str, dict[str, list[dict[str, str]]]] = {}
        for domain, eval_context, episodes, seed in (
            (
                "target",
                context,
                FINAL_TARGET_EPISODES,
                final_target_seed(context, context),
            ),
            ("D0", "D0", FINAL_D0_EPISODES, final_d0_seed(context)),
        ):
            aggregates: dict[str, dict[str, Any]] = {}
            rows: dict[str, list[dict[str, str]]] = {}
            for condition, (mode, runtime) in definitions.items():
                aggregates[condition], rows[condition] = _run_condition(
                    repo=repo,
                    config=config,
                    selected=selected,
                    checkpoint=checkpoints[condition],
                    context=eval_context,
                    condition=condition,
                    mode=mode,
                    runtime_filter=runtime,
                    episodes=episodes,
                    batch_size=min(args.eval_batch_size, episodes),
                    seed_base=seed,
                    output=output / context / domain,
                    device=args.device,
                    resume=args.resume,
                )
                csv_rows.append(
                    _flat_condition_row(
                        phase=f"final_{domain}",
                        source_context=context,
                        evaluation_context=eval_context,
                        policy=condition,
                        aggregate=aggregates[condition],
                    )
                )
            _assert_paired(aggregates, rows)
            effects = {
                "v31_new_vs_v31_current": _paired_ci(
                    rows["v31_A2_current"],
                    rows["v31_A2_new_hocbf"],
                    seed=bootstrap_seed(context, 31 if domain == "target" else 41),
                ),
                "trained_new_vs_v31_current": _paired_ci(
                    rows["v31_A2_current"],
                    rows["trained_A2_new_hocbf"],
                    seed=bootstrap_seed(context, 32 if domain == "target" else 42),
                ),
                "A1_current_vs_off": _paired_effect(rows["A1_off"], rows["A1_current"]),
                "A1_new_vs_off": _paired_effect(rows["A1_off"], rows["A1_new_hocbf"]),
                "A1_new_vs_current": _paired_ci(
                    rows["A1_current"],
                    rows["A1_new_hocbf"],
                    seed=bootstrap_seed(context, 33 if domain == "target" else 43),
                ),
            }
            domains[domain] = {"conditions": aggregates, "effects": effects}
            domain_rows[domain] = rows
            _atomic_json(output / context / f"{domain}_results.json", domains[domain])
        combined["contexts"][context] = domains
    conditions = (
        "base_current",
        "v31_A2_current",
        "v31_A2_new_hocbf",
        "trained_A2_new_hocbf",
        "A1_off",
        "A1_current",
        "A1_new_hocbf",
    )
    combined["three_context_target_means"] = {
        condition: {
            field: float(
                np.mean(
                    [
                        combined["contexts"][context]["target"]["conditions"][
                            condition
                        ][field]
                        for context in FORMAL_CONTEXTS
                    ]
                )
            )
            for field in (
                "success_rate",
                "post_intervention_fall_rate",
                "intervention_events_per_riser",
                "mean_qddot_correction_norm",
                "mean_qddot_correction_jerk",
                "unsafe_overlap_steps_per_riser",
                "mean_reached_riser",
                "mean_completion_time_s",
                "mean_cbf_compute_time_ms",
            )
        }
        for condition in conditions
    }
    means = combined["three_context_target_means"]
    combined["headline"] = {
        "highest_mean_success_condition": max(
            (condition for condition in conditions[:4]),
            key=lambda condition: means[condition]["success_rate"],
        ),
        "v31_A2_new_minus_current": means["v31_A2_new_hocbf"]["success_rate"]
        - means["v31_A2_current"]["success_rate"],
        "trained_new_minus_v31_current": means["trained_A2_new_hocbf"]["success_rate"]
        - means["v31_A2_current"]["success_rate"],
        "A1_current_on_minus_off": means["A1_current"]["success_rate"]
        - means["A1_off"]["success_rate"],
        "A1_new_on_minus_off": means["A1_new_hocbf"]["success_rate"]
        - means["A1_off"]["success_rate"],
        "breaks_72_percent_plateau": max(
            means[condition]["success_rate"] for condition in conditions[:4]
        )
        > 0.72,
    }
    combined["complete"] = True
    _atomic_json(output / "combined_results.json", combined)
    _write_csv(output / "condition_results.csv", csv_rows)


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    config_path = args.config.resolve()
    selected_path = args.selected.resolve()
    v31_root = args.v31_root.resolve()
    output = args.output_root.resolve()
    config = json.loads(config_path.read_text())
    selected = json.loads(selected_path.read_text())
    if (
        config.get("protocol_id") != PROTOCOL_ID
        or selected.get("protocol_id") != PROTOCOL_ID
    ):
        raise RuntimeError("v33 audit inputs differ")
    if args.eval_batch_size < 1 or any(
        value % args.eval_batch_size
        for value in (FROZEN_POLICY_EPISODES, FINAL_TARGET_EPISODES, FINAL_D0_EPISODES)
    ):
        raise ValueError("v33 audit batch size must divide 256 and 512")
    if args.phase == "frozen":
        _frozen(args, repo, config_path, selected, v31_root, output)
    else:
        _final(args, repo, config_path, selected, v31_root, output)


if __name__ == "__main__":
    main()
