"""Run the single held-out paired v34 target and D0 audit."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from proximal_v23_io import file_sha256
from velocity_cbf_v34_protocol import (
    CURRENT_CBF_MODE,
    FORMAL_CONTEXTS,
    OPTIMIZED_CBF_MODE,
    PROTOCOL_ID,
    V31_CHECKPOINT_SHA256,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--v31-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: str | bool) -> bool:
    return value if isinstance(value, bool) else value.lower() == "true"


def _identity(row: dict[str, str]) -> tuple[int, int]:
    return int(row["evaluation_seed"]), int(row["environment_id"])


def _checkpoint_v31(root: Path, context: str) -> Path:
    checkpoint = root / context / "A2" / "round_08.pt"
    if file_sha256(checkpoint) != V31_CHECKPOINT_SHA256[context]["A2"]:
        raise RuntimeError(f"v34 final v31 checkpoint differs for {context}")
    return checkpoint


def _aggregate(
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

    def episode_mean(field: str, source: list[dict[str, str]] | None = None) -> float:
        values = rows if source is None else source
        return float(np.mean([float(row[field]) for row in values])) if values else 0.0

    def step_mean(field: str) -> float:
        return sum(float(row[field]) * int(row["steps"]) for row in rows) / max(
            1, total_steps
        )

    compute = [
        float(summary["mean_cbf_compute_time_ms"])
        for summary in summaries
        if summary.get("mean_cbf_compute_time_ms") is not None
    ]
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
        "mean_return": episode_mean("return"),
        "mean_reached_riser": episode_mean("max_riser"),
        "mean_completion_time_s": episode_mean("completion_time_s"),
        "mean_success_completion_time_s": (
            float(np.mean(successful_times)) if successful_times else None
        ),
        "intervention_steps_per_riser": sum(
            int(row["intervention_count"]) for row in rows
        )
        / max(1, total_risers),
        "intervention_events_per_riser": sum(
            int(row["intervention_event_count"]) for row in rows
        )
        / max(1, total_risers),
        "mean_velocity_correction_norm": step_mean("mean_velocity_correction_norm"),
        "mean_velocity_correction_jerk": step_mean("mean_velocity_correction_jerk"),
        "mean_toe_riser_contact_impulse": episode_mean("toe_riser_contact_impulse"),
        "toe_riser_contact_force_peak": max(
            float(row["toe_riser_contact_force_peak"]) for row in rows
        ),
        "unsafe_overlap_steps_per_riser": sum(
            int(row["unsafe_overlap_steps"]) for row in rows
        )
        / max(1, total_risers),
        "post_intervention_fall_rate": sum(
            _bool(row["post_intervention_fall"]) for row in rows
        )
        / max(1, len(ever)),
        "post_intervention_episode_count": len(ever),
        "post_intervention_mean_abs_root_roll": episode_mean(
            "post_intervention_mean_abs_root_roll", ever
        ),
        "post_intervention_mean_abs_root_pitch": episode_mean(
            "post_intervention_mean_abs_root_pitch", ever
        ),
        "post_intervention_mean_support_foot_slip": episode_mean(
            "post_intervention_mean_support_foot_slip", ever
        ),
        "minimum_filtered_margin": min(
            float(row["minimum_filtered_margin"]) for row in rows
        ),
        "maximum_nominal_safe_target_error": max(
            float(row["maximum_nominal_safe_target_error"]) for row in rows
        ),
        "all_finite": all(_bool(row["all_finite"]) for row in rows),
        "mean_cbf_compute_time_ms": float(np.mean(compute)) if compute else None,
        "total_steps": total_steps,
        "total_reached_risers": total_risers,
    }


def _evaluate_batches(
    *,
    repo: Path,
    config: Path,
    selected: dict[str, Any],
    checkpoint: Path,
    evaluation_context: str,
    source_context: str,
    condition: str,
    mode: str,
    runtime_filter: bool,
    episodes: int,
    batch_size: int,
    seed_base: int,
    output: Path,
    device: str,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if episodes % batch_size:
        raise ValueError("v34 final episode count must divide by batch size")
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, str]] = []
    for repeat in range(episodes // batch_size):
        seed = seed_base + repeat
        run_dir = (
            output
            / "raw"
            / source_context
            / evaluation_context
            / condition
            / f"seed_{seed}"
        )
        summary_path = run_dir / "summary.json"
        episodes_path = run_dir / "episodes.csv"
        valid = False
        if resume and summary_path.is_file() and episodes_path.is_file():
            prior = json.loads(summary_path.read_text())
            valid = (
                prior.get("protocol_id") == PROTOCOL_ID
                and prior.get("seed") == seed
                and prior.get("num_episodes") == batch_size
                and prior.get("checkpoint_sha256") == file_sha256(checkpoint)
                and prior.get("runtime_filter") is runtime_filter
                and prior.get("cbf", {}).get("mode") == mode
            )
        if not valid:
            command = [
                sys.executable,
                str(repo / "experiments/scripts/evaluate_velocity_cbf_v34.py"),
                "--repo",
                str(repo),
                "--search-config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--context",
                evaluation_context,
                "--cbf-mode",
                mode,
                "--runtime-filter",
                "on" if runtime_filter else "off",
                "--num-envs",
                str(batch_size),
                "--num-episodes",
                str(batch_size),
                "--seed",
                str(seed),
                "--policy-label",
                condition,
                "--candidate",
                selected["candidate"] if mode == OPTIMIZED_CBF_MODE else "c000_current",
                "--device",
                device,
                "--output-json",
                str(summary_path),
                "--output-csv",
                str(episodes_path),
            ]
            if mode == OPTIMIZED_CBF_MODE:
                command.extend(
                    ["--parameters-json", json.dumps(selected["parameters"])]
                )
            completed = subprocess.run(
                command, cwd=repo, capture_output=True, text=True, check=False
            )
            if completed.returncode:
                diagnostic = "\n".join(
                    (completed.stdout + completed.stderr).splitlines()[-140:]
                )
                raise RuntimeError(
                    f"v34 final failed for {source_context}/{evaluation_context}/"
                    f"{condition}:\n{diagnostic}"
                )
        summary = json.loads(summary_path.read_text())
        batch_rows = _read_csv(episodes_path)
        if len(batch_rows) != batch_size:
            raise RuntimeError("v34 final evaluation batch is incomplete")
        summaries.append(summary)
        rows.extend(batch_rows)
        print(
            json.dumps(
                {
                    "phase": "held_out_final",
                    "source_context": source_context,
                    "evaluation_context": evaluation_context,
                    "condition": condition,
                    "batch": f"{repeat + 1}/{episodes // batch_size}",
                    "success": summary["success_rate"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    rows.sort(key=_identity)
    return _aggregate(condition, summaries, rows), rows


def _assert_paired(
    aggregates: dict[str, dict[str, Any]], rows: dict[str, list[dict[str, str]]]
) -> None:
    identities = [[_identity(row) for row in values] for values in rows.values()]
    signatures = [value["initial_state_signatures"] for value in aggregates.values()]
    if any(value != identities[0] for value in identities[1:]):
        raise RuntimeError("v34 final conditions do not share identities")
    if any(value != signatures[0] for value in signatures[1:]):
        raise RuntimeError("v34 final conditions do not share initial signatures")


def _paired_effect(
    baseline: list[dict[str, str]], comparison: list[dict[str, str]], seed: int
) -> dict[str, Any]:
    if [_identity(row) for row in baseline] != [_identity(row) for row in comparison]:
        raise RuntimeError("v34 paired comparison identities differ")
    before = np.asarray([_bool(row["success"]) for row in baseline], dtype=np.float64)
    after = np.asarray([_bool(row["success"]) for row in comparison], dtype=np.float64)
    delta = after - before
    rescue = int(np.sum((before == 0.0) & (after == 1.0)))
    interference = int(np.sum((before == 1.0) & (after == 0.0)))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(2000, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    return {
        "paired_episodes": len(delta),
        "success_delta": float(delta.mean()),
        "success_delta_pp": float(100.0 * delta.mean()),
        "bootstrap_95_percent_CI": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "rescue_count": rescue,
        "rescue_rate": rescue / len(delta),
        "interference_count": interference,
        "interference_rate": interference / len(delta),
    }


def _condition_spec(
    *,
    base_checkpoint: Path,
    v31_checkpoint: Path,
    trained_checkpoint: Path,
    selected_mode: str,
) -> dict[str, tuple[Path, str, bool]]:
    return {
        "base_current": (base_checkpoint, CURRENT_CBF_MODE, True),
        "v31_A2_current": (v31_checkpoint, CURRENT_CBF_MODE, True),
        "v31_A2_optimized": (v31_checkpoint, selected_mode, True),
        "trained_A2_optimized": (trained_checkpoint, selected_mode, True),
        "v31_A2_off": (v31_checkpoint, CURRENT_CBF_MODE, False),
        "trained_A2_off": (trained_checkpoint, selected_mode, False),
    }


def _mean_conditions(contexts: dict[str, Any], section: str) -> dict[str, Any]:
    fields = (
        "success_rate",
        "fall_rate",
        "mean_return",
        "mean_reached_riser",
        "mean_completion_time_s",
        "intervention_steps_per_riser",
        "intervention_events_per_riser",
        "mean_velocity_correction_norm",
        "mean_velocity_correction_jerk",
        "mean_toe_riser_contact_impulse",
        "unsafe_overlap_steps_per_riser",
        "post_intervention_fall_rate",
        "post_intervention_mean_support_foot_slip",
        "mean_cbf_compute_time_ms",
    )
    conditions = contexts[FORMAL_CONTEXTS[0]][section]["conditions"]
    output: dict[str, Any] = {}
    for condition in conditions:
        output[condition] = {}
        for field in fields:
            values = [
                contexts[context][section]["conditions"][condition].get(field)
                for context in FORMAL_CONTEXTS
            ]
            finite = [float(value) for value in values if value is not None]
            output[condition][field] = float(np.mean(finite)) if finite else None
    return output


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v34 held-out audit requires the frozen clean commit")
    config_path = args.search_config.resolve()
    selected_path = args.selected.resolve()
    config = json.loads(config_path.read_text())
    selected = json.loads(selected_path.read_text())
    if (
        config.get("protocol_id") != PROTOCOL_ID
        or selected.get("protocol_id") != PROTOCOL_ID
    ):
        raise RuntimeError("v34 final inputs differ")
    if selected.get("status") != "globally_selected_and_final_test_frozen":
        raise RuntimeError("v34 final parameters/identities are not frozen")
    output = args.output_root.resolve()
    marker = output / "execution_started.json"
    if marker.exists() and not args.resume:
        raise RuntimeError("v34 final audit was already invoked")
    if not marker.exists():
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            marker,
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "status": "single_final_invocation_started",
                "selected_sha256": file_sha256(selected_path),
                "git_commit": _git(repo, "rev-parse", "HEAD"),
                "started_unix_seconds": time.time(),
            },
        )
    else:
        prior = json.loads(marker.read_text())
        if prior.get("selected_sha256") != file_sha256(selected_path):
            raise RuntimeError("v34 final resume selection differs")
    contexts: dict[str, Any] = {}
    base_checkpoint = args.base_checkpoint.resolve()
    for context_index, source_context in enumerate(FORMAL_CONTEXTS):
        v31_checkpoint = _checkpoint_v31(args.v31_root.resolve(), source_context)
        trained_record = selected["trained_checkpoints"][source_context]
        trained_checkpoint = Path(trained_record["external_path"])
        if file_sha256(trained_checkpoint) != trained_record["sha256"]:
            raise RuntimeError(
                f"v34 final trained checkpoint differs for {source_context}"
            )
        specifications = _condition_spec(
            base_checkpoint=base_checkpoint,
            v31_checkpoint=v31_checkpoint,
            trained_checkpoint=trained_checkpoint,
            selected_mode=str(selected["mode"]),
        )
        context_payload: dict[str, Any] = {}
        for section, evaluation_context, episodes, batch_size, seed_group in (
            ("target", source_context, 512, 256, "target"),
            ("D0", "D0", 256, 256, "D0"),
        ):
            aggregates: dict[str, dict[str, Any]] = {}
            episode_rows: dict[str, list[dict[str, str]]] = {}
            seed_base = int(
                selected["held_out_identity_seeds"][seed_group][source_context]
            )
            for condition, (checkpoint, mode, runtime_filter) in specifications.items():
                aggregate, rows = _evaluate_batches(
                    repo=repo,
                    config=config_path,
                    selected=selected,
                    checkpoint=checkpoint,
                    evaluation_context=evaluation_context,
                    source_context=source_context,
                    condition=condition,
                    mode=mode,
                    runtime_filter=runtime_filter,
                    episodes=episodes,
                    batch_size=batch_size,
                    seed_base=seed_base,
                    output=output,
                    device=args.device,
                    resume=args.resume,
                )
                aggregates[condition] = aggregate
                episode_rows[condition] = rows
            _assert_paired(aggregates, episode_rows)
            effects = {
                "v31_optimized_minus_v31_current": _paired_effect(
                    episode_rows["v31_A2_current"],
                    episode_rows["v31_A2_optimized"],
                    34_100 + 10 * context_index + (0 if section == "target" else 1),
                ),
                "trained_optimized_minus_v31_current": _paired_effect(
                    episode_rows["v31_A2_current"],
                    episode_rows["trained_A2_optimized"],
                    34_200 + 10 * context_index + (0 if section == "target" else 1),
                ),
                "v31_current_on_minus_off": _paired_effect(
                    episode_rows["v31_A2_off"],
                    episode_rows["v31_A2_current"],
                    34_300 + 10 * context_index + (0 if section == "target" else 1),
                ),
                "trained_optimized_on_minus_off": _paired_effect(
                    episode_rows["trained_A2_off"],
                    episode_rows["trained_A2_optimized"],
                    34_400 + 10 * context_index + (0 if section == "target" else 1),
                ),
            }
            context_payload[section] = {
                "source_context": source_context,
                "evaluation_context": evaluation_context,
                "paired_identity_seed_base": seed_base,
                "conditions": aggregates,
                "paired_effects": effects,
            }
        contexts[source_context] = context_payload
        _atomic_json(
            output / source_context / "target_results.json", context_payload["target"]
        )
        _atomic_json(output / source_context / "D0_results.json", context_payload["D0"])
    target_means = _mean_conditions(contexts, "target")
    d0_means = _mean_conditions(contexts, "D0")
    baseline = float(target_means["v31_A2_current"]["success_rate"])
    direct = float(target_means["v31_A2_optimized"]["success_rate"])
    trained = float(target_means["trained_A2_optimized"]["success_rate"])
    combined = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "single_held_out_final_audit_complete",
        "selected_candidate": selected["candidate"],
        "selected_parameters": selected["parameters"],
        "contexts": contexts,
        "three_context_target_means": target_means,
        "three_source_context_D0_means": d0_means,
        "main_table": [
            {
                "method": "v31 A2 + current CBF",
                **{
                    context: contexts[context]["target"]["conditions"][
                        "v31_A2_current"
                    ]["success_rate"]
                    for context in FORMAL_CONTEXTS
                },
                "mean_success": baseline,
            },
            {
                "method": "v31 A2 + optimized CBF",
                **{
                    context: contexts[context]["target"]["conditions"][
                        "v31_A2_optimized"
                    ]["success_rate"]
                    for context in FORMAL_CONTEXTS
                },
                "mean_success": direct,
            },
            {
                "method": "new A2 + optimized CBF",
                **{
                    context: contexts[context]["target"]["conditions"][
                        "trained_A2_optimized"
                    ]["success_rate"]
                    for context in FORMAL_CONTEXTS
                },
                "mean_success": trained,
            },
        ],
        "headline": {
            "v31_current_mean_success": baseline,
            "direct_optimized_mean_success": direct,
            "trained_optimized_mean_success": trained,
            "direct_minus_v31_current": direct - baseline,
            "direct_minus_v31_current_pp": 100.0 * (direct - baseline),
            "trained_minus_v31_current": trained - baseline,
            "trained_minus_v31_current_pp": 100.0 * (trained - baseline),
            "development_target_met": trained - baseline >= 0.03,
        },
        "final_test_invocations": 1,
        "final_parameters_changed_after_test": False,
        "completed_unix_seconds": time.time(),
    }
    _atomic_json(output / "combined_results.json", combined)
    _atomic_json(
        output / "execution_completed.json",
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "combined_results_sha256": file_sha256(output / "combined_results.json"),
            "completed_unix_seconds": time.time(),
        },
    )
    print(json.dumps(combined["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
