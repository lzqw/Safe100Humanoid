"""Run the frozen three-context, three-method v31 formal audit once."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from cbf_teacher_v31_eval_io import (
    assert_paired,
    atomic_json,
    evaluate_condition,
    paired_ci,
    paired_repairs_regressions,
    paired_wide_rows,
    write_csv,
)
from cbf_teacher_v31_protocol import (
    BASE_CHECKPOINT_SHA256,
    BOOTSTRAP_SEED_BASE,
    FORMAL_BOOTSTRAP_SAMPLES,
    FORMAL_CONTEXTS,
    FORMAL_D0_EPISODES,
    FORMAL_D0_SEED_BASES,
    FORMAL_TARGET_EPISODES,
    FORMAL_TARGET_SEED_BASES,
    METHOD_ARMS,
    PREFERRED_EVAL_BATCH_SIZE,
    PROTOCOL_ID,
    arm_parameters,
)
from proximal_v23_io import file_sha256

TARGET_CONDITIONS = (
    "base_off",
    "base_on",
    "A0_off",
    "A0_on",
    "A1_off",
    "A1_on",
    "A2_off",
    "A2_on",
)
D0_CONDITIONS = ("base_on", "A0_on", "A1_on", "A2_on")
PUBLISHED_METRICS = (
    "success_rate",
    "fall_rate",
    "mean_return",
    "mean_reached_riser",
    "mean_completion_time_s",
    "mean_success_completion_time_s",
    "intervention_steps_per_riser",
    "counterfactual_would_intervene_fraction",
    "mean_correction_norm",
    "mean_counterfactual_correction_norm",
    "nominal_barrier_violation_steps_per_riser",
    "toe_riser_kick_events_per_riser",
    "unsafe_overlap_steps_per_riser",
    "mean_episode_minimum_nominal_barrier_margin",
    "global_minimum_nominal_barrier_margin",
    "kick_episode_rate",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _training_checkpoint(
    root: Path, context: str, arm: str, protocol_hash: str
) -> tuple[Path, dict[str, Any]]:
    directory = root / context / arm
    checkpoint = directory / "round_08.pt"
    summary_path = directory / "training_summary.json"
    if not checkpoint.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"formal v31 training output missing: {context}/{arm}")
    summary = json.loads(summary_path.read_text())
    checks = {
        "protocol": summary.get("protocol_id") == PROTOCOL_ID,
        "protocol_hash": summary.get("protocol", {}).get("sha256") == protocol_hash,
        "phase": summary.get("phase") == "formal",
        "context": summary.get("context") == context,
        "arm": summary.get("arm") == arm,
        "configuration": summary.get("arm_configuration") == arm_parameters(arm),
        "rounds": summary.get("rounds_completed") == 8,
        "checkpoint": summary.get("final_checkpoint_sha256") == file_sha256(checkpoint),
        "unconditional_final": summary.get("final_policy_rule")
        == "unconditional round 8 actor",
        "no_gates": summary.get("kl_stop_count") == 0
        and summary.get("kl_rollback_count") == 0
        and summary.get("performance_selection_count") == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid formal v31 run {context}/{arm}: {checks}")
    return checkpoint, summary


def _aggregate_row(
    context: str, domain: str, condition: str, value: dict[str, Any]
) -> dict[str, Any]:
    return {
        "context": context,
        "domain": domain,
        "condition": condition,
        "runtime_filter": value["runtime_filter"],
        "episodes": value["num_episodes"],
        **{field: value.get(field) for field in PUBLISHED_METRICS},
        "checkpoint_sha256": value["checkpoint_sha256"],
        "actor_state_sha256": value["actor_state_sha256"],
    }


def _delta(after: dict[str, Any], before: dict[str, Any], field: str) -> float:
    return float(after[field]) - float(before[field])


def _context_effects(
    context: str,
    target: dict[str, dict[str, Any]],
    d0: dict[str, dict[str, Any]],
    target_rows: dict[str, list[dict[str, str]]],
    d0_rows: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    comparisons: dict[str, tuple[str, str, dict[str, list[dict[str, str]]]]] = {}
    for arm in METHOD_ARMS:
        comparisons[f"{arm}_on_vs_base"] = ("base_on", f"{arm}_on", target_rows)
        comparisons[f"{arm}_off_vs_base"] = (
            "base_off",
            f"{arm}_off",
            target_rows,
        )
        comparisons[f"D0_{arm}_on_vs_base"] = ("base_on", f"{arm}_on", d0_rows)
    for arm in ("A1", "A2"):
        comparisons[f"{arm}_on_vs_A0"] = ("A0_on", f"{arm}_on", target_rows)
        comparisons[f"{arm}_off_vs_A0"] = ("A0_off", f"{arm}_off", target_rows)
        comparisons[f"D0_{arm}_on_vs_A0"] = ("A0_on", f"{arm}_on", d0_rows)

    intervals = {}
    repairs = {}
    deltas = {}
    context_offset = FORMAL_CONTEXTS.index(context) * 100
    for index, (name, (before, after, rows)) in enumerate(comparisons.items()):
        intervals[name] = paired_ci(
            rows[before],
            rows[after],
            field="success",
            seed=BOOTSTRAP_SEED_BASE + context_offset + index,
            samples=FORMAL_BOOTSTRAP_SAMPLES,
        )
        repairs[name] = paired_repairs_regressions(rows[before], rows[after])
        domain = d0 if rows is d0_rows else target
        deltas[name] = _delta(domain[after], domain[before], "success_rate")

    internalization = {}
    dependence_and_risk = {}
    for arm in METHOD_ARMS:
        off = target[f"{arm}_off"]
        base_off = target["base_off"]
        internalization[arm] = {
            "off_success_vs_base": _delta(off, base_off, "success_rate"),
            "off_would_intervene_vs_base": _delta(
                off, base_off, "counterfactual_would_intervene_fraction"
            ),
            "off_counterfactual_correction_vs_base": _delta(
                off, base_off, "mean_counterfactual_correction_norm"
            ),
            "off_nominal_violation_vs_base": _delta(
                off, base_off, "nominal_barrier_violation_steps_per_riser"
            ),
            "off_kick_events_vs_base": _delta(
                off, base_off, "toe_riser_kick_events_per_riser"
            ),
            "off_unsafe_overlap_vs_base": _delta(
                off, base_off, "unsafe_overlap_steps_per_riser"
            ),
            "off_minimum_margin_vs_base": _delta(
                off, base_off, "mean_episode_minimum_nominal_barrier_margin"
            ),
        }
        on = target[f"{arm}_on"]
        dependence_and_risk[arm] = {
            "on_interventions_per_riser": on["intervention_steps_per_riser"],
            "off_would_intervene_fraction": off[
                "counterfactual_would_intervene_fraction"
            ],
            "off_counterfactual_correction_norm": off[
                "mean_counterfactual_correction_norm"
            ],
            "off_kick_events_per_riser": off["toe_riser_kick_events_per_riser"],
            "off_unsafe_overlap_steps_per_riser": off["unsafe_overlap_steps_per_riser"],
            "off_nominal_violation_steps_per_riser": off[
                "nominal_barrier_violation_steps_per_riser"
            ],
        }
    return {
        "context": context,
        "success_deltas": deltas,
        "paired_success_95_CI": intervals,
        "repairs_and_regressions": repairs,
        "nominal_internalization": internalization,
        "cbf_dependence_and_risk": dependence_and_risk,
    }


def _mean(contexts: dict[str, dict[str, Any]], arm: str, condition: str) -> float:
    return float(
        np.mean(
            [
                contexts[context]["target"][f"{arm}_{condition}"]["success_rate"]
                for context in FORMAL_CONTEXTS
            ]
        )
    )


def _three_context_summary(contexts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    method_means = {}
    for arm in METHOD_ARMS:
        method_means[arm] = {
            "CBF_on_success": _mean(contexts, arm, "on"),
            "CBF_off_success": _mean(contexts, arm, "off"),
            "D0_success": float(
                np.mean(
                    [
                        contexts[context]["D0"][f"{arm}_on"]["success_rate"]
                        for context in FORMAL_CONTEXTS
                    ]
                )
            ),
            "CBF_on_interventions_per_riser": float(
                np.mean(
                    [
                        contexts[context]["target"][f"{arm}_on"][
                            "intervention_steps_per_riser"
                        ]
                        for context in FORMAL_CONTEXTS
                    ]
                )
            ),
            "CBF_off_would_intervene_fraction": float(
                np.mean(
                    [
                        contexts[context]["target"][f"{arm}_off"][
                            "counterfactual_would_intervene_fraction"
                        ]
                        for context in FORMAL_CONTEXTS
                    ]
                )
            ),
            "CBF_off_kick_events_per_riser": float(
                np.mean(
                    [
                        contexts[context]["target"][f"{arm}_off"][
                            "toe_riser_kick_events_per_riser"
                        ]
                        for context in FORMAL_CONTEXTS
                    ]
                )
            ),
            "CBF_off_unsafe_overlap_steps_per_riser": float(
                np.mean(
                    [
                        contexts[context]["target"][f"{arm}_off"][
                            "unsafe_overlap_steps_per_riser"
                        ]
                        for context in FORMAL_CONTEXTS
                    ]
                )
            ),
        }
    teacher_vs_A0 = {
        arm: {
            field: method_means[arm][field] - method_means["A0"][field]
            for field in method_means[arm]
        }
        for arm in ("A1", "A2")
    }
    highest = max(METHOD_ARMS, key=lambda arm: method_means[arm]["CBF_on_success"])
    base_d0_mean = float(
        np.mean(
            [
                contexts[context]["D0"]["base_on"]["success_rate"]
                for context in FORMAL_CONTEXTS
            ]
        )
    )
    return {
        "method_means": method_means,
        "teacher_minus_A0_means": teacher_vs_A0,
        "highest_mean_CBF_on_success_method": highest,
        "highest_mean_CBF_on_success": method_means[highest]["CBF_on_success"],
        "base_D0_success_mean": base_d0_mean,
        "D0_success_change_vs_base": {
            arm: method_means[arm]["D0_success"] - base_d0_mean for arm in METHOD_ARMS
        },
        "descriptive_only_no_pass_fail_gate": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight-summary", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--eval-batch-size", type=int, default=PREFERRED_EVAL_BATCH_SIZE
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    preflight_path = args.preflight_summary.resolve()
    base = args.base_checkpoint.resolve()
    training_root = args.training_root.resolve()
    output = args.output_dir.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v31 formal audit requires a clean worktree")
    protocol = json.loads(protocol_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    if protocol.get("status") != "frozen_before_v31_preflight_and_formal":
        raise RuntimeError("v31 formal audit requires the frozen protocol")
    if (
        preflight.get("protocol_id") != PROTOCOL_ID
        or not preflight.get("passed")
        or preflight.get("protocol_sha256") != file_sha256(protocol_path)
    ):
        raise RuntimeError("v31 formal audit requires the passed single preflight")
    if file_sha256(base) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v31 formal audit base checkpoint differs")
    if (output / "combined_results.json").exists() and not args.resume:
        raise RuntimeError("v31 formal audit output already exists")
    output.mkdir(parents=True, exist_ok=True)
    protocol_hash = file_sha256(protocol_path)

    checkpoints: dict[str, dict[str, Path]] = {}
    training_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for context in FORMAL_CONTEXTS:
        checkpoints[context] = {}
        training_summaries[context] = {}
        for arm in METHOD_ARMS:
            checkpoint, summary = _training_checkpoint(
                training_root, context, arm, protocol_hash
            )
            checkpoints[context][arm] = checkpoint
            training_summaries[context][arm] = summary

    all_rows = []
    context_payloads = {}
    for context in FORMAL_CONTEXTS:
        checkpoint_by_condition = {
            "base_off": base,
            "base_on": base,
            **{
                f"{arm}_{mode}": checkpoints[context][arm]
                for arm in METHOD_ARMS
                for mode in ("off", "on")
            },
        }
        target_aggregates = {}
        target_rows = {}
        for condition in TARGET_CONDITIONS:
            mode = "off" if condition.endswith("off") else "on"
            target_aggregates[condition], target_rows[condition] = evaluate_condition(
                repo=repo,
                protocol=protocol_path,
                checkpoint=checkpoint_by_condition[condition],
                context=context,
                condition=condition,
                runtime_filter=mode,
                episodes=FORMAL_TARGET_EPISODES,
                batch_size=min(args.eval_batch_size, FORMAL_TARGET_EPISODES),
                seed_base=FORMAL_TARGET_SEED_BASES[context],
                output_root=output,
                device=args.device,
                resume=args.resume,
            )
        assert_paired(target_aggregates, target_rows, label=f"formal/{context}/target")

        d0_checkpoint = {
            "base_on": base,
            **{f"{arm}_on": checkpoints[context][arm] for arm in METHOD_ARMS},
        }
        d0_aggregates = {}
        d0_rows = {}
        for condition in D0_CONDITIONS:
            d0_aggregates[condition], d0_rows[condition] = evaluate_condition(
                repo=repo,
                protocol=protocol_path,
                checkpoint=d0_checkpoint[condition],
                context="D0",
                condition=f"{context}_{condition}",
                runtime_filter="on",
                episodes=FORMAL_D0_EPISODES,
                batch_size=min(args.eval_batch_size, FORMAL_D0_EPISODES),
                seed_base=FORMAL_D0_SEED_BASES[context],
                output_root=output,
                device=args.device,
                resume=args.resume,
            )
        assert_paired(d0_aggregates, d0_rows, label=f"formal/{context}/D0")
        effects = _context_effects(
            context, target_aggregates, d0_aggregates, target_rows, d0_rows
        )
        context_payload = {
            "context": context,
            "methods": list(METHOD_ARMS),
            "target": target_aggregates,
            "D0": d0_aggregates,
            "effects": effects,
            "training": {
                arm: {
                    "summary_sha256": file_sha256(
                        training_root / context / arm / "training_summary.json"
                    ),
                    "final_checkpoint_sha256": training_summaries[context][arm][
                        "final_checkpoint_sha256"
                    ],
                }
                for arm in METHOD_ARMS
            },
        }
        context_payloads[context] = context_payload
        context_dir = output / context
        atomic_json(context_dir / "context_results.json", context_payload)
        condition_rows = [
            _aggregate_row(context, "target", name, target_aggregates[name])
            for name in TARGET_CONDITIONS
        ] + [
            _aggregate_row(context, "D0", name, d0_aggregates[name])
            for name in D0_CONDITIONS
        ]
        write_csv(context_dir / "condition_results.csv", condition_rows)
        write_csv(
            context_dir / "paired_episode_metrics.csv",
            paired_wide_rows(domain="target", conditions=target_rows)
            + paired_wide_rows(domain="D0", conditions=d0_rows),
        )
        all_rows.extend(condition_rows)

    summary = _three_context_summary(context_payloads)
    combined = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "protocol_sha256": protocol_hash,
        "methods": {arm: arm_parameters(arm) for arm in METHOD_ARMS},
        "contexts": context_payloads,
        "three_context_summary": summary,
        "all_nine_formal_runs_and_audits_published_without_result_rerun": True,
        "paired_CI_used_as_gate": False,
        "final_audit_used_for_policy_selection_or_modification": False,
    }
    atomic_json(output / "combined_results.json", combined)
    write_csv(output / "combined_results.csv", all_rows)
    atomic_json(
        output / "formal_audit_complete.json",
        {
            "protocol_id": PROTOCOL_ID,
            "combined_results_sha256": file_sha256(output / "combined_results.json"),
            "combined_results_csv_sha256": file_sha256(output / "combined_results.csv"),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
