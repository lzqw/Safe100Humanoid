"""Run the frozen v32 target/D0 audit without any pass/fail gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from cbf_teacher_v31_protocol import BASE_CHECKPOINT_SHA256
from cbf_teacher_v32_eval_io import (
    assert_paired,
    atomic_json,
    evaluate_condition,
    paired_ci,
    paired_repairs_regressions,
    paired_wide_rows,
    write_csv,
)
from cbf_teacher_v32_protocol import (
    BOOTSTRAP_SEED_BASE,
    CONTINUATION_SCHEDULES,
    FORMAL_BOOTSTRAP_SAMPLES,
    FORMAL_CONTEXTS,
    FORMAL_D0_EPISODES,
    FORMAL_D0_SEED_BASES,
    FORMAL_TARGET_EPISODES,
    FORMAL_TARGET_SEED_BASES,
    MIXED_D0_SEED_BASE,
    MIXED_SCHEDULE,
    PREFERRED_EVAL_BATCH_SIZE,
    PROTOCOL_ID,
    V31_A2_ROUND8_SHA256,
)
from proximal_v23_io import file_sha256

METHODS = ("v31_A2", "LongConstant", "LongDecay", "Mixed")
CONTINUATION_METHODS = ("v31_A2", "LongConstant", "LongDecay")
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
    root: Path,
    *,
    kind: str,
    context: str,
    schedule: str,
    protocol_hash: str,
) -> tuple[Path, dict[str, Any]]:
    directory = (
        root / "continuation" / context / schedule
        if kind == "continuation"
        else root / "mixed" / schedule
    )
    checkpoint = directory / "round_24.pt"
    summary_path = directory / "training_summary.json"
    if not checkpoint.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"missing v32 formal run {directory}")
    summary = json.loads(summary_path.read_text())
    checks = {
        "protocol": summary.get("protocol_id") == PROTOCOL_ID,
        "protocol_hash": summary.get("protocol", {}).get("sha256") == protocol_hash,
        "phase": summary.get("phase") == "formal",
        "kind": summary.get("kind") == kind,
        "context": summary.get("context") == context,
        "schedule": summary.get("schedule") == schedule,
        "rounds": summary.get("rounds_completed")
        == (16 if kind == "continuation" else 24),
        "final_round": summary.get("final_round") == 24,
        "fixed_final": summary.get("final_policy_rule")
        == "unconditional round 24 actor",
        "checkpoint": summary.get("final_checkpoint_sha256") == file_sha256(checkpoint),
        "no_gates": summary.get("performance_selection_count") == 0
        and summary.get("KL_stop_count") == 0
        and summary.get("KL_rollback_count") == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid v32 formal run {directory}: {checks}")
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
    comparisons = []
    for method in ("LongConstant", "LongDecay", "Mixed"):
        for mode in ("on", "off"):
            comparisons.append(
                (
                    f"{method}_{mode}_vs_v31_A2",
                    f"v31_A2_{mode}",
                    f"{method}_{mode}",
                    target,
                    target_rows,
                )
            )
    for method in ("LongConstant", "LongDecay"):
        comparisons.append(
            (
                f"D0_{method}_on_vs_v31_A2",
                "v31_A2_on",
                f"{method}_on",
                d0,
                d0_rows,
            )
        )
    intervals = {}
    repairs = {}
    deltas = {}
    context_offset = FORMAL_CONTEXTS.index(context) * 100
    for index, (name, before, after, aggregates, rows) in enumerate(comparisons):
        intervals[name] = paired_ci(
            rows[before],
            rows[after],
            field="success",
            seed=BOOTSTRAP_SEED_BASE + context_offset + index,
            samples=FORMAL_BOOTSTRAP_SAMPLES,
        )
        repairs[name] = paired_repairs_regressions(rows[before], rows[after])
        deltas[name] = _delta(aggregates[after], aggregates[before], "success_rate")
    dependence = {}
    for method in METHODS:
        on = target[f"{method}_on"]
        off = target[f"{method}_off"]
        dependence[method] = {
            "CBF_on_interventions_per_riser": on["intervention_steps_per_riser"],
            "CBF_on_mean_correction_norm": on["mean_correction_norm"],
            "CBF_off_success": off["success_rate"],
            "CBF_off_would_intervene_fraction": off[
                "counterfactual_would_intervene_fraction"
            ],
            "CBF_off_counterfactual_correction_norm": off[
                "mean_counterfactual_correction_norm"
            ],
            "CBF_off_nominal_violation_steps_per_riser": off[
                "nominal_barrier_violation_steps_per_riser"
            ],
            "CBF_off_kick_events_per_riser": off["toe_riser_kick_events_per_riser"],
            "CBF_off_unsafe_overlap_steps_per_riser": off[
                "unsafe_overlap_steps_per_riser"
            ],
        }
    return {
        "context": context,
        "success_deltas": deltas,
        "paired_success_95_CI": intervals,
        "repairs_and_regressions": repairs,
        "CBF_dependence_and_risk": dependence,
    }


def _mean_metric(
    contexts: dict[str, dict[str, Any]], method: str, mode: str, field: str
) -> float:
    return float(
        np.mean(
            [
                contexts[context]["target"][f"{method}_{mode}"][field]
                for context in FORMAL_CONTEXTS
            ]
        )
    )


def _summary(
    contexts: dict[str, dict[str, Any]], mixed_d0: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    means = {}
    for method in METHODS:
        means[method] = {
            "CBF_on_success": _mean_metric(contexts, method, "on", "success_rate"),
            "CBF_on_fall": _mean_metric(contexts, method, "on", "fall_rate"),
            "CBF_on_return": _mean_metric(contexts, method, "on", "mean_return"),
            "CBF_on_reached_riser": _mean_metric(
                contexts, method, "on", "mean_reached_riser"
            ),
            "CBF_off_success": _mean_metric(contexts, method, "off", "success_rate"),
            "CBF_on_interventions_per_riser": _mean_metric(
                contexts, method, "on", "intervention_steps_per_riser"
            ),
            "CBF_on_correction_norm": _mean_metric(
                contexts, method, "on", "mean_correction_norm"
            ),
            "CBF_off_would_intervene_fraction": _mean_metric(
                contexts, method, "off", "counterfactual_would_intervene_fraction"
            ),
            "CBF_off_nominal_violation_steps_per_riser": _mean_metric(
                contexts,
                method,
                "off",
                "nominal_barrier_violation_steps_per_riser",
            ),
        }
        if method in CONTINUATION_METHODS:
            means[method]["D0_success"] = float(
                np.mean(
                    [
                        contexts[context]["D0"][f"{method}_on"]["success_rate"]
                        for context in FORMAL_CONTEXTS
                    ]
                )
            )
        else:
            means[method]["D0_success"] = mixed_d0["Mixed_on"]["success_rate"]
    direction = {
        method: {
            "per_context_CBF_on_delta_vs_v31_A2": {
                context: _delta(
                    contexts[context]["target"][f"{method}_on"],
                    contexts[context]["target"]["v31_A2_on"],
                    "success_rate",
                )
                for context in FORMAL_CONTEXTS
            }
        }
        for method in ("LongConstant", "LongDecay", "Mixed")
    }
    for method, item in direction.items():
        values = item["per_context_CBF_on_delta_vs_v31_A2"]
        item["positive_contexts"] = sum(value > 0.0 for value in values.values())
        item["nonnegative_contexts"] = sum(value >= 0.0 for value in values.values())
        item["mean_delta"] = (
            means[method]["CBF_on_success"] - means["v31_A2"]["CBF_on_success"]
        )
    highest = max(METHODS, key=lambda item: means[item]["CBF_on_success"])
    return {
        "method_means": means,
        "direction_consistency": direction,
        "highest_mean_CBF_on_success_method": highest,
        "highest_mean_CBF_on_success": means[highest]["CBF_on_success"],
        "LongDecay_minus_LongConstant": {
            key: means["LongDecay"][key] - means["LongConstant"][key]
            for key in means["LongDecay"]
        },
        "mixed_D0_base_success": mixed_d0["base_on"]["success_rate"],
        "mixed_D0_change_vs_base": mixed_d0["Mixed_on"]["success_rate"]
        - mixed_d0["base_on"]["success_rate"],
        "descriptive_only_no_pass_fail_gate": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight-summary", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--v31-formal-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--eval-batch-size", type=int, default=PREFERRED_EVAL_BATCH_SIZE
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    preflight_path = args.preflight_summary.resolve()
    base = args.base_checkpoint.resolve()
    v31_root = args.v31_formal_root.resolve()
    training_root = args.training_root.resolve()
    output = args.output_dir.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("v32 formal audit requires a clean worktree")
    protocol = json.loads(protocol_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status") != "frozen_before_v32_preflight_and_formal"
        or preflight.get("protocol_id") != PROTOCOL_ID
        or not preflight.get("passed")
        or preflight.get("protocol_sha256") != file_sha256(protocol_path)
    ):
        raise RuntimeError("v32 audit requires frozen protocol and passed preflight")
    if file_sha256(base) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("v32 audit base checkpoint differs")
    if (output / "combined_results.json").exists() and not args.resume:
        raise RuntimeError("v32 formal audit already exists")
    output.mkdir(parents=True, exist_ok=True)
    protocol_hash = file_sha256(protocol_path)

    mixed_checkpoint, mixed_summary = _training_checkpoint(
        training_root,
        kind="mixed",
        context="mixed",
        schedule=MIXED_SCHEDULE,
        protocol_hash=protocol_hash,
    )
    context_payloads = {}
    all_rows = []
    for context in FORMAL_CONTEXTS:
        v31_checkpoint = v31_root / context / "A2" / "round_08.pt"
        if file_sha256(v31_checkpoint) != V31_A2_ROUND8_SHA256[context]:
            raise RuntimeError(f"v32 audit v31 input differs for {context}")
        checkpoints = {"v31_A2": v31_checkpoint, "Mixed": mixed_checkpoint}
        training = {"Mixed": mixed_summary}
        for schedule in CONTINUATION_SCHEDULES:
            checkpoint, summary = _training_checkpoint(
                training_root,
                kind="continuation",
                context=context,
                schedule=schedule,
                protocol_hash=protocol_hash,
            )
            checkpoints[schedule] = checkpoint
            training[schedule] = summary
        target = {}
        target_rows = {}
        for method in METHODS:
            for mode in ("off", "on"):
                condition = f"{method}_{mode}"
                target[condition], target_rows[condition] = evaluate_condition(
                    repo=repo,
                    protocol=protocol_path,
                    checkpoint=checkpoints[method],
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
        assert_paired(target, target_rows, label=f"v32 formal/{context}/target")
        d0 = {}
        d0_rows = {}
        for method in CONTINUATION_METHODS:
            condition = f"{method}_on"
            d0[condition], d0_rows[condition] = evaluate_condition(
                repo=repo,
                protocol=protocol_path,
                checkpoint=checkpoints[method],
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
        assert_paired(d0, d0_rows, label=f"v32 formal/{context}/D0")
        effects = _context_effects(context, target, d0, target_rows, d0_rows)
        payload = {
            "context": context,
            "target": target,
            "D0": d0,
            "effects": effects,
            "training": {
                method: {
                    "final_checkpoint_sha256": summary["final_checkpoint_sha256"],
                    "summary_protocol_sha256": summary["protocol"]["sha256"],
                }
                for method, summary in training.items()
            },
            "v31_A2_checkpoint_sha256": file_sha256(v31_checkpoint),
        }
        context_payloads[context] = payload
        context_dir = output / context
        atomic_json(context_dir / "context_results.json", payload)
        condition_rows = [
            _aggregate_row(context, "target", name, value)
            for name, value in target.items()
        ] + [_aggregate_row(context, "D0", name, value) for name, value in d0.items()]
        write_csv(context_dir / "condition_results.csv", condition_rows)
        write_csv(
            context_dir / "paired_episode_metrics.csv",
            paired_wide_rows(domain="target", conditions=target_rows)
            + paired_wide_rows(domain="D0", conditions=d0_rows),
        )
        all_rows.extend(condition_rows)

    mixed_d0 = {}
    mixed_d0_rows = {}
    for condition, checkpoint in (("base_on", base), ("Mixed_on", mixed_checkpoint)):
        mixed_d0[condition], mixed_d0_rows[condition] = evaluate_condition(
            repo=repo,
            protocol=protocol_path,
            checkpoint=checkpoint,
            context="D0",
            condition=f"mixed_{condition}",
            runtime_filter="on",
            episodes=FORMAL_D0_EPISODES,
            batch_size=min(args.eval_batch_size, FORMAL_D0_EPISODES),
            seed_base=MIXED_D0_SEED_BASE,
            output_root=output,
            device=args.device,
            resume=args.resume,
        )
    assert_paired(mixed_d0, mixed_d0_rows, label="v32 formal/mixed/D0")
    mixed_d0_effect = {
        "success_delta_vs_base": _delta(
            mixed_d0["Mixed_on"], mixed_d0["base_on"], "success_rate"
        ),
        "paired_success_95_CI": paired_ci(
            mixed_d0_rows["base_on"],
            mixed_d0_rows["Mixed_on"],
            field="success",
            seed=BOOTSTRAP_SEED_BASE + 500,
            samples=FORMAL_BOOTSTRAP_SAMPLES,
        ),
        "repairs_and_regressions": paired_repairs_regressions(
            mixed_d0_rows["base_on"], mixed_d0_rows["Mixed_on"]
        ),
    }
    mixed_d0_payload = {
        "conditions": mixed_d0,
        "effect": mixed_d0_effect,
    }
    atomic_json(output / "mixed_D0_results.json", mixed_d0_payload)
    write_csv(
        output / "mixed_D0_paired_episode_metrics.csv",
        paired_wide_rows(domain="D0", conditions=mixed_d0_rows),
    )
    all_rows.extend(
        _aggregate_row("mixed", "D0", name, value) for name, value in mixed_d0.items()
    )

    three_context = _summary(context_payloads, mixed_d0)
    combined = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "complete": True,
        "protocol_sha256": protocol_hash,
        "contexts": context_payloads,
        "mixed_D0": mixed_d0_payload,
        "three_context_summary": three_context,
        "all_seven_formal_runs_and_fixed_audits_published_without_result_rerun": True,
        "monitor_or_audit_used_for_policy_selection": False,
        "paired_CI_used_as_gate": False,
        "pass_fail_gate": False,
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
    print(json.dumps(three_context, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
