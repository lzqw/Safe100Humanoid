"""Resumable successive-halving development driver for v141."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from filter_free_v141_protocol import (
    BASE_ACTOR_LEARNING_RATE,
    DEV_EVALUATION_EPISODES,
    DEV_EVALUATION_SEED,
    DEV_TRAIN_SEED,
    DEVELOPMENT_THRESHOLDS,
    GENERATION_1_CANDIDATES,
    GENERATION_1_ROUNDS,
    METHOD_ID,
    NUM_ENVS,
    PROTOCOL_ID,
    RETENTION_CONTEXT,
    ROLLOUT_STEPS,
    SPECIALISTS,
    candidate_score,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--through-generation",
        type=int,
        default=0,
        help="Maximum generation to run; zero continues until both specialists pass.",
    )
    parser.add_argument(
        "--max-new-jobs",
        type=int,
        default=0,
        help="Stop after this many new subprocess jobs; zero means unlimited.",
    )
    parser.add_argument(
        "--ignore-success-through-generation",
        type=int,
        default=0,
        help=(
            "After an unsuccessful formal attempt, require a newly evaluated "
            "development candidate from a later generation."
        ),
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _new_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "method_id": METHOD_ID,
        "phase": "development",
        "status": "running",
        "current_job": None,
        "completed_jobs": [],
        "failed_jobs": [],
        "generations": {},
        "updated_unix_time": time.time(),
    }


class DevelopmentDriver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = args.repo.resolve()
        self.output = args.output_root.resolve()
        self.state_path = self.output / "run_state.json"
        self.output.mkdir(parents=True, exist_ok=True)
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.is_file()
            else _new_state()
        )
        self.new_jobs = 0

    def save_state(self) -> None:
        self.state["updated_unix_time"] = time.time()
        _atomic_json(self.state_path, self.state)

    def budget_exhausted(self) -> bool:
        return bool(
            self.args.max_new_jobs
            and self.new_jobs >= self.args.max_new_jobs
        )

    def run_job(self, job_id: str, command: list[str], expected: Path) -> bool:
        if expected.is_file():
            return True
        if self.budget_exhausted():
            self.state["status"] = "paused_at_job_budget"
            self.save_state()
            return False
        log_path = self.output / "logs" / f"{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.state["current_job"] = {
            "id": job_id,
            "command": command,
            "log": str(log_path),
            "started_unix_time": time.time(),
        }
        self.state["status"] = "running"
        self.save_state()
        started = time.monotonic()
        with log_path.open("a") as log:
            result = subprocess.run(
                command,
                cwd=self.repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        self.new_jobs += 1
        record = {
            "id": job_id,
            "returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "expected": str(expected),
            "log": str(log_path),
        }
        self.state["current_job"] = None
        if result.returncode or not expected.is_file():
            self.state["failed_jobs"].append(record)
            self.state["status"] = "infrastructure_or_job_failure"
            self.save_state()
            raise RuntimeError(f"v141 job failed: {job_id}; see {log_path}")
        self.state["completed_jobs"].append(record)
        self.save_state()
        return True

    def training_dir(self, generation: int, specialist: str, candidate: str) -> Path:
        return self.output / f"generation_{generation}" / specialist / candidate / "training"

    def evaluation_dir(
        self,
        generation: int,
        specialist: str,
        candidate: str,
        context: str,
        condition: str,
    ) -> Path:
        return (
            self.output
            / f"generation_{generation}"
            / specialist
            / candidate
            / "evaluation"
            / context
            / f"filter_{condition}"
        )

    def train_candidate(
        self, generation: int, specialist: str, configuration: dict[str, Any]
    ) -> Path | None:
        candidate = str(configuration["candidate"])
        directory = self.training_dir(generation, specialist, candidate)
        summary = directory / "training_summary.json"
        command = [
            str(self.args.python),
            "experiments/scripts/train_filter_free_v141.py",
            "--repo",
            str(self.repo),
            "--base-checkpoint",
            str(self.args.base_checkpoint.resolve()),
            "--output-dir",
            str(directory),
            "--specialist",
            specialist,
            "--candidate",
            candidate,
            "--seed",
            str(DEV_TRAIN_SEED + (0 if specialist == "F2" else 100)),
            "--device",
            self.args.device,
            "--rounds",
            str(configuration.get("rounds", GENERATION_1_ROUNDS)),
            "--num-envs",
            str(NUM_ENVS),
            "--rollout-steps",
            str(ROLLOUT_STEPS),
            "--target-fraction",
            str(configuration.get("target_fraction", 0.80)),
            "--intervention-ppo-eta",
            str(configuration["intervention_ppo_eta"]),
            "--correction-weight-mode",
            str(configuration["correction_weight_mode"]),
            "--correction-loss-weight",
            str(configuration["correction_loss_weight"]),
            "--dual-reward-scale",
            str(configuration["dual_reward_scale"]),
            "--actor-learning-rate",
            str(
                BASE_ACTOR_LEARNING_RATE
                * float(configuration.get("actor_learning_rate_multiplier", 1.0))
            ),
            "--moving-kl-beta",
            str(configuration.get("moving_kl_beta", 0.5)),
            "--actor-epochs",
            str(configuration.get("actor_epochs", 2)),
            "--exploration-std",
            str(configuration.get("exploration_std", 0.05)),
        ]
        if directory.exists() and not summary.is_file():
            command.append("--resume")
        job = f"g{generation}_{specialist}_{candidate}_train"
        if not self.run_job(job, command, summary):
            return None
        payload = json.loads(summary.read_text())
        return Path(payload["final_checkpoint"])

    def evaluate(
        self,
        *,
        generation: int,
        specialist: str,
        candidate: str,
        checkpoint: Path,
        context: str,
        condition: str,
    ) -> dict[str, Any] | None:
        directory = self.evaluation_dir(
            generation, specialist, candidate, context, condition
        )
        summary = directory / "summary.json"
        episodes = directory / "episodes.csv"
        command = [
            str(self.args.python),
            "experiments/scripts/evaluate_cbf_teacher_v31.py",
            "--repo",
            str(self.repo),
            "--protocol",
            str(self.args.protocol.resolve()),
            "--checkpoint",
            str(checkpoint),
            "--context",
            context,
            "--runtime-filter",
            condition,
            "--num-envs",
            str(DEV_EVALUATION_EPISODES),
            "--num-episodes",
            str(DEV_EVALUATION_EPISODES),
            "--seed",
            str(DEV_EVALUATION_SEED),
            "--device",
            self.args.device,
            "--instrument-current-velocity-cbf",
            "--output-json",
            str(summary),
            "--output-csv",
            str(episodes),
        ]
        job = f"g{generation}_{specialist}_{candidate}_{context}_{condition}"
        if not self.run_job(job, command, summary):
            return None
        return json.loads(summary.read_text())

    def candidate_result(
        self, generation: int, specialist: str, configuration: dict[str, Any]
    ) -> dict[str, Any] | None:
        candidate = str(configuration["candidate"])
        result_path = (
            self.output
            / f"generation_{generation}"
            / specialist
            / candidate
            / "candidate_result.json"
        )
        if result_path.is_file():
            return json.loads(result_path.read_text())
        checkpoint = self.train_candidate(generation, specialist, configuration)
        if checkpoint is None:
            return None
        target_off = self.evaluate(
            generation=generation,
            specialist=specialist,
            candidate=candidate,
            checkpoint=checkpoint,
            context=specialist,
            condition="off",
        )
        if target_off is None:
            return None
        retention_off = self.evaluate(
            generation=generation,
            specialist=specialist,
            candidate=candidate,
            checkpoint=checkpoint,
            context=RETENTION_CONTEXT,
            condition="off",
        )
        if retention_off is None:
            return None
        target_on = self.evaluate(
            generation=generation,
            specialist=specialist,
            candidate=candidate,
            checkpoint=checkpoint,
            context=specialist,
            condition="on",
        )
        if target_on is None:
            return None
        training = json.loads(
            (self.training_dir(generation, specialist, candidate) / "training_summary.json").read_text()
        )
        last_metrics = training["round_metrics"][-1]["metrics"]
        shield_gap = float(target_on["success_rate"]) - float(target_off["success_rate"])
        result = {
            "generation": generation,
            "specialist": specialist,
            "candidate": candidate,
            "configuration": configuration,
            "checkpoint": str(checkpoint),
            "target_off_success": float(target_off["success_rate"]),
            "target_on_success": float(target_on["success_rate"]),
            "f1_retention_off_success": float(retention_off["success_rate"]),
            "shield_gap": shield_gap,
            "would_intervene_fraction": float(
                target_off["counterfactual_would_intervene_fraction"]
            ),
            "counterfactual_correction_norm": float(
                target_off["mean_counterfactual_correction_norm"]
            ),
            "nominal_violation_steps_per_riser": float(
                target_off["nominal_barrier_violation_steps_per_riser"]
            ),
            "mean_reached_riser": float(target_off["mean_reached_riser"]),
            "fall_rate": float(target_off["fall_rate"]),
            "actor_moving_forward_kl": float(last_metrics["moving_forward_kl"]),
            "development_score": candidate_score(
                target_off=float(target_off["success_rate"]),
                f1_off=float(retention_off["success_rate"]),
                shield_gap=shield_gap,
            ),
        }
        _atomic_json(result_path, result)
        return result

    def baseline(self, context: str, condition: str) -> dict[str, Any] | None:
        directory = self.output / "baseline" / context / f"filter_{condition}"
        summary = directory / "summary.json"
        episodes = directory / "episodes.csv"
        command = [
            str(self.args.python),
            "experiments/scripts/evaluate_cbf_teacher_v31.py",
            "--repo",
            str(self.repo),
            "--protocol",
            str(self.args.protocol.resolve()),
            "--checkpoint",
            str(self.args.base_checkpoint.resolve()),
            "--context",
            context,
            "--runtime-filter",
            condition,
            "--num-envs",
            str(DEV_EVALUATION_EPISODES),
            "--num-episodes",
            str(DEV_EVALUATION_EPISODES),
            "--seed",
            str(DEV_EVALUATION_SEED),
            "--device",
            self.args.device,
            "--instrument-current-velocity-cbf",
            "--output-json",
            str(summary),
            "--output-csv",
            str(episodes),
        ]
        if not self.run_job(f"baseline_{context}_{condition}", command, summary):
            return None
        return json.loads(summary.read_text())

    @staticmethod
    def generation_2_candidates(top_three: list[dict[str, Any]]) -> list[dict[str, Any]]:
        templates = (
            {"target_fraction": 0.80, "actor_learning_rate_multiplier": 1.0, "actor_epochs": 2, "rounds": 4},
            {"target_fraction": 0.75, "actor_learning_rate_multiplier": 0.5, "actor_epochs": 3, "rounds": 4},
            {"target_fraction": 0.67, "actor_learning_rate_multiplier": 1.0, "actor_epochs": 3, "rounds": 6},
            {"target_fraction": 0.80, "actor_learning_rate_multiplier": 2.0, "actor_epochs": 4, "rounds": 6},
        )
        output: list[dict[str, Any]] = []
        for rank, result in enumerate(top_three, 1):
            base = dict(result["configuration"])
            for index, template in enumerate(templates, 1):
                candidate = {
                    **base,
                    **template,
                    "candidate": f"g2_r{rank}_v{index}_{base['candidate']}",
                    "parent_candidate": base["candidate"],
                }
                output.append(candidate)
        return output

    @staticmethod
    def configuration_signature(configuration: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(
            configuration.get(name, default)
            for name, default in (
                ("intervention_ppo_eta", 0.25),
                ("correction_loss_weight", 0.2),
                ("correction_weight_mode", "positive_advantage"),
                ("dual_reward_scale", 0.25),
                ("target_fraction", 0.80),
                ("actor_learning_rate_multiplier", 1.0),
                ("actor_epochs", 2),
                ("rounds", 2),
                ("exploration_std", 0.05),
                ("moving_kl_beta", 0.5),
            )
        )

    @classmethod
    def adaptive_candidates(
        cls,
        generation: int,
        best: dict[str, Any],
        baseline: dict[str, dict[str, Any]],
        tried: set[tuple[Any, ...]],
        *,
        maximum_candidates: int = 4,
    ) -> list[dict[str, Any]]:
        """Apply diagnostic rules A--E, then fill with untried local mutations."""
        base = dict(best["configuration"])
        target_delta = best["target_off_success"] - baseline["target_off"]["success_rate"]
        retention_delta = best["f1_retention_off_success"] - baseline["f1_off"]["success_rate"]
        gap = abs(best["shield_gap"])
        candidates: list[dict[str, Any]] = []
        weight_levels = (0.05, 0.1, 0.2, 0.4)
        eta_levels = (0.0, 0.25, 0.5, 1.0)
        current_weight = float(base["correction_loss_weight"])
        current_eta = float(base["intervention_ppo_eta"])
        parent = str(base["candidate"])

        def add(label: str, **updates: Any) -> None:
            configuration = {
                **base,
                **updates,
                "candidate": f"g{generation}_{label}_{len(candidates) + 1}",
                "parent_candidate": parent,
            }
            signature = cls.configuration_signature(configuration)
            if signature not in tried and all(
                cls.configuration_signature(item) != signature
                for item in candidates
            ):
                candidates.append(configuration)

        baseline_would = float(
            baseline["target_off"]["counterfactual_would_intervene_fraction"]
        )
        # Internalization bottleneck: task performance has transferred, but the
        # nominal actor still asks the runtime CBF to intervene too often. Use
        # the strongest allowed actor update while removing dual-reward
        # dominance; configuration selection still uses only development data.
        if (
            target_delta > 0.0
            and best["would_intervene_fraction"]
            > baseline_would
            * (1.0 - DEVELOPMENT_THRESHOLDS["would_intervene_reduction_fraction"])
        ):
            add(
                "internalize",
                intervention_ppo_eta=eta_levels[
                    max(0, eta_levels.index(current_eta) - 1)
                ],
                correction_weight_mode="positive_advantage",
                correction_loss_weight=weight_levels[
                    min(
                        len(weight_levels) - 1,
                        weight_levels.index(current_weight) + 1,
                    )
                ],
                dual_reward_scale=0.0,
                actor_learning_rate_multiplier=2.0,
                actor_epochs=4,
                rounds=6,
            )
        # A: the policy internalizes interventions but task success falls.
        if best["would_intervene_fraction"] < baseline_would and target_delta < 0.0:
            add(
                "A",
                correction_weight_mode="episode_success_positive_advantage",
                correction_loss_weight=weight_levels[
                    max(0, weight_levels.index(current_weight) - 1)
                ],
                rounds=6,
            )
        # B: target adaptation damages F1 retention.
        if target_delta > 0.0 and retention_delta < -0.015:
            add("B", target_fraction=0.67, moving_kl_beta=1.0, rounds=6)
        # C: the actor barely changes.
        if abs(target_delta) < 0.01 and best["actor_moving_forward_kl"] < 1.0e-4:
            add("C", actor_learning_rate_multiplier=2.0, actor_epochs=4, rounds=6)
        # D: success improves but the policy still depends on the shield.
        if target_delta > 0.0 and gap > 0.02:
            add(
                "D",
                intervention_ppo_eta=eta_levels[
                    max(0, eta_levels.index(current_eta) - 1)
                ],
                correction_loss_weight=weight_levels[
                    min(
                        len(weight_levels) - 1,
                        weight_levels.index(current_weight) + 1,
                    )
                ],
                rounds=6,
            )
        # E: safety-demand metrics improve without a task-success gain.
        baseline_correction = float(
            baseline["target_off"]["mean_counterfactual_correction_norm"]
        )
        baseline_violation = float(
            baseline["target_off"]["nominal_barrier_violation_steps_per_riser"]
        )
        if (
            target_delta <= 0.0
            and best["counterfactual_correction_norm"] < baseline_correction
            and best["nominal_violation_steps_per_riser"] < baseline_violation
        ):
            add(
                "E",
                dual_reward_scale=0.0,
                correction_weight_mode="positive_advantage",
                rounds=6,
            )

        mutations: list[tuple[str, dict[str, Any]]] = []
        for value in eta_levels:
            mutations.append((f"eta{value:g}", {"intervention_ppo_eta": value}))
        for value in weight_levels:
            mutations.append((f"corr{value:g}", {"correction_loss_weight": value}))
        for value in (
            "positive_advantage",
            "episode_success_positive_advantage",
            "intervention_only",
        ):
            mutations.append((f"weighting_{value}", {"correction_weight_mode": value}))
        for value in (0.0, 0.25, 1.0):
            mutations.append((f"dual{value:g}", {"dual_reward_scale": value}))
        for value in (0.80, 0.75, 0.67):
            mutations.append((f"target{value:g}", {"target_fraction": value}))
        for value in (0.5, 1.0, 2.0):
            mutations.append(
                (f"lr{value:g}", {"actor_learning_rate_multiplier": value})
            )
        for value in (2, 3, 4):
            mutations.append((f"epochs{value}", {"actor_epochs": value}))
        for value in (2, 4, 6):
            mutations.append((f"rounds{value}", {"rounds": value}))
        for value in (0.03, 0.05):
            mutations.append((f"std{value:g}", {"exploration_std": value}))
        for value in (0.25, 0.5, 1.0):
            mutations.append((f"kl{value:g}", {"moving_kl_beta": value}))

        for label, updates in mutations:
            if len(candidates) >= maximum_candidates:
                break
            add(label, **updates)

        # Once all single-variable mutations around the current best have been
        # tried, keep the staged search alive with a few untried combinations.
        # This remains successive halving: at most four candidates run in one
        # generation, and the full Cartesian product is never launched.
        domains = (
            ("intervention_ppo_eta", eta_levels),
            ("correction_loss_weight", weight_levels),
            (
                "correction_weight_mode",
                (
                    "positive_advantage",
                    "episode_success_positive_advantage",
                    "intervention_only",
                ),
            ),
            ("dual_reward_scale", (0.0, 0.25, 1.0)),
            ("target_fraction", (0.80, 0.75, 0.67)),
            ("actor_learning_rate_multiplier", (0.5, 1.0, 2.0)),
            ("actor_epochs", (2, 3, 4)),
            ("rounds", (2, 4, 6)),
            ("exploration_std", (0.03, 0.05)),
            ("moving_kl_beta", (0.25, 0.5, 1.0)),
        )
        base_signature = cls.configuration_signature(base)
        for distance in range(2, len(domains) + 1):
            if len(candidates) >= maximum_candidates:
                break
            for values in itertools.product(*(values for _, values in domains)):
                if sum(
                    value != base_value
                    for value, base_value in zip(values, base_signature)
                ) != distance:
                    continue
                add(
                    f"combo{distance}",
                    **{
                        name: value
                        for (name, _), value in zip(domains, values)
                    },
                )
                if len(candidates) >= maximum_candidates:
                    break
        return candidates[:maximum_candidates]

    @staticmethod
    def success_checks(
        result: dict[str, Any], baseline: dict[str, dict[str, Any]]
    ) -> dict[str, bool]:
        target_improvement = (
            result["target_off_success"] - baseline["target_off"]["success_rate"]
        )
        base_gap = abs(
            baseline["target_on"]["success_rate"]
            - baseline["target_off"]["success_rate"]
        )
        gap = abs(result["shield_gap"])
        return {
            "target_off_improves_2pp": target_improvement
            >= DEVELOPMENT_THRESHOLDS["target_off_improvement_pp"] / 100.0,
            "shield_gap_target": gap
            <= DEVELOPMENT_THRESHOLDS["shield_gap_pp"] / 100.0
            or gap
            <= base_gap
            * (1.0 - DEVELOPMENT_THRESHOLDS["shield_gap_reduction_fraction"]),
            "would_intervene_reduced_25pct": result["would_intervene_fraction"]
            <= baseline["target_off"]["counterfactual_would_intervene_fraction"]
            * (1.0 - DEVELOPMENT_THRESHOLDS["would_intervene_reduction_fraction"]),
            "f1_retention_within_1p5pp": result["f1_retention_off_success"]
            >= baseline["f1_off"]["success_rate"]
            - DEVELOPMENT_THRESHOLDS["f1_off_retention_loss_pp"] / 100.0,
        }

    def summarize_generation(
        self,
        generation: int,
        specialist: str,
        results: list[dict[str, Any]],
        baseline: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        ranked = sorted(results, key=lambda item: item["development_score"], reverse=True)
        for result in ranked:
            checks = self.success_checks(result, baseline)
            result["development_success_checks"] = checks
            result["development_success"] = all(checks.values())
        summary = {
            "generation": generation,
            "specialist": specialist,
            "baseline": baseline,
            "candidate_count": len(ranked),
            "ranking": ranked,
            "top_three": ranked[:3],
            "successful_candidates": [
                item["candidate"] for item in ranked if item["development_success"]
            ],
        }
        path = self.output / f"generation_{generation}" / specialist / "development_summary.json"
        _atomic_json(path, summary)
        self.state["generations"].setdefault(str(generation), {})[specialist] = {
            "summary": str(path),
            "candidate_count": len(ranked),
            "successful_candidates": summary["successful_candidates"],
        }
        self.save_state()
        return summary

    def run(self) -> None:
        generation_one = [dict(item, rounds=2, target_fraction=0.80, actor_epochs=2, actor_learning_rate_multiplier=1.0, exploration_std=0.05) for item in GENERATION_1_CANDIDATES]
        summaries: dict[tuple[int, str], dict[str, Any]] = {}
        for specialist in SPECIALISTS:
            results = []
            for configuration in generation_one:
                result = self.candidate_result(1, specialist, configuration)
                if result is None:
                    return
                results.append(result)
            baseline_items = {
                "target_off": self.baseline(specialist, "off"),
                "target_on": self.baseline(specialist, "on"),
                "f1_off": self.baseline(RETENTION_CONTEXT, "off"),
            }
            if any(value is None for value in baseline_items.values()):
                return
            baseline = {key: value for key, value in baseline_items.items() if value is not None}
            summaries[(1, specialist)] = self.summarize_generation(1, specialist, results, baseline)
        if self.args.through_generation == 1:
            self.state["status"] = "generation_1_complete"
            self.save_state()
            return

        for specialist in SPECIALISTS:
            g1 = summaries[(1, specialist)]
            configurations = self.generation_2_candidates(g1["top_three"])
            results = []
            for configuration in configurations:
                result = self.candidate_result(2, specialist, configuration)
                if result is None:
                    return
                results.append(result)
            baseline = g1["baseline"]
            summaries[(2, specialist)] = self.summarize_generation(2, specialist, results, baseline)
        if self.args.through_generation == 2:
            self.state["status"] = "generation_2_complete"
            self.save_state()
            return

        history: dict[str, list[dict[str, Any]]] = {
            specialist: [
                *summaries[(1, specialist)]["ranking"],
                *summaries[(2, specialist)]["ranking"],
            ]
            for specialist in SPECIALISTS
        }
        generation = 3
        success_cutoff = self.args.ignore_success_through_generation
        while True:
            selected: dict[str, dict[str, Any]] = {}
            for specialist in SPECIALISTS:
                successful = [
                    item
                    for item in history[specialist]
                    if item.get("development_success", False)
                    and int(item.get("generation", 0)) > success_cutoff
                ]
                if successful:
                    selected[specialist] = max(
                        successful,
                        key=lambda item: item["development_score"],
                    )
                    continue

                best = max(
                    history[specialist],
                    key=lambda item: item["development_score"],
                )
                tried = {
                    self.configuration_signature(item["configuration"])
                    for item in history[specialist]
                }
                configurations = self.adaptive_candidates(
                    generation,
                    best,
                    summaries[(1, specialist)]["baseline"],
                    tried,
                )
                if not configurations:
                    raise RuntimeError(
                        f"v141 adaptive search exhausted local mutations for {specialist}"
                    )
                results = []
                for configuration in configurations:
                    result = self.candidate_result(
                        generation, specialist, configuration
                    )
                    if result is None:
                        return
                    results.append(result)
                generation_summary = self.summarize_generation(
                    generation,
                    specialist,
                    results,
                    summaries[(1, specialist)]["baseline"],
                )
                summaries[(generation, specialist)] = generation_summary
                history[specialist].extend(generation_summary["ranking"])
                successful = [
                    item
                    for item in history[specialist]
                    if item.get("development_success", False)
                    and int(item.get("generation", 0)) > success_cutoff
                ]
                selected[specialist] = (
                    max(successful, key=lambda item: item["development_score"])
                    if successful
                    else max(
                        history[specialist],
                        key=lambda item: item["development_score"],
                    )
                )

            both_pass = all(
                item.get("development_success", False)
                for item in selected.values()
            )
            development_summary = {
                "protocol_id": PROTOCOL_ID,
                "method_id": METHOD_ID,
                "latest_generation": generation,
                "selected": selected,
                "both_specialists_pass": both_pass,
                "next_phase": (
                    "freeze_and_formal"
                    if both_pass
                    else "continue_mechanism_refinement"
                ),
            }
            _atomic_json(
                self.output / "development_summary.json",
                development_summary,
            )
            self.state["status"] = development_summary["next_phase"]
            self.save_state()
            if both_pass:
                return
            if (
                self.args.through_generation > 0
                and generation >= self.args.through_generation
            ):
                return
            generation += 1


def main() -> None:
    args = _parse_args()
    if args.through_generation < 0:
        raise ValueError("through-generation must be non-negative")
    if args.ignore_success_through_generation < 0:
        raise ValueError("ignore-success-through-generation must be non-negative")
    driver = DevelopmentDriver(args)
    try:
        driver.run()
    except Exception:
        driver.state["status"] = "infrastructure_or_job_failure"
        driver.save_state()
        raise


if __name__ == "__main__":
    main()
