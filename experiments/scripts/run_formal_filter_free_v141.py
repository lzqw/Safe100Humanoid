"""Run the frozen three-seed v141 formal train/evaluation matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cbf_teacher_v31_protocol import PROTOCOL_ID as EVALUATION_PROTOCOL_ID
from filter_free_v141_protocol import (
    BASE_ACTOR_LEARNING_RATE,
    BASE_CHECKPOINT_SHA256,
    FORMAL_EVALUATION_EPISODES,
    FORMAL_EVALUATION_SEED,
    FORMAL_TRAINING_SEEDS,
    METHOD_ID,
    NUM_ENVS,
    PROTOCOL_ID,
    RETENTION_CONTEXT,
    ROLLOUT_STEPS,
    SPECIALISTS,
    correction_loss_weight_for_round,
)
from proximal_v23_io import file_sha256


# Prospectively frozen by the published v140 protocol. Keeping this lightweight
# avoids importing the Isaac task stack in the orchestration process.
V140_TRAINING_SEEDS = (201_357_000, 201_357_001, 201_357_002)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--v140-training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--commit-and-push", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class FormalDriver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = args.repo.resolve()
        self.output = args.output_root.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / "run_state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.is_file()
            else {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "method_id": METHOD_ID,
                "phase": "formal",
                "status": "running",
                "current_job": None,
                "completed_jobs": [],
                "failed_jobs": [],
            }
        )
        self.frozen = json.loads(args.frozen_config.resolve().read_text())

    def save_state(self) -> None:
        self.state["updated_unix_time"] = time.time()
        _atomic_json(self.state_path, self.state)

    def run_job(self, job_id: str, command: list[str], expected: Path) -> None:
        if expected.is_file():
            return
        log = self.output / "logs" / f"{job_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        self.state["current_job"] = {
            "id": job_id,
            "command": command,
            "log": str(log),
            "started_unix_time": time.time(),
        }
        self.state["status"] = "running"
        self.save_state()
        started = time.monotonic()
        with log.open("a") as handle:
            result = subprocess.run(
                command,
                cwd=self.repo,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        record = {
            "id": job_id,
            "returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "expected": str(expected),
            "log": str(log),
        }
        self.state["current_job"] = None
        if result.returncode or not expected.is_file():
            self.state["failed_jobs"].append(record)
            self.state["status"] = "infrastructure_or_job_failure"
            self.save_state()
            raise RuntimeError(f"formal v141 job failed: {job_id}; see {log}")
        self.state["completed_jobs"].append(record)
        self.save_state()

    def validate(self) -> None:
        if file_sha256(self.args.base_checkpoint.resolve()) != BASE_CHECKPOINT_SHA256:
            raise RuntimeError("formal v141 base checkpoint differs from frozen v139")
        expected = {
            "protocol_id": PROTOCOL_ID,
            "method_id": METHOD_ID,
            "status": "frozen_before_formal_execution",
            "frozen_before_formal": True,
            "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
            "formal_training_seeds": list(FORMAL_TRAINING_SEEDS),
            "formal_evaluation_seed": FORMAL_EVALUATION_SEED,
            "formal_evaluation_episodes": FORMAL_EVALUATION_EPISODES,
            "formal_results_seen_before_freeze": False,
        }
        mismatches = {
            key: (self.frozen.get(key), value)
            for key, value in expected.items()
            if self.frozen.get(key) != value
        }
        if mismatches or set(self.frozen.get("specialists", {})) != set(SPECIALISTS):
            raise RuntimeError(
                f"formal v141 frozen configuration differs: {mismatches}"
            )
        source_hashes = self.frozen.get("source_files_sha256", {})
        if not source_hashes:
            raise RuntimeError("formal v141 freeze has no source hash manifest")
        source_mismatches: dict[str, tuple[str | None, str]] = {}
        for relative, expected_sha256 in source_hashes.items():
            path = self.repo / relative
            actual_sha256 = file_sha256(path) if path.is_file() else None
            if actual_sha256 != expected_sha256:
                source_mismatches[relative] = (
                    actual_sha256,
                    expected_sha256,
                )
        if source_mismatches:
            raise RuntimeError(
                f"formal v141 source changed after freeze: {source_mismatches}"
            )
        relative = self.args.frozen_config.resolve().relative_to(self.repo)
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=self.repo,
            check=True,
            capture_output=True,
        ).stdout
        if committed != self.args.frozen_config.resolve().read_bytes():
            raise RuntimeError("formal v141 configuration is not committed")
        ancestry = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                str(self.frozen.get("source_commit")),
                "HEAD",
            ],
            cwd=self.repo,
            check=False,
        )
        if ancestry.returncode != 0:
            raise RuntimeError(
                "formal v141 source commit is not in the current history"
            )
        if subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.repo,
            check=True,
            capture_output=True,
        ).stdout:
            raise RuntimeError("formal v141 execution requires a clean worktree")

    def train_v141(self, specialist: str, training_seed: int) -> tuple[Path, Path]:
        configuration = self.frozen["specialists"][specialist]["configuration"]
        directory = self.output / "training" / specialist / f"seed_{training_seed}"
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
            f"formal_{specialist}",
            "--seed",
            str(training_seed),
            "--device",
            self.args.device,
            "--rounds",
            str(configuration.get("rounds", 2)),
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
            "--correction-loss-weight-schedule",
            str(configuration.get("correction_loss_weight_schedule", "constant")),
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
        self.run_job(f"train_v141_{specialist}_{training_seed}", command, summary)
        training = json.loads(summary.read_text())
        checkpoint = Path(training["final_checkpoint"])
        expected_round = int(configuration.get("rounds", 2))
        expected_target_fraction = (
            round(NUM_ENVS * float(configuration.get("target_fraction", 0.80)))
            / NUM_ENVS
        )
        expected_summary = {
            "method_id": METHOD_ID,
            "candidate": f"formal_{specialist}",
            "specialist": specialist,
            "retention_context": RETENTION_CONTEXT,
            "seed": training_seed,
            "rounds": expected_round,
            "num_envs": NUM_ENVS,
            "rollout_steps": ROLLOUT_STEPS,
            "target_fraction": expected_target_fraction,
            "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
            "fixed_final_round_checkpoint": True,
            "best_so_far_selection": False,
        }
        expected_hyperparameters = {
            "intervention_ppo_eta": float(configuration["intervention_ppo_eta"]),
            "correction_weight_mode": str(configuration["correction_weight_mode"]),
            "correction_loss_weight": float(configuration["correction_loss_weight"]),
            "correction_loss_weight_schedule": str(
                configuration.get("correction_loss_weight_schedule", "constant")
            ),
            "correction_loss_weight_by_round": [
                correction_loss_weight_for_round(
                    float(configuration["correction_loss_weight"]),
                    str(
                        configuration.get("correction_loss_weight_schedule", "constant")
                    ),
                    round_index,
                )
                for round_index in range(1, expected_round + 1)
            ],
            "dual_reward_scale": float(configuration["dual_reward_scale"]),
            "actor_learning_rate": BASE_ACTOR_LEARNING_RATE
            * float(configuration.get("actor_learning_rate_multiplier", 1.0)),
            "moving_kl_beta": float(configuration.get("moving_kl_beta", 0.5)),
            "actor_epochs": int(configuration.get("actor_epochs", 2)),
            "exploration_std": float(configuration.get("exploration_std", 0.05)),
        }
        mismatches = {
            key: (training.get(key), value)
            for key, value in expected_summary.items()
            if training.get(key) != value
        }
        hyperparameters = training.get("hyperparameters", {})
        mismatches.update(
            {
                f"hyperparameters.{key}": (hyperparameters.get(key), value)
                for key, value in expected_hyperparameters.items()
                if hyperparameters.get(key) != value
            }
        )
        if (
            mismatches
            or checkpoint.name != f"round_{expected_round:02d}.pt"
            or not checkpoint.is_file()
            or training.get("final_checkpoint_sha256") != file_sha256(checkpoint)
        ):
            raise RuntimeError(
                f"formal v141 fixed training contract differs: {mismatches}"
            )
        return checkpoint, summary

    def v140_checkpoint(self, specialist: str, training_seed: int) -> Path:
        checkpoint = (
            self.args.v140_training_root.resolve()
            / specialist
            / "dual_safe_ft"
            / f"seed_{training_seed}"
            / "round_04.pt"
        )
        summary_path = checkpoint.with_name("training_summary.json")
        if not checkpoint.is_file() or not summary_path.is_file():
            raise FileNotFoundError(checkpoint)
        summary = json.loads(summary_path.read_text())
        arm = summary.get("paper_filter_free_ablation_training", {})
        expected = {
            "context": specialist,
            "seed": training_seed,
            "rounds": 4,
            "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
            "primary_checkpoint_round": 4,
            "primary_checkpoint_sha256": file_sha256(checkpoint),
            "training_runtime_filter": True,
            "training_filter_fraction": 1.0,
        }
        mismatches = {
            key: (summary.get(key), value)
            for key, value in expected.items()
            if summary.get(key) != value
        }
        if arm.get("arm") != "dual_safe_ft" or mismatches:
            raise RuntimeError(
                f"v140 Dual Safe-FT source differs for {specialist}/{training_seed}: "
                f"{mismatches}"
            )
        return checkpoint

    def evaluate(
        self,
        *,
        method: str,
        specialist: str,
        training_seed: int | None,
        checkpoint: Path,
        context: str,
        condition: str,
    ) -> dict[str, Any]:
        seed_name = "frozen" if training_seed is None else f"seed_{training_seed}"
        directory = (
            self.output
            / "evaluation"
            / method
            / specialist
            / seed_name
            / context
            / f"filter_{condition}"
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
            str(FORMAL_EVALUATION_EPISODES),
            "--num-episodes",
            str(FORMAL_EVALUATION_EPISODES),
            "--seed",
            str(FORMAL_EVALUATION_SEED),
            "--device",
            self.args.device,
            "--instrument-current-velocity-cbf",
            "--output-json",
            str(summary),
            "--output-csv",
            str(episodes),
        ]
        job = f"eval_{method}_{specialist}_{seed_name}_{context}_{condition}"
        self.run_job(job, command, summary)
        result = json.loads(summary.read_text())
        expected_summary = {
            "protocol_id": EVALUATION_PROTOCOL_ID,
            "context": context,
            "seed": FORMAL_EVALUATION_SEED,
            "num_envs": FORMAL_EVALUATION_EPISODES,
            "num_episodes": FORMAL_EVALUATION_EPISODES,
            "runtime_filter": condition == "on",
            "deterministic_policy_mean": True,
            "one_initial_episode_per_env": True,
            "original_observation_interface": True,
            "actor_observation_dim": 405,
        }
        mismatches = {
            key: (result.get(key), value)
            for key, value in expected_summary.items()
            if result.get(key) != value
        }
        initial_state_signature = result.get("initial_state_signature")
        if (
            mismatches
            or not isinstance(initial_state_signature, str)
            or len(initial_state_signature) != 64
        ):
            raise RuntimeError(
                f"formal v141 evaluation protocol differs for {job}: {mismatches}"
            )
        return {
            "method": method,
            "specialist": specialist,
            "training_seed": training_seed,
            "context": context,
            "context_role": "target" if context == specialist else "retention",
            "runtime_filter": condition,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": result["checkpoint_sha256"],
            "initial_state_signature": initial_state_signature,
            "deterministic_policy_mean": True,
            "evaluation_seed": FORMAL_EVALUATION_SEED,
            "evaluation_episodes": FORMAL_EVALUATION_EPISODES,
            "summary": str(summary),
            "episodes": str(episodes),
            "metrics": {
                key: result[key]
                for key in (
                    "success_rate",
                    "fall_rate",
                    "mean_reached_riser",
                    "counterfactual_would_intervene_fraction",
                    "mean_counterfactual_correction_norm",
                    "nominal_barrier_violation_steps_per_riser",
                    "filtered_barrier_violation_steps_per_riser",
                    "intervention_steps_per_riser",
                )
            },
        }

    def run(self) -> None:
        self.validate()
        training_summaries: list[dict[str, Any]] = []
        new_checkpoints: dict[tuple[str, int], Path] = {}
        for specialist in SPECIALISTS:
            for seed in FORMAL_TRAINING_SEEDS:
                checkpoint, summary = self.train_v141(specialist, seed)
                new_checkpoints[(specialist, seed)] = checkpoint
                training_summaries.append(
                    {
                        "method": "v141_intervention_aware",
                        "specialist": specialist,
                        "training_seed": seed,
                        "summary": str(summary),
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": file_sha256(checkpoint),
                        "initial_actor_sha256": json.loads(summary.read_text())[
                            "initial_actor_sha256"
                        ],
                    }
                )

        initial_actor_hashes = {
            item["initial_actor_sha256"] for item in training_summaries
        }
        if len(initial_actor_hashes) != 1:
            raise RuntimeError(
                f"formal v141 runs did not share the v139 actor: {initial_actor_hashes}"
            )

        records: list[dict[str, Any]] = []
        for specialist in SPECIALISTS:
            contexts = (specialist, RETENTION_CONTEXT)
            for condition in ("off", "on"):
                for context in contexts:
                    records.append(
                        self.evaluate(
                            method="frozen_v139",
                            specialist=specialist,
                            training_seed=None,
                            checkpoint=self.args.base_checkpoint.resolve(),
                            context=context,
                            condition=condition,
                        )
                    )
            for seed in V140_TRAINING_SEEDS:
                checkpoint = self.v140_checkpoint(specialist, seed)
                for condition in ("off", "on"):
                    for context in contexts:
                        records.append(
                            self.evaluate(
                                method="v140_dual_safe_ft",
                                specialist=specialist,
                                training_seed=seed,
                                checkpoint=checkpoint,
                                context=context,
                                condition=condition,
                            )
                        )
            for seed in FORMAL_TRAINING_SEEDS:
                checkpoint = new_checkpoints[(specialist, seed)]
                for condition in ("off", "on"):
                    for context in contexts:
                        records.append(
                            self.evaluate(
                                method="v141_intervention_aware",
                                specialist=specialist,
                                training_seed=seed,
                                checkpoint=checkpoint,
                                context=context,
                                condition=condition,
                            )
                        )

        raw = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "method_id": METHOD_ID,
            "frozen_configuration": str(self.args.frozen_config.resolve()),
            "frozen_configuration_sha256": file_sha256(
                self.args.frozen_config.resolve()
            ),
            "formal_training_seeds": list(FORMAL_TRAINING_SEEDS),
            "formal_evaluation_seed": FORMAL_EVALUATION_SEED,
            "formal_evaluation_episodes": FORMAL_EVALUATION_EPISODES,
            "paired_initial_conditions": True,
            "fixed_final_round_checkpoint": True,
            "best_so_far_selection": False,
            "training_summaries": training_summaries,
            "evaluation_records": records,
        }
        paired_signatures = {
            context: sorted(
                {
                    item["initial_state_signature"]
                    for item in records
                    if item["context"] == context
                }
            )
            for context in (RETENTION_CONTEXT, *SPECIALISTS)
        }
        if any(len(values) != 1 for values in paired_signatures.values()):
            raise RuntimeError(
                f"formal v141 paired initial states differ: {paired_signatures}"
            )
        raw["paired_initial_state_signatures"] = {
            context: values[0] for context, values in paired_signatures.items()
        }
        raw_path = self.output / "formal_raw_results.json"
        _atomic_json(raw_path, raw)
        publish_command = [
            str(self.args.python),
            "experiments/scripts/publish_filter_free_v141.py",
            "--repo",
            str(self.repo),
            "--raw-results",
            str(raw_path),
            "--frozen-config",
            str(self.args.frozen_config.resolve()),
            "--output-dir",
            str(self.args.publish_dir.resolve()),
        ]
        if self.args.commit_and_push:
            publish_command.append("--commit-and-push")
        completion = self.args.publish_dir.resolve() / "formal_results.json"
        self.run_job("publish_formal_v141", publish_command, completion)
        formal = json.loads(completion.read_text())
        self.state["status"] = (
            "complete"
            if formal.get("formal_success")
            else "formal_failed_return_development"
        )
        self.state["formal_success"] = bool(formal.get("formal_success"))
        self.save_state()


def main() -> None:
    args = _parse_args()
    driver = FormalDriver(args)
    try:
        driver.run()
    except Exception:
        driver.state["status"] = "infrastructure_or_job_failure"
        driver.save_state()
        raise


if __name__ == "__main__":
    main()
