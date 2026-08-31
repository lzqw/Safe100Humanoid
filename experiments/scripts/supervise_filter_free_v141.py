"""Drive v141 development, frozen formal evaluation, and publication to success."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BRANCH = "feature/online-safe-refinement"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v140-training-root", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--repo-development-dir",
        type=Path,
        default=Path("results/online/filter_free_v141/development"),
    )
    parser.add_argument(
        "--repo-formal-dir",
        type=Path,
        default=Path("results/online/filter_free_v141/formal"),
    )
    parser.add_argument(
        "--maximum-formal-attempts",
        type=int,
        default=0,
        help="Zero keeps returning to development until the fixed formal gates pass.",
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = args.repo.resolve()
        self.development = args.development_root.resolve()
        self.formal = args.formal_root.resolve()
        self.formal.mkdir(parents=True, exist_ok=True)
        self.state_path = self.formal / "supervisor_state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.is_file()
            else {
                "schema_version": 1,
                "status": "running",
                "formal_attempt": 1,
                "ignore_success_through_generation": 0,
                "current_stage": None,
                "completed_stages": [],
            }
        )

    def save(self) -> None:
        self.state["updated_unix_time"] = time.time()
        _atomic_json(self.state_path, self.state)

    def run_command(self, stage: str, command: list[str]) -> None:
        log = self.formal / "logs" / f"{stage}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        self.state["status"] = "running"
        self.state["current_stage"] = stage
        self.state["current_command"] = command
        self.state["current_stage_started_unix_time"] = time.time()
        self.save()
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
            "stage": stage,
            "returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "log": str(log),
        }
        self.state["current_stage"] = None
        self.state["current_command"] = None
        if result.returncode:
            self.state["status"] = "infrastructure_or_job_failure"
            self.state["last_failure"] = record
            self.save()
            raise RuntimeError(f"v141 supervisor stage failed: {stage}; see {log}")
        self.state["completed_stages"].append(record)
        self.save()

    def development_command(self) -> list[str]:
        command = [
            str(self.args.python),
            "experiments/scripts/run_filter_free_v141.py",
            "--repo",
            str(self.repo),
            "--python",
            str(self.args.python),
            "--base-checkpoint",
            str(self.args.base_checkpoint.resolve()),
            "--protocol",
            str(self.args.protocol.resolve()),
            "--output-root",
            str(self.development),
            "--device",
            self.args.device,
            "--through-generation",
            "0",
        ]
        cutoff = int(self.state.get("ignore_success_through_generation", 0))
        if cutoff:
            command.extend(
                ["--ignore-success-through-generation", str(cutoff)]
            )
        return command

    def frozen_path(self, attempt: int) -> Path:
        directory = self.args.repo_development_dir
        if not directory.is_absolute():
            directory = self.repo / directory
        return directory.resolve() / f"frozen_config_attempt_{attempt:02d}.json"

    def final_publish_dir(self) -> Path:
        directory = self.args.repo_formal_dir
        if not directory.is_absolute():
            directory = self.repo / directory
        return directory.resolve()

    def formal_failure_cutoff(self) -> int:
        state_path = self.development / "run_state.json"
        development_state = json.loads(state_path.read_text())
        generations = [
            int(value) for value in development_state.get("generations", {})
        ]
        if not generations:
            raise RuntimeError("cannot resume development without generation evidence")
        return max(generations)

    def run(self) -> None:
        while True:
            attempt = int(self.state["formal_attempt"])
            if (
                self.args.maximum_formal_attempts
                and attempt > self.args.maximum_formal_attempts
            ):
                raise RuntimeError("v141 maximum formal-attempt limit reached")

            self.run_command(
                f"attempt_{attempt:02d}_development",
                self.development_command(),
            )
            development_summary_path = self.development / "development_summary.json"
            development_summary = json.loads(development_summary_path.read_text())
            if (
                development_summary.get("both_specialists_pass") is not True
                or development_summary.get("next_phase") != "freeze_and_formal"
            ):
                raise RuntimeError("v141 development returned without passing both gates")

            frozen = self.frozen_path(attempt)
            self.run_command(
                f"attempt_{attempt:02d}_freeze",
                [
                    str(self.args.python),
                    "experiments/scripts/freeze_filter_free_v141.py",
                    "--repo",
                    str(self.repo),
                    "--development-root",
                    str(self.development),
                    "--output-json",
                    str(frozen),
                    "--commit-and-push",
                ],
            )

            attempt_root = self.formal / f"attempt_{attempt:02d}"
            staging = attempt_root / "publication_staging"
            self.run_command(
                f"attempt_{attempt:02d}_formal",
                [
                    str(self.args.python),
                    "experiments/scripts/run_formal_filter_free_v141.py",
                    "--repo",
                    str(self.repo),
                    "--python",
                    str(self.args.python),
                    "--base-checkpoint",
                    str(self.args.base_checkpoint.resolve()),
                    "--protocol",
                    str(self.args.protocol.resolve()),
                    "--frozen-config",
                    str(frozen),
                    "--v140-training-root",
                    str(self.args.v140_training_root.resolve()),
                    "--output-root",
                    str(attempt_root),
                    "--publish-dir",
                    str(staging),
                    "--device",
                    self.args.device,
                ],
            )
            result_path = staging / "formal_results.json"
            formal_result = json.loads(result_path.read_text())
            self.state.setdefault("formal_attempts", []).append(
                {
                    "attempt": attempt,
                    "frozen_configuration": str(frozen),
                    "result": str(result_path),
                    "formal_success": bool(formal_result.get("formal_success")),
                }
            )
            if formal_result.get("formal_success") is True:
                self.run_command(
                    f"attempt_{attempt:02d}_publish",
                    [
                        str(self.args.python),
                        "experiments/scripts/publish_filter_free_v141.py",
                        "--repo",
                        str(self.repo),
                        "--raw-results",
                        str(attempt_root / "formal_raw_results.json"),
                        "--frozen-config",
                        str(frozen),
                        "--output-dir",
                        str(self.final_publish_dir()),
                        "--commit-and-push",
                    ],
                )
                revision = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.repo,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                self.state.update(
                    {
                        "status": "complete",
                        "formal_success": True,
                        "published_commit": revision,
                        "published_directory": str(self.final_publish_dir()),
                    }
                )
                self.save()
                return

            # Only the binary gate outcome is carried back. Formal metrics do
            # not choose the next configuration; the next mutations use the
            # already specified development diagnostics and development seed.
            self.state["status"] = "formal_failed_return_development"
            self.state["formal_attempt"] = attempt + 1
            self.state["ignore_success_through_generation"] = (
                self.formal_failure_cutoff()
            )
            self.save()


def main() -> None:
    args = _parse_args()
    if args.maximum_formal_attempts < 0:
        raise ValueError("maximum-formal-attempts must be non-negative")
    supervisor = Supervisor(args)
    try:
        supervisor.run()
    except Exception:
        if supervisor.state.get("status") == "running":
            supervisor.state["status"] = "infrastructure_or_job_failure"
            supervisor.save()
        raise


if __name__ == "__main__":
    main()
