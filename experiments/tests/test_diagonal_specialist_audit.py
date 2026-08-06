"""Pure tests for the prospective diagonal-only specialist audit."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_specialists_diagonal_v18 import _validate_protocol
from audit_specialists_diagonal_v19 import (
  _validate_protocol as _validate_protocol_v19,
)
from diagonal_audit_stats import (
  hierarchical_paired_scene_interval,
  hierarchical_paired_scene_interval_v19,
  independent_diagonal_scene_gate,
  independent_diagonal_scene_gate_v19,
)


def test_scene_bootstrap_is_deterministic_and_preserves_positive_effect() -> None:
  groups = [
    torch.tensor(([1.0] * 20) + ([0.0] * 80), dtype=torch.float64),
    torch.tensor(([1.0] * 18) + ([0.0] * 82), dtype=torch.float64),
    torch.tensor(([1.0] * 22) + ([0.0] * 78), dtype=torch.float64),
  ]
  first = hierarchical_paired_scene_interval(
    groups, bootstrap_samples=1000, bootstrap_seed=811
  )
  second = hierarchical_paired_scene_interval(
    groups, bootstrap_samples=1000, bootstrap_seed=811
  )
  assert first == second
  assert first[0] == pytest.approx(0.20)
  assert first[1] > 0.0
  assert first[1] <= first[0] <= first[2]


def test_scene_bootstrap_rejects_unpaired_shapes_and_too_few_samples() -> None:
  with pytest.raises(ValueError, match="three adaptation seeds"):
    hierarchical_paired_scene_interval(
      [torch.zeros(8), torch.zeros(8)], bootstrap_samples=1000
    )
  with pytest.raises(ValueError, match="one non-zero size"):
    hierarchical_paired_scene_interval(
      [torch.zeros(8), torch.zeros(7), torch.zeros(8)], bootstrap_samples=1000
    )
  with pytest.raises(ValueError, match="at least 1000"):
    hierarchical_paired_scene_interval(
      [torch.zeros(8), torch.zeros(8), torch.zeros(8)], bootstrap_samples=999
    )


def test_independent_gate_accepts_small_positive_gain_without_two_pp_floor() -> None:
  gate = independent_diagonal_scene_gate(
    diagonal_success_delta=0.001,
    per_seed_success_deltas=[0.003, 0.001, -0.001],
    diagonal_fall_delta=0.029,
    d0_success_delta=-0.049,
  )
  assert gate["passed"] is True
  assert gate["positive_adaptation_seed_count"] == 2
  assert "diagonal_success_gain_above_2pp" not in gate["criteria"]


def test_v19_bootstrap_and_four_of_five_gate_are_deterministic() -> None:
  groups = [
    torch.tensor(([1.0] * count) + ([0.0] * (100 - count)), dtype=torch.float64)
    for count in (8, 10, 12, 14, 16)
  ]
  first = hierarchical_paired_scene_interval_v19(
    groups, bootstrap_samples=1000, bootstrap_seed=1900
  )
  second = hierarchical_paired_scene_interval_v19(
    groups, bootstrap_samples=1000, bootstrap_seed=1900
  )
  assert first == second
  assert first[0] == pytest.approx(0.12)
  gate = independent_diagonal_scene_gate_v19(
    diagonal_success_delta=0.001,
    per_seed_success_deltas=[0.01, 0.01, 0.01, 0.01, -0.01],
    diagonal_fall_delta=0.03,
    d0_success_delta=-0.05,
  )
  assert gate["passed"] is True
  assert gate["positive_adaptation_seed_count"] == 4
  failed = independent_diagonal_scene_gate_v19(
    diagonal_success_delta=0.001,
    per_seed_success_deltas=[0.01, 0.01, 0.01, 0.0, -0.01],
    diagonal_fall_delta=0.03,
    d0_success_delta=-0.05,
  )
  assert failed["passed"] is False
  assert failed["criteria"]["at_least_four_of_five_seed_gains_positive"] is False


@pytest.mark.parametrize(
  ("overrides", "failed_criterion"),
  [
    (
      {"diagonal_success_delta": 0.0},
      "mean_diagonal_success_gain_positive",
    ),
    (
      {"per_seed_success_deltas": [0.01, 0.0, -0.01]},
      "at_least_two_of_three_seed_gains_positive",
    ),
    (
      {"diagonal_fall_delta": 0.0300001},
      "diagonal_fall_increase_at_most_3pp",
    ),
    (
      {"d0_success_delta": -0.0500001},
      "d0_success_drop_at_most_5pp",
    ),
  ],
)
def test_independent_gate_fails_each_declared_boundary(
  overrides: dict[str, object], failed_criterion: str
) -> None:
  arguments: dict[str, object] = {
    "diagonal_success_delta": 0.01,
    "per_seed_success_deltas": [0.01, 0.01, 0.01],
    "diagonal_fall_delta": 0.0,
    "d0_success_delta": 0.0,
  }
  arguments.update(overrides)
  gate = independent_diagonal_scene_gate(**arguments)  # type: ignore[arg-type]
  assert gate["passed"] is False
  assert gate["criteria"][failed_criterion] is False


def test_frozen_protocol_accepts_only_the_declared_formal_runtime() -> None:
  repo = Path(__file__).resolve().parents[2]
  protocol = json.loads(
    (repo / "results/online/specialist_v18/protocol.json").read_text()
  )
  arguments = SimpleNamespace(
    smoke=False,
    adaptation_seeds=[42, 142, 242],
    audit_seed=3_100_000,
    bootstrap_seed=4_000_000,
    eval_batch_size=128,
    target_episodes=512,
    d0_episodes=256,
    bootstrap_samples=10000,
  )
  _validate_protocol(protocol, arguments)

  changed = copy.deepcopy(protocol)
  changed["evaluation"]["off_diagonal_evaluation"] = True
  with pytest.raises(ValueError, match="protocol file mismatch"):
    _validate_protocol(changed, arguments)

  changed = copy.deepcopy(arguments)
  changed.target_episodes = 256
  with pytest.raises(ValueError, match="runtime mismatch"):
    _validate_protocol(protocol, changed)


def test_frozen_v19_protocol_accepts_only_two_diagonals_and_five_seeds() -> None:
  repo = Path(__file__).resolve().parents[2]
  protocol = json.loads(
    (repo / "results/online/specialist_v19/protocol_revision3.json").read_text()
  )
  arguments = SimpleNamespace(
    smoke=False,
    adaptation_seeds=[53, 153, 253, 353, 453],
    audit_seed=5_300_000,
    bootstrap_seed=6_300_000,
    eval_batch_size=128,
    target_episodes=512,
    d0_episodes=256,
    bootstrap_samples=10000,
  )
  _validate_protocol_v19(protocol, arguments)
  changed = copy.deepcopy(protocol)
  changed["evaluation"]["filter_free_evaluation"] = True
  with pytest.raises(ValueError, match="protocol file mismatch"):
    _validate_protocol_v19(changed, arguments)
  changed_args = copy.deepcopy(arguments)
  changed_args.adaptation_seeds = [53, 153, 253]
  with pytest.raises(ValueError, match="runtime mismatch"):
    _validate_protocol_v19(protocol, changed_args)


def test_protocol_sealed_context_hashes_match_published_v17_evidence() -> None:
  repo = Path(__file__).resolve().parents[2]
  protocol = json.loads(
    (repo / "results/online/specialist_v18/protocol.json").read_text()
  )
  manifest = json.loads(
    (
      repo
      / "results/online/specialist_v17/formal/training_manifest.json"
    ).read_text()
  )
  for mode in ("lateral", "cbf", "balance"):
    context_path = repo / f"results/online/specialist_v17/contexts/{mode}.json"
    context = json.loads(context_path.read_text())
    sealed = protocol["sealed_inputs"]["contexts"][mode]
    assert hashlib.sha256(context_path.read_bytes()).hexdigest() == sealed[
      "file_sha256"
    ]
    assert context["parameters_sha256"] == sealed["parameters_sha256"]
    run_hashes = {
      record["deployment_context_parameters_sha256"]
      for record in manifest["runs"]
      if record["specialist_mode"] == mode
    }
    assert run_hashes == {sealed["parameters_sha256"]}
