"""Regression tests for the post-audit v20 telemetry disclosure."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/scripts"))

from collect_mechanism_telemetry_replay_v20 import (
  _compare_formal_replay,
  _selected_repairs,
  _trace_file_state,
)
from plot_specialist_v20_replay_disclosed import _disclosure_text


def test_v20_telemetry_selection_is_fixed_to_lowest_formal_repair() -> None:
  rows = [
    {
      "specialist_mode": "lateral",
      "evaluation_role": "target_diagonal_primary",
      "transition_class": "failure_to_success",
      "adaptation_seed": "73",
      "pair_index": "11",
    },
    {
      "specialist_mode": "lateral",
      "evaluation_role": "target_diagonal_primary",
      "transition_class": "failure_to_success",
      "adaptation_seed": "73",
      "pair_index": "3",
    },
  ]
  selected = _selected_repairs(rows, "lateral")
  assert selected[73] is not None
  assert selected[73]["pair_index"] == "3"
  assert all(selected[seed] is None for seed in (173, 273, 373, 473))


def test_v20_telemetry_records_outcome_divergence_without_raising() -> None:
  formal = {"success": "False", "fell": "True", "failure_type": "lateral"}
  replay = {"success": "True", "fell": "False", "failure_type": "success"}
  assert _compare_formal_replay(formal, replay) == {
    "success": False,
    "fell": False,
    "failure_type": False,
  }


def test_v20_telemetry_never_overwrites_partial_first_attempt(
  tmp_path: Path,
) -> None:
  trace_dir = tmp_path / "trace"
  trace_dir.mkdir()
  assert _trace_file_state(trace_dir) == "not_started"
  (trace_dir / "evaluation.json").write_text("{}\n")
  with pytest.raises(RuntimeError, match="will not be overwritten"):
    _trace_file_state(trace_dir)
  (trace_dir / "episodes.csv").write_text("x\n")
  (trace_dir / "telemetry.csv").write_text("x\n")
  assert _trace_file_state(trace_dir) == "complete_first_attempt"


def test_v20_mechanism_figure_discloses_replay_role() -> None:
  text = _disclosure_text(
    {
      "trace_reproduction": {
        "outcome_match_count": 4,
        "trace_count": 10,
      }
    }
  )
  assert "4/10" in text
  assert "descriptive replays" in text
  assert "paired CSV" in text
