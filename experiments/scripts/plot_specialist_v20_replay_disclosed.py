"""Run the frozen v20 plotter with explicit telemetry-replay disclosure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import plot_specialist_v20 as frozen_plotter

_ORIGINAL_MECHANISM = frozen_plotter._mechanism
_SELECTIONS: dict[str, dict[str, Any]] = {}


def _selection_path(audit_root: Path, mode: str) -> Path:
  return audit_root / mode / "mechanism_selection.json"


def _disclosure_text(selection: dict[str, Any]) -> str:
  reproduction = selection["trace_reproduction"]
  matched = reproduction["outcome_match_count"]
  total = reproduction["trace_count"]
  return (
    f"First-attempt same-initial-state telemetry replay: {matched}/{total} "
    "policy-role outcomes matched the formal audit. Curves are descriptive "
    "replays; formal outcomes come only from the paired CSV."
  )


def _disclosed_mechanism(
  mechanism_rows: list[dict[str, str]], mode: str
):
  figure = _ORIGINAL_MECHANISM(mechanism_rows, mode)
  selection = _SELECTIONS[mode]
  reproduction = selection["trace_reproduction"]
  if reproduction["selection_changed_after_formal_audit"] is not False:
    raise RuntimeError("v20 telemetry identity selection changed")
  if reproduction["retry_until_outcome_matches"] is not False:
    raise RuntimeError("v20 telemetry used outcome-matching retries")
  figure.suptitle(
    f"{mode}: fixed lowest-ID formal-repair identities (telemetry replay)"
  )
  figure.text(
    0.5,
    0.01,
    _disclosure_text(selection),
    ha="center",
    va="bottom",
    fontsize=8,
    wrap=True,
  )
  figure.subplots_adjust(bottom=0.24, top=0.82)
  return figure


def _audit_root_from_argv() -> Path:
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--audit-root", type=Path, required=True)
  parsed, _ = parser.parse_known_args()
  return parsed.audit_root.resolve()


def main() -> None:
  audit_root = _audit_root_from_argv()
  for mode in frozen_plotter.SPECIALIST_MODES:
    path = _selection_path(audit_root, mode)
    selection = json.loads(path.read_text())
    if "descriptive first-attempt" not in selection.get("evidence_role", ""):
      raise RuntimeError(f"v20 {mode} telemetry disclosure is absent")
    _SELECTIONS[mode] = selection
  frozen_plotter._mechanism = _disclosed_mechanism
  frozen_plotter.main()

  output_parser = argparse.ArgumentParser(add_help=False)
  output_parser.add_argument("--output-dir", type=Path, required=True)
  parsed, _ = output_parser.parse_known_args()
  manifest_path = parsed.output_dir.resolve() / "figure_manifest.json"
  manifest = json.loads(manifest_path.read_text())
  manifest["mechanism_telemetry_disclosure"] = {
    mode: {
      "selection_path": str(_selection_path(audit_root, mode)),
      "selection_sha256": frozen_plotter._sha256(
        _selection_path(audit_root, mode)
      ),
      "disclosure": _disclosure_text(_SELECTIONS[mode]),
    }
    for mode in frozen_plotter.SPECIALIST_MODES
  }
  manifest["reporting_wrapper"] = str(Path(__file__).resolve())
  manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
  print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
