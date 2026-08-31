"""Build the two-row immutable v23 lateral / v24 contact completion report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from proximal_v23_io import file_sha256
from verify_proximal_v24 import V23_FINAL_SHA256


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _summary(source: dict[str, Any], *, scene: str, version: str) -> dict[str, Any]:
    target = source["target"]
    d0 = source["D0"]
    return {
        "schema_version": 1,
        "version": version,
        "scene": scene,
        "target": {
            "paired_episodes": target["paired_conditions"],
            "baseline_success_rate": target["success"]["baseline_mean"],
            "final_success_rate": target["success"]["final_mean"],
            "success_delta": target["success"]["delta"],
            "baseline_fall_rate": target["fall"]["baseline_mean"],
            "final_fall_rate": target["fall"]["final_mean"],
            "fall_delta": target["fall"]["delta"],
            "repairs": target["repairs_regressions"]["repair_count"],
            "regressions": target["repairs_regressions"]["regression_count"],
        },
        "D0": {
            "paired_episodes": d0["paired_conditions"],
            "baseline_success_rate": d0["success"]["baseline_mean"],
            "final_success_rate": d0["success"]["final_mean"],
            "success_delta": d0["success"]["delta"],
            "baseline_fall_rate": d0["fall"]["baseline_mean"],
            "final_fall_rate": d0["fall"]["final_mean"],
            "fall_delta": d0["fall"]["delta"],
            "repairs": d0["repairs_regressions"]["repair_count"],
            "regressions": d0["repairs_regressions"]["regression_count"],
        },
        "development_gate": source["development_gate"],
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def _pp(value: float) -> str:
    return f"{100.0 * value:+.3f} pp"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v23-final-test", type=Path, required=True)
    parser.add_argument("--v24-final-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if file_sha256(args.v23_final_test) != V23_FINAL_SHA256:
        raise RuntimeError("combined report refuses a modified v23 final result")
    v23 = json.loads(args.v23_final_test.read_text())
    v24 = json.loads(args.v24_final_test.read_text())
    lateral = _summary(v23, scene="pure lateral", version="v23")
    contact = _summary(v24, scene="pure low foot friction", version="v24")
    contact_delta = contact["target"]["success_delta"]
    if contact["development_gate"]["passed"]:
        conclusion_code = "A_contact_effective_lateral_not_effective"
        conclusion_en = (
            "The method is task-effective for pure low-friction contact shift, but "
            "not for pure lateral command drift."
        )
        conclusion_zh = "方法对纯低摩擦 contact shift 有效，但对纯 lateral drift 无效。"
    elif contact_delta > 0.0:
        conclusion_code = "B_stable_but_small_contact_effect"
        conclusion_en = (
            "Both clean scenes show bounded updates, but the contact benefit is "
            "smaller than the registered task-effect threshold."
        )
        conclusion_zh = "两个干净场景都能有界更新，但 contact 提升小于预注册任务门槛。"
    else:
        conclusion_code = "C_bounded_updates_without_task_gain"
        conclusion_en = (
            "Moving-KL proximal PPO preserves a bounded online update path but did "
            "not improve task success in either representative deployment shift."
        )
        conclusion_zh = "Moving-KL proximal PPO 能保持有界在线更新，但在两个代表性部署偏移中均未提高任务成功率。"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lateral["source_file_sha256"] = file_sha256(args.v23_final_test)
    lateral["recomputed"] = False
    contact["source_file_sha256"] = file_sha256(args.v24_final_test)
    contact["recomputed"] = False
    _write_json(output_dir / "lateral_v23_summary.json", lateral)
    _write_json(output_dir / "contact_v24_summary.json", contact)
    rows = []
    for summary in (lateral, contact):
        target = summary["target"]
        rows.append(
            {
                "version": summary["version"],
                "scene": summary["scene"],
                "base_success_rate": target["baseline_success_rate"],
                "final_success_rate": target["final_success_rate"],
                "success_delta": target["success_delta"],
                "fall_delta": target["fall_delta"],
                "repairs": target["repairs"],
                "regressions": target["regressions"],
                "development_gate_passed": summary["development_gate"]["passed"],
            }
        )
    with (output_dir / "combined_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table_rows = []
    for summary in (lateral, contact):
        target = summary["target"]
        table_rows.append(
            "| "
            + " | ".join(
                (
                    f"{summary['version']} {summary['scene']}",
                    _percent(target["baseline_success_rate"]),
                    _percent(target["final_success_rate"]),
                    _pp(target["success_delta"]),
                    _pp(target["fall_delta"]),
                    f"{target['repairs']} / {target['regressions']}",
                    "PASS" if summary["development_gate"]["passed"] else "FAIL",
                )
            )
            + " |"
        )
    readme = f"""# CBF-Proximal Completion: v23 Lateral + v24 Contact

## English

v23 and v24 are independent, single-context development tests. They use the
same CBF-shielded, raw-action, moving-KL PPO algorithm, but **do not form a
joint gate**. v24 does not alter, rerun, or reinterpret the frozen v23 result.

| Scene | Base SR | Final SR | Delta SR | Delta fall | Repairs / regressions | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{table_rows[0]}
{table_rows[1]}

Conclusion `{conclusion_code}`: {conclusion_en}

The lateral row is copied from the byte-frozen v23 final JSON (SHA-256
`{V23_FINAL_SHA256}`), not reconstructed from simulator episodes. The contact
row comes from the single frozen v24 execution. Confidence intervals remain
report-only in both studies.

## 中文

v23 与 v24 是两个相互独立的单场景开发实验。两者使用同一套 CBF shield、
raw-action、moving-KL PPO 算法，但**不存在联合 gate**。v24 不修改、不重跑、
也不重新解释已冻结的 v23 lateral 负结果。

综合结论 `{conclusion_code}`：{conclusion_zh}

lateral 行直接复制自 byte-frozen v23 final JSON，不重新运行仿真；contact 行来自
唯一一次冻结的 v24 正式执行。两项实验的置信区间都只作报告，不参与 gate。
"""
    (output_dir / "README.md").write_text(readme)
    print(
        json.dumps(
            {
                "conclusion_code": conclusion_code,
                "v23_source_sha256": lateral["source_file_sha256"],
                "v24_source_sha256": contact["source_file_sha256"],
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
