# v20 Requirements Audit

This is a completion audit against the prospectively specified v20 task. A
`PASS` here means the experiment/control/evidence requirement was satisfied;
it does not mean the scientific performance claim passed.

| # | Requirement | Status | Evidence |
| ---: | --- | --- | --- |
| 1 | Fixed eight-round budget; no minimum accepted-update gate or rejection-patience stop | PASS | All 10 compact [`training`](training/) summaries report `actual_rounds = fixed_round_budget = 8`, `minimum_accepted_updates = null`, `rejection_patience = null`, and `zero_to_eight_retained_updates_are_valid = true`. Retained counts include valid two-update runs. |
| 2 | Lateral and contact-stability have independent queues, audits, verifiers, and conclusions | PASS | Separate training/audit trees and separate verified claims exist under [`audit/lateral`](audit/lateral/) and [`audit/contact_stability`](audit/contact_stability/). Both audits ran even though neither claim passed. `formal_results.json` declares no joint claim. |
| 3 | Preserve the v19 Revision-4 learning core | PASS | Per-run source hashes and structural checks record one actor/one critic, runtime CBF, raw-action PPO storage, 410-D actor input, frozen legacy columns, full-LR five-column adapter, the prescribed two rollout batches/start mixture, learning rates, PPO schedule, KL limits, and candidate fractions. No new critic/loss/anchor was introduced. |
| 4 | Preserve two-stage candidate evaluation and D0 rollback semantics | PASS | [`candidate_metrics.csv`](curves/candidate_metrics.csv) has exactly 240 rows (3 fractions × 8 rounds × 10 runs); each run used 64 screen pairs per fraction and 128 fresh confirm pairs. The target/fall/KL gate and D0 check remain in the frozen sources. No D0 rollback occurred in these formal runs. |
| 5 | Fresh contexts and randomness frozen before adaptation outcomes | PASS | [`fresh_randomness_preflight.json`](fresh_randomness_preflight.json) reports no adaptation/audit/bootstrap/calibration collision. Contexts 8312/8212 were the prospectively first qualifying base-only candidates. Context JSONs, hashes, seeds 73/173/273/373/473, audit seed 5,500,000, bootstrap seed 6,500,000, and protocol SHA-256 `74242b…` were committed at `1ded5b84…` before formal training. |
| 6 | Independent formal evaluation: 512 target + 256 D0 pairs/seed and 10,000 hierarchical bootstraps | PASS | Each mode publishes a 3,840-row paired CSV (2,560 target + 1,280 D0), an atomic audit summary, and `verified: true` independent reconstruction. Lateral claim: FAIL; contact claim: FAIL. Confidence intervals and all four point gates are published separately. |
| 7 | Complete round/candidate/replay/mechanism and per-episode tables | PASS | Combined row counts are 90 round, 240 candidate, 160 replay, and 11,873 mechanism rows. Each seed also publishes 9/24/16 aligned training tables. Each mode publishes its complete paired episode CSV. Schemas are checked by the builder/plotter/tests. |
| 8 | Mechanism telemetry, transition classes, repairs/regressions, deterministic identity selection | PASS WITH DISCLOSURE | Formal paired CSVs contain all four transition classes and per-seed/aggregate counts. Lowest formal repair pair indices were fixed. Same-state GPU replays reproduced 7/10 lateral and 5/10 contact outcomes; all actor/state identities matched, no selection changed, and no outcome-matching retry occurred. Mechanism curves are explicitly descriptive; formal CSV outcomes remain authoritative. |
| 9 | Deterministic PNG/PDF figures covering all requested analyses | PASS | [`figures/figure_manifest.json`](figures/figure_manifest.json) records 22 PNG and 22 PDF figures: success/fall, D0, candidate, KL, retained updates, replay integrity, adapter learning, forest, repairs/regressions, and repaired-mechanism curves for both modes. With `SOURCE_DATE_EPOCH=1786173901`, an independent second generation produced identical hashes for all 44 outputs. |
| 10 | Historical reference from frozen formal evidence, clearly non-comparable | PASS | [`historical_reference.csv`](historical_reference.csv) is generated from v17/v18 frozen formal summaries plus v19-R4 no-audit status. Every row says `directly_comparable_to_v20 = False`. The v17/v18/v19 evidence trees have no diff and retain their recorded hashes. |
| 11 | Reproducibility, hashes, logs, retries, and immutable historical evidence | PASS | [`training_manifest.json`](training_manifest.json) hashes 510 external artifacts (5.96 GB), [`run_log_manifest.json`](run_log_manifest.json) hashes all 26 logs, compact logs are included, per-run summaries record source/context/base/final actor/checkpoint hashes, and `SHA256SUMS` covers the committed package. Infrastructure amendments/retries are itemized in [`RUN_PROVENANCE.md`](RUN_PROVENANCE.md). |
| 12 | Regression tests and concise bilingual final report; push to existing Draft PR | PASS | Fixed-budget, zero-update validity, independent launcher/audit, seed collision, replay transaction/marginals, frozen actor columns, CSV matrices, plotting inputs, audit reconstruction, and telemetry-disclosure tests are present. The focused v20 suite reports 19 passed, the full recorded-environment suite reports 109 passed, and Ruff is clean. This bilingual [`README.md`](README.md) reports both negative conclusions without undrafting the PR. |

## Scientific gate reconstruction

The independent gate for each specialist is:

1. mean paired target success delta strictly greater than 0;
2. at least four of five adaptation-seed deltas strictly positive;
3. mean paired target fall delta at most +3 percentage points; and
4. mean paired D0 success delta at least −5 percentage points.

| Specialist | Mean target ΔSR > 0 | ≥4/5 positive | Target Δfall ≤ +3 pp | D0 ΔSR ≥ −5 pp | Claim |
| --- | --- | --- | --- | --- | --- |
| lateral | PASS (+0.273 pp) | **FAIL (2/5)** | PASS (−0.313 pp) | PASS (−1.484 pp) | **FAIL** |
| contact_stability | PASS (+0.156 pp) | **FAIL (2/5)** | PASS (−0.156 pp) | PASS (−0.703 pp) | **FAIL** |

`LCB95 > 0` is report-only strong evidence, not an engineering gate. It is
false for both modes.

## Evidence completeness checks

- Ten independent training branches: present and protocol-complete.
- Eight round checkpoints per branch: 80/80 present in the external manifest.
- Per-seed table shapes: 9 round, 24 candidate, 16 replay rows for all runs.
- Formal audit rows: 3,840/3,840 per mode; audit and verifier hashes agree.
- Protocol/source/context/actor identity checks: true in both verifications.
- Runtime CBF: on; off-diagonal/filter-free/macro/joint gates: absent.
- Replay exact-match failures: 0; maximum marginal imbalance: 0.
- Legacy first-layer input drift: 0 exactly across all rounds.
- Historical v17/v18/v19 evidence modifications: none.

## Disclosed post-freeze infrastructure events

The initial formal audit process failed before constructing the formal audit
environment because its loader inherited an irrelevant legacy checkpoint
payload. Audit amendment 1 enabled Brief-PPO loading; a non-formal smoke then
exposed inherited auxiliary-loss defaults. Audit amendment 2 restored the full
frozen Revision-4 runner configuration without changing training, evaluation,
actors, checkpoints, seeds, contexts, or gates. The amendment was committed and
tested before any formal audit outcome was observed.

After the verified lateral audit, the original mechanism collector showed that
GPU replay could diverge despite identical actor and initial-state signatures.
The failure and partial first attempts were frozen before adding a reporting
wrapper. The same frozen wrapper was used for contact. It records divergence,
never replaces formal outcomes, never changes the selected identity, and never
retries until matching. These are reporting-only events and do not alter either
formal conclusion.
