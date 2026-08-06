# v19 prospective protocol

This directory contains the frozen protocol and its prospective-preflight
audit trail for Observable Failure-Conditioned Brief PPO v19. Revision-2
calibration and every subsequent adaptation, diagonal audit, and independent
verification run use the exact frozen protocol commit.

Revision 1 correctly stopped before adaptation when none of its six lateral
contexts reached 80% mechanism purity. The exact base-only calibration summary
is retained in
[`preflight/revision1_lateral_calibration_failure.json`](preflight/revision1_lateral_calibration_failure.json).
Revision 2 changes the ambiguous event precedence, replaces lateral candidates
and evaluation randomness with unseen values, and is frozen before any
adapted-policy or formal-audit outcome.

## Revision-2 calibration result

Both first-qualifying contexts are now frozen from base-policy-only episodes:

| Specialist | Selected seed | Base success | Falls | Target purity | Second mechanism |
| --- | ---: | ---: | ---: | ---: | ---: |
| lateral | 7315 | 70.3125% | 152 | 90.7895% | 9.2105% |
| contact stability | 7217 | 77.9297% | 113 | 98.2301% | 1.7699% |

The exact contexts are [`contexts/lateral.json`](contexts/lateral.json) and
[`contexts/contact_stability.json`](contexts/contact_stability.json). Their
hashes, selection path, and external log/summary hashes are in
[`calibration_summary.json`](calibration_summary.json).

## Revision-2 formal-training stop

The frozen training queue stopped before the formal diagonal audit. Lateral
seed 43 completed eight rounds with five accepted updates, but lateral seed
143 completed all eight rounds with only one accepted update. This is below
the prospectively frozen minimum of three, so the runner raised an error and
the `set -e` queue stopped before the remaining eight adaptation runs.

No formal claim is made from revision 2. In particular, the per-run
128-episode evaluations are training diagnostics only and are not substitutes
for the unstarted 512-target/256-D0 paired audit. The exact compact round
records, artifact hashes, stop reason, and not-started seed list are retained
in
[`training/revision2_lateral_training_stop.json`](training/revision2_lateral_training_stop.json).

## Revision-3 development preflight (not formal)

The stopped run exposed two implementation-level weaknesses: replay reset did
not preserve the exact failure/success matches constructed in the banks, and
the five new zero-initialized observation columns inherited only the 0.1 first
layer learning-rate multiplier. The development successor preserves exact
pairs, jointly balances every required replay marginal, freezes the 405 legacy
input columns, and trains only the five new columns at the full actor learning
rate.

The allocator was replayed against both saved lateral banks and a saved contact
bank. Every 12-state sample was exactly paired; lateral achieved 6/6 for both
signs, support, and growth plus 4/4/4 riser stages, while contact achieved 6/6
for touchdown, slip, timing, and support. A first full-sized development attempt
then exposed one heuristic local optimum at 7/5 heading balance, so it was
intentionally stopped after round 1. An exact bounded-integer fallback and a
reproducing regression test now prevent that silent drift. The complete GPU
test suite passes 89/89. Exact diagnostics, hashes, development outcomes, and
the explicit no-formal-claim boundary are recorded in
[`development/revision3_preflight.json`](development/revision3_preflight.json).

No revision-3 formal protocol has been frozen and no revision-3 formal run has
started. The next check uses the already exposed weak seed 143 strictly as a
development ablation; it cannot enter a later formal claim.

Development smoke runs and exploratory context probes are intentionally not
copied here and are not formal evidence. A future prospectively refrozen run
may add a compact training manifest, paired CSV, audit summary, verifier
output, provenance, and checksums. Large checkpoints remain in the external
artifact store and are identified by SHA-256.
