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

Development smoke runs and exploratory context probes are intentionally not
copied here and are not formal evidence. A future prospectively refrozen run
may add a compact training manifest, paired CSV, audit summary, verifier
output, provenance, and checksums. Large checkpoints remain in the external
artifact store and are identified by SHA-256.
