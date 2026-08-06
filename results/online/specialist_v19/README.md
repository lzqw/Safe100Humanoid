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

Development smoke runs and exploratory context probes are intentionally not
copied here and are not formal evidence. After the fresh audit, this directory
will contain the compact training manifest, paired CSV,
audit summary, verifier output, provenance, and checksums. Large checkpoints
remain in the external artifact store and are identified by SHA-256.
