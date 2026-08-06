# v19 prospective protocol

This directory contains the frozen protocol and its prospective-preflight
audit trail for Observable Failure-Conditioned Brief PPO v19. Formal
calibration, ten independent
adaptation jobs, two diagonal paired audits, and independent verification must
all run from the exact protocol commit.

Revision 1 correctly stopped before adaptation when none of its six lateral
contexts reached 80% mechanism purity. The exact base-only calibration summary
is retained in
[`preflight/revision1_lateral_calibration_failure.json`](preflight/revision1_lateral_calibration_failure.json).
Revision 2 changes the ambiguous event precedence, replaces lateral candidates
and evaluation randomness with unseen values, and is frozen before any
adapted-policy or formal-audit outcome.

Development smoke runs and exploratory context probes are intentionally not
copied here and are not formal evidence. After the fresh audit, this directory
will contain the calibrated contexts, compact training manifest, paired CSV,
audit summary, verifier output, provenance, and checksums. Large checkpoints
remain in the external artifact store and are identified by SHA-256.
