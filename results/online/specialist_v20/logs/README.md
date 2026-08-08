# v20 compact logs

This directory contains v20 execution logs smaller than 1 MB, including
calibration, audit, verification, telemetry-disclosure, evidence-build, and
plotting records.

The five training logs per specialist and the two aggregate queue logs are
large and highly repetitive, so they are not duplicated in Git. Their absolute
execution paths, byte sizes, and SHA-256 hashes are recorded alongside the
included logs in [`../run_log_manifest.json`](../run_log_manifest.json).

The manifest is the provenance index; presence in this directory is a storage
decision, not a difference in evidentiary status.
