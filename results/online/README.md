# Online safe-refinement evidence

## Latest outcome-optimized velocity-CBF study (v34)

The complete bilingual v34 package is in
[`velocity_cbf_v34/`](velocity_cbf_v34/README.md). It contains the frozen
60-candidate search, top-8 confirmation, six eight-round training runs, the
single held-out F1/F2/F3 + D0 audit, four figures, and checksums.

The outcome-only selector fell back to the existing current CBF. On the single
held-out audit, v31 A2 + current CBF averaged 70.44% success and the newly
trained round-8 policies + selected current CBF averaged 68.88% (-1.56 pp).
Thus v34 did not meet the +3 pp development target; the package publishes this
as a negative result rather than relabeling a repeated current-CBF run as an
improvement.

This directory contains the curated, one-GPU evidence for the simulation-only
online refinement prototype. One candidate passed the declared D0/DQ/DQN gate,
but this must not be interpreted as a real-robot result or filter-free safety.

- `checkpoints/accepted_after_candidate_rejection.pt` is the transactionally
  retained base actor with the calibrated 799-D full privileged critic after a
  candidate failed the robust gate. It is a rollback artifact, not an improved
  actor.
- `checkpoints/accepted_round_001.pt` is the portable accepted actor, 799-D
  critic, and optimizer. Its embedded manifest records the large-batch gate.
- `evaluation/` contains the 128-episode D0/DQ/DQN base and rejected-candidate
  audits.
- `runs/` contains summaries from conservative PPO, critic calibration,
  pre-intervention credit, and true-intervention safe-BC variants.
- `evaluation/round2_rejected/` contains the large-batch evidence that rejected
  a resumed second round and retained `accepted_round_001.pt`.
- `smoke/` records strict checkpoint reload and GPU environment checks.
- `videos/accepted_round1/` contains one successful and one failed deterministic
  DQ rollout from the accepted policy with the runtime CBF enabled.

Absolute server paths in JSON files were replaced by portable placeholders.
See `docs/ONLINE_REFINEMENT_RESULTS.md` for interpretation.
