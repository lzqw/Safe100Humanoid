# Online safe-refinement evidence

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
- `smoke/` records strict checkpoint reload and GPU environment checks.
- `videos/accepted_round1/` contains one successful and one failed deterministic
  DQ rollout from the accepted policy with the runtime CBF enabled.

Absolute server paths in JSON files were replaced by portable placeholders.
See `docs/ONLINE_REFINEMENT_RESULTS.md` for interpretation.
