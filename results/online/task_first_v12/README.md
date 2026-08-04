# Task-First Constrained CBF-PPO v12 results

This directory contains the compact, machine-readable evidence published with
the v12 implementation. No rejected checkpoint is published as an accepted
policy.

The main conclusions are:

- closed-loop DQH/DQNH joystick commands resolve a major observability/control
  bottleneck in the open-loop DQ/DQN tasks;
- task-first constrained PPO can improve the DQH point estimates while also
  reducing CBF demand;
- the strongest candidate still fails the predeclared D0 retention floor and
  its paired DQH 95% intervals include zero, so the correct decision is
  `rollback`;
- success-gated CBF correction distillation did not outperform the
  no-correction arm under the tested interaction budget;
- the accepted actor remains unchanged and the runtime CBF cannot be removed.

Files:

- `key_results.json`: compact observability, formal-ablation, high-power audit,
  verification, and source-checksum ledger.
- `../evaluation/human_centering_v1/large/`: the four 512-episode CSV/JSON
  evaluations supporting the DQ/DQH and DQN/DQNH comparison.

See `docs/TASK_FIRST_CONSTRAINED_CBF_PPO_V12.md` for interpretation and
methodological details.
