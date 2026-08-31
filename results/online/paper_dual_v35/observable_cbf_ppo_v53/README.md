# v53 Multi-Round Observable CBF PPO

v53 closes two gaps between v51/v52 and the iterative CBF-RL training loop:
it refreshes on-policy paired filter-on/off rollouts for every update round and
fits the expanded privileged critic after each accepted actor update.  The
actor update remains restricted to the five deployable CBF-geometry input
columns, preserving the original 405-D policy exactly.

Each of four rounds used three new seeds with paired filter-on/off rollouts,
8 environments, and 512 steps.  Every round was accepted transactionally only
when all six rollout batches retained positive post-update PPO surrogate gain.

| Formal v53 result | Value |
|---|---:|
| Accepted rounds | **4/4** |
| Training transitions | 98,304 |
| RTX 4080 training time | 276.81 s |
| Positive post-update batches | **6/6 in every round** |
| Per-round reference KL | 7.27e-6, 5.78e-6, 2.65e-6, 9.41e-6 |
| Legacy 405-D actor parameter change | exactly zero |
| Untouched seed `201351512`, filter off | **39/64 (60.94%)** |
| Decision | rejected |

Although all four refreshed training rounds passed their offline gates and
each critic fit reduced mean-squared error, performance did not generalize to
the untouched deployment seed.  The 39/64 result is below the predeclared
48/64 threshold, so filter-on and additional validation seeds were not run.
This result shows that repeated local paired-surrogate improvement alone is
still not a reliable selection signal for the F2 terrain shift.

Provenance:

- Source commit: `2ff79db9683158172c88279c392b44ae5319e55e`.
- Training seeds: `201351101/111/121`, `201351201/211/221`,
  `201351301/311/321`, and `201351401/411/421`.
- Candidate checkpoint SHA-256:
  `9c09ff9e1be09ca5b2696a50bb2afe2c2081a731689531bcbcd17694a44469a9`.
- Candidate actor SHA-256:
  `4d784cb9dee754055c3d5102dedfa12e82695b8e29002cd1d4bc86d2838b00f3`.
- Base checkpoint SHA-256:
  `3ec45cd196447901cf815d0fa1ff400af1b519ed2bb85c2fc179458ce3e81d3f`.

Files:

- `training_summary.json`: complete four-round rollout, actor, critic, KL,
  seed, and integrity record.
- `round_metrics.json`: detailed per-round and per-batch metrics.
- `candidate.pt`: exact rejected 410-D actor / 843-D critic checkpoint.
- `untouched_seed201351512_filter_off_{summary.json,episodes.csv}`:
  deployment-gate summary and all 64 episode records.
- `execution_started.json`: immutable formal-run launch record.

Implementation: `experiments/scripts/refine_observable_cbf_ppo_v53.py`.
