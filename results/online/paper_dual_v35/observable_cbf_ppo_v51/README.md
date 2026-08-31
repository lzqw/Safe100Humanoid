# v51 Geometry-Aware Paired CBF PPO

v51 replaces direct safe-action regression with a paper-style paired PPO
update.  Three seeds each collect filter-on and filter-off rollouts, use the
bounded `raw_moderate` CBF dual reward with critic GAE, average the on/off
gradients within a seed, and then take a coordinate median across seeds.
Only the five new geometry-input columns are trainable; the original 405-D
policy is frozen exactly.

The implementation smoke and the first formal 49,152-transition run are both
published here.  The formal checkpoint missed the deployment threshold by one
episode and is not a final accepted actor.

| Smoke check, 3 seeds x 2 envs x 64 steps | Result |
|---|---:|
| Runtime | 20.42 s on RTX 4080 |
| Positive post-update PPO surrogate batches | **6/6** |
| Mean filter-on surrogate gain | +0.00219748 |
| Mean filter-off surrogate gain | +0.00217462 |
| Mean filter-on/off gradient cosine | 0.97255 |
| Reference forward KL | 0.00013189 (limit 0.001) |
| Legacy actor parameter change | exactly 0 |
| Offline smoke gate | **passed** |

The very short rollouts have a negative cross-seed paired-gradient cosine
(`-0.12620`), so this smoke proves only that the new training path executes and
improves every collected batch under its trust region.  No 64-episode
deployment evaluation was run for this smoke checkpoint.

## Formal run

The formal RTX 4080 run used three previously unused paired seeds, 16
environments per condition, and 512 steps per environment.

| Formal check | Result |
|---|---:|
| Runtime | 92.73 s |
| Training transitions | 49,152 |
| Positive post-update PPO surrogate batches | **6/6** |
| Mean filter-on surrogate gain | +0.00007335 |
| Mean filter-off surrogate gain | +0.00002667 |
| Reference forward KL | 0.00000298 (limit 0.001) |
| Untouched seed `201350992`, filter off | **47/64 (73.44%)** |
| Deployment decision | rejected; threshold is 48/64 |

The candidate was one successful episode below the registered 75% gate.
Filter-on and further seeds were therefore not run.  The formal update used
only about 0.3% of its KL budget, identifying update magnitude rather than KL
safety as the next concrete bottleneck.

Formal provenance:

- Training seeds: `201350722, 201350732, 201350742`.
- Optimization seed: `201350749`.
- Candidate checkpoint SHA-256: `0945ea857abd584ed6f0144c608a211ff5423fd9334597126455b461aa28ca48`.
- Candidate actor SHA-256: `30b0fc33e91c9b8a3569f51225171bbaf3d29a831513b34c3a82ea553b7a6ae4`.

Key provenance:

- Source commit: `0043d83982f4041c4a601bd99092dbae2da5db7c`.
- Candidate checkpoint SHA-256: `c8fefe7574a27ac1c720544676602b18d59b3401c3925004293596aac752d34c`.
- Candidate actor SHA-256: `a791714290e4cd8358d19bb8f111a5599a828967de97c3ac4ffa190837d73737`.
- Training script: `experiments/scripts/refine_observable_cbf_ppo_v51.py`.

Files:

- `training_summary.json`: all paired-rollout, GAE, gradient, KL, and integrity diagnostics.
- `candidate.pt`: exact smoke-only checkpoint; do not treat it as the final actor.
- `execution_started.json`: immutable launch record.
- `v51_formal_training_summary.json`: full formal paired-rollout and update record.
- `v51_formal_candidate.pt`: exact formal candidate checkpoint.
- `v51_formal_execution_started.json`: formal launch record.
- `untouched_seed201350992_filter_off_{summary.json,episodes.csv}`: failed untouched deployment gate.
