# v51 Geometry-Aware Paired CBF PPO

v51 replaces direct safe-action regression with a paper-style paired PPO
update.  Three seeds each collect filter-on and filter-off rollouts, use the
bounded `raw_moderate` CBF dual reward with critic GAE, average the on/off
gradients within a seed, and then take a coordinate median across seeds.
Only the five new geometry-input columns are trainable; the original 405-D
policy is frozen exactly.

This directory currently contains the implementation smoke result, not a
deployment-qualified checkpoint.

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

Key provenance:

- Source commit: `0043d83982f4041c4a601bd99092dbae2da5db7c`.
- Candidate checkpoint SHA-256: `c8fefe7574a27ac1c720544676602b18d59b3401c3925004293596aac752d34c`.
- Candidate actor SHA-256: `a791714290e4cd8358d19bb8f111a5599a828967de97c3ac4ffa190837d73737`.
- Training script: `experiments/scripts/refine_observable_cbf_ppo_v51.py`.

Files:

- `training_summary.json`: all paired-rollout, GAE, gradient, KL, and integrity diagnostics.
- `candidate.pt`: exact smoke-only checkpoint; do not treat it as the final actor.
- `execution_started.json`: immutable launch record.

