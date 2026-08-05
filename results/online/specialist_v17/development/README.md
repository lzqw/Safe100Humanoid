# v17 development evidence

This directory records the protocol-locked development result that justified
starting the nine formal specialist runs.  It is not formal evidence and does
not promote a policy on its own.

## Lateral seed-17 v6 result

The v6 change gives matched-success replay transitions a 1.5x actor weight in
the same scalar-reward PPO surrogate.  Failure-precursor transitions retain
their frozen 0.75x weight.  The implementation, validators, launcher, and tests
were pushed together in commit `f2ef800`; the remote simulator test suite passed
64/64 tests before training.

Training used the frozen lateral context and the full v17 protocol: 64
environments, 1,024 steps, five rounds, a 5e-6 maximum actor learning rate, a
1e-4 critic learning rate, one PPO epoch, clip 0.05, target KL 0.003, candidate
fractions 0.5/1.0/1.5, and a 512-episode paired target gate.  Rounds 1 and 2
were rejected.  Rounds 3, 4, and 5 passed the target and D0 gates.

The decisive audit was a fresh, strictly paired 512-episode lateral holdout on
seeds 1854000-1854003.  No gate or training seed was reused.

| Metric | pi0 | v6 | Change |
|---|---:|---:|---:|
| Success rate | 73.83% | 78.71% | +4.88 pp |
| Fall rate | 26.17% | 21.29% | -4.88 pp |
| Score, success - fall | 47.66% | 57.42% | +9.77 pp |

Per-block success changes were +5.47, +10.16, +3.91, and 0.00 percentage
points.  All initial-state signatures matched.  At the episode level there
were 73 repairs and 48 regressions (net +25).  A deterministic 200,000-resample
paired bootstrap gave a 95% interval of [+0.78, +9.18] percentage points for
the success-rate change; the exact two-sided McNemar p-value was 0.0287.

The complete compact record, including actor/checkpoint hashes, per-block
signatures, round decisions, failure-type counts, and raw-artifact hashes, is
in `lateral_seed17_v6.json`.  Formal status remains pending until all three
modes and all seeds (42/142/242) complete the frozen diagonal, D0, off-diagonal,
and hierarchical-bootstrap audit.
