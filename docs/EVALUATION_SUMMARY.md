# Evaluation summary

The 500-iteration 29-DoF engineering pre-run is not effective at stair
climbing. Across 128 deterministic episodes per condition, all methods reached
the 20-second timeout without falling but none reached the top platform.

| Condition | Success | Fall | Timeout | Mean max progress |
|---|---:|---:|---:|---:|
| CBF-trained, filter on | 0% | 0% | 100% | 0.0253 m |
| CBF-trained, filter off | 0% | 0% | 100% | 0.0252 m |
| Nominal, filter off | 0% | 0% | 100% | 0.0344 m |

These results establish that 500 iterations produce a stationary policy, not
the requested effect. They are retained as a negative short-training baseline.
Paper-spec training (12 lower-body actions, five-frame proprioception, bounded
Eq. (27) reward) was also evaluated at 500 and 1000 iterations. Filter-on
success remained 0%; mean max progress was respectively 0.0188 m and 0.0230 m.
The run was stopped because it had converged to standing. The active cold
restart includes the two common reward modifications explicitly named after
Eq. (27): stair-relative foot clearance and swing-foot force. It is evaluated
separately and is not yet complete.

The active curriculum run has produced an effective checkpoint. On an exact
13 cm, six-step staircase with 128 deterministic episodes:

| Checkpoint/condition | Success | Fall | Timeout | Mean max progress |
|---|---:|---:|---:|---:|
| CBF iter 500, filter on | 0.00% | 29.69% | 70.31% | 0.566 m |
| CBF iter 500, filter off | 0.00% | 26.56% | 73.44% | 0.545 m |
| CBF iter 1000, filter on | 68.75% | 15.62% | 84.38% | 2.880 m |
| CBF iter 1000, filter off | 73.44% | 10.94% | 89.06% | 2.926 m |
| CBF iter 1500, filter on | 95.31% | 4.69% | 95.31% | 3.309 m |
| CBF iter 1500, filter off | 95.31% | 4.69% | 95.31% | 3.315 m |
| Nominal iter 500 | 0.00% | 100.00% | 0.00% | 1.055 m |
| Nominal iter 1000 | 0.00% | 100.00% | 0.00% | 1.055 m |
| Nominal final (update 1500; file `model_1499.pt`) | 0.00% | 34.38% | 65.62% | 0.748 m |

The success definition is root progress at least 2.65 m, the start of the top
platform minus a 5 cm tolerance. A successful episode can still be recorded as
a timeout because success is evaluated from maximum progress while the simulator
continues until its normal termination; these columns are not exclusive.

This demonstrates effective stair climbing and filter-free deployment of the
CBF-trained policy. The equal-budget, same-seed Nominal arm completed all 1500
updates and did not solve the fixed 13 cm evaluation. Its final policy timed out
in 65.62% of episodes and fell in 34.38%; no episode reached the top-platform
success threshold. The CBF-trained policy therefore exceeds this specific
one-seed Nominal control by 95.31 percentage points in success rate. This is not
a multi-seed statistical claim or a reproduction of the paper's numerical
table.

By iteration 1500, filter-free nominal CBF violations fell to 3.23 events and
7.56 integrated violation units per episode, from 10.60 and 37.43 at iteration
1000. Filter-on/off success and fall rates are identical at this checkpoint.
The final Nominal policy has 242.37 violation events and 1006.18 integrated
violation units per episode under the same diagnostic, versus 3.23 and 7.56 for
the filter-free CBF-trained policy. The diagnostic is evaluated from the same
stair barrier even though the Nominal policy never applies the safety filter.

The Nominal final checkpoint is named `model_1499.pt` because the runner logs
iterations from zero: it is the state after 1500 PPO updates. All rows use 128
deterministic episodes, seed 42, and an independently rebuilt exact 0.13 m
staircase. Success and timeout are intentionally non-exclusive as described
above.
