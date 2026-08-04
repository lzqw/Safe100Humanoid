# Failure-Focused Brief PPO (v15)

v15 is a minimal extension of the v14 online-refinement path. It keeps one
405-D actor, one 838-D privileged value critic, one scalar reward, GAE,
clipped PPO, transactional candidate rollback, and the runtime CBF. It does
not add cost critics, a risk head, retention banks, or an actor-visible context
identifier.

## Frozen hidden deployment context

`calibrate_failure_focused_context_v15.py` deterministically generates contexts
from the ordered calibration seeds 1000--1019. Each context combines fixed
terrain geometry, joystick delay/filtering, action gain/bias/delay, and encoder
bias. The base policy is evaluated for 128 independent episodes and the first
context with 75--85% success is frozen. Selection reads only base-policy
success; it never evaluates an adapted policy. The frozen parameters have a
canonical SHA-256 identity and carry an auditable selection record.

The actor receives no context fields. PPO stores the sampled raw policy action
and its behavior log probability, while the environment applies the hidden
plant transform and the runtime CBF before execution.

```bash
bash experiments/scripts/calibrate_failure_focused_context_v15.sh
```

## Five-round refinement

Each adaptation seed uses 64 environments for 1024 steps per round:

- 54 persistent normal starts and 10 persistent target late-failure starts;
- target-only bank entries selected 50--150 steps before an actual
  DQH-Medium fall and restricted to the latter half of the staircase;
- hard-case actor weight 0.75;
- terminal fall penalty -2 and a second -2 redistributed over up to 100
  preceding transitions with decay 0.97, preserving an undiscounted total of
  -4 per fall;
- online dual CBF reward weight exactly zero in all five rounds;
- actor LR `5e-6`, critic LR `1e-4`, one epoch, clip 0.05, target KL 0.003,
  exploration standard deviation 0.35 times the base value;
- candidate fractions 0.5, 1.0, and 1.5, each screened on the same 128 paired
  target episodes;
- point acceptance only for improved `success - fall`, fall increase no more
  than 0.03, finite parameters and KL below 0.01, and CBF demand no more than
  1.25 times the old value;
- D0 checked every two rounds with a five-percentage-point tolerance;
  DQNH-Medium is never a training gate.

The target-only bank is built by base-policy failure-discovery rollouts at the
start of each adaptation run and is then refreshed only by subsequent target
falls. General intervention states and successful crossings are not inserted.

```bash
for seed in 42 142 242; do
  SAFE100_SEED="$seed" bash experiments/scripts/run_failure_focused_v15.sh
done
```

## Independent final audit

The final audit uses a disjoint seed range and paired initial conditions for
the online-start policy and each final policy:

| Domain | Episodes per adaptation seed | Role |
| --- | ---: | --- |
| DQH-Medium | 512 | target |
| D0 | 256 | retention |
| DQNH-Medium | 256 | report-only neighbor |

It hierarchically bootstraps adaptation seeds and paired episodes. Promotion
requires all three predeclared gates:

1. `LCB95[delta target success] > 0`;
2. `LCB95[delta D0 success + 0.05] >= 0`;
3. `UCB95[delta target fall] <= 0.03`.

```bash
bash experiments/scripts/run_final_audit_v15.sh
```

The supported claim is deliberately limited to refinement under one fixed,
training-unseen, algorithm-hidden composite deployment context. The runtime
CBF remains enabled during calibration, adaptation, candidate screening, and
the final audit.

## Result status

Formal calibration, all three adaptation seeds, and the independent final
audit are complete. The compact evidence package is published under
[`results/online/failure_focused_v15/`](../results/online/failure_focused_v15/README.md).

The calibration selected candidate seed 1000, the first candidate tested, at
`103/128 = 80.469%` base success. Selection used only base-policy success and
did not evaluate any adapted policy. The frozen parameter hash is
`4eafa5b94792f8b709e2da98ed3638a285790497a1e876e937373d55ce8d75bf`.

The final hierarchical paired audit used 512 target, 256 D0, and 256
report-only neighbor episodes per adaptation seed. Target success changed from
`82.747%` to `83.529%`, a paired delta of `+0.781` percentage points with 95%
CI `[-1.628, +3.190]`. Because the lower confidence bound is not above zero,
the target-improvement gate **failed**. Target fall changed by `-0.781` points
with CI `[-3.255, +1.563]`, passing its safety gate. D0 success changed by
`+0.260` points with CI `[-1.823, +2.474]`, passing the predeclared
five-point-margin noninferiority gate.

Consequently, v15 does **not** establish a statistically supported target
improvement. The predeclared decision tree selects **Branch B**: classify the
target falls and rebuild the hard-case bank around only the dominant failure
type. DQNH-Medium and CBF-demand changes remain report-only.
