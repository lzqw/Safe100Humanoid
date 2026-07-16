# Online Refinement Simulation Results

## Status

This report covers one RTX 4080 SUPER, training seed 42, simulation only. One
fall-aware, backtracked PPO candidate passed the transactional D0/DQ/DQN gate.
It is not a real-robot result, a multi-training-seed claim, or evidence that the
policy can remove its runtime CBF.

## OOD decomposition

The frozen base checkpoint is `results/models/cbf/model_1500.pt`.

| Domain | Episodes | Success | Fall | Mean reached riser | Interpretation |
|---|---:|---:|---:|---:|---|
| D0 | 4 | 100% | 0% | 6.00 / 6 | ID smoke |
| D1 | 4 | 75% | 25% | 16.75 / 18 | horizon-only shift |
| D2 | 4 | 100% | 0% | 6.00 / 6 | command-only shift |
| D3 | 4 | 0% | 100% | 10.75 / 18 | horizon + joystick |
| D4 | 4 | 0% | 100% | 10.25 / 18 | full target OOD |

The four-episode rows are calibration smoke tests, not confidence intervals.
They establish that horizon and command shifts interact: each isolated shift
is manageable, while their combination is too hard for brief refinement from
this checkpoint.

The 9-riser DQ quick target was therefore introduced without changing D4/D5.
Its 16-episode calibration was 62.5% success, 37.5% falls, and mean 8.625/9
risers, placing failures near the end of the trajectory.

## First candidate safety audit

The first conservative candidate used two critic burn-in passes and one
CBF-protected PPO update. Its update rollout contained actual CBF intervention
fraction 4.25%, nonzero ten-step backward credit, actor LR `1e-5`, and total KL
from the base actor about `0.00139`.

A 128-episode GPU audit rejected this candidate:

| Domain | Base success | Candidate success | Base fall | Candidate fall |
|---|---:|---:|---:|---:|
| D0 | 94.53% | 92.97% | 3.91% | 3.91% |
| DQ | 48.44% | 51.56% | 51.56% | 48.44% |
| DQN | 48.44% | 53.91% | 51.56% | 46.09% |

The DQ gain was only 3.12 percentage points (binomial standard error about
4.42 points), while intervention per reached riser increased from 0.865 to
0.989, above the configured 5% safety tolerance. The decision was `rollback`.
The retained checkpoint contains the base actor and the calibrated 799-D full
critic, not the rejected candidate actor.

A second three-round run used longer 768-step rollouts, lower exploration, and
stronger pre-intervention credit. All candidates were rolled back. The most
promising 64-episode candidate (round 2) improved DQ success from 48.44% to
54.69% and reduced DQ intervention/riser from 1.028 to 0.860, but missed the
neighbor gate by two DQN episodes. A separate 128-episode GPU audit did not
reproduce it: DQ success was 37.50% versus the base 48.44%, and intervention
rose to 1.117. It therefore remains rejected. This large-batch reversal is why
the repository does not report the small-batch candidate as an improvement.

The next experiment enables the separately labeled safe-action auxiliary only
on true CBF projection states; its result must pass the same gate.

In that experiment the auxiliary was genuinely active (`safe_bc_loss=0.0471`,
projection fraction 4.77%). Round 1 slightly reduced DQ intervention/riser
from 1.089 to 1.076 but reduced DQ success from 50.00% to 45.31%, so it was
rolled back. An initial round-2 attempt did not complete: an unrelated `srl100` pytest
process occupied about 12.3 GiB and MuJoCo-Warp exited with CUDA illegal memory
access. No result is inferred from the incomplete attempt. The experiment was
rerun after the GPU became idle; both completed candidates regressed DQ and
were rejected.

Increasing critic-only calibration to eight rounds produced explained variance
0.618 and return/value correlation 0.850 before the actor update, showing that
the expanded critic can be calibrated. That candidate still regressed DQ and
DQN and was rejected. Increasing the actor rollout to 32 environments (24,576
samples) stabilized success but left DQ unchanged at 45.31% and increased
intervention/riser from 0.936 to 1.035, so the gate again rolled back.

A final 32-environment run with safe-action BC weight 0.02 also failed the
gate: DQ success changed from 56.25% to 40.63%, while intervention/riser rose
from 0.892 to 1.041. These experiments all preceded the fall-only reward fix
below and remain rejected evidence.

## Accepted fall-aware round

The online environment originally disabled the generic termination penalty so
that successful top completion was not punished. That also removed the
explicit fall penalty. The corrected task adds a fall-only reward with weight
`-200`, leaving successful termination untouched. An 8-env, 768-step GPU
smoke observed nine real falls and verified that the signal entered storage.

The accepted candidate used 32 environments, 768 steps/environment (24,576
transitions), eight critic-only calibration rounds, and one two-epoch PPO
update. The update rollout contained 42 falls and 4.55% true CBF intervention.
Critic explained variance was 0.619 and return/value correlation was 0.809.
PPO mean KL was 0.000659 and clip fraction 0.155, below the configured 0.003
and 0.30 precheck limits. A policy line search selected fraction 0.50 of this
single PPO direction.

The machine-readable gate returned `accepted: true` with no rejection reasons:

| Domain | Episodes | Base success | Accepted success | Base fall | Accepted fall | Base intervention/riser | Accepted intervention/riser |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0 | 128 | 97.66% | 96.88% | 2.34% | 3.13% | 0.262 | 0.224 |
| DQ | 512 | 48.05% | 48.05% | 51.95% | 51.95% | 1.034 | 1.009 |
| DQN | 128 | 40.63% | 39.06% | 59.38% | 60.94% | 1.147 | 1.113 |

DQ task success and fall rate were retained while intervention/riser fell
2.42% and mean correction fell 9.46% (`0.002798` to `0.002533`). D0 and DQN
success changes remained inside their predeclared two-percentage-point bounds.
This is a modest safety-internalization improvement, not evidence that rare
falls were eliminated.

An optional corrected safe-action SGD auxiliary was tested after PPO. Its
0.10 line-search point looked favorable in 64 episodes but regressed DQ in a
larger audit; the final accepted actor therefore uses PPO backtracking only.

With the runtime filter disabled, a separate 256-episode audit measured 48.44%
base success and 46.48% accepted success. The 1.95-point difference is smaller
than one binomial standard error but provides no evidence of filter-free
improvement. Runtime CBF remains mandatory in this prototype.

## Final GPU validation and videos

The final clean regression used the RTX 4080 SUPER and produced:

- 14/14 pure tensor and gate tests passing;
- strict reload of actor, 799-D critic, and optimizer from the accepted
  checkpoint, with finite 12-D action and value outputs;
- an unchanged upstream six-riser task smoke of 100 GPU steps;
- an adversarial CBF check that repaired nominal margin `-29.1649` to filtered
  margin `+1.09e-6`;
- two deterministic DQ videos from the accepted actor and runtime CBF.

The accepted-policy videos deliberately include both outcomes. Seed 43 reaches
9/9 risers in 557 steps without falling. Seed 42 reaches riser 7 and then
falls at step 438. The previous base videos remain available for comparison;
single videos are qualitative evidence and are not substituted for the
512-episode DQ audit.

## Curated repository evidence

```text
results/online/evaluation/
results/online/runs/
results/online/smoke/
results/online/checkpoints/accepted_round_001.pt
results/online/evaluation/accepted_round1/final_candidate_gate.json
results/online/videos/accepted_round1/
```

The older rollback checkpoint is retained as negative transactional evidence.
The accepted checkpoint contains the backtracked actor, calibrated full critic,
and optimizer plus a portable acceptance manifest. Hashes are recorded in
`results/online/SHA256SUMS`.
