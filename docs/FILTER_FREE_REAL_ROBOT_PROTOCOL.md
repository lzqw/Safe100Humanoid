# Filter-free real-robot protocol

Status: protocol only; no physical-robot experiment has been executed or
claimed by this repository.

## Objective

The physical experiment tests whether CBF-protected online adaptation produces
a nominal policy that can later traverse stairs without runtime CBF action
projection. CBF is a training-time safety shield and learning signal, not the
final controller.

The primary physical metric is post-adaptation deterministic CBF-off success.
Secondary outcomes are pre-to-post CBF-off improvement, shield-removal gap,
nominal violation rate, would-intervene fraction, and correction norm.

## Allowed methods

Only these methods may collect real adaptation data:

| Method | Executed action during adaptation | CBF reward |
|---|---|---|
| Filter-only FT | CBF-filtered action | No |
| Dual Safe-FT | CBF-filtered action | Yes |

Nominal FT and Reward-only FT remain simulation-only because their online
exploration is not protected. Both real methods use the same frozen v139
initialization, actor, critic, raw-action PPO buffer semantics, data budget, and
command protocol. The buffer stores the nominal sampled action and its old log
probability even though the robot executes the filtered action.

## Mandatory physical safeguards

No robot process may be started until a human operator verifies and records all
of the following:

- overhead tether or independently rated fall-arrest support;
- reachable tested emergency stop;
- wide stairs with clear lateral fall space;
- low-speed command limits and joint/current limits;
- continuous operator monitoring;
- recovery termination independent of the CBF implementation;
- synchronized robot state, fixed-stair localization, and command watchdog;
- reviewed 12-to-29 joint mapping and approved uncontrolled-joint posture.

Missing, stale, non-finite, or unsynchronized state must trigger the separately
reviewed fail-closed behavior. It must never silently bypass the filter during
adaptation.

## Frozen run contract

Before the first traversal, write an immutable run manifest containing:

- Git commit and v139 checkpoint/actor SHA-256;
- ONNX artifact and validation-report SHA-256;
- robot identifier and controller/firmware versions;
- staircase geometry and localization calibration;
- observation, command, action, CBF, and watchdog periods;
- pre-registered shadow and CBF-off release gates;
- operator, tether, emergency-stop, and recovery checks;
- adaptation method, seed, round count, and traversals per round.

Use 2–4 fixed online rounds and 8–12 traversals per round. Round count and
traversal count are selected before data collection and are not adjusted using
performance. Save rounds 0/1/2/4 when four rounds are used; the last fixed round
is the final policy. There is no candidate line search or performance-based
checkpoint selection.

## Phase 1 — CBF-on adaptation

Runtime action projection is always enabled. For every control transition log:

- observation timestamp and freshness;
- nominal and executed action;
- old log probability of the nominal action;
- nominal and filtered barrier margins;
- intervention flag and correction norm;
- contact, riser, termination, fall, and recovery state;
- watchdog, operator takeover, and emergency-stop state.

Report per round: nominal/executed violations, falls, recovery takeovers,
minimum nominal/filtered margin, and completed traversals.

## Phase 2 — shadow filter-free check

The robot still executes the CBF-filtered action. The nominal action is evaluated
in shadow mode only. Record would-intervene fraction, correction norm, nominal
barrier violation, and the pre-registered shadow success predictor.

The operator may proceed only if every pre-registered release gate passes. A
failed gate ends the run; it does not authorize threshold changes or checkpoint
selection.

## Phase 3 — protected CBF-off evaluation

Disabling runtime CBF requires a separate explicit operator action after the
shadow gate. Tether, emergency stop, low-level limits, command watchdog,
operator monitoring, and independent recovery termination remain active.

Evaluate pre- and post-adaptation policies using the same command protocol:

- at least three fixed staircase settings;
- 10–20 traversals per setting, fixed in the manifest;
- deterministic actor mean;
- paired or counterbalanced initial conditions where physically feasible;
- no online parameter updates during final evaluation.

CBF-off means only that runtime CBF action projection is disabled. It does not
mean disabling joint limits, current limits, watchdogs, tether, emergency stop,
or recovery logic.

## Required result record

For each traversal retain success, fall, reached riser, return-equivalent task
score, completion time, nominal violation per riser, would-intervene fraction,
correction norm, unsafe overlap, toe-riser contact, operator takeover, and
termination reason.

The final physical table reports CBF-off success before and after adaptation,
CBF-on success, shield gap, nominal violations, adaptation falls, recoveries,
and uncertainty intervals for each staircase and method.

The filter-free claim is supported only if Dual Safe-FT has the best fixed-final
CBF-off result and the lowest nominal violation under the pre-registered
comparison. Otherwise report the outcome without making that claim.

## Archival layout

When a supervised physical run is eventually authorized, archive compact
evidence under:

```text
results/real/filter_free_v140/<run_id>/
  RUN_MANIFEST.json
  SAFETY_CHECKLIST.json
  traversal_results.csv
  round_safety.csv
  final_table.csv
  SUMMARY.md
  artifacts.sha256
```

Raw high-rate logs and redundant checkpoints may remain in protected experiment
storage, but their hashes and retention location must be recorded in the run
manifest. Never label a protocol-only or simulation-only directory as a real
robot result.
