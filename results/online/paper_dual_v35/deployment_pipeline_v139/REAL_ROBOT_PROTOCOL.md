# Real-robot continuation protocol

This file records the next hardware stage; no real-robot traversal was run as part of the
simulation result in this directory.

## Frozen starting point

- Initial policy: v139 checkpoint SHA-256
  `323f1e00b58d379b8746c0191a44272f2e1df134139050417c56e733cc484728`.
- Runtime CBF remains enabled for every real action during baseline evaluation and training.
- The robot executes `a_safe = F_CBF(s, a_raw)` while PPO storage records `a_raw`.
- No real filter-off traversal is allowed. Nominal improvement is measured with shadow CBF
  intervention, barrier-violation, and nominal-to-safe action-distance metrics.

## First hardware run

1. Freeze the real staircase geometry and record it before collecting data.
2. Evaluate `v139 + CBF` for the predeclared baseline traversal count.
3. Run exactly two protected adaptation rounds, each with 8–12 traversals.
4. Publish the round-2 final actor; do not select a checkpoint from hardware outcomes.
5. Re-evaluate the final actor with CBF enabled on the same fixed staircase protocol.

Primary metrics are protected success, fall/recovery takeover, mean reached riser,
completion time, intervention events per riser, intervention steps per riser, and mean
correction norm. Shadow metrics are nominal would-intervene fraction, nominal barrier
violations per riser, and nominal-to-safe action distance.

Hardware-specific safety limits, operator takeover criteria, emergency-stop checks, and
the exact baseline/evaluation traversal counts must be approved and frozen with the robot
operator before execution. Simulation evidence in this directory is directional rather
than sufficient evidence that adaptation is safe or beneficial on hardware.
