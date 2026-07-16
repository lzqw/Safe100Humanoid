# Third-party notices and method attribution

## Source-code base

This repository is derived from
[`unitreerobotics/unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab),
which is distributed under the Apache License 2.0. Its license is preserved as
`LICENSE` and `LICENCE`. The source snapshot used for this work was pinned to
commit `1425b15f73bd4095f0df53709d7c389c3eb9e790` before the stair CBF commits
were applied.

The upstream project uses or acknowledges:

- [MJLab](https://github.com/mujocolab/mjlab), Apache-2.0;
- [MuJoCo](https://github.com/google-deepmind/mujoco), Apache-2.0;
- [MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp), Apache-2.0;
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl), BSD-3-Clause;
- Unitree G1 model and asset files distributed with `unitree_rl_mjlab`.

Additional dependency licenses bundled by the upstream project are preserved
under `doc/license/`.

## Research methods

The safety-training design is based on:

> CBF-RL: Safety Filtering Reinforcement Learning in Training with Control
> Barrier Functions, arXiv:2510.14959.

The Flat Patch navigation, position-based velocity command, and terrain-edge
safety representation are inspired by:

> Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids,
> arXiv:2601.07718.

No InstinctLab or `instinct_rl` source code is included in this repository.
No motion dataset, AMP/WASABI component, or author-provided humanoid CBF code is
included.

The public CBF-RL navigation demo was inspected for experiment organization,
but its 2-D environment source was not copied into the G1 implementation.
