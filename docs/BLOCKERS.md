# Blockers and scope limits

- The CBF-RL authors publish a 2-D navigation demo, but not the full humanoid
  stair code, action-interface mapping, swing-foot gating, or humanoid reward
  coefficients. This prevents an exact code/numerical reproduction.
- Paper v6 gives the bounded reward form but not the humanoid `sigma`, class-K
  gain, activation band, or toe margin. These are documented assumptions.
- The implementation is simulation-only. It does not run Unitree deployment,
  real-robot control, networking, or sim-to-real.
- Long training is contingent on the shared RTX 4080 being free enough for the
  selected environment count; OOM will be reported and handled by reducing only
  `num_envs`.
- The shared server intermittently starts a CARLA workload using roughly
  7--12 GiB. It is not owned by this reproduction and is never terminated; the
  formal run therefore uses 1024 rather than the isolated 4096-env maximum.
