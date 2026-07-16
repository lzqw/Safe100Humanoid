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
- During the safe-BC round-2 gate, an unrelated `srl100` pytest process used
  about 12.3 GiB on the same GPU. MuJoCo-Warp then reported CUDA illegal memory
  access and the online run exited nonzero. The external process was inspected
  but not terminated; the failed gate must be rerun after GPU capacity returns.
- The current base checkpoint has zero success in small D3/D4 calibration
  batches and fails around risers 9--11. Brief online refinement on D4 is not
  yet statistically meaningful; DQ is used to validate the algorithm while a
  stronger long-horizon base policy is trained.
- MuJoCo-Warp GPU contact solving is not bitwise deterministic across repeated
  environment construction. Candidate gates therefore require aggregated GPU
  batches; four-episode smoke comparisons are not acceptance evidence.
- No real stair profile, joystick trace, hardware state estimator, emergency
  stop, or independent whole-body safety controller is connected. The online
  module remains a simulation of the future real-hardware workflow.
