# Reproduction report

Current status: implementation, integration verification, and the requested
one-seed CBF/Nominal training comparison are complete.

This is a method-level reproduction because the paper's complete humanoid CBF
implementation is not public. Every claimed result will be tied to a log,
checkpoint, CSV, or test artifact in this project.

Verified evidence so far:

- Pure tensor tests: 5 passed.
- Real MuJoCo-Warp adversarial Jacobian test: nominal margin about -29.17,
  post-filter margin about +1.1e-6.
- Engineering tasks run 4 environments for 200 finite steps with 29 actions.
- Paper-spec task runs 4 environments for 200 finite steps with 12 lower-body
  actions, actor shape 405 (five-frame history), critic shape 283.
- Paper-spec 5-update PPO checkpoint loads strictly and produces a finite 12-D
  action; the bounded Eq. (27) reward is nonzero when the filter intervenes.
- Capacity scan reached 4096 environments without OOM; the conservative formal
  setting is 1024 because unrelated CARLA workloads can return to the shared GPU.
- The initial 500-iteration CBF run is an explicitly labeled engineering pre-run
  because its intervention reward preceded the v6 Eq. (23) correction.
- Deterministic 128-episode evaluation of that engineering pre-run yielded 0%
  top-platform success for CBF/filter-on, CBF/filter-off, and Nominal. This is
  reported as an ineffective short baseline, not a positive result.
- Formal paper-spec training is active in `tmux:g1_stairs_paper` with seed 42,
  1024 environments and 5000 iterations per comparison arm.
- The first paper-interface attempt evaluated at 0% after both 500 and 1000
  iterations and was stopped rather than misreported. The active cold restart
  adds the paper-stated next-stair clearance reference and scheduled swing-foot
  force penalty, shared by CBF and Nominal; their unpublished coefficients are
  recorded as engineering assumptions.
- A clearance-only restart was stopped after 41 iterations because that reward
  cannot address waiting before the first riser. The active cold restart adds a
  Hiking-inspired `dont_wait` common reward; it passed unit, 4-env/200-step,
  and 5-update integration checks before launch at commit `6b93ac0`.
- After adding the five-level stair curriculum (commit `1769c55`), the
  1000-iteration CBF policy reached 68.75% success with the filter and 73.44%
  without it over 128 deterministic episodes on exact 13 cm stairs. The
  corresponding CSV/JSON files are under `reports/` and `artifacts/`. This is
  an effective one-seed method-level result, not a paper numerical reproduction.
- At iteration 1500 the same policy achieved 95.31% success and 4.69% falls both
  with and without runtime filtering on exact 13 cm stairs. The corresponding
  `model_1500.pt`, ONNX export, TensorBoard events, JSON summaries and per-episode
  CSVs are retained.
- The same-seed, same-environment, same-reward Nominal control completed 1500
  PPO updates. Its final `model_1499.pt` (zero-based runner numbering) achieved
  0% success, 34.38% falls and 65.62% timeouts over the same 128 deterministic
  exact-13-cm episodes. Its mean barrier violation event count/integral was
  242.37/1006.18, compared with 3.23/7.56 for the CBF-trained policy evaluated
  without runtime filtering.
- This is an effective one-seed method-level result. It is not a multi-seed
  significance result, paper numerical reproduction, or real-robot result.
- A deterministic exact-13-cm rollout was recorded from `model_1500.pt` with
  the runtime CBF filter disabled. It reached 3.537 m without falling and had
  zero barrier violations in 1000 steps. The H.264 video contains 1000 frames
  at 854x480, 50 FPS, and passed a full ffmpeg decode check. Video:
  `videos/g1-stairs-cbf-filter-off-seed42-step-0.mp4`; metrics:
  `artifacts/video_cbf_filter_off_seed42.json`.
