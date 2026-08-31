# CBF seed-242 formal-training interruption

- Observation time: 2026-08-05 20:41 (UTC+08)
- Affected formal job: `specialist=cbf`, `adaptation_seed=242`
- Last atomic record: round 3 (`post_round_003.pt` and three entries in
  `online_rounds.json`)
- Uncommitted work: round 4; no round-4 checkpoint or JSON record was written
- Failure class: infrastructure/runtime allocation failure
- Exception: `RuntimeError: Failed to create Texture2D` from Warp while creating
  a MuJoCo render context for paired candidate evaluation

Recovery rule applied:

1. Preserve the complete interrupted output and logs outside the canonical
   formal-training tree.
2. Do not reuse partial round-4 evaluations or the round-3 checkpoint.
3. Restart the entire CBF seed-242 job from the common frozen policy.
4. Use identical repository source, frozen context, seed, hyperparameters,
   gates, and five-round protocol.
5. Include only a fully completed retry at the canonical formal output path.
6. Preserve any further failed retry before another whole-job retry, with a
   maximum of three attempts.

The restart was triggered only by the infrastructure exception, not by a
candidate outcome. The clean retry completed all five rounds and is the sole
CBF/seed242 run included in `training_manifest.json` and the final audit.

The excluded attempt remains on the execution host under:

```text
/home/carla/LZQW/SAFE100/humanoid/artifacts/specialist_v17/interrupted/
  cbf_seed242_initial_texture2d_20260805T2041/
```
