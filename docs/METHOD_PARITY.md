# CBF-RL method parity

| Component | Paper/public demo | This implementation | Status |
|---|---|---|---|
| Safe set | Analytic `h(q) >= 0` | Swing-foot distance to next riser | Matched method |
| CBF condition | `grad(h)^T v + alpha h >= 0` | `-J_x(q) qdot + alpha h >= 0` | Matched stair equation |
| Safety filter | Closed-form single-halfspace QP | Batched closed-form PyTorch projection | Matched |
| Training mechanism | Filter plus a negative-margin term and bounded distance to the safe action/configuration | Eq. (27) using projected vs. nominal joint-position targets; QP stays in joint-velocity space | Matched equation; `sigma` assumed |
| Policy backend | PPO | MjlabOnPolicyRunner / rsl_rl PPO | Matched class |
| Humanoid action | Paper reports 12 lower-body actions | 12 hip/knee/ankle position targets; full 29-DoF G1 remains simulated | Matched policy dimension; upper body held at defaults |
| Navigation | Position goal to velocity command | Flat-patch tread target to body-frame velocity command | Hiking-inspired implementation |
| Edge representation | Stair/riser boundary | Exact risers recovered from generated tread patches | Equivalent for axis-aligned stairs |
| Reward sign | Eq. (23)/(27) penalizes negative margin and filter intervention | Direct implementation of Eq. (23); inactive constraints contribute zero | Matched |
| Deployment filter | Paper aims to remove it after learning | Training uses the filter; final evaluation reaches 95.31% success both with and without it | Verified, one seed |

Paper source: arXiv v6 (22 June 2026), SHA256
`6f97de25d8b4062382718c6fe8e1472614c506660c66666fc0379151fc20fd6a`.

The selected CBF is the paper's stair-specific barrier, not its planar obstacle
barrier. For the next riser tangent plane,
`h = x_stair - (x_swing_foot + toe_margin)` and the admissible joint velocity
halfspace is `-J_x^swing(q) qdot + alpha*h >= 0`. This is the most direct CBF
for the requested stair-climbing task because it targets the documented failure
mode: the swing toe clipping the next riser.
