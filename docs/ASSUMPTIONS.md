# Engineering assumptions

The CBF-RL paper and its public navigation demo do not publish the complete
humanoid stair implementation. The following choices are therefore explicit
method-level reproduction assumptions rather than claimed author parameters.

- Control interface: the policy's position target is converted to nominal joint
  velocity with `(q_target - q) / 0.02 s` before applying the CBF projection.
- Formal paper-spec policy controls the 12 hip/knee/ankle DoFs, uses a five-frame
  proprioceptive actor history, and reserves the one-frame height scan for the
  asymmetric critic. The full 29-DoF model remains simulated; uncontrolled
  upper-body joints hold their default position targets.
- Barrier: `h = x_riser - (x_swing_foot + 0.08 m)`.
- Class-K gain: `alpha = 10 s^-1`.
- The constraint activates within 0.30 m before a riser, remains recoverable up
  to 0.15 m behind it, and deactivates after foot height exceeds the tread by
  0.025 m.
- The paper states that stair height modifies the foot-clearance reference and
  that swing-foot force is penalized, but gives no coefficients. The corrected
  run uses a target 0.05 m above the upcoming tread, keeps the upstream -1.0
  clearance weight, and adds gait-scheduled swing-foot force with weight
  -0.001. These common rewards are identical for CBF and Nominal.
- To avoid the observed standing-to-timeout local optimum, the active run adds
  the Hiking-inspired `dont_wait` common reward: weight -1.0 times the normalized
  deficit below 0.1 m/s whenever commanded forward speed exceeds 0.2 m/s. This
  is a task-engineering assumption, not a claimed CBF-RL author coefficient.
- Swing foot is the non-contact foot with the longest current air time. The CBF
  is inactive in double support.
- Training configures five curriculum bins over 0.02--0.155 m rise; MJLab
  samples within bins, yielding about 0.041--0.131 m for seed 42. Evaluation
  rebuilds an exact fixed 0.13 m staircase. Run is 0.35 m and there are six
  steps; exact risers are recovered separately from each level's flat patches.
- The formal CBF run uses paper Eq. (23)/(27) exactly in structure:
  `min(psi_nominal, 0) + exp(-||qtarget_policy-qtarget_safe||^2 / sigma^2) - 1`.
  The QP itself acts in joint-velocity space, while this target-position
  distance follows the humanoid stair notation in Eq. (27).
  The humanoid section does not publish `sigma`; we use `sigma=0.5`, matching
  the value shown for the paper's single-integrator experiment. MJLab applies
  its common `dt` reward scaling. An earlier 500-iteration engineering pre-run
  used `-10*relu(-psi)-0.01*||intervention||^2`; it remains labeled separately
  and is not reported as paper-spec training.
- The comparison uses seed 42 only, per user request; it is an engineering
  effectiveness demonstration, not a statistical numerical reproduction.
