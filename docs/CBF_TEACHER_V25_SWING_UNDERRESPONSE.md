# v25 Swing-Foot Under-Clearance — Prospective Protocol

v25 is an independent successor to the frozen v23 lateral and v24 contact
negative results. It does not modify, rerun, recompute, or reinterpret either
result. The experiment tests the causal chain **CBF protects + CBF teaches +
PPO optimizes the task** in one shift deliberately aligned with the existing
toe/riser CBF.

v25 是 v23 lateral 和 v24 contact 已冻结负结果之后的独立实验；不修改、不重跑、
不重算、也不重新解释前两项结果。它只检验一条因果链：**CBF 保护 + CBF 教学 +
PPO 优化任务**。

Revision 1 was retired before any v25 simulator episode. A pre-execution audit
found that its adaptation entrypoint loaded the ordinary noisy training variant
while calibration and final evaluation loaded the fixed deployment variant. It
also identified toe/riser identity and pooled intervention-rate audit gaps.
Revision 2 fixes those issues prospectively, preserves revision 1 as immutable
history, and reuses the original fresh seed schedule because no v25 outcome was
observed.

Revision 2 was itself retired before any v25 simulator episode. A second
pre-execution audit found that the terminal verifier compared a reconstructed
gate-only dictionary against the selected-attempt dictionary that also contains
candidate metadata, so every qualifying result would have been mislabeled as a
verification failure. It also treated at least one updated round and at least one
teacher transition as mandatory evidence, although the frozen method permits
hard rollback and defines an empty teacher minibatch as exact zero. Revision 3
fixes only those result-independent verification semantics and adds complete
per-episode calibration evidence reconstruction before publication. The ordered
grid, gates, environment, algorithm, and all 105 fresh seeds remain unchanged;
revision 1 and revision 2 remain immutable zero-episode history.

Revision 3 was also retired before any v25 simulator episode. A synthetic
end-to-end package audit showed that the terminal verifier accepted tampering
of the frozen final identity schedule, single-adaptation/exclusion counts, and
several evidence hash bindings. Revision 4 closes those prospective audit gaps:
it binds both external checkpoints, every calibration marker and compact
candidate-evidence file, the training start/completion markers, all four final
condition contracts, the exact ordered 512-identity schedule, and the complete
derived result table. The audit still accepts scientifically valid negative
outcomes, including eight hard rollbacks or zero eligible teacher transitions.
No simulator episode was run under revisions 1–3, so the grid, gates,
environment, algorithm, and all 105 fresh seeds remain unchanged.

Revision 4 was likewise retired before any v25 simulator episode. A complete
producer-versus-verifier contract audit found that training records the full
fixed-deployment runtime audit object, while the terminal verifier incorrectly
expected the same field to be the scalar boolean `true`. That mismatch would
have mislabeled every otherwise valid terminal package as a verification
failure. Revision 5 makes the exact runtime audit object a shared verification
contract. This is an evidence-only correction: the ordered grid, gates,
environment, algorithm, checkpoint, and all 105 fresh seeds are unchanged.
Revisions 1–4 remain immutable zero-episode history.

Revision 5 was also retired before any v25 simulator episode. A complete
zero-simulator producer-to-verifier rehearsal found that Python's CSV writer
emitted CRLF records while Git's default text normalization stored LF records.
The formal freezer intentionally byte-binds committed calibration evidence, so
that normalization would have rejected authentic evidence immediately after it
was committed. Revision 6 gives every v25 CSV an explicit LF record terminator
and marks result CSVs as binary evidence in `.gitattributes`, preventing Git
from rewriting their bytes. The attribute file is itself part of the frozen
source boundary. No environment, grid, gate, algorithm, checkpoint, or seed was
changed; revisions 1–5 remain immutable zero-episode history.

Revision 6 was likewise retired before any v25 simulator episode. The same
zero-simulator rehearsal then reached the terminal verifier and exposed one
legacy-history incompatibility: revision 2's link to the original unnumbered
revision-1 protocol predates the `supersedes_revision` field, while the verifier
required that newer field on every link. Revision 7 accepts the missing field
only for that exact hash-bound legacy revision-1 path; all newer links still
require the explicit revision number. The experiment design and all 105 seeds
remain unchanged, and revisions 1–6 remain immutable zero-episode history.

## Fixed deployment shift

The environment remains the fixed nominal `DQHMED` staircase. Geometry,
friction, command, centerline controller, reward, actor observations, and CBF
riser metadata remain unchanged. The only deployment mismatch is a fixed
under-response gain on hip pitch, knee, and ankle pitch of the currently
swinging leg. The stance leg and every other action dimension retain gain 1.
All three phases load the same registered fixed-deployment (`play`) variant:
actor observation corruption, encoder bias, curriculum, physical
randomization, and pushes are disabled. Only the nominal reset pose/joint
events remain, providing the fresh paired initial conditions. This prevents
adaptation from silently adding a second observation-noise shift that is absent
from calibration and final evaluation.

The ordered candidate grid is `0.98, 0.96, ..., 0.50`. Before any adaptation,
the base policy is evaluated on 512 paired conditions with CBF off and on for
each severity. The first candidate satisfying all four inclusive gates is
frozen:

- toe/riser alignment coverage among CBF-off failures at least 80%;
- paired CBF rescue among CBF-off failures at least 60%;
- CBF-off success between 40% and 65%;
- CBF-on success between 80% and 95%.

A toe/riser kick is the debounced entry of a selected swing-toe/riser pair into
the exact CBF unsafe half-space (`h <= 0`). Debouncing is keyed by both foot and
riser, so a direct switch to another toe or riser remains a new event. This is
an exact geometry-derived clearance violation proxy rather than a generic
balance classifier.

## Action and teacher dataflow

The actor samples `a_policy`; PPO stores that raw action and its behavior log
probability. The fixed hidden plant map is applied before the runtime CBF, and
the environment executes the projected plant-side action. PPO never replaces
its ratio action with the filtered action.

Because the existing safe raw action is expressed after the hidden plant
gain, v25 inverts the plant transform before constructing the actor teacher.
Direct imitation of the post-plant value would apply the under-response twice.
Every transition therefore audits that forwarding the actor-coordinate
teacher back through the plant reproduces the CBF-safe action within `1e-6`.

An intervention is eligible only if the runtime CBF really changed the
action, the next riser is crossed within `H=50` control steps (1.0 s), and no
fall occurs in that horizon. Look-ahead never crosses an episode boundary.
An episode terminal supplies a complete outcome; an unfinished rollout tail
shorter than `H` is not treated as evidence of survival.
The correction weight is clipped at one using scale `s_D=0.05`. Teacher
targets are stop-gradient values. Each minibatch divides the weighted
Gaussian NLL numerator by its valid teacher-transition count; an empty
teacher minibatch contributes exact zero.

The actor objective is:

`-L_clip + 0.5 KL(pi_theta || pi_k) + 0.1 L_CBF-teacher`.

## Fixed adaptation

The formal run uses one original 405-D actor, one original 838-D privileged
critic, runtime CBF, raw-action PPO, a round-start moving reference, 8 rounds,
64 environments, and 1024 control steps per environment and round. Round 8 is
the final policy regardless of performance. Only corruption (non-finite
state, KL above 0.01, or action/teacher dataflow failure) permits a
transactional rollback.

The experiment excludes failure banks, state restarts, candidate line search,
performance gates, best-checkpoint selection, new observations, multiple
adaptation seeds, randomized per-riser geometry, and hidden episode-varying
shift variables.

## Four-condition final audit

The same 512 fresh initial conditions are evaluated under:

1. `pi0`, CBF off;
2. `pi0`, CBF on;
3. `pi8`, CBF on;
4. `pi8`, CBF off.

The primary outcomes are shielded success change, unshielded internalization
change, toe/riser kick change, and CBF interventions per riser. Development
success requires at least +5 percentage points in unshielded success,
nonnegative shielded success change, a strict unshielded kick-rate reduction,
and at least a 20% reduction in shielded interventions per riser. These gates
are post-training reports only and cannot select or alter the policy.
Interventions per riser is the pooled count ratio over all 512 episodes
(`total intervention steps / total risers crossed`), not an unweighted mean of
per-episode ratios.

The final evidence verifier reconstructs every gate field while separately
binding the selected-attempt metadata. It validates every updated round that
exists without requiring an update to occur, and validates teacher-signal
accounting without requiring the environment to produce a positive teacher
count. Thus a scientifically negative result or eight valid hard rollbacks is
reported honestly rather than converted into a technical verification failure.
It additionally byte-binds the complete prospective chain, every calibration
candidate, execution boundary markers, pi0/pi8 checkpoint actors, exact final
identity order, condition metadata, paired repair/regression counts, and all
reported primary deltas.

The motivating actuator-mismatch, action-projection aliasing, and CBF-learning
references are the primary arXiv records:

- <https://arxiv.org/abs/2504.06585>
- <https://arxiv.org/abs/2509.12833>
- <https://arxiv.org/abs/2510.14959>
