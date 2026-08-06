# Observable Failure-Conditioned Brief PPO v19

v19 keeps two independent branches initialized from the same deployed policy:

- `pi0 -> pi_lateral`
- `pi0 -> pi_contact_stability`

There is no CBF specialist. Each branch still has one actor, one privileged
critic, one scalar reward, raw-action clipped PPO, runtime CBF execution, and
transactional candidate rollback.

## Observable actor extension

The actor input grows from 405 to 410 values. Lateral appends normalized
centerline error, sine/cosine heading error, and both error rates. Contact
stability appends left/right contact, left/right contact-foot slip speed, and
phase/contact mismatch.

The legacy first-layer prefix and every other actor tensor are copied exactly.
The five new first-layer columns are zero and the appended normalizer state is
identity. The warm-start loader records the maximum copy, zero-column, and
normalizer errors, so the pre-adaptation `pi0` preservation claim is auditable.

## Update protocol

Each round freezes one behavior policy and collects two independent
`64 x 1024` rollout batches. Normal, failure, and matched-success advantages
are normalized separately across both batches, then weighted `1.0`, `1.0`,
and `1.25`. Four paired minibatch losses are averaged across the two batches.
Actor-gradient cosine is recorded for diagnosis only.

Each batch uses 40 normal, 12 failure-precursor, and 12 matched-success slots.
The actor learning rate is `5e-6`, critic learning rate `1e-4`, PPO has one
epoch, clip `0.05`, target KL `0.003`, and hard KL ceiling `0.01`.

Three fractions (`0.5`, `1.0`, `1.5`) are screened with 64 paired episodes
each. Only the best point estimate receives a fresh 128-pair confirmation. It
is accepted only for strictly positive success delta, fall increase at most
3 percentage points, and KL below `0.01`. Runs allow eight rounds, require at
least three accepted updates, and stop after two later consecutive rejections.

## Specialist-specific mechanisms

Lateral uses recovery-potential progress instead of redundant static
centerline penalties. Its failure bank balances both centerline and heading
signs, early/mid/late risers, both support feet, and error-growth strata.
Successes match riser, support, phase, and both signs.

Contact stability uses only moderate foot-friction reduction, a two-step
contact-sensor lag, a small gait-clock offset, and small name-resolved
left/right sagittal response asymmetry. Swing-foot velocity is explicitly not
counted as slip. Its recovery reward is active around touchdown, and its bank
balances both touchdown feet, both slip feet, and early/delayed contact. A
sustained severe contact-slip event is timestamped so later terminal lateral
drift does not overwrite the observable primary cause.

## Prospective evidence boundary

The immutable protocol is
[`results/online/specialist_v19/protocol.json`](../results/online/specialist_v19/protocol.json).
It freezes base-only calibration candidates, five adaptation seeds, fresh
audit/bootstrap seeds, and every engineering gate before formal calibration.

Formal evaluation contains only two diagonal claims. For each specialist and
adaptation seed it uses 512 paired target episodes and 256 paired D0 episodes.
Each claim independently requires mean target success delta above zero, at
least four of five positive seed deltas, mean fall increase at most 3 points,
and mean D0 success delta at least -5 points. The 95% interval is reported;
positive LCB is strong evidence, not a separate gate.

No off-diagonal, macro, filter-free, CBF-independence, or joint all-specialist
gate belongs to v19.

## Development validation (not formal evidence)

Both branches completed end-to-end GPU smoke runs. The contact smoke produced
a fully diverse bank across both touchdown feet, both slip feet, and both
contact-timing classes. Both tiny confirmation sets conservatively rejected
non-improving candidates. Focused tensor/config/protocol tests pass. These
checks validate implementation plumbing only and are excluded from formal
claims.
