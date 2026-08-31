# v17 requirement-by-requirement audit

This audit separates protocol compliance from the scientific acceptance result.
The protocol was completed as declared; the final performance claim failed.

| Requirement | Evidence | Protocol verdict |
|---|---|---|
| One common frozen policy for all specialists | The independent audit reports one base-file hash (`cb875d...a892a8`) and one initial actor hash (`194322...47b1`) across all nine runs. | PASS |
| Three independently trained specialists; never chained | Every run attests `independent_training_branch=true`; all initial actor hashes are identical and no final actor is used as another run's base. | PASS |
| Base-only context calibration | Each calibration records `base_policy_only=true`, `adapted_policy_evaluations_used=false`, 512 episodes, and disjoint calibration/training/audit seeds. | PASS |
| Calibration SR 70–85%, at least 100 falls, target at least 60%, second class at most 30% | lateral: 78.906%, 108, 85.185%, 9.259%; CBF: 78.906%, 108, 71.296%, 19.444%; balance: 77.344%, 116, 61.207%, 22.414%. | PASS |
| Frozen context parameters and hashes | Parameter hashes `e79dad...bf85`, `da07fd...e3d1`, and `5bf282...11e81`; context-file hashes match the audit inputs. | PASS |
| One actor and one privileged critic; no extra safety/risk critics | All nine structural checks record one actor, one critic, absent fall/intervention critics and risk head. | PASS |
| One scalar reward; no dual CBF reward | Training summaries identify one task/fall/mode-specific scalar reward and `dual_cbf_reward_weight=0`. | PASS |
| PPO stores raw policy actions/log-probabilities while runtime executes CBF-safe actions | All runs record `raw_policy_action_for_ppo=true`, `executed_action=runtime_cbf_safe_action`, and `runtime_cbf=true`; routing/storage consistency checks are present in every round. | PASS |
| Failure precursors plus matched-success counterexamples | Final banks contain only the declared failure class. Every success bank has the same size, a complete matched count, and unique source count. | PASS |
| Failure windows | lateral 50–150, CBF 10–48, balance 20–78 steps before fall, all within the predeclared mode windows. | PASS |
| 70/15/15 start mixture | Every run records integer allocation normal/failure/success = 44/10/10 for 64 environments. | PASS |
| PPO batch and optimizer profile | `run_specialist_v17.sh` fixes 64 environments × 1024 steps, five rounds, candidate fractions 0.5/1.0/1.5, actor LR 5e-6, critic LR 1e-4. Structural checks record one PPO epoch, clip 0.05, target KL 0.003, and layer multipliers 0.10/0.25/0.50/1.0. | PASS |
| KL limits | Across 135 candidate evaluations, maximum mean KL was 0.001214 and maximum recorded pre-update minibatch KL was 0.000664; both are below the declared 0.003 target and 0.01 hard ceiling. | PASS |
| Target-only training gate, fall tolerance, KL, and D0 catastrophic check | Every round contains three paired 512-episode candidates, target score/fall/KL decisions, and a D0 check. Two candidates were rolled back by D0. | PASS |
| No other-specialist, neighbor, or CBF-demand training gates | All nine summaries record these three gate flags as `false`. | PASS |
| Isolation is not implemented through retention anchors/banks or multiple critics | Structural checks record all retention/base/neighbor anchor weights as zero, retention bank count zero, absent actor references, and no extra critics. | PASS |
| Exactly 3 modes × 3 seeds × 5 rounds | The compact training manifest contains nine formal runs, seeds 42/142/242, and round indices 1–5 for every run. | PASS |
| Final diagonal audit | 512 paired episodes per adaptation seed and specialist scene, independent of training gates. | PASS |
| Final D0 audit | 256 paired episodes per adaptation seed; all three aggregate D0 drops are better than the -5pp floor. | PASS |
| Off-diagonal audit is report-only | 256 paired episodes per off-diagonal cell; the audit records `off_diagonal_results_are_report_only=true`. | PASS |
| Paired data integrity | 11,520 CSV data rows; recomputed CSV SHA-256 equals the hash embedded in the final summary; baseline/final initial-state signatures were checked during execution. | PASS |
| Hierarchical macro bootstrap | 10,000 samples, seed 2,800,000, over scenes, adaptation seeds, and paired episodes. | PASS |
| Formal scene acceptance | Requires diagonal gain >2pp, at least 2/3 positive seeds, fall increase ≤3pp, and D0 drop ≤5pp. No scene passed all criteria. | **FAIL (outcome)** |
| Formal macro acceptance | Mean +0.673pp, LCB95 -1.215pp, UCB95 +2.344pp; requires LCB95 >0. | **FAIL (outcome)** |
| Final claim | Requires all three scene gates and macro gate. | **FAIL (outcome)** |

## Hard-case bank audit

| Mode / seed | Failure entries | Window | Failure purity | Matched successes | Unique success sources |
|---|---:|---|---|---:|---:|
| lateral / 42 | 256 | 50–150 | 256 lateral | 256 | 256 |
| lateral / 142 | 256 | 50–150 | 256 lateral | 256 | 256 |
| lateral / 242 | 256 | 50–150 | 256 lateral | 256 | 256 |
| CBF / 42 | 256 | 10–48 | 256 high-CBF | 256 | 256 |
| CBF / 142 | 256 | 10–47 | 256 high-CBF | 256 | 256 |
| CBF / 242 | 256 | 10–48 | 256 high-CBF | 256 | 256 |
| balance / 42 | 174 | 20–72 | 174 balance/phase | 174 | 174 |
| balance / 142 | 170 | 20–78 | 170 balance/phase | 170 | 170 |
| balance / 242 | 161 | 20–73 | 161 balance/phase | 161 | 161 |

The brief specifies bank semantics and time windows but no minimum final entry
count. The balance banks were therefore not enlarged post hoc.

## Infrastructure interruption

The first CBF/seed242 attempt stopped before an atomic round-4 record because
Warp failed to create a texture. The incomplete output and logs were preserved,
the restart decision was based only on the infrastructure exception, and the
entire job was rerun from the common frozen policy with unchanged source,
context, seed, hyperparameters, and gates. See
[`INTERRUPTION_PROVENANCE.md`](INTERRUPTION_PROVENANCE.md).

## Verification command

```text
python -m pytest -q experiments/tests/test_cbf_math.py \
  experiments/tests/test_online_refinement.py
69 passed in 10.47s
```

Machine-readable evidence is in [`training_manifest.json`](training_manifest.json)
and [`audit/final_audit_compact.json`](audit/final_audit_compact.json).
