# v18 requirement-by-requirement audit

This audit separates training-method compliance, evaluation-protocol
compliance, and the three independent scientific outcomes.

| Requirement | Authoritative evidence | Verdict |
|---|---|---|
| One common frozen base policy | The sealed training manifest and fresh summary contain base checkpoint SHA-256 `cb875d...a892a8` and initial actor SHA-256 `194322...47b1` for all nine jobs. | PASS |
| Three specialists initialize independently; no chained checkpoint | Every sealed run records `independent_training_branch=true`; all initial actor hashes match, while each final checkpoint is mode/seed specific. | PASS |
| Independent actor, critic, optimizer, and banks | Each run has its own checkpoint/summary. Structural checks record one actor and one critic, and the nine run directories are independent. | PASS |
| Base-only context calibration | All three context files record `base_policy_only=true`, `adapted_policy_evaluations_used=false`, 512 episodes, first-qualifying selection, and disjoint calibration/training/audit seeds. | PASS |
| Calibration gates | lateral: 78.906% SR, 108 falls, 85.185% target, 9.259% second; CBF: 78.906%, 108, 71.296%, 19.444%; balance: 77.344%, 116, 61.207%, 22.414%. | PASS |
| Frozen context hashes | File and parameter hashes are fixed in `protocol.json`; the formal preflight matched them against both context files and all nine training records. | PASS |
| Mode-specific scalar failure signals | The sealed v17 source/config implements lateral centerline/heading/edge, CBF intervention/nominal-margin, and balance attitude/angular-velocity/slip/contact-mismatch signals without cross-mode components. | PASS |
| Failure precursor and matched-success banks | The training manifest records target-pure failure banks and complete matched-success banks for every run, including unique success sources and mode-specific windows. | PASS |
| 70/15/15 initial-state mixture | Every run records the exact integer allocation normal/failure/success = 44/10/10 for 64 environments. | PASS |
| CBF-shielded raw-action PPO | Every run records `raw_policy_action_for_ppo=true`, runtime CBF enabled, and executed safe actions; routing/storage checks were preserved in all rounds. | PASS |
| One scalar reward and one privileged critic | Structural checks record one critic, no fall/intervention critics, no risk head, no retention actor/reference bank, and zero auxiliary anchor/distillation weights. | PASS |
| Brief PPO profile | 64 environments × 1024 steps, five rounds, actor LR `5e-6`, critic LR `1e-4`, one PPO epoch, clip 0.05, target KL 0.003, hard ceiling 0.01, candidate fractions 0.5/1.0/1.5. | PASS |
| Target-only training selection plus broad D0 check | All 45 rounds use only the current target score/fall/KL gate and a D0 catastrophic check; no other-specialist, neighbor, or CBF-demand gate is active. | PASS |
| Final evaluation is diagonal-only | The fresh summary records `off_diagonal_evaluation_performed=false`; the paired CSV contains only `target_diagonal_primary` and `d0_sanity`. | PASS |
| Three adaptation seeds | Exactly seeds 42, 142, and 242 occur for each of lateral, CBF, and balance. | PASS |
| 512 paired target episodes per seed | Independent CSV verification finds nine target groups of exactly 512 rows with complete pair indices. | PASS |
| 256 paired D0 episodes per seed | Independent CSV verification finds nine D0 groups of exactly 256 rows with complete pair indices. | PASS |
| Paired simulator randomness | Every base/final raw evaluation passed identical initial-state-signature checks; the CSV stores final-minus-base outcomes row by row. | PASS |
| Mean success gain and 95% CI reported | Each independent claim contains a two-level paired-bootstrap mean/LCB95/UCB95 tuple reconstructed exactly from the CSV using 10,000 samples. | PASS |
| CI is report-only | Protocol and summary both record `individual_confidence_interval_used_as_gate=false`; all intervals are still published. | PASS |
| Independent acceptance, no all-three conjunction | Three separate `claim_passed` values are present; `joint_conclusion.defined=false` and `all_three_required=false`. | PASS |
| No macro gate | `macro_average_computed=false` and no macro statistic appears in the formal result. | PASS |
| No uniform 2pp floor | The independent gate requires only a strictly positive mean; the verifier confirms no `diagonal_success_gain_above_2pp` criterion. | PASS |
| No off-diagonal or CBF-independence requirement | Neither is run or used by the v18 claims. | PASS |
| Fresh evidence rather than v17 relabelling | v18 uses audit seed 3,100,000; v17 used 1,900,000. Sealed actors are unchanged, and the v17 negative audit remains published unchanged. | PASS |
| Exact frozen source | Formal summary records clean tracked worktree/index and commit `108e6013...fabfe7`; all evaluation-relevant historical source hashes matched the sealed training manifest. | PASS |
| Complete paired data | CSV SHA-256 `64ad56...a00db`, exactly 6,912 rows in 18 groups; all rates, deltas, intervals, gates, and pass/fail lists independently reconstruct. | PASS |
| Lateral independent claim | Mean success +3.060pp; seed deltas +0.391/+4.102/+4.688pp; fall -3.060pp; D0 +1.042pp. | **PASS (outcome)** |
| CBF independent claim | Mean success -0.651pp; two seeds are positive, but the required aggregate point estimate is not. Fall +0.651pp and D0 -0.260pp pass their safeguards. | **FAIL (outcome)** |
| Balance independent claim | Mean success +1.107pp; seed deltas +0.781/+0.977/+1.563pp; fall -1.107pp; D0 +0.391pp. | **PASS (outcome)** |

All three target-success confidence intervals include zero. This does not change
the frozen v18 point-estimate gates, but it limits the strength of the supported
claim and must accompany any presentation of the two passing outcomes.
