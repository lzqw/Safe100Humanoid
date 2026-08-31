# v30 Completion Audit

Audit date: 2026-08-25 (Asia/Shanghai)  
Published evidence commit inspected: `82dd1d883afed5cb3788da403db3501bfcfacd0a`

## Outcome

The exact frozen v30 objective is **not complete**. Development and F1 are complete, but the predeclared F2/F3 formal matrix, three-context audit, checkpoint monitor, and paper-value decision are unavailable. The package deliberately reports `complete: false`.

## Requirement-by-requirement evidence

| Requirement | Status | Authoritative evidence |
|---|---|---|
| Preserve v25–v29 without modification | PASS | Current Git trees match every frozen tree hash in `protocol.json`: v25 `8d4384a…`, v26 `011f430…`, v27 `368144c…`, v28 `bae47e7…`, v29 `f214aad…`. |
| Freeze source, contexts, seeds and selected teacher before formal work | PASS | `protocol.json` has status `frozen_before_v30_formal`, SHA-256 `25a33f90…e965`, and selected arm A2. |
| Run only one smoke and satisfy the specified tensor/routing/finite/backward checks | PASS | `smoke/smoke_summary.json` records `passed: true`, all named checks true, and no KL gate. |
| Complete A0–A5 development, eight rounds each | PASS | `development/arm_summary.json` records `complete: true`; `training/development_round_metrics.csv` contains 48 aggregate round rows. |
| Select one teacher using the frozen round-8 rule | PASS | `development/selected_teacher.json` records A2 and the ordered selection rule; no KL or intermediate checkpoint was used. |
| Run F1 control and teacher once for eight rounds | PASS | Both F1 completion records contain 8 rounds; `formal/F1/execution_status.json` publishes their final checkpoint and training-summary hashes. |
| Run the complete F1 final paired audit | PASS | `formal/F1/context_results.json` and `condition_results.csv` cover 15 batches and 3,840 episodes with paired CIs. |
| Run F2 control and teacher once for eight rounds | **FAIL** | A0 completed 8/8; A2 produced four aggregate rounds, then `formal/F2/run_failure.json` recorded a behavior-log-probability routing error of `0.0005512237548828125`. |
| Run F3 control and teacher once for eight rounds | **FAIL** | Both arms failed before rollout with `shape (12,3) into shape (10,3)`; see `formal/F3/execution_status.json`. |
| Audit all three contexts using the frozen final policies | **INCOMPLETE** | Only F1 has audit output. F2 lacks an A2 round-8 checkpoint; F3 has no checkpoints. `formal/combined_results.json` contains only F1 and is explicitly incomplete. |
| Report the three-context paper-value criteria | **UNAVAILABLE** | Two of three contexts are missing; `paper_value_assessment` is `null`, so no three-context verdict is claimed. |
| Run the read-only F1 round-0–8 monitor only after formal training | **NOT RUN** | The required precondition was not met. `monitor/NOT_RUN.json` records the reason; no empty or fabricated curve is published. |
| Publish the prescribed aggregate evidence and figures | **PARTIAL** | Development, all available formal aggregates, failures, 28 formal round rows, F1-only figures, and checksums are published. Figures requiring F2/F3 or the monitor are intentionally absent. |

## Why the exact frozen run cannot be completed in place

F2/A2 stopped for an action-routing audit, one of the specification's permitted reasons to stop a run. Completing it now would require a second adaptation attempt, a threshold/source change, or a non-predeclared resume procedure.

F3 cannot construct the frozen environment: its 11-step profile produces 12 stair-target patches including the top platform, while the inherited allocation reserves 10. Producing F3 checkpoints requires changing source/configuration after the formal source freeze and then attempting the arms again.

Those actions conflict with the predeclared boundaries that freeze the source and allow one adaptation per method/context. They must not be backfilled and labeled as the original v30 formal matrix.

## Scientifically valid next paths

1. Preserve v30 as the immutable incomplete/failure result now published.
2. If new authority is given, create a separately labeled protocol revision (for example v30.1 or v31), predeclare the F3 allocation fix and F2 routing-audit treatment, freeze new source/seeds/identities, and run a new matrix. Its results must remain separate from v30.

No v30 simulator, training, audit, monitor, or packaging process remains active on the 4080 host at the time of this audit.
