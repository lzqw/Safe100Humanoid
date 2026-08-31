# v20 Run Provenance

## Evidence boundary

The learning code, contexts, seeds, and protocol were frozen before any v20
formal adaptation outcome. Training was executed only from the clean protocol
worktree at commit `1ded5b84f1c4b8605fd285ef3138c0363db20ee4`.
An audit-loader infrastructure failure was repaired in a separate clean
worktree and frozen before any formal audit episode outcome. Formal audits were
executed only from audit commit
`1428f57ff58ad8bb76da8e6c2fd1f5a22d8bd21c`.

Reporting-only telemetry divergence was handled in a third clean worktree at
`b575cf34aa76649b6c9f793514fbe3e221a32dcf`. That wrapper cannot modify formal
outcomes, change selected identities, or retry until an outcome matches.

## Commit chronology

| Commit | Time (Asia/Shanghai) | Role and outcome boundary |
| --- | --- | --- |
| `6c75cdce5dbef03cf99a1544df9ea2f37f389c62` | 2026-08-07 10:47 | Implement fixed-budget, independent-specialist v20 code and tests. No formal v20 outcomes existed. |
| `a65cc0d9e8f9d67b3fb41dfb3a5d53eee3ece210` | 2026-08-07 10:59 | Correct prospective contact candidate IDs from invalid 8217…8224 coverage to valid 8212…8219 before calibration. No candidate outcome had been observed. |
| `7206d120d30610f562921ea43c805e19244d6a51` | 2026-08-07 11:02 | Align base-only calibration runner with the frozen Revision-4 full-rate adapter before fresh calibration. |
| `1ded5b84f1c4b8605fd285ef3138c0363db20ee4` | 2026-08-07 11:13 | Freeze selected contexts, hashes, seeds, source files, and formal protocol before adaptation. |
| `665a71dc54045b7efc10fa3a2aff392fe2468b8c` | 2026-08-08 14:29 | Audit-only checkpoint-loader fix after a pre-outcome infrastructure failure. |
| `731836ce011f4c6d2d815de621835a737c060099` | 2026-08-08 14:31 | Freeze audit amendment 1. A non-formal smoke subsequently exposed incomplete inherited runner flags. |
| `5f6674ad8c1ffcf84256deb0ef1fa1a93d39c2a5` | 2026-08-08 14:35 | Restore the complete Revision-4 Brief-PPO audit runner configuration. |
| `1428f57ff58ad8bb76da8e6c2fd1f5a22d8bd21c` | 2026-08-08 14:36 | Freeze complete audit amendment 2 before any formal audit outcome. |
| `22654dd3eae5d40dadf3f0e7576e0f496b88065b` | 2026-08-08 15:23 | After the verified lateral audit, disclose same-state GPU telemetry replay divergence; formal artifacts remained unchanged. |
| `b575cf34aa76649b6c9f793514fbe3e221a32dcf` | 2026-08-08 15:25 | Freeze the first-attempt replay disclosure wrapper before contact training/audit outcomes. |

The result/evidence commit comes after all outcomes and is reporting-only. It
contains no training or audit algorithm change.

## Frozen identities

| Item | Value |
| --- | --- |
| Protocol SHA-256 | `74242b1131499f7163ef1985d25a94ca06b0758f6a3d5f2dea9f56ac200c28da` |
| Audit amendment SHA-256 | `5700b77302093ce708a9be2aa264e36267f7220d05e5037ca4eff109c85377dc` |
| Telemetry amendment SHA-256 | `c10b5d828ee6ddd33f245faa6add84e602d824210075fbf9f5c9144e5f0be150` |
| Base checkpoint SHA-256 | `cb875d571e126d418c1908dcb4a2ef97851e6aa9e0a50dfcf7c42eabf5a892a8` |
| Expanded initial actor SHA-256 | `f8e27e9b3bb92dd33a460f38ecbd72f8b0fe03809683f697d54d3549626d69bb` |
| Lateral context | generator seed 8312; file SHA-256 `27f16a28bb0f362c784c26962af412ffa15b728fd8946876e91189350c6eb45d` |
| Contact context | generator seed 8212; file SHA-256 `cd0423b8f7ece3962f4d8fff4458c2f4639a303d40da48cc57b919a9b05d0f34` |
| Adaptation seeds | 73, 173, 273, 373, 473 |
| Audit / bootstrap seeds | 5,500,000 / 6,500,000 |
| Formal rounds | exactly 8 per run |

The context-file hashes above are independently checked in every relevant
training summary and audit verification. Context parameter hashes and complete
calibration evidence are in [`protocol.json`](protocol.json) and
[`calibration/`](calibration/).

## Execution environment

| Component | Recorded value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 SUPER, 16,376 MiB |
| NVIDIA driver | 550.144.03 |
| Kernel | Linux 6.8.0-124-generic x86_64 |
| Python | 3.11.15 |
| Training worktree | `/home/carla/LZQW/SAFE100/humanoid/worktrees/v20_formal` at `1ded5b84…`, clean after execution |
| Audit worktree | `/home/carla/LZQW/SAFE100/humanoid/worktrees/v20_audit_amendment` at `1428f57…`, clean after execution |
| Telemetry worktree | `/home/carla/LZQW/SAFE100/humanoid/worktrees/v20_telemetry_amendment` at `b575cf34…`, clean after execution |

## Execution timeline

All timestamps in this table are UTC.

| Event | Start | Finish |
| --- | --- | --- |
| Lateral seed 73 | 2026-08-07 03:15:02 | 03:47:57 |
| Lateral seed 173 | 03:47:57 | 04:19:10 |
| Lateral seed 273 | 04:19:10 | 04:51:03 |
| Lateral seed 373 | 04:51:03 | 05:22:36 |
| Lateral seed 473 | 05:22:36 | 05:53:56 |
| Verified lateral audit | post-amendment rerun on 2026-08-08 | 07:13:43 |
| Disclosed lateral telemetry | 2026-08-08 | 07:28:49 |
| Contact seed 73 | 2026-08-08 07:30:40 | 08:08:41 |
| Contact seed 173 | 08:08:41 | 08:46:30 |
| Contact seed 273 | 08:46:30 | 09:23:54 |
| Contact seed 373 | 09:23:54 | 10:01:42 |
| Contact seed 473 | 10:01:42 | 10:39:30 |
| Verified contact audit | 10:39:30 | 11:10:22 |
| Disclosed contact telemetry | after verification | 11:16:19 |

## Infrastructure failure and retry record

### Audit loader

The first lateral audit launch failed before constructing the formal audit
environment, before writing any formal row, and before observing any formal
episode outcome. The 410-D interface was configured, but the generic loader
attempted to restore an irrelevant legacy 405-D retention-actor payload.
Amendment 1 enabled Brief-PPO loading. A five-seed non-formal smoke then showed
that inherited auxiliary-loss defaults violated the Brief-PPO constructor
guard. Amendment 2 restored all frozen Revision-4 runner flags before runner
construction.

No actor/checkpoint tensor, training logic, evaluation logic, seed, context,
episode count, gate, or bootstrap rule changed. The complete amendment passed
the 105-test GPU suite before the formal audit was run. The amendment JSON
records source-file hashes and the exact pre-outcome boundary.

### Mechanism replay

The verified lateral audit completed once. The old mechanism collector then
failed because a replay with the same actor hash and initial-state signature
had a different outcome. No formal audit was rerun and the partial first
attempt was preserved. A reporting-only wrapper was added and frozen; it uses
the already fixed lowest repair identity, reuses complete existing first
attempts, makes one attempt for missing roles, and records outcome divergence.

The contact audit also completed once and verified successfully. Its old
collector failed on the first seed-73 baseline trace for the same reason. The
already frozen wrapper reused that first attempt and completed the remaining
nine traces once each. Formal contact summary/paired/verification hashes were
checked unchanged before and after telemetry.

| Mode | Traces | Outcome matches | Divergences | Preserved/reused first attempts | Actor/state identity matches | Outcome-matching retries |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| lateral | 10 | 7 | 3 | 3 | 10/10 | 0 |
| contact_stability | 10 | 5 | 5 | 1 | 10/10 | 0 |

Formal audit outcomes in the paired CSVs remain authoritative. Telemetry
traces are descriptive only.

## Artifact and log manifests

[`training_manifest.json`](training_manifest.json) was produced by the frozen
builder from the full training and audit roots. It contains 510 records and
hashes 5,964,965,319 bytes. This includes all 80 round checkpoints, 10 accepted
checkpoints, full training summaries, raw formal evaluation JSON/CSV/actors,
compact tables, audit summaries, verifications, and telemetry traces.

The Git package intentionally contains compact per-seed summaries/tables and
all key audit/figure evidence, not multi-gigabyte checkpoints. Each compact
summary records the external full-summary and accepted-checkpoint path, size,
and SHA-256. [`run_log_manifest.json`](run_log_manifest.json) hashes every v20
log. Logs below 1 MB are included under [`logs/`](logs/); large repetitive
training/queue logs remain hash-addressed external artifacts.

Figure generation used `SOURCE_DATE_EPOCH=1786173901`. A second output tree was
generated from the same published CSV/JSON inputs and all 44 PNG/PDF hashes
matched the primary tree exactly.

## Key formal hashes

| Artifact | SHA-256 |
| --- | --- |
| Lateral final summary | `9dff606fad5f68b052c3e4b7372bfc4b179ff9616b942f3a4e54e1e9ed15a39d` |
| Lateral paired CSV | `170eb8ce10aeea0924880661ee3d0405472cb35f16225e0d06ab62381d4ef45e` |
| Lateral verification | `040359b3d533149476704ecd6a9241b5abf7a4cd478c3a7542d4dcdaec250d55` |
| Lateral mechanism CSV | `9a37735f8e5b1c961878013c29e9a5f718813f5d06f3b64d28a0750d35f904f8` |
| Contact final summary | `8bf00ce908c59147f834fc7e3fd7b560c315c69ea0767d6646f7a9afccc18bb1` |
| Contact paired CSV | `e258f16a9049dac85439f2c0e494e318a6ef0979d2580f9d7ded8df67c1b7ae4` |
| Contact verification | `41d2f91dea151c946030062309667410fb5ce6325d31dc6e69bbe74b014f64c5` |
| Contact mechanism CSV | `4af085316cdb8e566a1843a94e55f0c21368a8c41688587bee66979e1c7212ad` |

The package-wide [`SHA256SUMS`](SHA256SUMS) is generated last, after the final
reports, and excludes itself by definition.

## Validation commands

The completion audit reruns, at minimum:

```bash
pytest -q experiments/tests/test_specialist_v20.py \
  experiments/tests/test_specialist_v20_telemetry_amendment.py
ruff check experiments/scripts/collect_mechanism_telemetry_replay_v20.py \
  experiments/scripts/plot_specialist_v20_replay_disclosed.py \
  experiments/tests/test_specialist_v20.py \
  experiments/tests/test_specialist_v20_telemetry_amendment.py
sha256sum -c results/online/specialist_v20/SHA256SUMS
```

The final full suite in the recorded GPU environment passed 109/109 tests. A
generic local environment lacked the project `mjlab` dependency and could not
collect the full suite; the focused dependency-light v20 suite passed 19/19
locally. Both copied paired CSVs were then independently reconstructed again
from the local repository and returned `verified: true` with 3,840 rows each.

The existing draft PR is the publication boundary. A result/evidence commit is
pushed to `feature/online-safe-refinement`; the PR is not converted out of
Draft because a point estimate is positive.
