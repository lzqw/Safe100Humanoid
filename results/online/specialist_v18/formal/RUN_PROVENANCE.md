# v18 diagonal-audit run provenance

## Protocol boundary

- Historical v17 formal audit was already known and remains unchanged.
- Protocol revision 1 commit: `a5834b3c4843e06f655f51b1349ff1e516fea1a0`.
- Its smoke preflight stopped before environment construction because two full
  context-parameter hashes had been transcribed incorrectly from abbreviated
  display strings. No episode or policy outcome was produced.
- Revision 2 copied the exact hashes from the sealed context files and training
  manifest, added a direct regression test, and froze all other protocol values
  unchanged.
- Formal protocol/source commit:
  `108e6013d8d1282b095dafcbb14fa16d73fabfe7`.
- Protocol file SHA-256:
  `48d9ad0ab3232927601b5a6c69c3f362bbc1184c7bdc59e7d199e6084eb66d2e`.

## Clean execution checkout

The existing GPU simulation directory was a deliberately overlaid dirty
worktree without a Git remote, so it was not used as formal source. A separate
clean checkout was created at:

```text
/home/carla/LZQW/SAFE100/humanoid/worktrees/diagonal_v18_108e601
```

The audit program required its HEAD to equal the frozen commit and required a
clean tracked worktree and index. The final summary records both checks as
passed. Existing checkpoint/context artifacts were read from the shared
artifact tree and verified by full SHA-256 before evaluation.

## Validation and smoke

The exact checkout passed:

```text
78 passed in 10.13s
```

A non-formal end-to-end smoke used audit seed 3,090,000, one target and one D0
pair per actor, and produced exactly 18 rows. It exercised all nine actor loads,
all three contexts, common D0 baseline reuse, paired-signature checks, CSV
writing, bootstrap code, and independent gates. Smoke outcomes were not used
to change the formal protocol and are not formal evidence.

## Formal run

- Host: `4080`, NVIDIA GeForce RTX 4080 SUPER.
- Runtime CBF: enabled for base and final actors.
- Start: 2026-08-06 10:45:42 UTC+08.
- Summary written: 2026-08-06 11:39:34 UTC+08.
- Audit seed: 3,100,000.
- Bootstrap seed: 4,000,000.
- Adaptation seeds: 42, 142, 242.
- Target pairs: 512 per mode/seed.
- D0 pairs: 256 per mode/seed.
- Raw artifacts: 96 JSON and 96 CSV blocks.
- Published paired rows: 6,912.
- Infrastructure failures/retries: none.
- Traceback/runtime/CUDA/Texture2D errors: none.
- Formal log SHA-256:
  `dc8e9040ed9ee08f31dbab4f4cd9d96e83f1c01b931edcbfdc6ed2761439e73c`.

The uncompressed raw block tree remains on the execution host at:

```text
/home/carla/LZQW/SAFE100/humanoid/artifacts/specialist_v18/diagonal_audit/raw
```

The complete aggregate raw evaluations and their initial-state signatures are
embedded in `diagonal_audit_summary.json`; all row-level binary outcomes and
failure types are published in `paired_episode_metrics.csv`.

## Independent reconstruction

`verify_diagonal_audit_v18.py` reads only the published protocol, training
manifest, complete summary, and paired CSV. It independently verified:

- exact protocol/source/input hashes and absence of off-diagonal/macro/joint
  claims;
- 6,912 rows and the complete 18-group schema;
- 512 target and 256 D0 pairs for every mode/seed;
- every per-seed base/final rate and paired delta;
- all twelve two-level bootstrap intervals;
- all four criteria in each independent gate;
- passed specialists `lateral`, `balance`; failed specialist `cbf`.
