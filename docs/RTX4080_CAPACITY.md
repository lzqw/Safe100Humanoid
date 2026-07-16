# RTX 4080 capacity

The GPU is an RTX 4080 SUPER with 16,376 MiB. The 29-DoF engineering task
completed two PPO updates at every tested scale through 4096 environments; no
row OOMed.

| Environments | Last samples/s | Peak VRAM MiB | Interpretation |
|---:|---:|---:|---|
| 32 | 285 | 7,035 | CARLA overlap polluted peak |
| 128 | 5,183 | 8,225 | CARLA overlap polluted peak |
| 256 | 6,555 | 14,181 | CARLA overlap polluted peak |
| 512 | 13,386 | 14,363 | CARLA overlap polluted peak |
| 1024 | 26,540 | 14,517 | CARLA overlap polluted peak |
| 2048 | 56,782 | 2,786 | isolated GPU measurement |
| 4096 | 68,529 | 4,744 | isolated GPU measurement |

The raw CSVs are `artifacts/capacity_sweep.csv`,
`artifacts/capacity_sweep_2048.csv`, and
`artifacts/capacity_sweep_4096.csv`. Because an unrelated CARLA process
intermittently consumes 7--12 GiB and must not be terminated, the formal run
uses a conservative 1024 environments. The live paper-spec run uses about
1.34 GiB process VRAM and initially sustains roughly 37--38k samples/s.

Effective PPO rollout batch: `1024 * 24 = 24,576` transitions. With four PPO
mini-batches, each mini-batch contains 6,144 transitions before epoch reuse.
