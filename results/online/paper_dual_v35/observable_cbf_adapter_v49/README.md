# v49 Deployable CBF-Geometry Actor Adapter

v49 fixes an observability mismatch found after v48: the 405-D actor received
only proprioceptive history, while the runtime CBF used swing-toe position and
mapped riser geometry.  Five current, deployable coordinates are appended to
the actor.  The old checkpoint is expanded with exactly zero first-layer
columns, so the initial 410-D actor is identical to the base actor.  All five
coordinates are zero when CBF geometry is inactive; after training, behavior
outside that region remains exactly the base policy.

The short RTX 4080 run used three paired filter-on/off training seeds with
eight environments each.  Only episodes where filter-on succeeded and the
matched filter-off episode failed supplied safe-action targets.  The update
trained 2,560 new input-column parameters and left all legacy actor parameters
unchanged.

| F2, seed `201350932`, 64 episodes | Filter off | Filter on | Decision |
|---|---:|---:|---|
| Original 405-D base | 46/64 (71.88%) | not rerun | control |
| v49, all shielded successes as teachers | 44/64 (68.75%) | not run | rejected |
| v49b, matched rescued episodes only | **48/64 (75.00%)** | **51/64 (79.69%)** | paired gate passed |

The paired pilot improves filter-off by two successes over the exact base
control and leaves only a three-episode (4.69 pp) filter-on/off gap.  It is the
first F2 candidate in this sequence to reach the 75% deployment gate while
also keeping the paired filtered result above the gate.  However, the
prospectively unused seed `201350952` then reached only **45/64 (70.31%)**
filter-off.  Its filter-on condition was not run after this failed gate.
Therefore v49b is retained as a positive observability ablation but rejected
as the final robust actor.

Key provenance:

- Training time: 89.10 s.
- Training data: 10,180 transitions; 190 rescued-episode teacher transitions.
- Teacher correction cosine: 0.0000 to 0.8355.
- Active-state forward KL: 0.004647.
- Candidate SHA-256: `0cfe45c3324d8c4e11254ec8ba292781dc4cc125d4cd3662f688430caf5ced91`.
- Actor SHA-256: `53f568f5f430ffae316574cdcaba0c174a5da28a750d579927169c216dce174f`.

Files:

- `comparison_summary.json`: compact decision record and base/on/off comparison.
- `v49b_training_summary.json`: exact training configuration and offline metrics.
- `v49b_gate_seed201350932_filter_{off,on}_summary.json`: aggregate gates.
- `v49b_gate_seed201350932_filter_{off,on}_episodes.csv`: episode-level evidence.
- `v49b_untouched_seed201350952_filter_off_{summary.json,episodes.csv}`: failed untouched confirmation.
- `v49b_candidate.pt`: exact 410-D candidate checkpoint.
- `v49_all_success_*`: rejected ablation that learned from every shielded success.

The actor observation and training implementation is in
`src/tasks/stairs_cbf/mdp.py`, `src/tasks/stairs_cbf/config.py`, and
`experiments/scripts/refine_observable_cbf_adapter_v49.py`.  The v34 evaluator
accepts `--actor-observation-interface deployable-cbf-geometry-410` for this
checkpoint.
