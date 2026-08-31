"""KL-targeted geometry-aware paired CBF PPO.

v51's formal consensus update used only 0.3% of its allowed reference KL.
v52 keeps the identical rollouts, CBF dual reward, GAE gradients, frozen legacy
policy, and hard KL cap, but expands the consensus-gradient step until it
reaches a requested KL target or any paired rollout loses positive surrogate
gain.
"""

from refine_observable_cbf_ppo_v51 import main


METHOD_ID = "observable-cbf-geometry-kl-targeted-paired-dual-gae-v52"


if __name__ == "__main__":
  main(method_id=METHOD_ID, require_target_reference_kl=True)
