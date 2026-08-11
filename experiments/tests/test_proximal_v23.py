"""Regression tests for the independent CBF-Proximal PPO v23 path."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/scripts"))

from proximal_v23_protocol import (  # noqa: E402
  ADAPTATION_SEED,
  FINAL_D0_SEED,
  FINAL_TARGET_SEED,
  all_v23_fresh_seed_values,
  development_gate,
  formal_algorithm_parameters,
  fresh_randomness_report,
  repair_regression_counts,
)
from src.tasks.stairs_cbf.config import g1_online_stairs_env_cfg  # noqa: E402
from src.tasks.stairs_cbf.deployment_context import (  # noqa: E402
  apply_frozen_deployment_context,
  load_calibrated_v22_context,
)
from src.tasks.stairs_cbf.proximal_context import (  # noqa: E402
  apply_cbf_proximal_context,
)
from src.tasks.stairs_cbf.proximal import (  # noqa: E402
  CbfProximalPpoAlgorithmCfg,
  ProximalHardRollback,
  diagonal_gaussian_forward_kl,
  module_state_is_finite,
  optimizer_state_is_finite,
)


CONTEXT = (
  REPO / "results/online/specialist_v22/calibration/L_effect/context.json"
)


def test_forward_kl_matches_torch_and_has_declared_orientation() -> None:
  current_mean = torch.tensor([[0.2, -0.4], [0.5, 0.1]])
  current_std = torch.tensor([[0.12, 0.18], [0.20, 0.09]])
  reference_mean = torch.tensor([[0.0, -0.1], [0.2, -0.2]])
  reference_std = torch.tensor([[0.20, 0.11], [0.13, 0.22]])
  actual = diagonal_gaussian_forward_kl(
    (current_mean, current_std), (reference_mean, reference_std)
  )
  expected = torch.distributions.kl_divergence(
    torch.distributions.Normal(current_mean, current_std),
    torch.distributions.Normal(reference_mean, reference_std),
  ).sum(dim=-1)
  reverse = torch.distributions.kl_divergence(
    torch.distributions.Normal(reference_mean, reference_std),
    torch.distributions.Normal(current_mean, current_std),
  ).sum(dim=-1)
  assert torch.allclose(actual, expected, atol=1.0e-7)
  assert not torch.allclose(actual, reverse)


def test_forward_kl_stops_reference_gradient_and_rejects_bad_std() -> None:
  current_mean = torch.tensor([[0.2, -0.4]], requires_grad=True)
  current_std = torch.tensor([[0.12, 0.18]], requires_grad=True)
  reference_mean = torch.tensor([[0.0, -0.1]], requires_grad=True)
  reference_std = torch.tensor([[0.20, 0.11]], requires_grad=True)
  diagonal_gaussian_forward_kl(
    (current_mean, current_std), (reference_mean, reference_std)
  ).sum().backward()
  assert current_mean.grad is not None
  assert current_std.grad is not None
  assert reference_mean.grad is None
  assert reference_std.grad is None
  with pytest.raises(ProximalHardRollback, match="non-positive"):
    diagonal_gaussian_forward_kl(
      (current_mean.detach(), torch.zeros_like(current_std)),
      (reference_mean.detach(), reference_std.detach()),
    )


def test_optimizer_and_module_finite_audits_detect_corruption() -> None:
  module = torch.nn.Linear(2, 1)
  optimizer = torch.optim.Adam(module.parameters(), lr=1.0e-3)
  module(torch.ones(1, 2)).sum().backward()
  optimizer.step()
  assert module_state_is_finite(module)
  assert optimizer_state_is_finite(optimizer)
  state = next(iter(optimizer.state.values()))
  state["exp_avg"].fill_(float("inf"))
  assert not optimizer_state_is_finite(optimizer)
  module.weight.data.fill_(float("nan"))
  assert not module_state_is_finite(module)


def test_proximal_context_keeps_original_interface_and_base_reward() -> None:
  payload = load_calibrated_v22_context(CONTEXT)
  cfg = g1_online_stairs_env_cfg("DQHMED")
  metadata = apply_cbf_proximal_context(cfg, payload)
  assert "deployable_failure" not in cfg.observations
  assert "specialist_failure_signal" not in cfg.rewards
  assert cfg.rewards["cbf_dual"].weight == 1.0
  assert cfg.rewards["fall_termination"].weight == -200.0
  assert metadata["actor_context_fields_added"] == 0
  assert metadata["cbf_proximal_interface"] == {
    "original_observation_interface": True,
    "deployable_failure_group_absent": True,
    "specialist_reward_term_absent": True,
    "historical_adapter_removed_before_environment_construction": True,
  }

  historical_cfg = g1_online_stairs_env_cfg("DQHMED")
  historical = apply_frozen_deployment_context(
    historical_cfg, payload, role="target"
  )
  assert "deployable_failure" in historical_cfg.observations
  assert historical["actor_context_fields_added"] == 5


def test_proximal_config_freezes_required_defaults() -> None:
  params = formal_algorithm_parameters()
  cfg = CbfProximalPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=params["ppo_clip"],
    entropy_coef=params["entropy_coefficient"],
    num_learning_epochs=params["maximum_actor_epochs"],
    num_mini_batches=params["mini_batches"],
    learning_rate=params["actor_learning_rate"],
    schedule="fixed",
    gamma=params["gamma"],
    lam=params["gae_lambda"],
    desired_kl=params["target_kl"],
    max_grad_norm=params["maximum_gradient_norm"],
  )
  assert cfg.class_name.endswith(":CbfProximalPPO")
  assert cfg.actor_new_feature_count == 0
  assert cfg.freeze_log_std
  assert cfg.maximum_std == 0.25
  assert cfg.moving_kl_beta == 0.5
  assert cfg.hard_kl_ceiling == 0.01
  assert cfg.critic_learning_epochs == 2
  assert all(
    value == 0.0
    for value in (
      cfg.pre_intervention_weight,
      cfg.intervention_advantage_weight,
      cfg.base_anchor_weight,
      cfg.d0_retention_anchor_weight,
      cfg.neighbor_retention_anchor_weight,
      cfg.safe_bc_weight,
      cfg.matched_success_preservation_beta,
      cfg.correction_distillation_weight,
    )
  )


def test_v23_randomness_is_unique_and_fresh() -> None:
  seeds = all_v23_fresh_seed_values()
  assert ADAPTATION_SEED in seeds
  assert FINAL_TARGET_SEED in seeds
  assert FINAL_D0_SEED in seeds
  assert len(seeds) == len(set(seeds))
  report = fresh_randomness_report(REPO)
  assert report["passed"]
  assert report["collisions"] == []


def test_v23_gate_is_report_only_and_final_round_is_not_selected() -> None:
  passed = development_gate(
    target_success_delta=0.03,
    target_fall_delta=0.01,
    d0_success_delta=-0.05,
  )
  assert passed["passed"]
  assert not passed["confidence_intervals_are_gates"]
  failed = development_gate(
    target_success_delta=0.029,
    target_fall_delta=0.011,
    d0_success_delta=-0.051,
  )
  assert not failed["passed"]

  source = (REPO / "experiments/scripts/refine_proximal_v23.py").read_text()
  tree = ast.parse(source)
  imported = {
    alias.name
    for node in ast.walk(tree)
    if isinstance(node, ast.Import)
    for alias in node.names
  }
  imported.update(
    node.module or ""
    for node in ast.walk(tree)
    if isinstance(node, ast.ImportFrom)
  )
  assert not any("hard_cases" in name for name in imported)
  assert "select_best_so_far" not in source
  assert "_collect_and_update_specialist" not in source
  assert "candidate_fractions" not in source


def test_repair_regression_counts_use_paired_identities() -> None:
  metrics = repair_regression_counts(
    [False, False, True, True], [True, False, False, True]
  )
  assert metrics["repair_count"] == 1
  assert metrics["regression_count"] == 1
  assert metrics["repair_rate_given_base_failure"] == 0.5
  assert metrics["regression_rate_given_base_success"] == 0.5
  assert metrics["net_success_change"] == 0
