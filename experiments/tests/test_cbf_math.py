from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.tasks.stairs_cbf.cbf_math import (
  contact_or_scheduled_swing_foot,
  dual_cbf_reward,
  next_riser,
  next_riser_clearance_reference,
  project_halfspace,
  sloped_toe_clearance_constraint,
  stair_barrier,
)
from src.tasks.stairs_cbf.edge_detection import (
  riser_edges_from_tread_patches,
  select_active_riser,
)
from src.tasks.stairs_cbf.paper_dual_v35 import (
  PAPER_DUAL_CANDIDATES,
  capped_norm_balance_auxiliary_gradients,
  configure_paper_training_domain_randomization,
  normalize_filter_group_advantages,
  split_filter_actor_objective_masks,
  task_priority_project_auxiliary_gradients,
)
from src.tasks.stairs_cbf.paper_occupancy_corrected_v127 import (
  crossfit_occupancy_corrected_advantages,
)
from src.tasks.stairs_cbf.paper_early_start_v128 import (
  aligned_filtered_rollout_decision,
)
from src.tasks.stairs_cbf.paper_continuous_kl_v129 import (
  continuous_ppo_kl_learning_rate,
)
from src.tasks.stairs_cbf.paper_shield_withdrawal_v130 import (
  withdrawal_deployment_rollout_decision,
)
from src.tasks.stairs_cbf.paper_deterministic_aligned_v131 import (
  deterministic_alignment_diagnostics,
)
from src.tasks.stairs_cbf.paper_scaled_continuation_v132 import (
  paper_scale_diagnostics,
)
from src.tasks.stairs_cbf.paper_scaled_stage_two_v133 import (
  stage_two_scale_diagnostics,
)
from src.tasks.stairs_cbf.paper_uniform_swa_v134 import (
  uniform_actor_mlp_average,
)
from src.tasks.stairs_cbf.paper_high_parallel_v135 import (
  high_parallel_scale_diagnostics,
)
from src.tasks.stairs_cbf.paper_early_peak_v136 import (
  earliest_exact_peak_decision,
)
from src.tasks.stairs_cbf.paper_clearance_margin_v138 import (
  CLEARANCE_MARGIN_M,
  configure_paper_clearance_margin,
)


def test_halfspace_projection_repairs_violation():
  nominal = torch.tensor([[2.0, -1.0], [0.2, 0.4]])
  normal = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
  rhs = torch.tensor([-0.5, 0.0])
  projected, before, after = project_halfspace(nominal, normal, rhs)
  assert before[0] < 0
  assert torch.all(after >= -1.0e-6)
  assert torch.allclose(projected[0], torch.tensor([0.5, -1.0]))
  assert torch.allclose(projected[1], nominal[1])


def test_v137_preactivates_only_toe_off_and_keeps_airborne_foot_priority():
  contact = torch.tensor(
    [
      [True, True],   # right foot is scheduled to swing at phase 0.10
      [True, True],   # double-stance window at phase 0.52
      [True, False],  # physical right swing wins regardless of gait phase
      [False, True],  # physical left swing wins regardless of gait phase
    ]
  )
  air_time = torch.tensor(
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.12], [0.20, 0.0]]
  )
  episode_steps = torch.tensor([3, 16, 20, 3])

  selected, preactivated = contact_or_scheduled_swing_foot(
    contact,
    air_time,
    episode_steps,
    step_dt=0.02,
    period=0.6,
    stance_fraction=0.56,
  )

  assert selected.tolist() == [1, -1, 1, 0]
  assert preactivated.tolist() == [True, False, False, False]


def test_v138_changes_only_paper_next_riser_clearance_margin():
  original = {
    "action_name": "joint_pos",
    "asset_name": "robot",
    "default_height": 0.10,
    "height_above_tread": 0.05,
    "reference_mode": "next_riser",
    "lookahead_distance": 0.60,
  }
  term = SimpleNamespace(params=dict(original))
  env_cfg = SimpleNamespace(rewards={"foot_clearance": term})

  audit = configure_paper_clearance_margin(env_cfg)

  assert term.params == {**original, "height_above_tread": CLEARANCE_MARGIN_M}
  assert audit["clearance_margin_increase_m"] == pytest.approx(0.03)
  assert audit["cbf_changed"] is False


def test_inactive_constraint_is_identity():
  nominal = torch.tensor([[2.0, 3.0]])
  normal = torch.tensor([[-1.0, 0.0]])
  projected, _, _ = project_halfspace(
    nominal, normal, torch.tensor([0.0]), active=torch.tensor([False])
  )
  assert torch.equal(projected, nominal)


def test_stair_geometry_sign_and_next_edge():
  foot_x = torch.tensor([0.8, 1.1, 1.45])
  origin_x = torch.tensor([0.5, 0.5, 0.5])
  index, edge_x, top_z, valid = next_riser(
    foot_x, origin_x, 0.6, 0.35, 0.13, 6
  )
  assert index.tolist() == [0, 0, 1]
  assert torch.allclose(edge_x, torch.tensor([1.1, 1.1, 1.45]))
  assert torch.allclose(top_z, torch.tensor([0.13, 0.13, 0.26]))
  assert valid.all()
  h = stair_barrier(foot_x, edge_x, toe_margin=0.08)
  assert h[0] > 0 and h[1] < 0


def test_sloped_clearance_constraint_couples_lift_and_forward_motion():
  horizontal = torch.tensor([0.20, 0.00])
  foot_z = torch.tensor([0.10, 0.18])
  top_z = torch.tensor([0.18, 0.18])
  jac_x = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
  jac_z = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
  barrier, normal = sloped_toe_clearance_constraint(
    horizontal,
    foot_z,
    top_z,
    jac_x,
    jac_z,
    top_clearance=0.02,
    slope=0.5,
  )
  assert torch.allclose(barrier, torch.tensor([0.0, -0.02]))
  assert torch.allclose(normal, torch.tensor([[-0.5, 1.0], [-0.5, 1.0]]))


def test_riser_extraction_and_selection():
  patches = torch.tensor(
    [[[1.275, 0.0, 0.13], [1.625, 0.0, 0.26], [1.975, 0.0, 0.39]]]
  )
  edges, tops = riser_edges_from_tread_patches(patches, 0.35, 3)
  assert torch.allclose(edges, torch.tensor([[1.10, 1.45, 1.80]]))
  index, h, top, active = select_active_riser(
    torch.tensor([1.03]),
    torch.tensor([0.05]),
    edges,
    tops,
    toe_margin=0.08,
    top_clearance=0.025,
    activation_distance=0.30,
    recovery_distance=0.15,
  )
  assert active.item() and index.item() == 0
  assert h.item() < 0 and torch.allclose(top, torch.tensor([0.13]))


def test_next_riser_clearance_reference_persists_to_top_platform():
  root_x = torch.tensor([0.20, 0.40, 1.20, 2.10])
  origin_z = torch.zeros(4)
  edge_x = torch.tensor([[1.0, 1.5, 2.0]]).expand(4, -1)
  edge_top_z = torch.tensor([[0.18, 0.36, 0.54]]).expand(4, -1)

  reference, active, index = next_riser_clearance_reference(
    root_x,
    origin_z,
    edge_x,
    edge_top_z,
    default_height=0.10,
    height_above_tread=0.05,
    lookahead_distance=0.60,
  )

  torch.testing.assert_close(reference, torch.tensor([0.10, 0.23, 0.41, 0.59]))
  assert active.tolist() == [False, True, True, True]
  assert index.tolist() == [0, 0, 1, 2]


def test_dual_reward_matches_paper_and_is_bounded_without_violation():
  margin = torch.tensor([-0.25, 0.10, -4.0])
  intervention = torch.tensor([0.5, 0.0, 10.0])
  active = torch.tensor([True, True, False])
  reward = dual_cbf_reward(margin, intervention, active, sigma=0.5)
  expected_first = -0.25 + torch.exp(torch.tensor(-1.0)) - 1.0
  assert torch.allclose(reward[0], expected_first)
  assert reward[1] == 0.0
  assert reward[2] == 0.0
  assert -1.0 <= float(dual_cbf_reward(
    torch.tensor([0.1]), torch.tensor([100.0]), torch.tensor([True]), sigma=0.5
  )[0]) <= 0.0


def test_dual_reward_supports_independent_paper_demo_weights():
  margin = torch.tensor([-0.2, 0.1])
  correction = torch.tensor([0.5, 0.0])
  active = torch.tensor([True, True])

  reward = dual_cbf_reward(
    margin,
    correction,
    active,
    sigma=0.5,
    margin_weight=10.0,
    intervention_weight=100.0,
  )

  expected = 10.0 * margin[0] + 100.0 * (torch.exp(torch.tensor(-1.0)) - 1.0)
  torch.testing.assert_close(reward[0], expected)
  assert reward[1] == 0.0


def test_dual_reward_rejects_negative_component_weights():
  with pytest.raises(ValueError, match="weights must be non-negative"):
    dual_cbf_reward(
      torch.zeros(1),
      torch.zeros(1),
      torch.ones(1, dtype=torch.bool),
      sigma=0.5,
      margin_weight=-1.0,
    )


def test_v35_paper_demo_candidate_matches_public_demo_scaling():
  assert PAPER_DUAL_CANDIDATES["raw_demo"] == {
    "correction_space": "raw_action",
    "sigma": 0.5,
    "margin_weight": 10.0,
    "intervention_weight": 100.0,
  }


def test_v35_paper_stair_candidate_uses_reduced_order_foot_distance():
  assert PAPER_DUAL_CANDIDATES["paper_stair_exact"] == {
    "correction_space": "foot_task",
    "sigma": 0.05,
    "margin_weight": 1.0,
    "intervention_weight": 1.0,
  }
  assert PAPER_DUAL_CANDIDATES["paper_stair_demo_scale"] == {
    "correction_space": "foot_task",
    "sigma": 0.05,
    "margin_weight": 10.0,
    "intervention_weight": 100.0,
  }
  assert PAPER_DUAL_CANDIDATES["paper_stair_sloped_demo_scale"] == {
    "correction_space": "foot_task",
    "sigma": 0.05,
    "margin_weight": 10.0,
    "intervention_weight": 100.0,
  }
  assert PAPER_DUAL_CANDIDATES["paper_stair_sloped_exact"] == {
    "correction_space": "foot_task",
    "sigma": 0.05,
    "margin_weight": 1.0,
    "intervention_weight": 1.0,
  }
  assert PAPER_DUAL_CANDIDATES["paper_stair_sloped_unit_balanced"] == {
    "correction_space": "foot_task",
    "sigma": 0.05,
    "margin_weight": 0.1,
    "intervention_weight": 1.0,
  }
  assert PAPER_DUAL_CANDIDATES["paper_stair_sloped_mid_balanced"] == {
    "correction_space": "foot_task",
    "sigma": 0.05,
    "margin_weight": 0.25,
    "intervention_weight": 1.0,
  }
  assert PAPER_DUAL_CANDIDATES["paper_stair_sloped_proximity_balanced"] == {
    "correction_space": "foot_task",
    "sigma": 0.05,
    "margin_weight": 0.1,
    "intervention_weight": 2.0,
  }


def test_v67_normalizes_mixed_filter_advantages_per_execution_group():
  advantages = torch.tensor(
    [[1.0, 10.0, 3.0, 14.0], [5.0, 18.0, 7.0, 22.0]]
  )
  filter_mask = torch.tensor([True, False, True, False])

  normalized, metrics = normalize_filter_group_advantages(
    advantages, filter_mask
  )

  for mask in (filter_mask, ~filter_mask):
    torch.testing.assert_close(normalized[:, mask].mean(), torch.tensor(0.0))
    torch.testing.assert_close(
      normalized[:, mask].std(unbiased=False), torch.tensor(1.0)
    )
  assert metrics["filter_group_balanced_advantages"] == 1.0
  assert metrics["filter_on_advantage_count"] == 4.0
  assert metrics["filter_off_advantage_count"] == 4.0


def test_v67_rejects_empty_mixed_filter_advantage_group():
  with pytest.raises(ValueError, match="both be non-empty"):
    normalize_filter_group_advantages(
      torch.ones(2, 3), torch.ones(3, dtype=torch.bool)
    )


def test_v127_crossfits_state_occupancy_and_keeps_only_filtered_actor_credit():
  time = torch.linspace(-1.0, 1.0, 12).view(12, 1)
  environment = torch.tensor([-0.3, 0.2, -0.1, 0.4]).view(1, 4)
  on = torch.stack(
    (
      -1.0 + time + environment,
      time.square() + environment,
      time - environment,
    ),
    dim=-1,
  )
  off = on + torch.tensor([2.0, 0.6, -0.4])
  features = torch.empty(12, 8, 3)
  filter_mask = torch.tensor([True, False, True, False, True, False, True, False])
  features[:, filter_mask] = on
  features[:, ~filter_mask] = off
  advantages = torch.arange(96, dtype=torch.float32).reshape(12, 8)

  corrected, metrics = crossfit_occupancy_corrected_advantages(
    features, advantages, filter_mask
  )

  assert torch.equal(corrected[:, ~filter_mask], torch.zeros(12, 4))
  torch.testing.assert_close(
    corrected[:, filter_mask].mean(), torch.tensor(0.0), atol=1.0e-6, rtol=0
  )
  torch.testing.assert_close(
    corrected[:, filter_mask].std(unbiased=False),
    torch.tensor(2.0),
    atol=1.0e-5,
    rtol=0,
  )
  assert metrics["occupancy_correction_active"] is True
  assert metrics["occupancy_classifier_balanced_accuracy"] > 0.75
  assert metrics["occupancy_density_ratio_effective_sample_fraction"] > 0.0
  assert metrics["occupancy_actor_filter_off_advantage_max_abs"] == 0.0
  assert metrics["occupancy_critic_uses_all_transitions"] is True


def test_v128_selects_only_the_best_aligned_filtered_training_rollout():
  first = aligned_filtered_rollout_decision(
    candidate_round=1,
    success_count=80,
    episode_count=120,
    mean_reached_riser=8.0,
    incumbent_round=None,
    incumbent_success_count=None,
    incumbent_episode_count=None,
    incumbent_mean_reached_riser=None,
  )
  assert first["selected"] is True
  assert first["selection_uses_training_rollout_only"] is True
  assert first["selection_changes_training_trajectory"] is False

  worse_progress = aligned_filtered_rollout_decision(
    candidate_round=2,
    success_count=100,
    episode_count=150,
    mean_reached_riser=7.9,
    incumbent_round=1,
    incumbent_success_count=80,
    incumbent_episode_count=120,
    incumbent_mean_reached_riser=8.0,
  )
  assert worse_progress["selected"] is False

  better_rate = aligned_filtered_rollout_decision(
    candidate_round=3,
    success_count=101,
    episode_count=150,
    mean_reached_riser=7.8,
    incumbent_round=1,
    incumbent_success_count=80,
    incumbent_episode_count=120,
    incumbent_mean_reached_riser=8.0,
  )
  assert better_rate["selected"] is True
  assert better_rate["reason"] == "higher_success_rate"


def test_v129_controls_next_round_lr_without_rollback_or_actor_mutation():
  reduced, high_metrics = continuous_ppo_kl_learning_rate(5.0e-6, 2.4e-3)
  assert reduced == pytest.approx(2.5e-6)
  assert high_metrics["v129_kl_controller_bounded_scale"] == pytest.approx(0.5)

  restored, low_metrics = continuous_ppo_kl_learning_rate(reduced, 1.5e-4)
  assert restored == pytest.approx(3.125e-6)
  assert low_metrics["v129_kl_controller_bounded_scale"] == pytest.approx(1.25)
  assert low_metrics["v129_kl_controller_changes_actor_state"] is False
  assert low_metrics["v129_kl_controller_uses_rollback"] is False


def test_v130_selects_only_sufficient_filter_off_training_rollouts():
  ineligible = withdrawal_deployment_rollout_decision(
    candidate_round=2,
    runtime_filter_fraction=0.75,
    filter_off_success_count=25,
    filter_off_episode_count=40,
    filter_off_mean_reached_riser=7.5,
    incumbent_round=None,
    incumbent_success_count=None,
    incumbent_episode_count=None,
    incumbent_mean_reached_riser=None,
  )
  assert ineligible["eligible"] is False
  assert ineligible["selected"] is False

  eligible = withdrawal_deployment_rollout_decision(
    candidate_round=3,
    runtime_filter_fraction=0.5,
    filter_off_success_count=48,
    filter_off_episode_count=64,
    filter_off_mean_reached_riser=8.2,
    incumbent_round=None,
    incumbent_success_count=None,
    incumbent_episode_count=None,
    incumbent_mean_reached_riser=None,
  )
  assert eligible["eligible"] is True
  assert eligible["selected"] is True
  assert eligible["selection_group"] == "filter_off"
  assert eligible["selection_changes_training_trajectory"] is False


def test_v131_reduces_exploration_variance_while_retaining_nonzero_sampling():
  diagnostics = deterministic_alignment_diagnostics()
  assert diagnostics["v131_training_action_std"] == pytest.approx(0.03)
  assert diagnostics["v131_reference_action_std"] == pytest.approx(0.05)
  assert diagnostics["v131_standard_deviation_ratio"] == pytest.approx(0.6)
  assert diagnostics["v131_exploration_variance_ratio"] == pytest.approx(0.36)
  assert diagnostics["v131_equal_kl_mean_shift_ratio"] == pytest.approx(0.6)


def test_v132_doubles_v129_transition_scale_without_changing_rollout_length():
  diagnostics = paper_scale_diagnostics()
  assert diagnostics["v132_num_envs"] == 128
  assert diagnostics["v132_rollout_steps"] == 1024
  assert diagnostics["v132_rounds"] == 8
  assert diagnostics["v132_transition_count"] == 1_048_576
  assert diagnostics["v132_parallel_scale_ratio"] == pytest.approx(2.0)
  assert diagnostics["v132_transition_scale_ratio"] == pytest.approx(2.0)


def test_v133_extends_the_same_scaled_objective_to_a_second_stage():
  diagnostics = stage_two_scale_diagnostics()
  assert diagnostics["v133_stage_index"] == 2
  assert diagnostics["v133_stage_transition_count"] == 1_048_576
  assert diagnostics["v133_scaled_stage_cumulative_transition_count"] == 2_097_152
  assert diagnostics["v133_objective_changed_from_v132"] is False
  assert diagnostics["v133_optimizer_protocol_changed_from_v132"] is False


def test_v134_uniformly_averages_only_actor_mlp_snapshots():
  states = [
    {
      "obs_normalizer.count": torch.tensor(7),
      "distribution.std_param": torch.tensor([0.05]),
      "mlp.0.weight": torch.tensor([[value, 2.0 * value]]),
      "mlp.0.bias": torch.tensor([value]),
    }
    for value in (1.0, 2.0, 3.0)
  ]
  averaged, diagnostics = uniform_actor_mlp_average(states)
  torch.testing.assert_close(
    averaged["mlp.0.weight"], torch.tensor([[2.0, 4.0]])
  )
  torch.testing.assert_close(averaged["mlp.0.bias"], torch.tensor([2.0]))
  assert torch.equal(averaged["obs_normalizer.count"], torch.tensor(7))
  assert torch.equal(
    averaged["distribution.std_param"], torch.tensor([0.05])
  )
  assert diagnostics["v134_snapshot_count"] == 3
  assert diagnostics["v134_uniform_snapshot_weight"] == pytest.approx(1 / 3)
  assert diagnostics["v134_non_mlp_actor_state_exactly_preserved"] is True


def test_v135_triples_synchronous_v129_scale_without_more_updates():
  diagnostics = high_parallel_scale_diagnostics()
  assert diagnostics["v135_num_envs"] == 192
  assert diagnostics["v135_rollout_steps"] == 1024
  assert diagnostics["v135_rounds"] == 8
  assert diagnostics["v135_transition_count"] == 1_572_864
  assert diagnostics["v135_parallel_scale_ratio"] == pytest.approx(3.0)
  assert diagnostics["v135_transition_scale_ratio"] == pytest.approx(3.0)
  assert diagnostics["v135_sequential_update_count_changed_from_v129"] is False


def test_v136_selects_the_earliest_exact_peak_without_extra_evaluation():
  decision = earliest_exact_peak_decision(
    [
      {"rollout_round": 2, "success_count": 194, "episode_count": 264},
      {"rollout_round": 4, "success_count": 198, "episode_count": 269},
      {"rollout_round": 8, "success_count": 198, "episode_count": 269},
    ]
  )
  assert decision["rollout_round"] == 4
  assert decision["success_rate"] == pytest.approx(198 / 269)
  assert decision["exact_peak_tie_count"] == 2
  assert decision["exact_peak_tied_rollout_rounds"] == [4, 8]
  assert decision["selection_tie_break"] == (
    "earliest_rollout_minimum_update_count"
  )
  assert decision["additional_evaluation_count"] == 0


def test_v68_routes_nominal_worlds_to_ppo_and_filtered_worlds_to_teacher():
  filter_mask = torch.tensor([True, True, False, False])
  ppo, teacher = split_filter_actor_objective_masks(filter_mask, 3)

  assert ppo.shape == teacher.shape == (3, 4)
  assert torch.equal(teacher, filter_mask.unsqueeze(0).expand(3, -1))
  assert torch.equal(ppo, ~teacher)
  assert not bool((ppo & teacher).any())
  assert bool((ppo | teacher).all())


def test_v68_rejects_single_execution_group_actor_routing():
  with pytest.raises(ValueError, match="both execution groups"):
    split_filter_actor_objective_masks(
      torch.zeros(4, dtype=torch.bool), 2
    )


def test_v69_projects_only_conflicting_teacher_gradient_component():
  deployment = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0]))
  teacher = (torch.tensor([-1.0, 1.0]), torch.tensor([0.0, 0.2]))

  projected, metrics = task_priority_project_auxiliary_gradients(
    deployment, teacher
  )

  original_dot = sum(
    (left * right).sum() for left, right in zip(deployment, teacher)
  )
  projected_dot = sum(
    (left * right).sum() for left, right in zip(deployment, projected)
  )
  assert original_dot < 0.0
  torch.testing.assert_close(projected_dot, torch.tensor(0.0), atol=1.0e-6, rtol=0)
  assert metrics["auxiliary_gradient_conflict"] == 1.0
  assert 0.0 < metrics["auxiliary_gradient_retained_fraction"] < 1.0


def test_v69_leaves_aligned_teacher_gradient_unchanged():
  deployment = (torch.tensor([1.0, 2.0]),)
  teacher = (torch.tensor([3.0, 4.0]),)

  projected, metrics = task_priority_project_auxiliary_gradients(
    deployment, teacher
  )

  torch.testing.assert_close(projected[0], teacher[0])
  assert metrics["auxiliary_gradient_conflict"] == 0.0
  assert metrics["auxiliary_gradient_retained_fraction"] == pytest.approx(1.0)


def test_v70_norm_balances_projected_teacher_with_a_hard_scale_cap():
  deployment = (torch.tensor([3.0, 4.0]),)
  teacher = (torch.tensor([1.0, 0.0]),)

  balanced, metrics = capped_norm_balance_auxiliary_gradients(
    deployment, teacher, target_ratio=0.5
  )

  torch.testing.assert_close(balanced[0], torch.tensor([2.5, 0.0]))
  assert metrics["auxiliary_gradient_balance_scale"] == pytest.approx(2.5)
  assert metrics["balanced_auxiliary_to_primary_norm_ratio"] == pytest.approx(0.5)
  assert metrics["auxiliary_gradient_scale_capped"] == 0.0

  capped, capped_metrics = capped_norm_balance_auxiliary_gradients(
    (torch.tensor([10.0]),),
    (torch.tensor([0.1]),),
    target_ratio=1.0,
    maximum_scale=4.0,
  )
  torch.testing.assert_close(capped[0], torch.tensor([0.4]))
  assert capped_metrics["auxiliary_gradient_balance_scale"] == pytest.approx(4.0)
  assert capped_metrics["auxiliary_gradient_scale_capped"] == 1.0


def test_v57_restores_native_static_training_domain_randomization():
  from src.tasks.stairs_cbf.config import g1_online_stairs_env_cfg

  cfg = g1_online_stairs_env_cfg("DQHMED", play=True)
  assert cfg.observations["actor"].enable_corruption is False
  assert not ({"encoder_bias", "foot_friction", "base_com"} & cfg.events.keys())

  metadata = configure_paper_training_domain_randomization(
    cfg, "paper_static", strength=0.25
  )

  assert metadata["enabled"] is True
  assert metadata["external_pushes"] is False
  assert cfg.observations["actor"].enable_corruption is True
  assert {"encoder_bias", "foot_friction", "base_com"} <= cfg.events.keys()
  assert "push_robot" not in cfg.events
  assert metadata["strength"] == 0.25
  assert metadata["foot_friction_range"] == [0.825, 1.15]
  assert metadata["encoder_bias_range"] == [-0.00375, 0.00375]
  assert cfg.observations["actor"].terms["base_ang_vel"].noise.n_max == 0.05
