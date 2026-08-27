from __future__ import annotations

import pytest
import torch

from src.tasks.stairs_cbf.cbf_math import (
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
  configure_paper_training_domain_randomization,
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
