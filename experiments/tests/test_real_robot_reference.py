from __future__ import annotations

import torch

from src.tasks.stairs_cbf.real_robot_reference import (
  ACTION_DIM,
  ACTOR_TERM_WIDTHS,
  G1_DEFAULT_JOINT_POSITION,
  LOWER_BODY_ACTION_OFFSET,
  ActorObservationHistory,
  embed_lower_body_target,
  nominal_lower_body_target,
  project_stair_cbf_action,
)


def _state(value: float):
  return {
    "base_ang_vel": [value] * 3,
    "projected_gravity": [value + 1.0] * 3,
    "command": [0.2, 0.0, 0.0],
    "episode_step": int(value),
    "joint_position": [base + value for base in G1_DEFAULT_JOINT_POSITION],
    "joint_velocity": [value + 2.0] * 29,
    "previous_raw_action": [value + 3.0] * ACTION_DIM,
  }


def test_reference_bridge_freezes_observation_action_and_cbf_semantics():
  history = ActorObservationHistory()
  first = history.push(**_state(0.0))
  second = history.push(**_state(1.0))
  assert first.shape == (1, 405)
  assert second.shape == (1, 405)

  offset = 0
  expected_latest = {
    "base_ang_vel": torch.ones(3),
    "projected_gravity": torch.full((3,), 2.0),
    "command": torch.tensor([0.2, 0.0, 0.0]),
    "joint_pos_relative": torch.ones(29),
    "joint_vel_relative": torch.full((29,), 3.0),
    "previous_raw_action": torch.full((12,), 4.0),
  }
  for name, width in ACTOR_TERM_WIDTHS:
    block = second[0, offset : offset + 5 * width].reshape(5, width)
    if name != "phase":
      assert torch.allclose(block[-1], expected_latest[name])
      oldest = first[0, offset : offset + width].expand_as(block[:-1])
      assert torch.allclose(block[:-1], oldest)
    offset += 5 * width
  assert offset == 405

  raw = torch.linspace(-0.5, 0.5, ACTION_DIM)
  nominal = nominal_lower_body_target(raw)
  full = embed_lower_body_target(nominal, G1_DEFAULT_JOINT_POSITION)
  assert torch.allclose(full[:ACTION_DIM], nominal)
  assert torch.allclose(
    full[ACTION_DIM:], torch.tensor(G1_DEFAULT_JOINT_POSITION[ACTION_DIM:])
  )

  jacobian = torch.zeros(2, 2, ACTION_DIM)
  jacobian[0, 0, 0] = 1.0
  jacobian[0, 1, 1] = 1.0
  result = project_stair_cbf_action(
    raw_action=[1.0] + [0.0] * (ACTION_DIM - 1),
    lower_body_joint_position=LOWER_BODY_ACTION_OFFSET,
    foot_position_xz=[[0.90, 0.0], [0.70, 0.0]],
    foot_jacobian_xz=jacobian,
    foot_contact=[False, True],
    edge_x=[1.0],
    edge_top_z=[0.18],
  )
  assert result.active and result.intervened
  assert result.selected_foot == 0 and result.selected_riser == 0
  assert result.projected_margin >= -1.0e-5
  assert result.safe_raw_action[0] < result.nominal_raw_action[0]
  assert result.safe_raw_action[1] > result.nominal_raw_action[1]
  assert torch.allclose(result.safe_raw_action[2:], result.nominal_raw_action[2:])
