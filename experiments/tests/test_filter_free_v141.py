"""Pure unit checks for intervention-aware filter-free refinement v141."""

from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[2]


def _pure_helpers() -> dict[str, object]:
    source = (REPO / "src/tasks/stairs_cbf/filter_free_v141.py").read_text()
    tree = ast.parse(source)
    names = {
        "normalize_context_group_advantages",
        "intervention_aware_ppo_weights",
        "successful_episode_transition_mask",
        "correction_distillation_weights",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    namespace = {
        "torch": torch,
        "math": math,
        "CORRECTION_WEIGHT_MODES": (
            "intervention_only",
            "positive_advantage",
            "episode_success_positive_advantage",
        ),
    }
    exec(  # noqa: S102 - selected local pure helper AST only
        compile(ast.fix_missing_locations(ast.Module(functions, type_ignores=[])), "v141", "exec"),
        namespace,
    )
    return namespace


HELPERS = _pure_helpers()


def test_v141_intervention_ppo_weights() -> None:
    mask = torch.tensor([[False, True, False, True]])
    actual = HELPERS["intervention_aware_ppo_weights"](mask, 0.25)
    assert torch.equal(actual, torch.tensor([[1.0, 0.25, 1.0, 0.25]]))
    with pytest.raises(ValueError):
        HELPERS["intervention_aware_ppo_weights"](mask, 1.1)


def test_v141_group_advantages_are_separately_standardized() -> None:
    advantages = torch.tensor(
        [[1.0, 10.0, 3.0, 14.0], [5.0, 18.0, 7.0, 22.0]]
    )
    target = torch.tensor([True, False, True, False])
    normalized, metrics = HELPERS["normalize_context_group_advantages"](
        advantages, target
    )
    assert float(normalized[:, target].mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(normalized[:, ~target].mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(normalized[:, target].std(unbiased=False)) == pytest.approx(1.0)
    assert metrics["context_group_advantages_standardized_separately"] == 1.0


def test_v141_episode_success_and_correction_weights() -> None:
    episode_ids = torch.tensor([[0], [0], [1], [1], [2]])
    success_terminal = torch.tensor([[False], [True], [False], [False], [False]])
    successful = HELPERS["successful_episode_transition_mask"](
        episode_ids, success_terminal
    )
    assert torch.equal(
        successful.squeeze(1), torch.tensor([True, True, False, False, False])
    )
    intervention = torch.tensor([[True], [True], [True], [False], [True]])
    correction = torch.tensor([[0.05], [0.025], [0.1], [0.1], [0.01]])
    advantage = torch.tensor([[2.0], [-1.0], [3.0], [4.0], [5.0]])
    weights = HELPERS["correction_distillation_weights"](
        intervention,
        correction,
        advantage,
        correction_scale=0.05,
        mode="episode_success_positive_advantage",
        successful_episode_mask=successful,
    )
    assert torch.equal(weights.squeeze(1), torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0]))


def test_v141_protocol_covers_required_search_values() -> None:
    path = REPO / "experiments/scripts/filter_free_v141_protocol.py"
    spec = importlib.util.spec_from_file_location("v141_protocol", path)
    assert spec is not None and spec.loader is not None
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    candidates = protocol.GENERATION_1_CANDIDATES
    assert {item["intervention_ppo_eta"] for item in candidates} == {0.0, 0.25, 0.5, 1.0}
    assert {item["dual_reward_scale"] for item in candidates} == {0.0, 0.25, 1.0}
    assert {item["correction_weight_mode"] for item in candidates} == {
        "intervention_only",
        "positive_advantage",
        "episode_success_positive_advantage",
    }


def test_v30_soft_weight_hook_preserves_legacy_default() -> None:
    source = (REPO / "src/tasks/stairs_cbf/teacher_v30.py").read_text()
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_actor_ppo_transition_weights"
    )
    assert isinstance(method.body[-1], ast.Return)
    assert isinstance(method.body[-1].value, ast.Constant)
    assert method.body[-1].value.value is None
