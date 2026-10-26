"""Correctness checks for the separate real-range PPO role policies."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="range PPO tests require the optional rl extra")

import torch

from violet.algorithms.ppo import RolloutBuffer, Transition, compute_gae
from violet.range.learning import (
    RangeObservation,
    RangePPOAgent,
    RangePPOConfig,
    RangeTransition,
    clipped_surrogate_loss,
)
from violet.range.train_real import PassiveBluePolicy, RandomRangePolicy


def test_truncation_bootstraps_value_but_ends_gae_trace() -> None:
    adv, returns = compute_gae(
        [1.0],
        [0.5],
        [False],
        0.0,
        0.9,
        0.95,
        truncations=[True],
        next_values=[2.0],
    )
    assert np.allclose(adv, [2.3])
    assert np.allclose(returns, [2.8])
    terminal_adv, _ = compute_gae(
        [1.0],
        [0.5],
        [True],
        0.0,
        0.9,
        0.95,
        truncations=[False],
        next_values=[2.0],
    )
    assert np.allclose(terminal_adv, [0.5])


def test_single_transition_rollout_keeps_policy_advantage() -> None:
    buffer = RolloutBuffer()
    buffer.add(
        Transition(
            obs=np.zeros(2, dtype=np.float32),
            mask=np.asarray([True, True]),
            action=0,
            log_prob=-0.69,
            value=0.2,
            reward=1.0,
            terminated=False,
            truncated=True,
            next_value=0.3,
        )
    )
    buffer.finalize(0.0, 0.99, 0.95)
    assert buffer.advantages is not None
    assert buffer.advantages[0] == pytest.approx(1.097)


def _fill_rollout(agent: RangePPOAgent, rewards: tuple[float, ...]) -> None:
    if agent.role == "red":
        observation = RangeObservation.red(
            step=1,
            budget=4,
            authenticated_alice=True,
            authenticated_bob=False,
            last_success=True,
            last_status=200,
            verified_objectives=0,
        )
    else:
        observation = RangeObservation.blue(
            step=1,
            budget=1,
            requests=4,
            failed_requests=1,
            service_healthy=True,
            verified_attack=False,
            verified_prevention=False,
            controls_applied=0,
        )
    mask = np.ones(len(agent.actions), dtype=bool)
    for idx, reward in enumerate(rewards):
        decision = agent.act(observation, action_mask=mask, training=True)
        agent.record(
            RangeTransition(
                observation.array(), mask, decision, reward, 0.25, idx == len(rewards) - 1, False
            )
        )


def test_probability_ratio_and_clipping_match_ppo_objective() -> None:
    old_logp = torch.log(torch.tensor([0.5, 0.5]))
    new_logp = torch.log(torch.tensor([0.75, 0.25]))
    advantages = torch.tensor([2.0, -2.0])
    loss, ratio, clip_fraction = clipped_surrogate_loss(new_logp, old_logp, advantages, 0.2)
    assert torch.allclose(ratio, torch.tensor([1.5, 0.5]))
    # The positive sample clips at 1.2 and the negative sample uses the clipped
    # -1.6 surrogate at ratio 0.5.
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(-0.4))
    assert torch.allclose(clip_fraction, torch.tensor(1.0))


def test_red_and_blue_are_independent_and_both_optimizer_updates_change_parameters() -> None:
    config = RangePPOConfig(seed=13, epochs=2, minibatch_size=4)
    red = RangePPOAgent("red", config)
    blue = RangePPOAgent("blue", config)
    assert set(red.model.parameters()).isdisjoint(set(blue.model.parameters()))
    before_red = [p.detach().clone() for p in red.model.parameters()]
    before_blue = [p.detach().clone() for p in blue.model.parameters()]
    _fill_rollout(red, (0.1, -0.2, 1.0, -0.1))
    red_result = red.update()
    assert red_result["transitions"] == 4
    assert red_result["updates"] == 1
    assert red_result["optimizer_steps"] == 2
    assert red_result["optimizer_steps_total"] == 2
    assert np.isfinite(red_result["gradient_norm"])
    assert np.isfinite(red_result["advantage_std"])
    assert all(
        np.isfinite(value)
        for key, value in red_result.items()
        if key not in {"updates", "transitions"}
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in red.model.parameters()
    )
    assert any(
        not torch.equal(a, b) for a, b in zip(before_red, red.model.parameters(), strict=True)
    )
    assert all(torch.equal(a, b) for a, b in zip(before_blue, blue.model.parameters(), strict=True))
    _fill_rollout(blue, (-0.1, 0.2, 1.0, -0.2))
    blue_result = blue.update()
    assert blue_result["transitions"] == 4
    assert blue_result["updates"] == 1
    assert blue_result["optimizer_steps"] == 2
    assert blue_result["optimizer_steps_total"] == 2
    assert np.isfinite(blue_result["gradient_norm"])
    assert np.isfinite(blue_result["advantage_std"])
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in blue.model.parameters()
    )
    assert any(
        not torch.equal(a, b) for a, b in zip(before_blue, blue.model.parameters(), strict=True)
    )


def test_empty_and_invalid_rollouts_do_not_update() -> None:
    agent = RangePPOAgent("red", RangePPOConfig(seed=15))
    before = [p.detach().clone() for p in agent.model.parameters()]
    assert agent.update() == {
        "updates": 0,
        "transitions": 0,
        "optimizer_steps": 0,
        "optimizer_steps_total": 0,
    }
    observation = RangeObservation.red(
        step=0,
        budget=2,
        authenticated_alice=False,
        authenticated_bob=False,
        last_success=False,
        last_status=None,
        verified_objectives=0,
    )
    mask = np.zeros(len(agent.actions), dtype=bool)
    decision = agent.act(
        observation, action_mask=np.ones(len(agent.actions), dtype=bool), training=True
    )
    with pytest.raises(ValueError, match="masked"):
        agent.record(RangeTransition(observation.array(), mask, decision, 1.0, 0.0, True, False))
    assert all(torch.equal(a, b) for a, b in zip(before, agent.model.parameters(), strict=True))


def test_inference_is_deterministic_and_does_not_mutate_parameters(tmp_path) -> None:
    agent = RangePPOAgent("blue", RangePPOConfig(seed=7))
    _fill_rollout(agent, (-0.1, 0.2, 1.0, -0.2))
    agent.update()
    observation = RangeObservation.blue(
        step=1,
        budget=1,
        requests=4,
        failed_requests=1,
        service_healthy=True,
        verified_attack=False,
        verified_prevention=False,
        controls_applied=0,
    )
    before = [p.detach().clone() for p in agent.model.parameters()]
    first = agent.act(observation, training=False)
    second = agent.act(observation, training=False)
    assert first.action == second.action
    assert all(torch.equal(a, b) for a, b in zip(before, agent.model.parameters(), strict=True))
    checkpoint = tmp_path / "blue.pt"
    agent.save(checkpoint)
    restored = RangePPOAgent("blue", RangePPOConfig(seed=99))
    restored.load(checkpoint, inference_only=True)
    assert restored.act(observation, training=False).action == first.action
    resumed = RangePPOAgent("blue", RangePPOConfig(seed=101))
    resumed.load(checkpoint)
    assert resumed.policy_version == agent.policy_version
    assert resumed.optimizer.state_dict()["state"]
    assert all(
        torch.equal(saved, loaded)
        for saved, loaded in zip(agent.model.parameters(), resumed.model.parameters(), strict=True)
    )
    with pytest.raises(ValueError, match="role"):
        RangePPOAgent("red").load(checkpoint, inference_only=True)


def test_stochastic_action_streams_are_reproducible_per_seed() -> None:
    config = RangePPOConfig(seed=31)
    first_agent = RangePPOAgent("red", config)
    second_agent = RangePPOAgent("red", config)
    observation = RangeObservation.red(
        step=1,
        budget=3,
        authenticated_alice=False,
        authenticated_bob=False,
        last_success=False,
        last_status=None,
        verified_objectives=0,
    )
    sequence_a = [first_agent.act(observation, training=True).action for _ in range(8)]
    sequence_b = [second_agent.act(observation, training=True).action for _ in range(8)]
    assert sequence_a == sequence_b


def test_selected_probability_and_entropy_are_finite_and_mask_aware() -> None:
    agent = RangePPOAgent("blue", RangePPOConfig(seed=23))
    observation = RangeObservation.blue(
        step=1,
        budget=2,
        requests=1,
        failed_requests=0,
        service_healthy=True,
        verified_attack=False,
        verified_prevention=False,
        controls_applied=0,
    )
    mask = np.array([False, True, False, False])
    decision = agent.act(observation, action_mask=mask, training=True)
    assert decision.action == 1
    assert decision.action_probability == pytest.approx(1.0)
    assert decision.entropy == pytest.approx(0.0)


def test_random_baseline_is_seeded_and_samples_only_valid_actions() -> None:
    observation = RangeObservation.blue(
        step=1,
        budget=2,
        requests=1,
        failed_requests=0,
        service_healthy=True,
        verified_attack=False,
        verified_prevention=False,
        controls_applied=0,
    )
    mask = np.array([True, False, True, False])
    first = RandomRangePolicy("blue", 42)
    second = RandomRangePolicy("blue", 42)
    decisions_a = [first.act(observation, action_mask=mask, training=False) for _ in range(12)]
    decisions_b = [second.act(observation, action_mask=mask, training=False) for _ in range(12)]
    assert [decision.action for decision in decisions_a] == [
        decision.action for decision in decisions_b
    ]
    assert all(decision.action in {0, 2} for decision in decisions_a)
    assert all(decision.action_probability == pytest.approx(0.5) for decision in decisions_a)
    assert all(decision.entropy == pytest.approx(np.log(2.0)) for decision in decisions_a)


def test_passive_blue_calibration_uses_valid_noop_action() -> None:
    observation = RangeObservation.blue(
        step=1,
        budget=2,
        requests=1,
        failed_requests=0,
        service_healthy=True,
        verified_attack=False,
        verified_prevention=False,
        controls_applied=0,
    )
    decision = PassiveBluePolicy().act(
        observation, action_mask=np.array([True, False, False, False]), training=False
    )
    assert decision.action == 0
