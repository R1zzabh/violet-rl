from violet.range.rewards import blue_reward, red_reward


def test_red_gets_no_finding_reward_from_http_success_alone() -> None:
    reward = red_reward(None)
    assert reward["total"] == 0
    assert reward["components"]["confirmed_unauthorized_read"] == 0


def test_red_positive_reward_requires_confirmed_evidence() -> None:
    reward = red_reward(
        {"status": "confirmed", "evidence": ["request-1", "target-event-1"]},
    )
    assert reward["total"] == 1.0
    assert reward["evidence"] == ["request-1", "target-event-1"]


def test_blue_detection_and_control_rewards_require_attributed_evidence() -> None:
    false_alarm = blue_reward(
        detection={"status": "suspected", "source_target_event_id": "event-1"},
        evaluator_label=False,
    )
    assert false_alarm["total"] == -1.0
    assert blue_reward(detection=None, evaluator_label=None)["total"] == 0
    verified = blue_reward(
        detection={"status": "suspected", "source_target_event_id": "event-1"},
        evaluator_label=True,
        control_verified=True,
    )
    assert verified["total"] == 1.0


def test_blue_reward_penalizes_legitimate_access_disruption_separately() -> None:
    reward = blue_reward(
        detection=None,
        evaluator_label=None,
        control_verified=False,
        denied_legitimate_request=True,
    )
    assert reward["components"]["denied_legitimate_request"] == -1.0
    assert reward["total"] == -1.0
