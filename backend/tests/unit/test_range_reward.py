"""Outcome-grounded rewards for the bounded HTTP exercise."""

from violet.range.train_real import red_reward_components


def test_red_reward_does_not_reward_irrelevant_authentication() -> None:
    components = red_reward_components(
        verified_foreign_read=False,
        action_id=4,
        successful_http_response=True,
        principal_ref="bob",
    )
    assert components == {
        "verified_foreign_read": 0.0,
        "successful_alice_authentication": 0.0,
        "request_failure_cost": 0.0,
    }


def test_red_reward_uses_verified_finding_and_alice_setup_evidence() -> None:
    setup = red_reward_components(
        verified_foreign_read=False,
        action_id=3,
        successful_http_response=True,
        principal_ref="alice",
    )
    assert setup["successful_alice_authentication"] == 0.05
    verified = red_reward_components(
        verified_foreign_read=True,
        action_id=6,
        successful_http_response=True,
        principal_ref="alice",
    )
    assert verified["verified_foreign_read"] == 1.0


def test_red_reward_penalizes_a_failed_request_without_success_credit() -> None:
    components = red_reward_components(
        verified_foreign_read=False,
        action_id=6,
        successful_http_response=False,
        principal_ref="alice",
    )
    assert components["request_failure_cost"] == -0.1
    assert components["successful_alice_authentication"] == 0.0
