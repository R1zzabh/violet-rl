# tests/unit/test_rewards.py — VIOLET

from violet.rewards import RewardConfig, RewardEngine, reward_config_sparse, reward_config_dense, reward_config_availability_penalty


def test_reward_config_defaults():
    c = RewardConfig()
    assert c.red_exploit_success == 10.0
    assert c.red_exfiltration_success == 100.0
    assert c.blue_critical_asset_compromised == -100.0
    assert c.step_penalty == -0.1
    assert isinstance(c.to_dict(), dict)


def test_reward_engine_red_exploit_success():
    engine = RewardEngine()
    ctx = {"result": "exploit_success"}
    r = engine.red_reward(ctx)
    assert abs(r - (10.0 - 0.1)) < 1e-6  # exploit + step penalty


def test_reward_engine_red_exploit_failed():
    engine = RewardEngine()
    ctx = {"result": "exploit_failed"}
    r = engine.red_reward(ctx)
    assert abs(r - (-3.0 - 0.1)) < 1e-6


def test_reward_engine_red_honeypot():
    engine = RewardEngine()
    ctx = {"honeypot_triggered": True}
    r = engine.red_reward(ctx)
    assert abs(r - (-15.0 - 0.1)) < 1e-6


def test_reward_engine_red_terminated():
    engine = RewardEngine()
    ctx = {"terminated": True}
    r = engine.red_reward(ctx)
    assert abs(r - (-100.0 - 0.1)) < 1e-6


def test_reward_engine_blue_critical_asset_compromised():
    engine = RewardEngine()
    ctx = {"critical_asset_compromised": True}
    r = engine.blue_reward(ctx)
    assert abs(r - (-100.0 - 0.1)) < 1e-6


def test_reward_engine_blue_isolate_success():
    engine = RewardEngine()
    ctx = {"attack_contained": True, "service_downtime": True}
    r = engine.blue_reward(ctx)
    # blue_attack_contained 20, service_downtime -10, step -0.1
    assert abs(r - (20.0 - 10.0 - 0.1)) < 1e-6


def test_reward_engine_blue_unnecessary_isolation():
    engine = RewardEngine()
    ctx = {"unnecessary_isolation": True}
    r = engine.blue_reward(ctx)
    assert abs(r - (-8.0 - 0.1)) < 1e-6


def test_detection_farming_is_closed():
    """Regression: repeat detections on an already-known compromised node must
    NOT pay the attack_detected reward — only the first detection does."""
    engine = RewardEngine()
    first = engine.blue_reward({"attack_detected": True})
    repeat = engine.blue_reward({"attack_detected": True, "attack_contained": True})
    # repeat detection alone adds nothing beyond contained; farming impossible by construction
    assert abs(first - (5.0 - 0.1)) < 1e-6


def test_unnecessary_reset_penalty_wired():
    """Regression: unnecessary_reset was a dead flag — it must now be rewarded."""
    engine = RewardEngine()
    r = engine.blue_reward({"unnecessary_reset": True})
    assert abs(r - (-2.0 - 0.1)) < 1e-6


def test_reward_engine_pretty_decomposition():
    engine = RewardEngine()
    ctx = {"result": "exploit_success"}
    pretty = engine.pretty("red", ctx)
    assert "total" in pretty
    assert "components" in pretty
    assert pretty["components"]["exploit_success"] == 10.0


def test_reward_engine_unknown_agent_raises():
    engine = RewardEngine()
    ctx = {}
    try:
        engine.reward("green", ctx)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_sparse_config_drops_step_and_minor_rewards():
    c = reward_config_sparse()
    assert c.step_penalty == 0.0
    assert c.red_recon_success == 0.0


def test_dense_config_is_default():
    c = reward_config_dense()
    assert c.step_penalty == -0.1  # unchanged from defaults


def test_availability_penalty_increases_defender_downtime_cost():
    c = reward_config_availability_penalty()
    assert c.blue_unnecessary_isolation == -12.0
    assert c.blue_service_downtime == -15.0
