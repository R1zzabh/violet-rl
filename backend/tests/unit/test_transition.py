# tests/unit/test_transition.py — VIOLET


from violet.rewards import RewardEngine
from violet.scenarios import v1_enterprise_network
from violet.transition import TransitionContext, TransitionEngine
from violet.types import CompromiseStatus, IsolationStatus


def make_engine():
    return TransitionEngine(v1_enterprise_network(), RewardEngine(), seed=42)


def test_scan_reveals_node():
    engine = make_engine()
    ctx = TransitionContext(step=1)
    action = _make_action("red", "SCAN", "web")
    engine.execute_red_action(action, ctx)
    web = engine.network.nodes["web"]
    assert web.attacker_discovered is True
    assert ctx.result == "recon_success"


def test_exploit_compromises_vulnerable_node():
    engine = make_engine()
    ctx = TransitionContext(step=1)
    action = _make_action("red", "EXPLOIT", "web")
    engine.execute_red_action(action, ctx)
    web = engine.network.nodes["web"]
    # web has high vulnerability rate (0.8) so exploit should succeed with RNG 42
    assert web.is_compromised()
    assert ctx.result == "exploit_success"


def test_exploit_fails_on_low_vuln_node():
    engine = make_engine()
    ctx = TransitionContext(step=1)
    action = _make_action("red", "EXPLOIT", "database")
    engine.execute_red_action(action, ctx)
    db = engine.network.nodes["database"]
    # database has low vulnerability (0.2), should mostly fail
    assert not db.is_compromised() or ctx.result == "exploit_failed"


def test_priv_esc_on_compromised_node():
    engine = make_engine()
    ctx = TransitionContext(step=1)
    # First compromise web
    web_action = _make_action("red", "EXPLOIT", "web")
    engine.execute_red_action(web_action, ctx)
    assert engine.network.nodes["web"].is_compromised()

    ctx2 = TransitionContext(step=2)
    esc_action = _make_action("red", "PRIVILEGE_ESCALATE", "web")
    engine.execute_red_action(esc_action, ctx2)
    web = engine.network.nodes["web"]
    assert web.compromise_status == CompromiseStatus.PRIVILEGES_ESCALATED
    assert ctx2.result == "priv_esc_success"


def test_lateral_movement_reachable():
    engine = make_engine()
    ctx = TransitionContext(step=1)
    # Compromise web first
    w = _make_action("red", "EXPLOIT", "web")
    engine.execute_red_action(w, ctx)

    ctx2 = TransitionContext(step=2)
    lat_action = _make_action("red", "LATERAL_MOVE", "app")
    engine.execute_red_action(lat_action, ctx2)
    app = engine.network.nodes["app"]
    assert app.is_compromised() or ctx2.result in ("lateral_failed",)


def test_lateral_target_is_destination():
    """Regression: LATERAL_MOVE target must be the DESTINATION node.

    The engine picks a source foothold that can reach it. Compromise web,
    then move precisely to app.
    """
    engine = make_engine()
    ctx = TransitionContext(step=1)
    engine.execute_red_action(_make_action("red", "EXPLOIT", "web"), ctx)
    assert engine.network.nodes["web"].is_compromised()

    ctx2 = TransitionContext(step=2)
    engine.execute_red_action(_make_action("red", "LATERAL_MOVE", "app"), ctx2)
    app = engine.network.nodes["app"]
    assert app.is_compromised() or ctx2.result == "lateral_failed"
    if app.is_compromised():
        ev = ctx2.events[-1]
        assert ev["details"]["source"] == "web"
        assert ev["details"]["destination"] == "app"


def test_lateral_fails_without_foothold_path():
    """Regression: lateral without a foothold path fails cleanly.

    The move must fail with lateral_failed — no random success, no crash.
    """
    engine = make_engine()
    ctx = TransitionContext(step=1)
    # No foothold at all: lateral move to database must fail
    engine.execute_red_action(_make_action("red", "LATERAL_MOVE", "database"), ctx)
    assert ctx.result == "lateral_failed"
    assert not engine.network.nodes["database"].is_compromised()


def test_lateral_fails_if_destination_already_compromised():
    engine = make_engine()
    ctx = TransitionContext(step=1)
    engine.execute_red_action(_make_action("red", "EXPLOIT", "web"), ctx)
    assert engine.network.nodes["web"].is_compromised()

    ctx2 = TransitionContext(step=2)
    engine.execute_red_action(_make_action("red", "LATERAL_MOVE", "web"), ctx2)
    assert ctx2.result == "lateral_failed"
    assert ctx2.events[-1]["result"] == "lateral_failed"


def test_honeypot_triggers_on_decoy():
    """Exploiting a honeypot triggers it — but only when REACHABLE.

    Regression note: decoy-1 is reachable only from web. An earlier version of
    this test exploited the decoy with no foothold, which was possible only
    because EXPLOIT ignored reachability (the teleport-exploit loophole).
    """
    engine = make_engine()
    ctx = TransitionContext(step=1)
    # Establish a foothold adjacent to the decoy (web -> decoy-1)
    engine.execute_red_action(_make_action("red", "EXPLOIT", "web"), ctx)
    assert engine.network.nodes["web"].is_compromised()

    ctx2 = TransitionContext(step=2)
    action = _make_action("red", "EXPLOIT", "decoy-1")
    engine.execute_red_action(action, ctx2)
    assert ctx2.honeypot_triggered is True
    assert ctx2.result == "honeypot_triggered"


def test_exploit_requires_reachability():
    """Regression: EXPLOIT must not be a teleport.

    A node with no network path from the attacker (no internet-facing
    exposure, no adjacent foothold) cannot be exploited. This closes the
    loophole where an attacker compromised the
    critical asset directly from the internet in one action.
    """
    engine = make_engine()
    ctx = TransitionContext(step=1)
    # No foothold yet: database is not internet-facing -> must fail cleanly
    engine.execute_red_action(_make_action("red", "EXPLOIT", "database"), ctx)
    assert ctx.result == "exploit_failed"
    assert not engine.network.nodes["database"].is_compromised()
    assert "no network path" in str(ctx.events[-1].get("details", {}))

    # After a foothold on web (not adjacent to database), still no direct exploit
    ctx2 = TransitionContext(step=2)
    engine.execute_red_action(_make_action("red", "EXPLOIT", "web"), ctx2)
    assert engine.network.nodes["web"].is_compromised()

    ctx3 = TransitionContext(step=3)
    engine.execute_red_action(_make_action("red", "EXPLOIT", "database"), ctx3)
    assert ctx3.result == "exploit_failed"
    assert not engine.network.nodes["database"].is_compromised()


def test_isolate_disconnects_node():
    engine = make_engine()
    ctx = TransitionContext(step=1)
    action = _make_action("blue", "ISOLATE", "web")
    engine.execute_blue_action(action, ctx)
    web = engine.network.nodes["web"]
    assert web.isolated == IsolationStatus.ISOLATED
    assert ctx.result == "isolate_success"


def test_isolate_blocks_lateral_movement():
    """Regression: an isolated foothold must not project lateral movement.

    Without this check the attacker could tunnel through isolation.
    """
    engine = make_engine()
    ctx = TransitionContext(step=1)
    engine.execute_red_action(_make_action("red", "EXPLOIT", "web"), ctx)
    assert engine.network.nodes["web"].is_compromised()

    ctx2 = TransitionContext(step=2)
    engine.execute_blue_action(_make_action("blue", "ISOLATE", "web"), ctx2)
    assert engine.network.nodes["web"].isolated == IsolationStatus.ISOLATED

    ctx3 = TransitionContext(step=3)
    engine.execute_red_action(_make_action("red", "LATERAL_MOVE", "app"), ctx3)
    assert ctx3.result == "lateral_failed"
    assert not engine.network.nodes["app"].is_compromised()


def test_patch_mitigates_all_vulnerabilities():
    """Regression: PATCH must mitigate all vulnerabilities on the node.

    PATCH previously kept priv-esc-capable vulns exploitable, so a patched
    web server could still be exploited at its original 0.8 rate.
    """
    engine = make_engine()
    ctx = TransitionContext(step=1)
    engine.execute_blue_action(_make_action("blue", "PATCH", "web"), ctx)
    web = engine.network.nodes["web"]
    if ctx.result == "patch_success":
        assert web.vulnerabilities == []
        assert web.vulnerability_rate <= 0.05


def test_isolate_already_isolated_is_unnecessary():
    engine = make_engine()
    ctx = TransitionContext(step=1)
    # isolate twice
    action = _make_action("blue", "ISOLATE", "web")
    engine.execute_blue_action(action, ctx)
    assert engine.network.nodes["web"].isolated == IsolationStatus.ISOLATED
    ctx2 = TransitionContext(step=2)
    action2 = _make_action("blue", "ISOLATE", "web")
    engine.execute_blue_action(action2, ctx2)
    assert ctx2.unnecessary_isolation is True
    assert ctx2.result == "isolate_already"


def test_reset_clears_compromise():
    engine = make_engine()
    ctx = TransitionContext(step=1)
    w = _make_action("red", "EXPLOIT", "web")
    engine.execute_red_action(w, ctx)
    assert engine.network.nodes["web"].is_compromised()

    ctx2 = TransitionContext(step=2)
    reset_action = _make_action("blue", "RESET", "web")
    engine.execute_blue_action(reset_action, ctx2)
    web = engine.network.nodes["web"]
    assert web.compromise_status == CompromiseStatus.NONE
    assert web.is_compromised() is False
    assert ctx2.host_recovered is True


def test_terminate_on_critical_asset():
    engine = make_engine()
    # Compromise database (critical asset) directly — should_terminate must fire.
    engine.network.nodes["database"].compromise_status = CompromiseStatus.COMPROMISED
    assert engine.should_terminate() is True


def test_event_emission():
    engine = make_engine()
    ctx = TransitionContext(step=1)
    action = _make_action("red", "SCAN", "web")
    engine.execute_red_action(action, ctx)
    assert len(ctx.events) >= 1
    ev = ctx.events[0]
    assert ev["agent"] == "red"
    assert ev["action"] == "SCAN"
    assert ev["target"] == "web"


def _make_action(agent: str, action_type: str, target: str):
    from violet.types import Action, ActionTarget, BlueActionType, RedActionType

    atype: RedActionType | BlueActionType = (
        RedActionType[action_type] if agent == "red" else BlueActionType[action_type]
    )
    return Action(agent=agent, action_type=atype, target=ActionTarget(node_id=target))
