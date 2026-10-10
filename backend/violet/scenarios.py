# violet/scenarios.py — predefined network topologies / scenarios

from __future__ import annotations

from typing import Any

from violet.network import Network, Node, Service, Vulnerability
from violet.types import ServiceStatus


def v1_enterprise_network() -> Network:
    """Phase 1 V1 scenario: ~7 node enterprise network as described in the spec.

    Internet → WEB/MAIL → APP → DATABASE, with EMPLOYEE-1/2 downstream of DB.
    One critical asset (DATABASE), realistic vulnerability values, at least one honeypot
    placeholder for Phase 2.

    This is deliberately small so we can verify the simulator BEFORE training.
    """
    net = Network()

    internet = Node(
        id="internet",
        label="INTERNET",
        node_type="gateway",
        value=0.0,
        vulnerability_rate=0.0,
        is_critical_asset=False,
        connections=[],  # internet is a source; reachable only from connected nodes
        metadata={"description": "external entry point"},
    )
    web = Node(
        id="web",
        label="WEB SERVER",
        node_type="host",
        value=20.0,
        vulnerability_rate=0.8,
        connections=["app", "mail"],
        metadata={"description": "public-facing web server"},
    )
    mail = Node(
        id="mail",
        label="MAIL SERVER",
        node_type="host",
        value=10.0,
        vulnerability_rate=0.5,
        connections=["web"],
        metadata={"description": "mail server, less exposed"},
    )
    app = Node(
        id="app",
        label="APP SERVER",
        node_type="host",
        value=50.0,
        vulnerability_rate=0.5,
        connections=["web", "database"],
        metadata={"description": "application server, internal"},
    )
    database = Node(
        id="database",
        label="DATABASE",
        node_type="database",
        value=100.0,
        vulnerability_rate=0.2,
        connections=["app"],
        is_critical_asset=True,
        metadata={"description": "high-value asset; compromise terminates episode"},
    )
    employee1 = Node(
        id="employee-1",
        label="EMPLOYEE-1",
        node_type="host",
        value=15.0,
        vulnerability_rate=0.6,
        connections=["database"],
        metadata={"description": "internal workstation"},
    )
    employee2 = Node(
        id="employee-2",
        label="EMPLOYEE-2",
        node_type="host",
        value=15.0,
        vulnerability_rate=0.6,
        connections=["database"],
        metadata={"description": "internal workstation"},
    )

    # Pre-seed a known vulnerability on the web server (realistic for a public-facing node)
    web.vulnerabilities.append(
        Vulnerability(
            id="WEB-CVE-2024-001",
            description="public-facing web app RCE",
            base_success_rate=0.8,
            privilege_escalation_available=True,
        )
    )
    app.vulnerabilities.append(
        Vulnerability(
            id="APP-CVE-2024-002",
            description="internal app service vulnerability",
            base_success_rate=0.5,
            privilege_escalation_available=False,
        )
    )
    database.vulnerabilities.append(
        Vulnerability(
            id="DB-CVE-2024-003",
            description="database service vulnerability",
            base_success_rate=0.2,
            privilege_escalation_available=True,
        )
    )

    # Add a service to each host (primarily for Blue observability / defense)
    for node in (web, mail, app, database, employee1, employee2):
        node.services.append(
            Service(name=f"{node.id}-primary", status=ServiceStatus.ACTIVE, port=80)
        )

    nodes = [internet, web, mail, app, database, employee1, employee2]
    for n in nodes:
        net.add_node(n)

    # Connections already listed above via Node.connections; we also set
    # bidirectional/host connections explicitly so the graph is symmetric where intended.
    # internet -> web, internet -> mail (only from internet side)
    net.connect_unidirectional("internet", "web")
    net.connect_unidirectional("internet", "mail")

    # web <-> app <-> database <-> employee-1/2
    net.connect_bidirectional("web", "app")
    net.connect_bidirectional("app", "database")
    net.connect_bidirectional("database", "employee-1")
    net.connect_bidirectional("database", "employee-2")

    # mail <-> web (already partially set; ensure bidirectional)
    net.connect_bidirectional("web", "mail")

    # Phase 2 placeholder: a honeypot on a decoy node
    decoy = Node(
        id="decoy-1",
        label="DECOY-1",
        node_type="honeypot",
        value=1.0,
        vulnerability_rate=0.9,
        is_honeypot=True,
        connections=["web"],
        metadata={"description": "decoy host to detect/reaction test"},
    )
    decoy.services.append(Service(name="decoy-ssh", status=ServiceStatus.ACTIVE, port=22))
    decoy.vulnerabilities.append(
        Vulnerability(
            id="DECOY-OPEN-SSH",
            description="intentionally vulnerable SSH service",
            base_success_rate=0.9,
            privilege_escalation_available=True,
        )
    )
    net.add_node(decoy)
    net.connect_unidirectional("web", "decoy-1")

    return net


def v1_scenario_config() -> dict[str, Any]:
    """Structured description of the V1 scenario for configs / docs / dashboard."""
    return {
        "name": "v1_enterprise",
        "description": "7-node enterprise network: Internet, WEB, MAIL, APP, DATABASE, EMPLOYEE-1, EMPLOYEE-2, plus a DECOY honeypot.",
        "node_count": 8,
        "critical_assets": ["database"],
        "topology": {
            "internet": {"outgoing": ["web", "mail"]},
            "web": {"outgoing": ["app", "mail", "decoy-1"], "incoming": ["internet"]},
            "mail": {"outgoing": ["web"], "incoming": ["internet"]},
            "app": {"outgoing": ["web", "database"], "incoming": ["web"]},
            "database": {"outgoing": ["app", "employee-1", "employee-2"], "incoming": ["app"]},
            "employee-1": {"outgoing": ["database"], "incoming": ["database"]},
            "employee-2": {"outgoing": ["database"], "incoming": ["database"]},
            "decoy-1": {"outgoing": [], "incoming": ["web"]},
        },
        "honeypots": ["decoy-1"],
    }
