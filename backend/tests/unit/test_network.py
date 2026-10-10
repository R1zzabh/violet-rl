# tests/unit/test_network.py — VIOLET

from violet.network import Network, Node
from violet.types import CompromiseStatus, IsolationStatus


def test_node_defaults():
    n = Node(id="web", label="WEB")
    assert n.id == "web"
    assert n.value == 10.0
    assert n.compromise_status == CompromiseStatus.NONE
    assert n.isolated == IsolationStatus.CONNECTED
    assert n.is_compromised() is False


def test_node_compromise_behavior():
    n = Node(id="db", label="DATABASE", is_critical_asset=True, value=100.0)
    assert n.is_compromised() is False

    n.compromise_status = CompromiseStatus.COMPROMISED
    assert n.is_compromised() is True


def test_network_add_and_connect():
    net = Network()
    a = Node(id="a", label="A")
    b = Node(id="b", label="B")
    net.add_node(a)
    net.add_node(b)
    net.connect_bidirectional("a", "b")

    assert "b" in net.nodes["a"].connections
    assert "a" in net.nodes["b"].connections


def test_unidirectional_connect():
    net = Network()
    src = Node(id="internet", label="INTERNET")
    dst = Node(id="web", label="WEB")
    net.add_node(src)
    net.add_node(dst)
    net.connect_unidirectional("internet", "web")

    assert "web" in net.nodes["internet"].connections
    assert "internet" not in net.nodes["web"].connections


def test_isolation_blocks_reachability():
    net = Network()
    a = Node(id="a", label="A")
    b = Node(id="b", label="B")
    net.add_node(a)
    net.add_node(b)
    net.connect_bidirectional("a", "b")

    assert net._is_reachable("a", "b", set()) is True

    a.isolated = IsolationStatus.ISOLATED
    assert net._is_reachable("a", "b", set()) is False


def test_reachable_from():
    net = Network()
    web = Node(id="web", label="WEB")
    app = Node(id="app", label="APP")
    db = Node(id="database", label="DATABASE")
    net.add_node(web)
    net.add_node(app)
    net.add_node(db)
    net.connect_bidirectional("web", "app")
    net.connect_bidirectional("app", "database")

    reachable = net.reachable_from("web")
    assert "app" in reachable
    assert "database" in reachable


def test_snapshot():
    net = Network()
    n = Node(id="web", label="WEB", value=20.0, vulnerability_rate=0.8)
    net.add_node(n)
    snap = net.snapshot()
    assert snap["nodes"]["web"]["value"] == 20.0
    assert snap["nodes"]["web"]["compromise_status"] == "none"


def test_clone_independence():
    net = Network()
    n = Node(id="web", label="WEB")
    net.add_node(n)
    clone = net.clone()
    clone.nodes["web"].compromise_status = CompromiseStatus.COMPROMISED
    assert net.nodes["web"].compromise_status == CompromiseStatus.NONE
    assert clone.nodes["web"].is_compromised() is True


def test_critical_assets_compromised():
    net = Network()
    db = Node(id="database", label="DATABASE", is_critical_asset=True)
    web = Node(id="web", label="WEB")
    net.add_node(db)
    net.add_node(web)

    assert net.critical_assets_compromised() == 0
    db.compromise_status = CompromiseStatus.COMPROMISED
    assert net.critical_assets_compromised() == 1


def test_compromised_value():
    net = Network()
    web = Node(id="web", label="WEB", value=20.0)
    db = Node(id="database", label="DATABASE", value=100.0, is_critical_asset=True)
    net.add_node(web)
    net.add_node(db)

    assert net.compromised_value() == 0.0
    web.compromise_status = CompromiseStatus.COMPROMISED
    assert net.compromised_value() == 20.0
