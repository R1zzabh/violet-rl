"""Audit Compose network membership and bounded service reachability."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, cast

SERVICES = {
    "backend": {"control", "api-host", "trainer-control"},
    "range-trainer": {"trainer-control"},
    "range-executor": {"range"},
    "violet-ledger": {"range", "target-data"},
    "db": {"control"},
    "target-db": {"target-data"},
    "range-migrate": {"control"},
    "target-migrate": {"target-data"},
    "sqli-db": {"sqli-data"},
    "sqli-migrate": {"sqli-data"},
    "sqli-target": {"sqli-range", "sqli-data"},
    "sqli-executor": {"sqli-range"},
}


def command(*args: str) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def container_id(service: str) -> str:
    value = command("docker", "compose", "ps", "-aq", service)
    if not value:
        raise RuntimeError(f"Compose service {service!r} is not running")
    return value


def inspect(service: str) -> dict[str, Any]:
    return cast(
        "dict[str, Any]", json.loads(command("docker", "inspect", container_id(service)))[0]
    )


def networks(service: str) -> set[str]:
    return {name.rsplit("_", 1)[-1] for name in inspect(service)["NetworkSettings"]["Networks"]}


def probe(
    service: str, allowed: list[tuple[str, int]], denied: list[tuple[str, int]]
) -> dict[str, Any]:
    source = """
import json, socket, struct, sys
allowed=json.loads(sys.argv[1]); denied=json.loads(sys.argv[2]); out={"allowed":{},"denied":{}}
def can_connect(host, port):
    try:
        with socket.create_connection((host, port), timeout=1.5): return True
    except OSError: return False
for host,port in allowed: out["allowed"][f"{host}:{port}"]=can_connect(host,port)
for host,port in denied: out["denied"][f"{host}:{port}"]=can_connect(host,port)
try:
    for row in open('/proc/net/route').read().splitlines()[1:]:
        fields=row.split()
        if fields[1]=='00000000':
            gateway=socket.inet_ntoa(struct.pack('<L', int(fields[2],16)))
            out['denied'][f'host-gateway:{gateway}:22']=can_connect(gateway,22)
            out['denied'][f'host-gateway:{gateway}:8000']=can_connect(gateway,8000)
            break
except OSError: pass
print(json.dumps(out))
"""
    raw = command(
        "docker",
        "compose",
        "exec",
        "-T",
        service,
        "python",
        "-c",
        source,
        json.dumps(allowed),
        json.dumps(denied),
    )
    return cast("dict[str, Any]", json.loads(raw))


def schema_creation_denied(service: str, database_url_env: str) -> bool:
    source = """
import os, sys
from sqlalchemy import create_engine
from sqlalchemy.exc import DBAPIError
engine=create_engine(os.environ[sys.argv[1]])
table='violet_runtime_ddl_probe'
try:
    with engine.connect() as connection:
        connection.exec_driver_sql(f'CREATE TABLE {table} (id integer)')
        connection.commit()
except DBAPIError as error:
    if 'permission denied' not in str(error).lower(): raise
    print('denied')
else:
    with engine.begin() as connection:
        connection.exec_driver_sql(f'DROP TABLE IF EXISTS {table}')
    print('allowed')
finally:
    engine.dispose()
"""
    result = command(
        "docker",
        "compose",
        "exec",
        "-T",
        service,
        "python",
        "-c",
        source,
        database_url_env,
    )
    return result == "denied"


def main() -> int:
    failures: list[str] = []
    report: dict[str, Any] = {
        "schema_id": "violet.range.isolation_report.v1",
        "services": {},
        "skipped_services": [],
    }
    services = dict(SERVICES)
    # The trainer image is optional during lightweight isolation checks because
    # CPU Torch is large. When it is running, its boundaries are always checked.
    trainer_running = command("docker", "compose", "ps", "-q", "range-trainer")
    if not trainer_running:
        services.pop("range-trainer")
        if os.environ.get("VIOLET_REQUIRE_TRAINER") == "1":
            failures.append("range-trainer is required but is not running")
        else:
            report["skipped_services"].append("range-trainer: service is not running")
    for service, expected in services.items():
        actual = networks(service)
        report["services"][service] = {
            "expected_networks": sorted(expected),
            "actual_networks": sorted(actual),
        }
        if actual != expected:
            failures.append(f"{service} networks {sorted(actual)} != {sorted(expected)}")
        exposed = inspect(service).get("NetworkSettings", {}).get("Ports", {})
        published = {port: bindings for port, bindings in exposed.items() if bindings}
        report["services"][service]["published_ports"] = published
        if service == "backend":
            expected_ports = {"8000/tcp"}
            if set(published) != expected_ports:
                failures.append(
                    f"backend published ports {sorted(published)} != {sorted(expected_ports)}"
                )
            for port, bindings in published.items():
                if any(binding.get("HostIp") != "127.0.0.1" for binding in bindings):
                    failures.append(
                        f"backend port {port} is not bound only to loopback: {bindings}"
                    )
        elif (
            service
            in {
                "db",
                "target-db",
                "violet-ledger",
                "range-executor",
                "range-trainer",
                "range-migrate",
                "target-migrate",
                "sqli-db",
                "sqli-target",
                "sqli-executor",
                "sqli-migrate",
            }
            and published
        ):
            failures.append(f"{service} unexpectedly publishes ports: {published}")

    executor = probe(
        "range-executor",
        [("violet-ledger", 8001)],
        [
            ("backend", 8000),
            ("db", 5432),
            ("target-db", 5432),
            ("1.1.1.1", 443),
            ("169.254.169.254", 80),
        ],
    )
    probes = {"executor": executor}
    actors = [("executor", executor)]
    if trainer_running:
        trainer = probe(
            "range-trainer",
            [("backend", 8000)],
            [
                ("db", 5432),
                ("target-db", 5432),
                ("violet-ledger", 8001),
                ("1.1.1.1", 443),
                ("169.254.169.254", 80),
            ],
        )
        probes["trainer"] = trainer
        actors.append(("trainer", trainer))
        if not trainer["allowed"].get("backend:8000"):
            failures.append("trainer cannot reach the VIOLET API")

    sqli_executor = probe(
        "sqli-executor",
        [("sqli-target", 8002)],
        [
            ("backend", 8000),
            ("db", 5432),
            ("target-db", 5432),
            ("sqli-db", 5432),
            ("violet-ledger", 8001),
            ("1.1.1.1", 443),
            ("169.254.169.254", 80),
        ],
    )
    sqli_target = probe(
        "sqli-target",
        [("sqli-db", 5432)],
        [
            ("backend", 8000),
            ("db", 5432),
            ("target-db", 5432),
            ("violet-ledger", 8001),
            ("1.1.1.1", 443),
            ("169.254.169.254", 80),
        ],
    )
    probes["sqli_executor"] = sqli_executor
    probes["sqli_target"] = sqli_target
    actors.extend([("sqli_executor", sqli_executor), ("sqli_target", sqli_target)])
    if trainer_running:
        trainer = probes["trainer"]
        for destination in ("sqli-target:8002", "sqli-db:5432"):
            if trainer["allowed"].get(destination):
                failures.append(f"trainer unexpectedly reached {destination}")
    if not sqli_executor["allowed"].get("sqli-target:8002"):
        failures.append("SQLi executor cannot reach its fixed HTTP target")
    if not sqli_target["allowed"].get("sqli-db:5432"):
        failures.append("SQLi target cannot reach its private database")
    report["probes"] = probes
    ddl_permissions = {
        "backend": schema_creation_denied("backend", "DATABASE_URL"),
        "violet-ledger": schema_creation_denied("violet-ledger", "TARGET_DATABASE_URL"),
        "sqli-target": schema_creation_denied("sqli-target", "SQLI_DATABASE_URL"),
    }
    report["runtime_schema_creation_denied"] = ddl_permissions
    for service, denied in ddl_permissions.items():
        if not denied:
            failures.append(f"{service} runtime database role can create schema objects")
    sqli_dml_source = """
import os, sys
from sqlalchemy import create_engine
from sqlalchemy.exc import DBAPIError
engine=create_engine(os.environ[sys.argv[1]])
try:
    with engine.connect() as connection:
        connection.exec_driver_sql('UPDATE sqli_records SET handle=handle WHERE false')
        connection.commit()
except DBAPIError as error:
    if 'permission denied' not in str(error).lower(): raise
    print('denied')
else:
    print('allowed')
finally:
    engine.dispose()
"""
    sqli_dml = command(
        "docker",
        "compose",
        "exec",
        "-T",
        "sqli-target",
        "python",
        "-c",
        sqli_dml_source,
        "SQLI_DATABASE_URL",
    )
    report["sqli_runtime_dml_denied"] = sqli_dml == "denied"
    if sqli_dml != "denied":
        failures.append("SQLi target runtime database role can update fixture rows")
    if not executor["allowed"].get("violet-ledger:8001"):
        failures.append("executor cannot reach the allowlisted Violet Ledger service")
    for actor, result in actors:
        for destination, reachable in result["denied"].items():
            if reachable:
                failures.append(f"{actor} reached prohibited destination {destination}")
    report["status"] = "failed" if failures else "passed"
    report["failures"] = failures
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
