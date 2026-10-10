"""Violet Ledger's intentionally narrow challenge and defense behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

if TYPE_CHECKING:
    from violet.target_app.control import TargetRuntime

SYNTHETIC_CREDENTIALS = {
    "synthetic_alice": "alice",
    "synthetic_bob": "bob",
}


def authenticate(runtime: TargetRuntime, credential_ref: str) -> str:
    principal = SYNTHETIC_CREDENTIALS.get(credential_ref)
    if principal is None:
        raise HTTPException(status_code=401, detail="invalid synthetic credentials")
    if principal in runtime.blocked_principals:
        raise HTTPException(status_code=423, detail="synthetic account blocked")
    return principal


def read_document(
    runtime: TargetRuntime,
    principal: str | None,
    document_ref: str,
) -> dict[str, Any]:
    if principal is None:
        raise HTTPException(status_code=401, detail="authentication required")
    if principal in runtime.blocked_principals:
        raise HTTPException(status_code=423, detail="synthetic account blocked")
    count = runtime.request_counts.get(principal, 0) + 1
    runtime.request_counts[principal] = count
    if principal in runtime.rate_limited_principals and count > 3:
        raise HTTPException(status_code=429, detail="synthetic actor rate limited")
    document = runtime.database.get_document(document_ref)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    if runtime.ownership_hardened and document["owner_ref"] != principal:
        raise HTTPException(status_code=403, detail="document owner check denied access")
    return document
