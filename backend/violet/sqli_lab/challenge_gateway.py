"""Private in-memory bridge for challenge setup on the isolated SQLi target."""

from __future__ import annotations

from uuid import UUID

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

TARGET_URL = "http://sqli-target:8002"


class ChallengeSecret(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phrase: str = Field(min_length=1, max_length=96)


app = FastAPI(title="VIOLET volatile challenge gateway", docs_url=None, redoc_url=None)


@app.post("/challenge/{challenge_id}")
async def set_challenge(challenge_id: UUID, body: ChallengeSecret) -> dict[str, str]:
    try:
        async with httpx.AsyncClient(base_url=TARGET_URL, timeout=3.0) as client:
            response = await client.post(
                f"/internal/challenges/{challenge_id}", json={"phrase": body.phrase}
            )
            response.raise_for_status()
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="isolated target challenge setup failed") from None
    return {"challenge_id": str(challenge_id), "status": "ready"}


@app.delete("/challenge/{challenge_id}")
async def clear_challenge(challenge_id: UUID) -> dict[str, str]:
    try:
        async with httpx.AsyncClient(base_url=TARGET_URL, timeout=3.0) as client:
            response = await client.delete(f"/internal/challenges/{challenge_id}")
            response.raise_for_status()
    except httpx.HTTPError:
        raise HTTPException(status_code=503, detail="isolated target challenge cleanup failed") from None
    return {"challenge_id": str(challenge_id), "status": "cleared"}
