# violet/api/app.py — FastAPI + WebSocket for VIOLET

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from violet.scenarios import v1_enterprise_network, v1_scenario_config
from violet.types import ATTACKER_AGENT, DEFENDER_AGENT
from violet.env import VIOLETEnv


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing heavy needed for V1; the env is created per-simulation.
    yield
    # Shutdown: cleanup if needed.


app = FastAPI(
    title="VIOLET API",
    description="Backend API + WebSocket for the VIOLET multi-agent cyber simulator.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "version": "0.1.0", "environment": "violet_v1"}


@app.get("/scenarios")
async def list_scenarios() -> Dict[str, Any]:
    return {
        "scenarios": [
            {
                "id": "v1_enterprise",
                "name": "V1 Enterprise Network",
                "description": v1_scenario_config()["description"],
                "node_count": v1_scenario_config()["node_count"],
                "critical_assets": v1_scenario_config()["critical_assets"],
            }
        ]
    }


@app.get("/scenarios/{scenario_id}")
async def get_scenario(scenario_id: str) -> Dict[str, Any]:
    if scenario_id == "v1_enterprise":
        return {"scenario": v1_scenario_config()}
    return JSONResponse(status_code=404, content={"error": "scenario not found"})


# ------------------------------------------------------------------
# Simulation Runs (ephemeral in-memory for V1 — backed by DB later)
# ------------------------------------------------------------------
active_simulations: Dict[str, Any] = {}


async def _run_simulation_loop(
    sim_id: str,
    env: VIOLETEnv,
    red_agent: Any,
    blue_agent: Any,
    websocket: WebSocket,
    max_cycles: int = 100,
) -> None:
    """Run the environment step-by-step and push events to the WebSocket."""
    obs, infos = env.reset(seed=42)
    red_agent.reset()
    blue_agent.reset()

    episode_events: List[Dict[str, Any]] = []
    red_reward = 0.0
    blue_reward = 0.0
    terminated = False

    await websocket.send_json({
        "type": "episode_start",
        "sim_id": sim_id,
        "seed": 42,
        "max_cycles": max_cycles,
        "network": env.network.snapshot() if env.network else None,
    })

    for step in range(max_cycles):
        if terminated:
            break

        red_action = red_agent.select_action(env, infos.get(ATTACKER_AGENT, {}))
        blue_action = blue_agent.select_action(env, infos.get(DEFENDER_AGENT, {}))
        actions = {
            ATTACKER_AGENT: red_action,
            DEFENDER_AGENT: blue_action,
        }
        obs, rewards, terminations, truncations, infos = env.step(actions)

        red_reward += rewards.get(ATTACKER_AGENT, 0.0)
        blue_reward += rewards.get(DEFENDER_AGENT, 0.0)
        terminated = any(terminations.values()) or any(truncations.values())

        # Push per-step events
        for agent_name in (ATTACKER_AGENT, DEFENDER_AGENT):
            for ev in infos.get(agent_name, {}).get("events", []):
                episode_events.append(ev)
                await websocket.send_json({
                    "type": "event",
                    "sim_id": sim_id,
                    "step": ev.get("step"),
                    "agent": ev.get("agent"),
                    "action": ev.get("action"),
                    "target": ev.get("target"),
                    "result": ev.get("result"),
                    "details": ev.get("details", {}),
                })

        # Push state snapshot periodically (every 5 steps or on termination)
        if step % 5 == 0 or terminated:
            await websocket.send_json({
                "type": "state",
                "sim_id": sim_id,
                "step": step + 1,
                "network": env.network.snapshot() if env.network else None,
                "red_reward": red_reward,
                "blue_reward": blue_reward,
                "terminated": terminated,
                "winner": "red" if (env.network and env.network.critical_assets_compromised() > 0) else None,
            })

        await asyncio.sleep(0.05)  # pacing so the UI is watchable

    await websocket.send_json({
        "type": "episode_end",
        "sim_id": sim_id,
        "step": step + 1,
        "red_reward": red_reward,
        "blue_reward": blue_reward,
        "winner": "red" if (env.network and env.network.critical_assets_compromised() > 0) else None,
        "terminated": terminated,
    })


@app.post("/simulation/start")
async def start_simulation(
    scenario_id: str = "v1_enterprise",
    red_policy: str = "scripted_aggressive",
    blue_policy: str = "random",
    checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Start a live simulation streamed to a WebSocket client.

    red_policy="rl" loads a trained PPO checkpoint via the RLAgent wrapper.
    Pass checkpoint=<path> to select a specific run directory (defaults to the
    best checkpoint of runs/ppo_red_v1). No checkpoint found -> 404-style error,
    never a silent fallback to fake behavior.
    """
    sim_id = f"sim-{int(time.time())}-{len(active_simulations)}"

    env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=100, seed=42)
    from violet.agents import (
        RandomRedAgent,
        RandomBlueAgent,
        ScriptedAggressiveRedAgent,
        RuleBasedBlueAgent,
        RLAgent,
    )

    if red_policy == "random":
        red_agent: Any = RandomRedAgent()
    elif red_policy == "rl":
        ckpt_dir = Path(checkpoint) if checkpoint else Path("runs/ppo_red_v1/checkpoints/best")
        red_agent = RLAgent("red", role="red")
        if not red_agent.load_checkpoint(str(ckpt_dir), env=env):
            return JSONResponse(
                status_code=404,
                content={
                    "error": "checkpoint not found",
                    "checkpoint": str(ckpt_dir),
                    "hint": "run `violet train` first, or pass checkpoint=<dir>",
                },
            )
    else:
        red_agent = ScriptedAggressiveRedAgent(target_node_id="database")

    if blue_policy == "random":
        blue_agent: Any = RandomBlueAgent()
    elif blue_policy == "rule_based":
        blue_agent = RuleBasedBlueAgent()
    else:
        blue_agent = RandomBlueAgent()

    active_simulations[sim_id] = {
        "env": env,
        "red_agent": red_agent,
        "blue_agent": blue_agent,
        "red_policy": red_policy,
        "blue_policy": blue_policy,
        "scenario_id": scenario_id,
        "started_at": time.time(),
    }

    return {
        "sim_id": sim_id,
        "scenario": scenario_id,
        "red_policy": red_policy,
        "blue_policy": blue_policy,
        "message": "connect to /ws/sim/{sim_id} to stream",
    }


@app.websocket("/ws/sim/{sim_id}")
async def simulation_websocket(websocket: WebSocket, sim_id: str):
    """WebSocket stream for a live simulation."""
    if sim_id not in active_simulations:
        await websocket.close(code=1008, reason="simulation not found")
        return

    await websocket.accept()
    state = active_simulations[sim_id]
    env = state["env"]
    red_agent = state["red_agent"]
    blue_agent = state["blue_agent"]

    try:
        await _run_simulation_loop(
            sim_id=sim_id,
            env=env,
            red_agent=red_agent,
            blue_agent=blue_agent,
            websocket=websocket,
            max_cycles=100,
        )
    except WebSocketDisconnect:
        pass
    finally:
        active_simulations.pop(sim_id, None)


@app.get("/simulation/{sim_id}/status")
async def simulation_status(sim_id: str) -> Dict[str, Any]:
    """Status of an active simulation (V1: only returns metadata, not live state)."""
    state = active_simulations.get(sim_id)
    if state is None:
        return {"sim_id": sim_id, "status": "not_found"}
    return {
        "sim_id": sim_id,
        "status": "running",
        "scenario": state["scenario_id"],
        "red_policy": state["red_policy"],
        "blue_policy": state["blue_policy"],
        "started_at": state["started_at"],
    }
