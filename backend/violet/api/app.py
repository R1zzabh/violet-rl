# violet/api/app.py — FastAPI + WebSocket for VIOLET
#
# Live simulation architecture:
#   POST /simulation/start   -> creates a paused, idle Session (no episode running)
#   WS   /ws/sim/{sim_id}    -> control + event channel (JSON request/response)
#
# Control messages (client -> server):
#   {"type": "start",   "seed": 42}
#   {"type": "pause"}
#   {"type": "resume"}
#   {"type": "step",    "count": 1}
#   {"type": "reset"}
#   {"type": "set_speed", "delay_ms": 400}
#
# Stream messages (server -> client):
#   {"type": "episode_start", ...}   episode begins (includes network snapshot)
#   {"type": "event", ...}           one real environment event
#   {"type": "state", ...}           after every step: network, rewards, last actions
#   {"type": "episode_end", ...}     terminal/truncation outcome
#   {"type": "paused"} / {"type": "resumed"} / {"type": "reset_done"} / {"type": "error"}

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from violet.api.range import (
    close_range_store,
    initialize_range_store,
)
from violet.api.range import router as range_router
from violet.env import VIOLETEnv
from violet.scenarios import v1_enterprise_network, v1_scenario_config
from violet.types import ATTACKER_AGENT, DEFENDER_AGENT

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    initialize_range_store()
    try:
        yield
    finally:
        close_range_store()


app = FastAPI(
    title="VIOLET API",
    description="Backend API + WebSocket for the VIOLET multi-agent cyber simulator.",
    version="0.2.0",
    lifespan=lifespan,
)

# The frontend (Next.js) is a distinct origin in both dev (localhost:3000) and
# docker (frontend service), so browser fetches need explicit CORS. WebSockets
# are not subject to CORS preflight, but REST calls are.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(range_router, prefix="/range", tags=["real range"])


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "version": "0.2.0", "environment": "violet_v1"}


@app.get("/scenarios")
async def list_scenarios() -> dict[str, Any]:
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
async def get_scenario(scenario_id: str) -> Response:
    if scenario_id == "v1_enterprise":
        return JSONResponse(content={"scenario": v1_scenario_config()})
    return JSONResponse(status_code=404, content={"error": "scenario not found"})


@app.get("/scenarios/{scenario_id}/topology")
async def get_scenario_topology(scenario_id: str) -> Response:
    """Return the full scenario description and a fresh ground-truth snapshot.

    Includes node metadata, vulnerabilities, services, and connections
    (static scenario data).
    """
    if scenario_id != "v1_enterprise":
        return JSONResponse(status_code=404, content={"error": "scenario not found"})
    network = v1_enterprise_network()
    return JSONResponse(content={"scenario": v1_scenario_config(), "network": network.snapshot()})


# ------------------------------------------------------------------
# Agents registry (only agents that actually exist in violet.agents)
# ------------------------------------------------------------------
RED_AGENTS = [
    {
        "id": "scripted_aggressive",
        "label": "Scripted Aggressive",
        "rl": False,
        "description": "Fixed attack strategy targeting DATABASE directly.",
    },
    {
        "id": "scripted_stealth",
        "label": "Scripted Stealth",
        "rl": False,
        "description": "Fixed low-and-slow attack strategy.",
    },
    {
        "id": "scripted_lateral",
        "label": "Scripted Lateral",
        "rl": False,
        "description": "Fixed lateral-movement strategy.",
    },
    {
        "id": "random",
        "label": "Random",
        "rl": False,
        "description": "Uniformly random valid actions.",
    },
    {
        "id": "rl",
        "label": "RL (PPO)",
        "rl": True,
        "description": "Trained PPO checkpoint (runs/ppo_red_v1).",
    },
]

BLUE_AGENTS = [
    {
        "id": "rule_based",
        "label": "Rule-Based",
        "rl": False,
        "description": "Deterministic monitor/isolate/patch rules.",
    },
    {
        "id": "reactive",
        "label": "Reactive",
        "rl": False,
        "description": "Responds only after detection events.",
    },
    {
        "id": "proactive",
        "label": "Proactive",
        "rl": False,
        "description": "Patrols and hardens before alerts fire.",
    },
    {
        "id": "random",
        "label": "Random",
        "rl": False,
        "description": "Uniformly random valid actions.",
    },
]


@app.get("/agents")
async def list_agents() -> dict[str, Any]:
    return {"red": RED_AGENTS, "blue": BLUE_AGENTS}


# ------------------------------------------------------------------
# Simulation sessions (ephemeral in-memory for V1 — backed by DB later)
# ------------------------------------------------------------------
active_sessions: dict[str, SimulationSession] = {}


def _load_rl_red_agent(env: VIOLETEnv, checkpoint: str | None) -> tuple[Any | None, str]:
    from violet.agents import RLAgent

    ckpt_dir = Path(checkpoint) if checkpoint else RUNS_DIR / "ppo_red_v1" / "checkpoints" / "best"
    agent = RLAgent("red", role="red")
    if not agent.load_checkpoint(str(ckpt_dir), env=env):
        return None, str(ckpt_dir)
    return agent, str(ckpt_dir)


def _make_red_agent(
    env: VIOLETEnv, red_policy: str, checkpoint: str | None
) -> tuple[Any | None, str | None]:
    from violet.agents import (
        RandomRedAgent,
        ScriptedAggressiveRedAgent,
        ScriptedLateralRedAgent,
        ScriptedStealthRedAgent,
    )

    if red_policy == "random":
        return RandomRedAgent(), None
    if red_policy == "rl":
        return _load_rl_red_agent(env, checkpoint)
    if red_policy == "scripted_stealth":
        return ScriptedStealthRedAgent(target_node_id="database"), None
    if red_policy == "scripted_lateral":
        return ScriptedLateralRedAgent(target_node_id="database"), None
    return ScriptedAggressiveRedAgent(target_node_id="database"), None


def _make_blue_agent(blue_policy: str) -> Any:
    from violet.agents import (
        ProactiveBlueAgent,
        RandomBlueAgent,
        ReactiveBlueAgent,
        RuleBasedBlueAgent,
    )

    if blue_policy == "rule_based":
        return RuleBasedBlueAgent()
    if blue_policy == "reactive":
        return ReactiveBlueAgent()
    if blue_policy == "proactive":
        return ProactiveBlueAgent()
    return RandomBlueAgent()


class SimulationSession:
    """One controllable live simulation.

    The environment only advances inside `advance_step()`, which is driven by
    explicit client commands over the WebSocket (start/resume/step). There is
    no background task advancing the env on its own, so "pause" is the natural
    state and "resume" simply steps in a loop with pacing.
    """

    def __init__(
        self,
        sim_id: str,
        scenario_id: str,
        red_policy: str,
        blue_policy: str,
        checkpoint: str | None,
        seed: int,
        max_cycles: int = 100,
    ):
        self.sim_id = sim_id
        self.scenario_id = scenario_id
        self.red_policy = red_policy
        self.blue_policy = blue_policy
        self.checkpoint = checkpoint
        self.seed = seed
        self.max_cycles = max_cycles
        self.step_delay_ms = 400  # default pacing for auto-play

        self.env = VIOLETEnv(scenario_fn=v1_enterprise_network, max_cycles=max_cycles, seed=seed)
        self.red_agent, self.checkpoint_error = _make_red_agent(self.env, red_policy, checkpoint)
        self.blue_agent = _make_blue_agent(blue_policy)

        self.red_reward = 0.0
        self.blue_reward = 0.0
        self.current_step = 0
        self.running = False  # auto-play flag
        self.episode_over = False
        self.winner: str | None = None
        self.pause_event = asyncio.Event()  # set when a pause is requested
        self.step_lock = asyncio.Lock()  # serializes env access across tasks

    # ----- commands -------------------------------------------------
    def reset_episode(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.env.reset(seed=self.seed)
        if self.red_agent is not None:
            self.red_agent.reset()
        self.blue_agent.reset()
        self.red_reward = 0.0
        self.blue_reward = 0.0
        self.current_step = 0
        self.episode_over = False
        self.winner = None
        self.running = False

    def select_actions(self) -> dict[str, int]:
        infos = getattr(self.env, "infos", {})
        assert self.red_agent is not None, "red agent unavailable (checked at session start)"
        red_action = self.red_agent.select_action(self.env, infos.get(ATTACKER_AGENT, {}))
        blue_action = self.blue_agent.select_action(self.env, infos.get(DEFENDER_AGENT, {}))
        return {ATTACKER_AGENT: red_action, DEFENDER_AGENT: blue_action}

    def advance_step(self) -> tuple[bool, str | None]:
        """Advance the environment exactly one step.

        Returns (terminated, error). error is a short string when the
        environment could not be advanced (never silently swallowed).
        """
        if self.episode_over:
            return True, None
        if self.red_agent is None:
            return False, "red agent unavailable (checkpoint load failed)"

        actions = self.select_actions()
        _obs, rewards, terminations, truncations, _infos = self.env.step(actions)

        self.red_reward += float(rewards.get(ATTACKER_AGENT, 0.0))
        self.blue_reward += float(rewards.get(DEFENDER_AGENT, 0.0))
        self.current_step = self.env.current_step

        terminated = any(terminations.values()) or any(truncations.values())
        if terminated:
            self.episode_over = True
            self.running = False
            if self.env.network and self.env.network.critical_assets_compromised() > 0:
                self.winner = "red"
            elif (
                self.env.network
                and self.env.network.critical_assets_compromised() == 0
                and any(terminations.values())
            ):
                self.winner = "blue"
            # truncation without critical-asset compromise -> undecided
        return terminated, None

    # ----- stream payloads ------------------------------------------
    def network_snapshot(self) -> dict[str, Any] | None:
        return self.env.network.snapshot() if self.env.network else None

    def event_messages(self, infos: dict[str, Any]) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        for agent_name in (ATTACKER_AGENT, DEFENDER_AGENT):
            for ev in infos.get(agent_name, {}).get("events", []):
                msgs.append(
                    {
                        "type": "event",
                        "sim_id": self.sim_id,
                        "step": ev.get("step"),
                        "agent": ev.get("agent"),
                        "action": ev.get("action"),
                        "target": ev.get("target"),
                        "result": ev.get("result"),
                        "details": ev.get("details", {}),
                    }
                )
        return msgs

    def state_message(self, terminated: bool = False) -> dict[str, Any]:
        red_action = ""
        blue_action = ""
        if self.env.last_result:
            red_action = self.env.last_result.get(ATTACKER_AGENT, "")
        return {
            "type": "state",
            "sim_id": self.sim_id,
            "step": self.current_step,
            "network": self.network_snapshot(),
            "red_reward": round(self.red_reward, 2),
            "blue_reward": round(self.blue_reward, 2),
            "last_red_action": red_action,
            "last_blue_action": blue_action,
            "terminated": terminated or self.episode_over,
            "winner": self.winner,
            "max_cycles": self.max_cycles,
        }

    def episode_end_message(self) -> dict[str, Any]:
        outcome = self.winner
        if outcome is None:
            outcome = "timeout"
        return {
            "type": "episode_end",
            "sim_id": self.sim_id,
            "step": self.current_step,
            "red_reward": round(self.red_reward, 2),
            "blue_reward": round(self.blue_reward, 2),
            "winner": self.winner,
            "outcome": outcome,
            "terminated": True,
        }


async def _ws_send(ws: WebSocket, msg: dict[str, Any]) -> None:
    await ws.send_json(msg)


async def _advance_and_stream(session: SimulationSession, ws: WebSocket) -> bool:
    """Advance exactly one environment step and stream the resulting messages.

    Returns True when the episode ended with this step (or could not advance).
    The step lock serializes env access between the auto-play task and
    client-initiated "step" commands.
    """
    async with session.step_lock:
        if session.episode_over:
            return True
        terminated, error = session.advance_step()
        if error:
            await _ws_send(ws, {"type": "error", "message": error})
            session.running = False
            return True
        for msg in session.event_messages(session.env.infos):
            await _ws_send(ws, msg)
        await _ws_send(ws, session.state_message(terminated=terminated))
        if terminated:
            await _ws_send(ws, session.episode_end_message())
            return True
    return False


async def _auto_play(session: SimulationSession, ws: WebSocket) -> None:
    """Auto-play task: advance the episode while session.running is set.

    Runs concurrently with the WebSocket receive loop so pause/step/reset
    commands are processed immediately instead of queueing behind pacing.
    """
    try:
        while session.running and not session.episode_over:
            ended = await _advance_and_stream(session, ws)
            if ended or not session.running:
                return
            # Pacing, interrupted immediately by a pause request.
            session.pause_event.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    session.pause_event.wait(), timeout=session.step_delay_ms / 1000.0
                )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover — never swallow stream errors
        session.running = False
        with contextlib.suppress(Exception):
            await _ws_send(ws, {"type": "error", "message": f"stream error: {exc}"})


@app.post("/simulation/start")
async def start_simulation(
    scenario_id: str = "v1_enterprise",
    red_policy: str = "scripted_aggressive",
    blue_policy: str = "rule_based",
    checkpoint: str | None = None,
    seed: int = 42,
    max_cycles: int = 100,
) -> Response:
    """Create an idle simulation session.

    The session is created PAUSED and NO episode is running until the client
    sends {"type": "start"} over the WebSocket. For red_policy="rl" the
    checkpoint is loaded eagerly so a missing checkpoint fails here with 404
    instead of mid-stream.
    """
    if scenario_id != "v1_enterprise":
        return JSONResponse(status_code=404, content={"error": "scenario not found"})

    session = SimulationSession(
        sim_id=f"sim-{int(time.time())}-{len(active_sessions)}",
        scenario_id=scenario_id,
        red_policy=red_policy,
        blue_policy=blue_policy,
        checkpoint=checkpoint,
        seed=seed,
        max_cycles=max_cycles,
    )

    # Eager checkpoint validation for RL red — never a silent fallback.
    if red_policy == "rl" and session.red_agent is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "checkpoint not found",
                "checkpoint": session.checkpoint_error,
                "hint": "run `violet train` first, or pass checkpoint=<dir>",
            },
        )

    active_sessions[session.sim_id] = session
    return JSONResponse(
        content={
            "sim_id": session.sim_id,
            "scenario": scenario_id,
            "red_policy": red_policy,
            "blue_policy": blue_policy,
            "seed": seed,
            "max_cycles": max_cycles,
            "message": 'connect to /ws/sim/{sim_id} and send {"type":"start"} to begin',
        }
    )


@app.get("/simulation/{sim_id}/status")
async def simulation_status(sim_id: str) -> dict[str, Any]:
    session = active_sessions.get(sim_id)
    if session is None:
        return {"sim_id": sim_id, "status": "not_found"}
    return {
        "sim_id": sim_id,
        "status": "running" if session.running else ("ended" if session.episode_over else "paused"),
        "scenario": session.scenario_id,
        "red_policy": session.red_policy,
        "blue_policy": session.blue_policy,
        "seed": session.seed,
        "step": session.current_step,
    }


@app.delete("/simulation/{sim_id}")
async def delete_simulation(sim_id: str) -> dict[str, Any]:
    removed = active_sessions.pop(sim_id, None)
    return {"sim_id": sim_id, "deleted": removed is not None}


@app.websocket("/ws/sim/{sim_id}")
async def simulation_websocket(websocket: WebSocket, sim_id: str) -> None:
    """Bidirectional control + event stream for a live simulation."""
    session = active_sessions.get(sim_id)
    if session is None:
        await websocket.close(code=1008, reason="simulation not found")
        return

    await websocket.accept()

    run_task: asyncio.Task[None] | None = None

    async def stop_run_task() -> None:
        nonlocal run_task
        session.running = False
        session.pause_event.set()
        if run_task is not None and not run_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(run_task), timeout=5.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                run_task.cancel()
        run_task = None

    try:
        await _ws_send(
            websocket,
            {
                "type": "connected",
                "sim_id": sim_id,
                "scenario": session.scenario_id,
                "red_policy": session.red_policy,
                "blue_policy": session.blue_policy,
                "seed": session.seed,
                "max_cycles": session.max_cycles,
            },
        )

        while True:
            raw = await websocket.receive_json()

            msg_type = raw.get("type")

            if msg_type == "start":
                # (Re)start: stop any running auto-play, reset to a fresh episode.
                await stop_run_task()
                seed = raw.get("seed")
                session.reset_episode(seed=int(seed) if seed is not None else None)
                await _ws_send(
                    websocket,
                    {
                        "type": "episode_start",
                        "sim_id": sim_id,
                        "seed": session.seed,
                        "max_cycles": session.max_cycles,
                        "network": session.network_snapshot(),
                    },
                )
                session.running = True
                session.pause_event.clear()
                run_task = asyncio.create_task(_auto_play(session, websocket))

            elif msg_type == "pause":
                await stop_run_task()
                await _ws_send(websocket, {"type": "paused", "step": session.current_step})

            elif msg_type == "resume":
                if session.episode_over:
                    await _ws_send(websocket, {"type": "error", "message": "episode already ended"})
                    continue
                if run_task is None or run_task.done():
                    session.running = True
                    session.pause_event.clear()
                    run_task = asyncio.create_task(_auto_play(session, websocket))
                    await _ws_send(websocket, {"type": "resumed", "step": session.current_step})
                else:
                    await _ws_send(websocket, {"type": "resumed", "step": session.current_step})

            elif msg_type == "step":
                if session.episode_over:
                    await _ws_send(websocket, {"type": "error", "message": "episode already ended"})
                    continue
                # Manual stepping always pauses auto-play first.
                await stop_run_task()
                count = max(1, min(int(raw.get("count", 1)), 50))
                for _ in range(count):
                    if session.episode_over:
                        break
                    ended = await _advance_and_stream(session, websocket)
                    if ended:
                        break

            elif msg_type == "reset":
                await stop_run_task()
                session.reset_episode()
                await _ws_send(
                    websocket,
                    {
                        "type": "reset_done",
                        "step": 0,
                        "network": session.network_snapshot(),
                        "red_reward": 0.0,
                        "blue_reward": 0.0,
                    },
                )

            elif msg_type == "set_speed":
                delay = int(raw.get("delay_ms", 400))
                session.step_delay_ms = max(0, min(delay, 5000))
                await _ws_send(websocket, {"type": "speed_set", "delay_ms": session.step_delay_ms})

            else:
                await _ws_send(
                    websocket, {"type": "error", "message": f"unknown message type: {msg_type}"}
                )

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover — surfaced to the client, never silent
        with contextlib.suppress(Exception):
            await _ws_send(websocket, {"type": "error", "message": f"server error: {exc}"})
    finally:
        if run_task is not None and not run_task.done():
            run_task.cancel()
        session.running = False
        active_sessions.pop(sim_id, None)


# ------------------------------------------------------------------
# Training runs (real artifacts from runs/<run_id>/)
# ------------------------------------------------------------------


def _read_runs_registry() -> list[dict[str, Any]]:
    """Scan runs/ for training runs. Only real artifacts are reported."""
    runs: list[dict[str, Any]] = []
    if not RUNS_DIR.exists():
        return runs
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics" / "training.jsonl"
        if not metrics_path.exists():
            continue
        n_updates = 0
        last: dict[str, Any] = {}
        try:
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        last = json.loads(line)
                        n_updates += 1
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
        if n_updates == 0:
            continue
        ckpt_best = run_dir / "checkpoints" / "best"
        runs.append(
            {
                "run_id": run_dir.name,
                "algorithm": last.get("algorithm", "ppo") if last.get("algorithm") else "ppo",
                "updates": n_updates,
                "global_step": last.get("global_step"),
                "mean_episode_reward": last.get("mean_episode_reward"),
                "attack_success_rate": last.get("attack_success_rate"),
                "has_checkpoint": ckpt_best.exists(),
                "metrics_file": str(metrics_path),
            }
        )
    return runs


@app.get("/runs")
async def list_runs() -> dict[str, Any]:
    return {"runs": _read_runs_registry()}


@app.get("/evaluation/{run_id}")
async def evaluation_summary(run_id: str) -> Response:
    """Real evaluation summary from runs/<run_id>/evaluation/eval_summary.json."""
    path = RUNS_DIR / run_id / "evaluation" / "eval_summary.json"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "no evaluation artifacts for run"})
    try:
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
        return JSONResponse(content=data)
    except (OSError, json.JSONDecodeError) as exc:
        return JSONResponse(
            status_code=500, content={"error": f"unreadable evaluation artifact: {exc}"}
        )


@app.get("/behavior/{run_id}")
async def behavior_report(run_id: str) -> Response:
    """Real behavior-discovery report from runs/<run_id>/evaluation/behavior_report.json."""
    path = RUNS_DIR / run_id / "evaluation" / "behavior_report.json"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "no behavior artifacts for run"})
    try:
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
        return JSONResponse(content=data)
    except (OSError, json.JSONDecodeError) as exc:
        return JSONResponse(
            status_code=500, content={"error": f"unreadable behavior artifact: {exc}"}
        )


@app.get("/runs/{run_id}/metrics")
async def run_metrics(run_id: str, max_points: int = 400) -> Response:
    """Stream a run's real training.jsonl metrics, downsampled client-side-free."""
    if run_id not in {r["run_id"] for r in _read_runs_registry()}:
        return JSONResponse(status_code=404, content={"error": "run not found"})
    metrics_path = RUNS_DIR / run_id / "metrics" / "training.jsonl"
    records: list[dict[str, Any]] = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    total = len(records)
    if total > max_points:
        stride = total / max_points
        records = [records[int(i * stride)] for i in range(max_points)]
    ckpt_meta_path = RUNS_DIR / run_id / "checkpoints" / "best" / "checkpoint_meta.json"
    checkpoint_meta: dict[str, Any] | None = None
    if ckpt_meta_path.exists():
        try:
            with open(ckpt_meta_path) as f:
                checkpoint_meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            checkpoint_meta = None
    return JSONResponse(
        content={
            "run_id": run_id,
            "total_updates": total,
            "downsampled_to": len(records),
            "metrics": records,
            "checkpoint_meta": checkpoint_meta,
        }
    )
