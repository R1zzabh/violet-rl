"""Smoke test for the VIOLET simulation WebSocket control protocol.

Run from backend/: .venv/bin/python scripts/ws_smoke.py
"""

import asyncio
import json
import urllib.request

BASE = "http://127.0.0.1:8000"


def http_json(path: str, method: str = "GET") -> dict:
    req = urllib.request.Request(BASE + path, method=method)
    return json.loads(urllib.request.urlopen(req).read())


async def drive(ws, scenario_qs: str) -> None:
    import websockets

    resp = http_json(f"/simulation/start?{scenario_qs}", method="POST")
    sim_id = resp["sim_id"]
    print(f"sim_id={sim_id}")

    async with websockets.connect(f"ws://127.0.0.1:8000/ws/sim/{sim_id}") as sock:
        hello = json.loads(await sock.recv())
        assert hello["type"] == "connected", hello

        # START -> should stream episode_start, events, states until end or pace
        await sock.send(json.dumps({"type": "start", "seed": 42}))
        first_state = None
        events = 0
        while True:
            msg = json.loads(await sock.recv())
            if msg["type"] == "event":
                events += 1
            if msg["type"] == "state":
                first_state = msg
                break
            if msg["type"] in ("episode_end", "error"):
                print("early:", msg)
                return
        print(
            f"start: events={events} step={first_state['step']} nodes={len(first_state['network']['nodes'])}"
        )

        # PAUSE
        await sock.send(json.dumps({"type": "pause"}))
        msg = json.loads(await sock.recv())
        print("pause ->", msg["type"], "step", msg.get("step"))

        # STEP x3
        steps = []
        for _ in range(3):
            await sock.send(json.dumps({"type": "step"}))
            got_state = None
            while got_state is None:
                m = json.loads(await sock.recv())
                if m["type"] == "state":
                    got_state = m
                elif m["type"] in ("episode_end", "error"):
                    print("early:", m)
                    got_state = "END"
                    break
            if got_state == "END":
                break
            steps.append(got_state["step"])
        print("steps:", steps)

        # RESET
        await sock.send(json.dumps({"type": "reset"}))
        m = json.loads(await sock.recv())
        print(
            "reset ->",
            m["type"],
            "step",
            m.get("step"),
            "nodes",
            len(m.get("network", {}).get("nodes", {})),
        )

        # RESUME (fresh episode after reset) — run to completion
        await sock.send(json.dumps({"type": "resume"}))
        final = None
        n_events = 0
        while final is None:
            m = json.loads(await sock.recv())
            if m["type"] == "event":
                n_events += 1
            if m["type"] == "episode_end":
                final = m
            if m["type"] == "error":
                print("error:", m)
                return
        print(
            f"resume: final outcome={final['outcome']} step={final['step']} red={final['red_reward']} blue={final['blue_reward']} events={n_events}"
        )


async def main() -> None:
    print("== scripted aggressive vs rule_based ==")
    await drive(
        None, "scenario_id=v1_enterprise&red_policy=scripted_aggressive&blue_policy=rule_based"
    )
    print("== rl (PPO checkpoint) vs rule_based ==")
    await drive(None, "scenario_id=v1_enterprise&red_policy=rl&blue_policy=rule_based")


if __name__ == "__main__":
    asyncio.run(main())
