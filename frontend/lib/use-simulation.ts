"use client";
// lib/use-simulation.ts — React state machine for the live battle WebSocket.
//
// Owns: config fetch, session creation, WS lifecycle, and the reducer for all
// server messages. Components consume read-only state + command callbacks.

import { useCallback, useEffect, useReducer, useRef } from "react";
import {
  ApiError,
  SimSocket,
  api,
  type AgentInfo,
  type NetworkSnapshot,
  type ScenarioSummary,
  type SimServerMessage,
} from "./api";

export type ConnPhase =
  | "idle"
  | "loading-config"
  | "config-error"
  | "ready"
  | "creating-session"
  | "connecting"
  | "live"
  | "paused"
  | "ended"
  | "backend-down";

export type SimEvent = {
  seq: number;
  step: number | null;
  agent: string;
  action: string;
  target: string | null;
  result: string | null;
};

export type SimulationState = {
  phase: ConnPhase;
  configError: string;
  scenarios: ScenarioSummary[];
  redAgents: AgentInfo[];
  blueAgents: AgentInfo[];
  // selected config
  scenarioId: string;
  redPolicy: string;
  bluePolicy: string;
  seed: number;
  maxCycles: number;
  // live state
  wsStatus: "connecting" | "open" | "closed" | "error" | null;
  simId: string | null;
  step: number;
  network: NetworkSnapshot | null;
  redReward: number;
  blueReward: number;
  lastRedAction: string;
  lastBlueAction: string;
  winner: string | null;
  outcome: string | null;
  events: SimEvent[];
};

const MAX_EVENTS = 200;

const initialState: SimulationState = {
  phase: "idle",
  configError: "",
  scenarios: [],
  redAgents: [],
  blueAgents: [],
  scenarioId: "v1_enterprise",
  redPolicy: "scripted_aggressive",
  bluePolicy: "rule_based",
  seed: 42,
  maxCycles: 100,
  wsStatus: null,
  simId: null,
  step: 0,
  network: null,
  redReward: 0,
  blueReward: 0,
  lastRedAction: "",
  lastBlueAction: "",
  winner: null,
  outcome: null,
  events: [],
};

type Action =
  | { t: "config-loading" }
  | { t: "config-loaded"; scenarios: ScenarioSummary[]; redAgents: AgentInfo[]; blueAgents: AgentInfo[] }
  | { t: "config-error"; message: string }
  | { t: "set-config"; key: "scenarioId" | "redPolicy" | "bluePolicy" | "seed" | "maxCycles"; value: string | number }
  | { t: "session-created"; simId: string }
  | { t: "ws-status"; status: SimulationState["wsStatus"] }
  | { t: "server"; msg: SimServerMessage }
  | { t: "backend-down"; message: string };

function reducer(state: SimulationState, a: Action): SimulationState {
  switch (a.t) {
    case "config-loading":
      return { ...state, phase: "loading-config", configError: "" };
    case "config-loaded":
      return {
        ...state,
        phase: "ready",
        scenarios: a.scenarios,
        redAgents: a.redAgents,
        blueAgents: a.blueAgents,
        // Only offer agents that actually exist (RL red marked unavailable until a checkpoint exists).
        redPolicy: state.redPolicy,
        bluePolicy: state.bluePolicy,
      };
    case "config-error":
      return { ...state, phase: "config-error", configError: a.message };
    case "set-config":
      return { ...state, [a.key]: a.value } as SimulationState;
    case "session-created":
      return { ...state, phase: "connecting", simId: a.simId, events: [], step: 0, winner: null, outcome: null };
    case "ws-status":
      return {
        ...state,
        wsStatus: a.status,
        phase:
          a.status === "error" || a.status === "closed"
            ? state.phase === "live" || state.phase === "paused" || state.phase === "ended"
              ? state.phase
              : "backend-down"
            : state.phase,
      };
    case "backend-down":
      return { ...state, phase: "backend-down", configError: a.message };
    case "server":
      return applyServer(state, a.msg);
  }
}

function applyServer(state: SimulationState, msg: SimServerMessage): SimulationState {
  switch (msg.type) {
    case "connected":
      return { ...state, phase: "paused", wsStatus: "open" };
    case "episode_start":
      return {
        ...state,
        phase: "live",
        step: 0,
        network: msg.network ?? state.network,
        events: [],
        redReward: 0,
        blueReward: 0,
        winner: null,
        outcome: null,
        lastRedAction: "",
        lastBlueAction: "",
      };
    case "event": {
      const ev: SimEvent = {
        seq: state.events.length ? state.events[state.events.length - 1].seq + 1 : 1,
        step: msg.step,
        agent: msg.agent,
        action: msg.action,
        target: msg.target,
        result: msg.result,
      };
      return { ...state, events: [...state.events, ev].slice(-MAX_EVENTS) };
    }
    case "state":
      return {
        ...state,
        phase: state.phase === "ended" ? state.phase : state.phase === "connecting" ? "live" : state.phase,
        step: msg.step,
        network: msg.network ?? state.network,
        redReward: msg.red_reward,
        blueReward: msg.blue_reward,
        lastRedAction: msg.last_red_action,
        lastBlueAction: msg.last_blue_action,
        winner: msg.winner,
      };
    case "episode_end":
      return {
        ...state,
        phase: "ended",
        winner: msg.winner,
        outcome: msg.outcome,
        step: msg.step,
        redReward: msg.red_reward,
        blueReward: msg.blue_reward,
      };
    case "paused":
      return { ...state, phase: "paused", step: msg.step };
    case "resumed":
      return { ...state, phase: "live" };
    case "reset_done":
      return {
        ...state,
        phase: "paused",
        step: 0,
        network: msg.network,
        redReward: 0,
        blueReward: 0,
        events: [],
        winner: null,
        outcome: null,
        lastRedAction: "",
        lastBlueAction: "",
      };
    case "error":
      return { ...state, configError: msg.message };
    default:
      return state;
  }
}

export type ConfigKey = "scenarioId" | "redPolicy" | "bluePolicy" | "seed" | "maxCycles";

export function useSimulation() {
  const [state, dispatch] = useReducer(
    (s: SimulationState, action: Action): SimulationState => reducer(s, action),
    initialState,
  );
  const sockRef = useRef<SimSocket | null>(null);

  // Load config (scenarios + agents) once.
  useEffect(() => {
    let cancelled = false;
    dispatch({ t: "config-loading" });
    (async () => {
      try {
        const [scenarios, agents] = await Promise.all([api.scenarios(), api.agents()]);
        if (!cancelled) {
          dispatch({ t: "config-loaded", scenarios: scenarios.scenarios, redAgents: agents.red, blueAgents: agents.blue });
        }
      } catch (e) {
        if (!cancelled) {
          const message = e instanceof ApiError ? e.message : String(e);
          dispatch({ t: "backend-down", message });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const createSession = useCallback(async () => {
    dispatch({ t: "session-created", simId: "pending" });
    try {
      const resp = await api.startSimulation({
        scenario_id: state.scenarioId,
        red_policy: state.redPolicy,
        blue_policy: state.bluePolicy,
        seed: state.seed,
        max_cycles: state.maxCycles,
      });
      dispatch({ t: "session-created", simId: resp.sim_id });
      const sock = new SimSocket(resp.sim_id);
      sockRef.current = sock;
      sock.onStatus((status) => dispatch({ t: "ws-status", status }));
      sock.onMessage((msg) => dispatch({ t: "server", msg }));
      sock.send({ type: "start", seed: state.seed });
    } catch (e) {
      const message = e instanceof ApiError ? e.message : String(e);
      dispatch({ t: "backend-down", message });
    }
  }, [state.scenarioId, state.redPolicy, state.bluePolicy, state.seed, state.maxCycles]);

  const teardown = useCallback(() => {
    sockRef.current?.close();
    sockRef.current = null;
  }, []);

  useEffect(() => () => teardown(), [teardown]);

  const stop = useCallback(() => {
    sockRef.current?.send({ type: "pause" });
  }, []);

  const resume = useCallback(() => {
    sockRef.current?.send({ type: "resume" });
  }, []);

  const stepOnce = useCallback(() => {
    sockRef.current?.send({ type: "step", count: 1 });
  }, []);

  const reset = useCallback(() => {
    sockRef.current?.send({ type: "reset" });
  }, []);

  const restart = useCallback(
    async (seed?: number) => {
      teardown();
      await new Promise((r) => setTimeout(r, 100));
      if (seed !== undefined) dispatch({ t: "set-config", key: "seed", value: seed });
      await createSession();
    },
    [teardown, createSession],
  );

  const setConfig = useCallback(
    (key: ConfigKey, value: string | number) => dispatch({ t: "set-config", key, value }),
    [],
  );

  return { state, dispatch, createSession, stop, resume, stepOnce, reset, restart, setConfig };
}
