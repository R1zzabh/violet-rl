// lib/api.ts — typed REST + WebSocket client for the VIOLET backend.
//
// The base URL is configurable via NEXT_PUBLIC_API_URL (set in docker-compose).
// Defaults to the local dev backend on 127.0.0.1:8000.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export function wsUrl(path: string): string {
  const url = new URL(path, API_BASE);
  const proto = url.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${url.host}${url.pathname}`;
}

// ---------------------------------------------------------------------------
// Wire types (mirror the FastAPI payloads — see backend/violet/api/app.py)
// ---------------------------------------------------------------------------

export type ScenarioSummary = {
  id: string;
  name: string;
  description: string;
  node_count: number;
  critical_assets: string[];
};

export type NodeSnapshot = {
  id: string;
  label: string;
  node_type: string;
  value: number;
  compromise_status: string; // none | discovered | compromised | privileges_escalated | exfiltrated
  alert_level: string; // none | low | medium | high
  service_status: string; // active | down | patched
  isolated: string; // connected | isolated
  is_honeypot: boolean;
  is_critical_asset: boolean;
  connections: string[];
  incoming: string[];
  vulnerabilities: string[];
  services: string[];
};

export type NetworkSnapshot = {
  step: number;
  nodes: Record<string, NodeSnapshot>;
  firewall_rules: string[];
  active_alerts: Record<string, string>;
};

export type AgentInfo = {
  id: string;
  label: string;
  rl: boolean;
  description: string;
};

export type RunSummary = {
  run_id: string;
  algorithm: string;
  updates: number;
  global_step: number | null;
  mean_episode_reward: number | null;
  attack_success_rate: number | null;
  has_checkpoint: boolean;
  metrics_file: string;
};

export type TrainingMetric = {
  update: number;
  global_step?: number | null;
  mean_episode_reward?: number | null;
  attack_success_rate?: number | null;
  loss?: number;
  policy_loss?: number;
  value_loss?: number;
  entropy?: number;
  approx_kl?: number;
  clip_frac?: number;
  episodes_completed?: number;
};

export type RunMetricsResponse = {
  run_id: string;
  total_updates: number;
  downsampled_to: number;
  metrics: TrainingMetric[];
  checkpoint_meta: Record<string, unknown> | null;
};

export type EvalMatchupSummary = {
  n_episodes: number;
  attack_success_rate: number;
  red_win_rate: number;
  blue_win_rate: number;
  undecided_rate: number;
  mean_red_reward: number;
  mean_blue_reward: number;
  mean_episode_length: number;
};

export type EvalMatchup = {
  matchup: string;
  red_agent: string;
  blue_agent: string;
  summary: EvalMatchupSummary;
};

export type EvalSummary = {
  run_id: string;
  generated_at: string;
  n_episodes_per_matchup: number;
  seed_range: number[];
  matchups: EvalMatchup[];
};

export type BehaviorPatternReport = {
  id: string;
  condition: string;
  observed_behavior: string;
  frequency: string;
  baseline: string;
  improvement: string;
  confidence: string;
  side_effects?: string[];
};

export type BehaviorReport = {
  run_id: string;
  red_behaviors: BehaviorPatternReport[];
  blue_behaviors: BehaviorPatternReport[];
  red_action_distribution: { condition: string; total: number; actions: Record<string, number> };
  blue_action_distribution: { condition: string; total: number; actions: Record<string, number> };
  red_action_distribution_conditioned: { condition: string; total: number; actions: Record<string, number> };
  blue_action_distribution_conditioned: { condition: string; total: number; actions: Record<string, number> };
  red_heatmap: Record<string, Record<string, number>>;
  blue_heatmap: Record<string, Record<string, number>>;
};

export type ScenarioDetail = {
  scenario: Record<string, unknown>;
  network: NetworkSnapshot;
};

export type RangeRun = {
  run_id: string;
  status: string;
  target_version: string;
  created_at: string;
  event_count: number;
};
export type RangeEvent = {
  run_id: string;
  seq: number;
  event_type: string;
  episode_id: string | null;
  occurred_at: string;
  payload: Record<string, unknown>;
};

// ---------------------------------------------------------------------------
// REST helpers — always surface a clean error; callers own the UI state.
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function getJson<T>(path: string): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  } catch {
    throw new ApiError(`Backend unreachable at ${API_BASE}`, 0);
  }
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = (await resp.json()) as { error?: string };
      if (body.error) detail = body.error;
    } catch {
      /* keep HTTP status detail */
    }
    throw new ApiError(detail, resp.status);
  }
  return (await resp.json()) as T;
}

export const api = {
  scenarios: () => getJson<{ scenarios: ScenarioSummary[] }>("/scenarios"),
  scenarioTopology: (id: string) =>
    getJson<ScenarioDetail>(`/scenarios/${encodeURIComponent(id)}/topology`),
  agents: () => getJson<{ red: AgentInfo[]; blue: AgentInfo[] }>("/agents"),
  runs: () => getJson<{ runs: RunSummary[] }>("/runs"),
  runMetrics: (runId: string, maxPoints = 400) =>
    getJson<RunMetricsResponse>(
      `/runs/${encodeURIComponent(runId)}/metrics?max_points=${maxPoints}`,
    ),
  evaluation: (runId: string) =>
    getJson<EvalSummary>(`/evaluation/${encodeURIComponent(runId)}`),
  behavior: (runId: string) =>
    getJson<BehaviorReport>(`/behavior/${encodeURIComponent(runId)}`),
  rangeRuns: () => getJson<{ runs: RangeRun[] }>("/range/runs"),
  rangeEvents: (runId: string) =>
    getJson<{ run_id: string; events: RangeEvent[] }>(
      `/range/runs/${encodeURIComponent(runId)}/events?limit=1000`,
    ),
  startSimulation: (params: {
    scenario_id?: string;
    red_policy?: string;
    blue_policy?: string;
    seed?: number;
    max_cycles?: number;
  }) =>
    fetch(`${API_BASE}/simulation/start?${new URLSearchParams(
      Object.entries(params).map(([k, v]) => [k, String(v)]),
    )}`, { method: "POST" }).then(async (resp) => {
      if (!resp.ok) {
        let detail = `HTTP ${resp.status}`;
        try {
          const body = (await resp.json()) as { error?: string; hint?: string };
          if (body.error) detail = body.hint ? `${body.error} (${body.hint})` : body.error;
        } catch {
          /* keep status */
        }
        throw new ApiError(detail, resp.status);
      }
      return (await resp.json()) as { sim_id: string; message: string };
    }),
};

// ---------------------------------------------------------------------------
// WebSocket control protocol
// ---------------------------------------------------------------------------

export type SimServerMessage =
  | { type: "connected"; sim_id: string; scenario: string; red_policy: string; blue_policy: string; seed: number; max_cycles: number }
  | { type: "episode_start"; sim_id: string; seed: number; max_cycles: number; network: NetworkSnapshot }
  | { type: "event"; sim_id: string; step: number | null; agent: string; action: string; target: string | null; result: string | null; details: Record<string, unknown> }
  | {
      type: "state";
      sim_id: string;
      step: number;
      network: NetworkSnapshot | null;
      red_reward: number;
      blue_reward: number;
      last_red_action: string;
      last_blue_action: string;
      terminated: boolean;
      winner: string | null;
      max_cycles: number;
    }
  | { type: "episode_end"; sim_id: string; step: number; red_reward: number; blue_reward: number; winner: string | null; outcome: string; terminated: boolean }
  | { type: "paused"; step: number }
  | { type: "resumed"; step: number }
  | { type: "reset_done"; step: number; network: NetworkSnapshot; red_reward: number; blue_reward: number }
  | { type: "speed_set"; delay_ms: number }
  | { type: "error"; message: string };

export type SimClientMessage =
  | { type: "start"; seed?: number }
  | { type: "pause" }
  | { type: "resume" }
  | { type: "step"; count?: number }
  | { type: "reset" }
  | { type: "set_speed"; delay_ms: number };

export class SimSocket {
  private ws: WebSocket | null = null;
  private listeners = new Set<(msg: SimServerMessage) => void>();
  private statusListeners = new Set<(status: "connecting" | "open" | "closed" | "error") => void>();
  private outbox: SimClientMessage[] = []; // messages queued before the handshake completes
  private closedByUser = false;

  constructor(simId: string) {
    const url = wsUrl(`/ws/sim/${encodeURIComponent(simId)}`);
    this.setStatus("connecting");
    this.ws = new WebSocket(url);
    this.ws.onopen = () => {
      this.setStatus("open");
      // Flush anything sent while the socket was still CONNECTING.
      const pending = this.outbox.splice(0);
      for (const msg of pending) this.send(msg);
    };
    this.ws.onerror = () => this.setStatus("error");
    this.ws.onclose = () => {
      this.setStatus("closed");
      this.ws = null;
    };
    this.ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data as string) as SimServerMessage;
        for (const fn of this.listeners) fn(msg);
      } catch {
        for (const fn of this.listeners) fn({ type: "error", message: "unparseable server message" });
      }
    };
  }

  onMessage(fn: (msg: SimServerMessage) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  onStatus(fn: (status: "connecting" | "open" | "closed" | "error") => void): () => void {
    this.statusListeners.add(fn);
    return () => this.statusListeners.delete(fn);
  }

  private setStatus(s: "connecting" | "open" | "closed" | "error") {
    for (const fn of this.statusListeners) fn(s);
  }

  send(msg: SimClientMessage): boolean {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
      return true;
    }
    if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
      this.outbox.push(msg);
    }
    return false;
  }

  close(): void {
    this.closedByUser = true;
    this.ws?.close();
  }
}
