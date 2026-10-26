import { API_BASE } from "./api";

export type RangeEvent = {
  event_id: string;
  run_id: string;
  episode_id: string | null;
  seq: number;
  event_type: string;
  occurred_at: string;
  payload: Record<string, unknown>;
  previous_hash: string;
  event_hash: string;
};

export type TrainingRun = {
  training_run_id: string;
  role: "red" | "blue" | "joint";
  mode: "train" | "evaluate";
  status: "queued" | "running" | "completed" | "stopped" | "failed";
  config: Record<string, unknown>;
  stop_requested: boolean;
  result: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
};

export type SqliChallengeSession = {
  challenge_id: string;
  run_id: string;
  episode_id: string | null;
  training_run_id: string | null;
  status: "queued" | "preparing" | "running" | "completed" | "incomplete" | "failed";
  created_at: string | null;
  result: Record<string, unknown> | null;
};

export type SqliCheckpoint = {
  training_run_id: string;
  status: "completed";
  created_at: string;
  config: Record<string, unknown>;
  result: Record<string, unknown>;
};

export type TargetStatus = {
  schema_id: string;
  target_id: string;
  target_version: string;
  status: "available" | "unavailable";
  last_health_check_age_seconds: number | null;
  state_version: number;
  active_controls: Record<string, unknown>;
};

export type SqliTargetStatus = {
  scenario_id: string;
  target_version: string;
  status: "available" | "unavailable";
  last_health_check_age_seconds?: number | null;
  reason?: string;
  executor?: string;
  target?: Record<string, unknown>;
};

export type ManualAction =
  | { operation: "DISCOVER_ROUTE"; route_candidate_id: "home" | "login" | "document" }
  | { operation: "AUTH_TEST"; credential_ref: "synthetic_alice" | "synthetic_bob" | "synthetic_invalid" }
  | { operation: "CHECK_DOCUMENT_ACCESS"; document_ref: "doc_alice" | "doc_bob"; session_alias: string }
  | { operation: "CHECK_TARGET_HEALTH" };

export type RangeJobResult = {
  job: { job_id: string; action: ManualAction; request_id: string; route_catalog_id?: string };
  result: {
    status: string;
    method: string | null;
    route_catalog_id: string | null;
    http_status: number | null;
    response_headers: Record<string, string>;
    response_excerpt: string;
    request_id: string;
    target_event_id: string | null;
    session_alias: string | null;
    principal_ref: "alice" | "bob" | null;
    error_code: string | null;
  };
  finding?: Record<string, unknown>;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new Error(`VIOLET API is unreachable at ${API_BASE}`);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as { detail?: string };
    throw new Error(body.detail ?? `Range API returned HTTP ${response.status}`);
  }
  return await response.json() as T;
}

export const rangeApi = {
  targetStatus: () => request<TargetStatus>("/range/target/status"),
  trainingRuns: (limit = 50) => request<{ items: TrainingRun[] }>(`/range/training-runs?limit=${limit}`),
  demoCampaign: (restart = false) => request<TrainingRun>(`/range/sqli/demo-campaign${restart ? "?restart=true" : ""}`, { method: "POST" }),
  createTrainingRun: (data: {
    role: "red" | "blue" | "joint";
    mode: "train" | "evaluate";
    run_kind: "teaching" | "research";
    episodes: number;
    max_steps: number;
    seed: number;
    opponent: "scripted" | "random" | "none";
    scenario_id?: "violet_ledger_v1" | "sqli_lab_v1";
    checkpoint_run_id?: string;
  }) => request<TrainingRun>("/range/training-runs", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(data),
  }),
  trainingRun: (id: string) => request<TrainingRun>(`/range/training-runs/${encodeURIComponent(id)}`),
  stopTrainingRun: (id: string) => request<TrainingRun>(`/range/training-runs/${encodeURIComponent(id)}/stop`, { method: "POST" }),
  resetTarget: (data: { seed: number; run_id: string; episode_id: string }) => request<{ command_id: string; status: string }>("/range/reset", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(data),
  }),
  control: (id: string) => request<Record<string, unknown>>(`/range/controls/${encodeURIComponent(id)}`),
  submitManualAction: (data: {
    action: ManualAction;
    run_id: string;
    episode_id: string;
    step_id: number;
  }) => request<{ job_id: string; status: string; request_id: string; target_event_id?: string }>("/range/jobs", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ ...data, purpose: "manual" }),
  }),
  job: (id: string) => request<RangeJobResult | { job_id: string; status: string }>(`/range/jobs/${encodeURIComponent(id)}`),
  events: (id: string, afterSeq = 0) => request<{ run_id: string; after_seq: number; events: RangeEvent[] }>(
    `/range/runs/${encodeURIComponent(id)}/events?after_seq=${afterSeq}&limit=500`,
  ),
  episodes: (id: string) => request<{ run_id: string; items: { episode_id: string; episode_index: number; seed: number | null; status: string; config: Record<string, unknown> }[] }>(`/range/runs/${encodeURIComponent(id)}/episodes?limit=500`),
  episodeTimeline: (runId: string, episodeId: string) => request<{ run_id: string; episode_id: string; items: RangeEvent[] }>(`/range/runs/${encodeURIComponent(runId)}/episodes/${encodeURIComponent(episodeId)}/timeline?limit=500`),
  sqliSummary: (id: string, afterSeq = 0, limit = 1000) => request<{ run_id: string; after_seq: number; next_after_seq: number; items: RangeEvent[] }>(`/range/runs/${encodeURIComponent(id)}/sqli/summary?after_seq=${afterSeq}&limit=${limit}`),
  createSqliChallenge: (secretPhrase: string) => request<SqliChallengeSession>("/range/sqli/challenges", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ secret_phrase: secretPhrase }),
  }),
  sqliChallenge: (id: string) => request<SqliChallengeSession>(`/range/sqli/challenges/${encodeURIComponent(id)}`),
  sqliCheckpoint: () => request<SqliCheckpoint | null>("/range/sqli/checkpoint"),
  streamUrl: (id: string, afterSeq = 0) =>
    `${API_BASE}/range/runs/${encodeURIComponent(id)}/stream?after_seq=${afterSeq}`,
  sqliTargetStatus: () => request<SqliTargetStatus>("/range/sqli/target/status"),
};
