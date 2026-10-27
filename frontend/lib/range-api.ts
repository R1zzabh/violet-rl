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

export type TrainingUpdateRecord = {
  training_run_id: string;
  role: "red" | "blue";
  update_index: number;
  global_step: number;
  metrics: Record<string, unknown>;
  checkpoint_sha256: string | null;
  created_at: string;
};

export type PilotTrainingRunStatus = {
  seed: number;
  training_run_id: string;
  status: TrainingRun["status"];
  stop_requested: boolean;
  requested_episodes: number;
  completed_episodes: number;
  attempted_episodes: number;
  policy_versions: { red?: number; blue?: number } | null;
  checkpoint_sha256: string | null;
  checkpoint_available: boolean;
};

export type PilotSeedEvaluation = {
  training_seed: number;
  training_run_id: string;
  training_status: TrainingRun["status"];
  completed_training_episodes: number;
  evaluation_run_id: string | null;
  evaluation_status: TrainingRun["status"] | "not_started";
  evaluation_seeds: number[];
  requested_matches: number;
  completed_matches: number;
  incomplete_matches: number;
  red_wins: number;
  blue_wins: number;
  undecided_matches: number;
  red_win_rate: number | null;
  blue_win_rate: number | null;
  red_win_rate_wilson_95: [number, number] | null;
  blue_win_rate_wilson_95: [number, number] | null;
  verified_disclosures: number;
  verified_exposure_rate: number | null;
  verified_prevention_rate: number | null;
  legitimate_service_availability: number | null;
};

export type SqliPilotStatus = {
  campaign_id: string;
  status: "training" | "ready_for_evaluation" | "evaluation" | "completed" | "incomplete";
  started_at: string | null;
  training_deadline: string | null;
  evaluation_deadline: string | null;
  interaction_protocol_schema_id: string;
  reward_schema_id: string;
  training: {
    requested_per_seed: number;
    planned_seeds: number[];
    completed_updates: number;
    requested_updates: number;
    runs: PilotTrainingRunStatus[];
  };
  evaluation: {
    requested_matches: number;
    completed_matches: number;
    incomplete_matches: number;
    attempted_matches: number;
    red_wins: number;
    blue_wins: number;
    undecided_matches: number;
    red_win_rate: number | null;
    blue_win_rate: number | null;
    red_win_rate_wilson_95: [number, number] | null;
    blue_win_rate_wilson_95: [number, number] | null;
    verified_disclosures: number;
    verified_exposure_rate: number | null;
    verified_prevention_rate: number | null;
    legitimate_service_availability: number | null;
    per_seed: PilotSeedEvaluation[];
    balanced: boolean;
    balance_verdict: "balanced" | "incomplete" | "red_dominant" | "blue_dominant" | "unbalanced";
    balance_rule: string;
    uncertainty_method: string;
  };
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
  status: "running" | "completed" | "stopped";
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

export type ServerTraceStage = {
  stage_id: string;
  status: string;
  occurred_at: string;
  duration_ms: number | null;
  inputs: Record<string, unknown>;
  result: Record<string, unknown>;
};

export type ServerTrace = {
  schema_id: string;
  target_event_id: string;
  run_id: string | null;
  episode_id: string | null;
  decision_id: string | null;
  request_id: string | null;
  purpose: string;
  route_id: string;
  method: string;
  http_status: number;
  occurred_at: string;
  duration_ms: number;
  trace: ServerTraceStage[];
  evidence: { evidence_status: string; target_version: string };
};

export type LearningReport = {
  schema_id: string;
  evaluation_id: string;
  status: "running" | "completed" | "incomplete";
  source_training_run_id: string;
  matched_seed_schedule: number[];
  test_distribution: Record<string, unknown>;
  initial: Record<string, unknown> | null;
  later: Record<string, unknown> | null;
  paired_changes: Record<string, {
    matched_fixtures: number;
    mean_change: number | null;
    bootstrap_ci_95: [number, number] | null;
  }>;
  family_results: Record<string, unknown>;
  claim_basis: string;
};

export type ExposureExperiment = {
  schema_id: string;
  experiment_id: string;
  status: "training" | "ready_for_evaluation" | "evaluation" | "completed" | "incomplete";
  family: string;
  source_training_run_id: string;
  equal_training_budget: boolean;
  matched_training_seeds: number[];
  training: Record<string, Record<string, unknown> | null>;
  held_out_evaluation: {
    test_distribution_unchanged: boolean;
    same_seed_schedule: boolean;
    branches: Record<string, Record<string, unknown> | null>;
  };
  interpretation: string;
};

export type SqliEpisodePhase = "training" | "evaluation" | "challenge" | "other";
export type SqliEpisodeOutcomeKey = "red" | "blue" | "tie" | "no_winner" | "unknown";
export type SqliEpisodeIndexItem = {
  episode_id: string;
  run_id: string;
  episode_index: number;
  status: string;
  seed: number | null;
  created_at: string;
  timestamp: string;
  phase: SqliEpisodePhase;
  run_mode: string;
  run_type: string;
  run_status: string;
  outcome_key: SqliEpisodeOutcomeKey;
  outcome: {
    status: string | null;
    winner: string | null;
    reason: string | null;
    verified_exposure: boolean | null;
    challenge_disclosed: boolean | null;
    steps: number | null;
    red_return: number | null;
    blue_return: number | null;
  } | null;
  policy_versions_at_start: Record<string, unknown> | null;
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
  trainingRuns: (limit = 100, offset = 0) => request<{ items: TrainingRun[]; next_offset: number; has_more: boolean }>(`/range/training-runs?limit=${limit}&offset=${offset}`),
  demoCampaign: (restart = false) => request<TrainingRun>(`/range/sqli/demo-campaign${restart ? "?restart=true" : ""}`, { method: "POST" }),
  startPilot: () => request<SqliPilotStatus>("/range/sqli/pilots", {
    method: "POST", headers: { "content-type": "application/json" }, body: "{}",
  }),
  pilotStatus: (id: string) => request<SqliPilotStatus>(`/range/sqli/pilots/${encodeURIComponent(id)}`),
  startPilotEvaluation: (id: string) => request<SqliPilotStatus>(`/range/sqli/pilots/${encodeURIComponent(id)}/evaluation`, { method: "POST" }),
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
    campaign_label?: "live_demo_v1" | "retained_baseline_v1" | "classroom_evaluation_v2";
    baseline_training_run_id?: string;
    baseline_checkpoint_run_id?: string;
  }) => request<TrainingRun>("/range/training-runs", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(data),
  }),
  trainingRun: (id: string) => request<TrainingRun>(`/range/training-runs/${encodeURIComponent(id)}`),
  trainingUpdates: (id: string) => request<{ training_run_id: string; items: TrainingUpdateRecord[] }>(`/range/training-runs/${encodeURIComponent(id)}/updates`),
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
  episodes: (id: string, cursor: { afterIndex?: number; beforeIndex?: number } = {}, limit = 100) => {
    const query = new URLSearchParams({ limit: String(limit) });
    if (cursor.afterIndex !== undefined) query.set("after_index", String(cursor.afterIndex));
    if (cursor.beforeIndex !== undefined) query.set("before_index", String(cursor.beforeIndex));
    return request<{ run_id: string; after_index?: number | null; before_index?: number | null; next_after_index: number | null; next_before_index: number | null; has_more: boolean; has_older: boolean; has_newer: boolean; items: { episode_id: string; episode_index: number; seed: number | null; status: string; config: Record<string, unknown> }[] }>(`/range/runs/${encodeURIComponent(id)}/episodes?${query}`);
  },
  episodeTimeline: (runId: string, episodeId: string, afterSeq = 0, limit = 500) => request<{ run_id: string; episode_id: string; after_seq: number; next_after_seq: number; has_more: boolean; items: RangeEvent[] }>(`/range/runs/${encodeURIComponent(runId)}/episodes/${encodeURIComponent(episodeId)}/timeline?after_seq=${afterSeq}&limit=${limit}`),
  sqliEpisodes: (filters: {
    runId?: string;
    phase?: SqliEpisodePhase | "all";
    status?: "running" | "completed" | "incomplete" | "failed" | "all";
    outcome?: SqliEpisodeOutcomeKey | "all";
    cursor?: string | null;
    limit?: number;
  } = {}) => {
    const query = new URLSearchParams({ limit: String(filters.limit ?? 50) });
    if (filters.runId) query.set("run_id", filters.runId);
    if (filters.phase && filters.phase !== "all") query.set("phase", filters.phase);
    if (filters.status && filters.status !== "all") query.set("status", filters.status);
    if (filters.outcome && filters.outcome !== "all") query.set("outcome", filters.outcome);
    if (filters.cursor) query.set("cursor", filters.cursor);
    return request<{ cursor: string | null; next_cursor: string | null; has_more: boolean; items: SqliEpisodeIndexItem[] }>(`/range/sqli/episodes?${query}`);
  },
  sqliSummary: (id: string, afterSeq = 0, limit = 1000) => request<{ run_id: string; after_seq: number; next_after_seq: number; has_more: boolean; items: RangeEvent[] }>(`/range/runs/${encodeURIComponent(id)}/sqli/summary?after_seq=${afterSeq}&limit=${limit}`),
  createSqliChallenge: (secretPhrase: string) => request<SqliChallengeSession>("/range/sqli/challenges", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ secret_phrase: secretPhrase }),
  }),
  sqliChallenge: (id: string) => request<SqliChallengeSession>(`/range/sqli/challenges/${encodeURIComponent(id)}`),
  sqliCheckpoint: () => request<SqliCheckpoint | null>("/range/sqli/checkpoint"),
  streamUrl: (id: string, afterSeq = 0) =>
    `${API_BASE}/range/runs/${encodeURIComponent(id)}/stream?after_seq=${afterSeq}`,
  sqliTargetStatus: () => request<SqliTargetStatus>("/range/sqli/target/status"),
  serverTrace: (targetEventId: string) => request<ServerTrace>(`/range/server/traces/${encodeURIComponent(targetEventId)}`),
  startLearningReport: (data: { training_run_id: string; evaluation_id: string; matches: number }) => request<LearningReport>("/range/learning-reports", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(data),
  }),
  learningReport: (id: string) => request<LearningReport>(`/range/learning-reports/${encodeURIComponent(id)}`),
  startExposureExperiment: (data: {
    source_training_run_id: string;
    experiment_id: string;
    episodes_per_branch: number;
    seed: number;
    control_vulnerability_probability: number;
    increased_vulnerability_probability: number;
    evaluation_matches: number;
  }) => request<ExposureExperiment>("/range/experiments/exposure", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(data),
  }),
  exposureExperiment: (id: string) => request<ExposureExperiment>(`/range/experiments/${encodeURIComponent(id)}`),
  evaluateExposureExperiment: (id: string, evaluation_matches: number) => request<ExposureExperiment>(`/range/experiments/${encodeURIComponent(id)}/evaluation`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ evaluation_matches }),
  }),
};
