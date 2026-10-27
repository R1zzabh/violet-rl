import { expect, test, type Page, type Route } from "@playwright/test";

const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const currentRunId = "run-current-violet";
const legacyRunId = "run-legacy-violet";
const currentEpisodeId = "episode-current-0";
const nextEpisodeId = "episode-current-1";
const legacyEpisodeId = "episode-legacy-0";
const fixedTime = "2026-10-26T10:00:00+00:00";
const secretPhrase = "violet-e2e-secret-must-not-render";

function event(
  runId: string,
  episodeId: string,
  seq: number,
  eventType: string,
  payload: Record<string, unknown>,
) {
  return {
    event_id: `${episodeId}-${seq}`,
    run_id: runId,
    episode_id: episodeId,
    seq,
    event_type: eventType,
    occurred_at: fixedTime,
    payload,
    previous_hash: "0".repeat(64),
    event_hash: "1".repeat(64),
  };
}

const currentTimeline = [
  event(currentRunId, currentEpisodeId, 1, "sqli_episode_started", {
    interaction_protocol_schema_id: "sqli.round_protocol.v2",
    policy_versions: { red: 5, blue: 5 },
    seed: 17,
    max_steps: 40,
  }),
  event(currentRunId, currentEpisodeId, 2, "sqli_red_decision", {
    decision_id: "red-decision-0",
    step_id: 0,
    policy_id: "red-ppo-current",
    policy_version: 5,
    action: { field: "handle", base_term: "rhea", close_quote: true, connector: "OR", predicate: "tautology", comment: true },
  }),
  event(currentRunId, currentEpisodeId, 3, "sqli_blue_decision", {
    decision_id: "blue-decision-0",
    step_id: 0,
    policy_version: 5,
    selected_action: { operation: "PARAMETERIZE_QUERY" },
  }),
  event(currentRunId, currentEpisodeId, 4, "sqli_control_acknowledged", {
    decision_id: "blue-decision-0",
    step_id: 0,
    action: "PARAMETERIZE_QUERY",
    acknowledgement: { status: "applied", controls: { parameterized: true } },
  }),
  event(currentRunId, currentEpisodeId, 5, "sqli_target_response", {
    purpose: "policy",
    decision_id: "red-decision-0",
    step_id: 0,
    request_id: "request-current",
    target_event_id: "target-current",
    method: "POST",
    route: "/api/lookup",
    http_status: 200,
    status: "completed",
    evidence_status: "verified",
    request: { field: "handle", query_value: "rhea' OR 1=1 -- " },
    response: { rows: [{ record_ref: "rec-002", owner_ref: "other" }, { record_ref: "rec-003", owner_ref: "other" }] },
    telemetry: { evidence_status: "verified", unauthorized_rows: 2, row_count: 2 },
  }),
  event(currentRunId, currentEpisodeId, 6, "sqli_target_response", {
    purpose: "verifier_attack",
    step_id: 0,
    request_id: "verifier-attack",
    target_event_id: "target-verifier-attack",
    http_status: 200,
    response: { rows: [] },
    telemetry: { evidence_status: "verified", unauthorized_rows: 0 },
  }),
  event(currentRunId, currentEpisodeId, 7, "sqli_target_response", {
    purpose: "verifier_legitimate",
    step_id: 0,
    request_id: "verifier-legitimate",
    target_event_id: "target-verifier-legitimate",
    http_status: 200,
    response: { rows: [{ record_ref: "rec-001", owner_ref: "researcher" }] },
    telemetry: { evidence_status: "verified", unauthorized_rows: 0 },
  }),
  event(currentRunId, currentEpisodeId, 8, "sqli_verification", {
    step_id: 0,
    evidence_verified: true,
    verified_prevention: true,
    unauthorized_rows_returned: 0,
    legitimate_query_served: true,
  }),
  event(currentRunId, currentEpisodeId, 9, "sqli_ppo_update", {
    role: "blue",
    updated: true,
    update_status: "applied",
    policy_version_before: 5,
    policy_version: 6,
    optimization_passes: 4,
    checkpoint_sha256: "a".repeat(64),
    metrics: { sample_count: 1, accepted_transitions: 1 },
  }),
  event(currentRunId, currentEpisodeId, 10, "sqli_policy_probe", {
    role: "blue",
    same_reference_observation: true,
    policy_version_before: 5,
    policy_version_after: 6,
    reference_observation: [0, 0, 1],
    before_action_probabilities: [[0.25, 0.25, 0.25, 0.25]],
    after_action_probabilities: [[0.4, 0.2, 0.2, 0.2]],
  }),
  event(currentRunId, currentEpisodeId, 11, "sqli_episode_outcome", {
    phase: "training",
    status: "completed",
    winner: "red",
    verified_exposure: true,
    exposure_verification: { evidence_verified: true, verified_exposure: true },
    episode_index: 1,
    seed: 17,
  }),
];

const nextTimeline = [
  event(currentRunId, nextEpisodeId, 1, "sqli_episode_started", {
    interaction_protocol_schema_id: "sqli.round_protocol.v2",
    policy_versions: { red: 6, blue: 6 },
  }),
  event(currentRunId, nextEpisodeId, 2, "sqli_blue_decision", {
    policy_version: 6,
    selected_action: { operation: "MONITOR" },
  }),
];

const legacyTimeline = [
  event(legacyRunId, legacyEpisodeId, 1, "sqli_episode_started", { seed: 9, max_steps: 40 }),
  event(legacyRunId, legacyEpisodeId, 2, "sqli_red_decision", {
    decision_id: "legacy-red-decision",
    step_id: 0,
    policy_version: 2,
    action: { field: "handle", base_term: "rhea", close_quote: true, connector: "OR", predicate: "tautology", comment: true },
  }),
  event(legacyRunId, legacyEpisodeId, 3, "sqli_target_response", {
    purpose: "policy",
    decision_id: "legacy-red-decision",
    step_id: 0,
    request_id: "legacy-request",
    target_event_id: "legacy-target-event",
    http_status: 200,
    request: { field: "handle", query_value: "rhea' OR 1=1 -- " },
    response: { rows: [{ record_ref: "rec-002" }] },
    telemetry: { evidence_status: "verified", unauthorized_rows: 1 },
  }),
  event(legacyRunId, legacyEpisodeId, 4, "sqli_blue_decision", {
    decision_id: "legacy-blue-decision",
    step_id: 0,
    policy_version: 2,
    selected_action: { operation: "ISOLATE_GUEST_LOOKUP" },
  }),
  event(legacyRunId, legacyEpisodeId, 5, "sqli_control_acknowledged", {
    decision_id: "legacy-blue-decision",
    step_id: 0,
    action: "ISOLATE_GUEST_LOOKUP",
    acknowledgement: { status: "applied" },
  }),
  event(legacyRunId, legacyEpisodeId, 6, "sqli_verification", {
    step_id: 0,
    evidence_verified: true,
    verified_prevention: true,
    unauthorized_rows_returned: 0,
    legitimate_query_served: true,
  }),
];

const summaryEvents = {
  [currentRunId]: [currentTimeline.at(-1)!],
  [legacyRunId]: [event(legacyRunId, legacyEpisodeId, 1, "sqli_episode_outcome", {
    phase: "training",
    status: "completed",
    winner: "red",
    verified_exposure: true,
    exposure_verification: { evidence_verified: true, verified_exposure: true },
    episode_index: 1,
    seed: 9,
  })],
};

const runRecords = [
  {
    training_run_id: currentRunId,
    role: "joint",
    mode: "train",
    status: "completed",
    config: { scenario_id: "sqli_lab_v1", interaction_protocol_schema_id: "sqli.round_protocol.v2", campaign_revision: "balanced_adversarial_ppo4_v3" },
    stop_requested: false,
    result: { policy_versions: { red: 6, blue: 6 } },
    created_at: fixedTime,
    updated_at: fixedTime,
  },
  {
    training_run_id: legacyRunId,
    role: "joint",
    mode: "train",
    status: "completed",
    config: { scenario_id: "sqli_lab_v1", campaign_revision: "red_syntax_progress_ppo4_v2" },
    stop_requested: false,
    result: {},
    created_at: "2025-10-26T10:00:00+00:00",
    updated_at: "2025-10-26T10:00:00+00:00",
  },
];

async function fulfillJson(route: Route, body: unknown) {
  await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
}

async function mockArchiveApi(page: Page) {
  const targetRequests: string[] = [];
  await page.route(`${apiBase}/**`, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() !== "GET") {
      targetRequests.push(`${request.method()} ${url.pathname}`);
      await route.fulfill({ status: 403, body: "archive views must remain read only" });
      return;
    }
    if (url.pathname === "/range/training-runs") {
      await fulfillJson(route, { items: runRecords, next_offset: 2, has_more: false });
      return;
    }
    if (url.pathname === "/range/sqli/target/status") {
      await fulfillJson(route, { scenario_id: "sqli_lab_v1", target_version: "sqli_target_v1", status: "available", last_health_check_age_seconds: 0.2 });
      return;
    }
    if (url.pathname.endsWith("/sqli/summary")) {
      const runId = url.pathname.split("/")[3];
      const items = summaryEvents[runId as keyof typeof summaryEvents] ?? [];
      await fulfillJson(route, { run_id: runId, after_seq: 0, next_after_seq: items.at(-1)?.seq ?? 0, has_more: false, items });
      return;
    }
    if (url.pathname.endsWith("/timeline")) {
      const parts = url.pathname.split("/");
      const runId = parts[3];
      const episodeId = parts[5];
      const items = episodeId === currentEpisodeId ? currentTimeline
        : episodeId === nextEpisodeId ? nextTimeline
          : legacyTimeline;
      await fulfillJson(route, { run_id: runId, episode_id: episodeId, after_seq: 0, next_after_seq: items.at(-1)?.seq ?? 0, has_more: false, items });
      return;
    }
    if (url.pathname === "/range/sqli/episodes") {
      const item = {
        episode_id: currentEpisodeId,
        run_id: currentRunId,
        episode_index: 0,
        status: "completed",
        seed: 17,
        created_at: fixedTime,
        timestamp: fixedTime,
        phase: "training",
        run_mode: "train",
        run_type: "teaching",
        run_status: "completed",
        outcome_key: "red",
        outcome: { status: "completed", winner: "red", reason: "verified_unauthorized_read", verified_exposure: true, challenge_disclosed: false, steps: 1, red_return: 1, blue_return: 1 },
        policy_versions_at_start: { red: 5, blue: 5 },
      };
      await fulfillJson(route, { cursor: null, next_cursor: null, has_more: false, items: [item] });
      return;
    }
    if (url.pathname.endsWith("/episodes")) {
      const runId = url.pathname.split("/")[3];
      const items = runId === legacyRunId
        ? [{ episode_id: legacyEpisodeId, episode_index: 0, seed: 9, status: "completed", config: {} }]
        : [
          { episode_id: currentEpisodeId, episode_index: 0, seed: 17, status: "completed", config: {} },
          { episode_id: nextEpisodeId, episode_index: 1, seed: 18, status: "completed", config: {} },
        ];
      const afterIndex = url.searchParams.get("after_index");
      const selected = afterIndex === null ? items : items.filter((item) => item.episode_index > Number(afterIndex));
      await fulfillJson(route, {
        run_id: runId,
        after_index: afterIndex === null ? null : Number(afterIndex),
        before_index: null,
        next_after_index: selected.at(-1)?.episode_index ?? null,
        next_before_index: selected[0]?.episode_index ?? null,
        has_more: false,
        has_older: false,
        has_newer: false,
        items: selected,
      });
      return;
    }
    if (url.pathname === `/range/training-runs/${currentRunId}` || url.pathname === `/range/training-runs/${legacyRunId}`) {
      await fulfillJson(route, runRecords.find((item) => item.training_run_id === url.pathname.split("/").at(-1)));
      return;
    }
    await route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: `Unhandled mock route: ${url.pathname}` }) });
  });
  return targetRequests;
}

test("Breaches tells the verified incident in order and labels Red-first history", async ({ page }) => {
  const targetRequests = await mockArchiveApi(page);
  await page.goto("/breaches");
  await expect(page.getByRole("heading", { name: "Breaches" })).toBeVisible();
  await page.getByRole("button", { name: /Select recipe/ }).first().click();
  await expect(page.getByRole("heading", { name: "Blue control before Red request" })).toBeVisible();
  await expect(page.getByText("2 unauthorized rows", { exact: false }).first()).toBeVisible();
  await expect(page.getByText("event #4", { exact: false })).toBeVisible();
  await expect(page.getByText("event #5", { exact: false })).toBeVisible();
  await expect(page.getByText(/before Red's request/)).toBeVisible();
  await expect(page.getByText(/Episode #2 · Blue v6/)).toBeVisible();
  await expect(page.getByText(/first recorded action was MONITOR/)).toBeVisible();
  await page.getByRole("button", { name: "Refresh archive" }).click();
  await expect(page.getByRole("heading", { name: "Blue control before Red request" })).toBeVisible();

  await expect(page.getByLabel("Training run").locator("option", { hasText: "Legacy Red-first" })).toHaveCount(1);
  await page.getByRole("button", { name: /Occurrence 2 · episode/ }).click();
  await expect(page.getByRole("heading", { name: "Legacy Red-first episode" })).toBeVisible();
  await expect(page.getByText(/Red's request reached the target before Blue acted/)).toBeVisible();
  expect(targetRequests).toEqual([]);
});

test("Episodes replay supports keyboard, refresh, phrase redaction, and sends no target traffic", async ({ page }) => {
  const targetRequests = await mockArchiveApi(page);
  const playbackEvents = [
    event(currentRunId, currentEpisodeId, 1, "sqli_episode_started", { secret_phrase: secretPhrase, policy_versions: { red: 5, blue: 5 }, seed: 17, max_steps: 40 }),
    event(currentRunId, currentEpisodeId, 2, "sqli_red_decision", { step_id: 0, policy_version: 5, secret_phrase: secretPhrase, action: { field: "handle" } }),
    event(currentRunId, currentEpisodeId, 3, "sqli_ppo_update", { role: "blue", updated: true, update_status: "applied", policy_version_before: 5, policy_version: 6, optimization_passes: 4, metrics: { sample_count: 1 } }),
    event(currentRunId, currentEpisodeId, 4, "sqli_episode_outcome", { phase: "training", status: "completed", winner: "red", episode_index: 1, seed: 17 }),
  ];
  await page.route(`${apiBase}/range/runs/${currentRunId}/episodes/${currentEpisodeId}/timeline*`, async (route) => {
    await fulfillJson(route, { run_id: currentRunId, episode_id: currentEpisodeId, after_seq: 0, next_after_seq: 4, has_more: false, items: playbackEvents });
  });

  await page.goto("/episodes");
  await expect(page.getByRole("heading", { name: "Episodes" })).toBeVisible();
  await expect(page.getByText("1 / 4", { exact: true })).toBeVisible();
  const replay = page.getByLabel(/Episode event replay/);
  await replay.focus();
  await replay.press("ArrowRight");
  await expect(page.getByText("2 / 4", { exact: true })).toBeVisible();
  await page.getByText("Exact persisted event payload · challenge phrases redacted").click();
  await expect(page.getByText("[redacted]")).toBeVisible();
  await expect(page.getByText(secretPhrase, { exact: false })).toHaveCount(0);
  await page.reload();
  await expect(page.getByText("1 / 4", { exact: true })).toBeVisible();
  expect(targetRequests).toEqual([]);
});

test("Breaches recipe index remains usable on a narrow viewport", async ({ page }) => {
  const targetRequests = await mockArchiveApi(page);
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto("/breaches");
  await expect(page.getByRole("heading", { name: "Breaches" })).toBeVisible();
  await expect(page.getByRole("list", { name: "Attack recipe index" })).toBeVisible();
  const overflows = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  expect(overflows).toBe(false);
  await page.getByRole("button", { name: /Select recipe/ }).first().click();
  await expect(page.getByRole("heading", { name: "Blue control before Red request" })).toBeVisible();
  expect(targetRequests).toEqual([]);
});

test("the pilot starts only after an explicit user action", async ({ page }) => {
  await page.addInitScript(() => window.localStorage.clear());
  const pilotPosts: string[] = [];
  const trainingRuns = [17, 18, 19].map((seed) => ({
    seed,
    training_run_id: `training-${seed}`,
    status: "queued",
    stop_requested: false,
    requested_episodes: 50,
    completed_episodes: 0,
    attempted_episodes: 0,
    policy_versions: null,
    checkpoint_sha256: null,
    checkpoint_available: false,
  }));
  const pilotStatus = {
    campaign_id: "pilot-violet-e2e",
    status: "training",
    started_at: fixedTime,
    training_deadline: "2026-10-26T13:00:00+00:00",
    evaluation_deadline: "2026-10-26T14:00:00+00:00",
    interaction_protocol_schema_id: "sqli.round_protocol.v2",
    reward_schema_id: "sqli.reward.v3",
    training: {
      requested_per_seed: 50,
      planned_seeds: [17, 18, 19],
      completed_updates: 0,
      requested_updates: 150,
      runs: trainingRuns,
    },
    evaluation: {
      requested_matches: 45,
      completed_matches: 0,
      incomplete_matches: 45,
      attempted_matches: 0,
      red_wins: 0,
      blue_wins: 0,
      undecided_matches: 0,
      red_win_rate: null,
      blue_win_rate: null,
      red_win_rate_wilson_95: null,
      blue_win_rate_wilson_95: null,
      verified_disclosures: 0,
      verified_exposure_rate: null,
      verified_prevention_rate: null,
      legitimate_service_availability: null,
      per_seed: [],
      balanced: false,
      balance_verdict: "incomplete",
      balance_rule: "Both sides must win at least 20% of completed matches.",
      uncertainty_method: "Wilson score 95% interval.",
    },
  };
  await page.route(`${apiBase}/**`, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/range/sqli/target/status") {
      await fulfillJson(route, { scenario_id: "sqli_lab_v1", target_version: "sqli_target_v1", status: "available" });
      return;
    }
    if (url.pathname === "/range/sqli/pilots" && request.method() === "POST") {
      pilotPosts.push(request.method());
      await fulfillJson(route, pilotStatus);
      return;
    }
    if (url.pathname === "/range/sqli/pilots/pilot-violet-e2e") {
      await fulfillJson(route, pilotStatus);
      return;
    }
    await route.fulfill({ status: 404, contentType: "application/json", body: JSON.stringify({ detail: `Unhandled mock route: ${url.pathname}` }) });
  });

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Balanced adversarial pilot" })).toBeVisible();
  await expect(page.getByText("Explicit start · no work runs on page load")).toBeVisible();
  expect(pilotPosts).toEqual([]);

  await page.getByRole("button", { name: "Start three-seed pilot" }).click();
  await expect(page.getByText("Training in progress")).toBeVisible();
  expect(pilotPosts).toEqual(["POST"]);
});
