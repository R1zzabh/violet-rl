import type { RangeEvent, SqliEpisodeIndexItem } from "./range-api";

export type Role = "red" | "blue";
export const object = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
export const number = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;
export const records = (value: unknown): Record<string, unknown>[] => Array.isArray(value) ? value.map(object) : [];
export const fixed = (value: unknown, digits = 4): string => number(value)?.toFixed(digits) ?? "unavailable";
export const percent = (value: unknown): string => number(value) === null ? "unavailable" : `${(number(value)! * 100).toFixed(2)}%`;

export function redact(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(redact);
  if (value !== null && typeof value === "object") return Object.fromEntries(Object.entries(object(value)).map(([key, item]) => [
    key, /secret|phrase|password|token|authorization|cookie/i.test(key) ? "[redacted]" : redact(item),
  ]));
  return value;
}

const CATALOG = {
  red: [
    ["field", ["handle", "department", "city", "email"]],
    ["base term", ["rhea", "alice", "bob", "engineering", "finance"]],
    ["close quote", ["no", "yes"]],
    ["connector", ["none", "OR", "AND"]],
    ["predicate", ["none", "tautology", "contradiction"]],
    ["comment", ["no", "yes"]],
  ],
  blue: [["control", ["MONITOR", "PARAMETERIZE_QUERY", "RATE_LIMIT_GUEST", "ISOLATE_GUEST_LOOKUP"]]],
} as const;

export type ProbabilityChange = { component: string; option: string; before: number; after: number; delta: number; observations: number };
export type RoleReport = {
  role: Role;
  decisions: RangeEvent[];
  transitions: RangeEvent[];
  update: RangeEvent | null;
  metrics: Record<string, unknown>;
  settings: Record<string, unknown>;
  diagnostics: Record<string, unknown>[];
  applied: boolean;
  beforeVersion: number | null;
  afterVersion: number | null;
  reward: number | null;
  rewardComponents: Record<string, number>;
  actionCounts: { label: string; count: number }[];
  changes: ProbabilityChange[];
  diagnosticCoverage: number | null;
  narrative: string;
};

export function actionLabel(value: unknown): string {
  const action = object(redact(value));
  if (typeof action.operation === "string") return action.operation.replaceAll("_", " ");
  return Object.entries(action).filter(([key]) => key !== "schema_id").map(([key, item]) => `${key.replaceAll("_", " ")}: ${String(item)}`).join(" · ") || "Action not recorded";
}

export function isPolicyDecision(event: RangeEvent): boolean {
  return ["sqli_red_decision", "sqli_blue_decision"].includes(event.event_type)
    && event.payload.phase !== "control_submitted"
    && typeof event.payload.decision_id === "string"
    && number(event.payload.policy_version) !== null;
}

function probabilityChanges(diagnostics: Record<string, unknown>[], role: Role, knownSchema: boolean): ProbabilityChange[] {
  const groups = new Map<string, ProbabilityChange>();
  for (const item of diagnostics) {
    const before = Array.isArray(item.old_component_probabilities) ? item.old_component_probabilities : [];
    const after = Array.isArray(item.new_component_probabilities) ? item.new_component_probabilities : [];
    before.forEach((raw, head) => {
      if (!Array.isArray(raw) || !Array.isArray(after[head])) return;
      raw.forEach((value, option) => {
        const old = number(value);
        const next = number(after[head][option]);
        if (old === null || next === null) return;
        const catalog = knownSchema ? CATALOG[role][head] : undefined;
        const key = `${head}/${option}`;
        const group = groups.get(key) ?? {
          component: catalog?.[0] ?? `component ${head + 1}`,
          option: catalog?.[1][option] ?? `option ${option + 1}`,
          before: 0, after: 0, delta: 0, observations: 0,
        };
        group.before += old;
        group.after += next;
        group.observations += 1;
        groups.set(key, group);
      });
    });
  }
  return [...groups.values()].map((group) => ({ ...group,
    before: group.before / group.observations, after: group.after / group.observations,
    delta: (group.after - group.before) / group.observations,
  })).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
}

export function buildRoleReport(events: RangeEvent[], role: Role): RoleReport {
  const own = (kind: string) => events.filter((event) => event.event_type === kind && event.payload.role === role);
  const decisions = events.filter((event) => event.event_type === `sqli_${role}_decision` && isPolicyDecision(event));
  const transitions = own("sqli_transition");
  const update = own("sqli_ppo_update").at(-1) ?? null;
  const metrics = object(update?.payload.metrics);
  const settings = object(update?.payload.ppo_settings ?? metrics.ppo_settings);
  const applied = update?.payload.updated === true && update.payload.update_status !== "skipped_incomplete_joint_episode";
  // Only diagnostics from a recorded applied update describe optimizer inputs.
  const diagnostics = applied ? records(update?.payload.transition_diagnostics) : [];
  const outcome = events.findLast((event) => event.event_type === "sqli_episode_outcome");
  const beforeVersion = number(update?.payload.policy_version_before) ?? number(decisions[0]?.payload.policy_version);
  const afterVersion = number(update?.payload.policy_version) ?? (applied ? null : beforeVersion);
  const rewardValues = transitions.map((event) => number(event.payload.reward)).filter((value): value is number => value !== null);
  const reward = number(outcome?.payload[`${role}_return`]) ?? (rewardValues.length ? rewardValues.reduce((sum, item) => sum + item, 0) : null);
  const rewardComponents: Record<string, number> = {};
  for (const transition of transitions) for (const [key, value] of Object.entries(object(transition.payload.reward_components))) {
    const n = number(value);
    if (n !== null) rewardComponents[key] = (rewardComponents[key] ?? 0) + n;
  }
  const counts = new Map<string, number>();
  for (const decision of decisions) {
    const label = actionLabel(decision.payload.selected_action ?? decision.payload.action);
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  const changes = probabilityChanges(diagnostics, role, outcome?.payload.action_schema_id === "sqli.composed_action.v1"
    || transitions.some((event) => object(event.payload.action).schema_id === "sqli.composed_action.v1"));
  const sampleCount = number(metrics.sample_count);
  const diagnosticCoverage = sampleCount !== null && sampleCount > 0 ? diagnostics.length / sampleCount : null;
  const strongest = changes[0];
  const narrative = !update
    ? "A PPO update has not been recorded for this episode yet. Decisions and rewards below show the experience collected so far."
    : !applied
      ? "This episode did not produce an eligible joint update. Its experience is retained for inspection; no learning is attributed to this episode."
      : strongest
        ? `On ${strongest.observations} saved observations, ${strongest.component} “${strongest.option}” changed from ${percent(strongest.before)} to ${percent(strongest.after)} on average (${strongest.delta >= 0 ? "+" : ""}${(strongest.delta * 100).toFixed(2)} percentage points). This is a measured change on this batch, with the saved observations and recurrent states held fixed.`
        : "The optimizer update is recorded, but saved before/after action distributions are unavailable. A strategy change cannot be described from losses alone.";
  return { role, decisions, transitions, update, metrics, settings, diagnostics, applied, beforeVersion, afterVersion,
    reward, rewardComponents, actionCounts: [...counts].map(([label, count]) => ({label, count})).sort((a, b) => b.count - a.count),
    changes, diagnosticCoverage, narrative };
}

export function checkpointEvidence(events: RangeEvent[]) {
  const updates = [buildRoleReport(events, "red"), buildRoleReport(events, "blue")];
  const hashes = updates.map((role) => role.applied ? role.update?.payload.checkpoint_sha256 : null);
  const outcome = events.findLast((event) => event.event_type === "sqli_episode_outcome");
  const hash = hashes[0];
  const verified = typeof hash === "string" && /^[a-f0-9]{64}$/i.test(hash)
    && hashes[1] === hash && outcome?.payload.checkpoint_sha256 === hash;
  const recordedHashes = [...hashes, outcome?.payload.checkpoint_sha256].filter((value): value is string => typeof value === "string");
  return { verified, hash: typeof hash === "string" ? hash : null,
    mismatch: new Set(recordedHashes).size > 1 };
}

export function nextPolicyEvidence(events: RangeEvent[], nextEvents: RangeEvent[]) {
  const start = nextEvents.find((event) => event.event_type === "sqli_episode_started");
  const versions = object(start?.payload.policy_versions);
  const roles = ["red", "blue"] as const;
  const comparisons = roles.map((role) => {
    const report = buildRoleReport(events, role);
    const actual = number(versions[role]);
    return { role, expected: report.applied ? report.afterVersion : null, actual,
      matched: report.applied && report.afterVersion !== null && actual === report.afterVersion
        && start !== undefined && report.update !== null && start.seq > report.update.seq };
  });
  return { observed: Boolean(start), matched: comparisons.every((role) => role.matched), comparisons, start };
}

export function markdownReport(episode: SqliEpisodeIndexItem, events: RangeEvent[], nextEvents: RangeEvent[]): string {
  const lines = ["# Violet episode learning report", "", `Run: ${episode.run_id}`, `Episode: ${episode.episode_index + 1} (${episode.episode_id})`,
    `Recorded status: ${episode.status}`, "", "This report describes recorded training signals and measured policy changes. Performance improvement requires held-out evaluation.", ""];
  for (const role of ["red", "blue"] as const) {
    const report = buildRoleReport(events, role);
    lines.push(`## ${role.toUpperCase()} policy`, "", `Policy version: ${report.beforeVersion ?? "unavailable"} → ${report.afterVersion ?? "unavailable"}`,
      `Update: ${report.applied ? "applied" : "not applied / not recorded"}; reward: ${report.reward ?? "unavailable"}`, "", report.narrative, "",
      "### Reward components", "", ...Object.entries(report.rewardComponents).map(([key, value]) => `- ${key}: ${value}`), "",
      "### Optimizer diagnostics", "", "```json", JSON.stringify(redact(report.metrics), null, 2), "```", "",
      "### Mean action probabilities on saved observations", "", "| Component | Option | Before | After | Change (pp) | Observations |", "|---|---|---:|---:|---:|---:|");
    for (const change of report.changes) lines.push(`| ${change.component} | ${change.option} | ${percent(change.before)} | ${percent(change.after)} | ${(change.delta * 100).toFixed(4)} | ${change.observations} |`);
    lines.push("", "### Exact optimizer inputs by decision", "", "```json", JSON.stringify(redact(report.diagnostics), null, 2), "```", "");
  }
  lines.push("## Checkpoint and next episode", "", "```json", JSON.stringify(redact({checkpoint: checkpointEvidence(events), nextEpisode: nextPolicyEvidence(events, nextEvents)}), null, 2), "```", "",
    "## Recorded decision and transition evidence", "", "```json", JSON.stringify(redact(events.filter((event) => /decision|transition|verification|outcome/.test(event.event_type))), null, 2), "```");
  return lines.join("\n");
}
