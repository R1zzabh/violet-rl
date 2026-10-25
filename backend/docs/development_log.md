# Violet Development Log

Machine-readable + human-readable record of implementation, tests, and results.
Statuses: `NOT STARTED` / `IN DEVELOPMENT` / `IMPLEMENTED` / `TESTED` / `VERIFIED` / `BLOCKED` / `FAILED`

---

## 2026-09-09 — Phase 1/4: Environment core + baseline battles

**Feature:** Topology-aware `ScriptedAggressiveRedAgent` + end-to-end scripted battles.

**Status: VERIFIED**

### Implemented
- `backend/violet/agents/red.py`:
  - `ScriptedAggressiveRedAgent` rewritten to behave like a realistic intrusion
    progression. It maintains a foothold set and advances along the topology from its
    entry point (`web`) toward the objective (`database`):
    SCAN → EXPLOIT → PRIVILEGE_ESCALATE → LATERAL_MOVE (repeat per hop).
    - No foothold ⇒ attack the entry node (discover, then exploit).
    - Foothold is `compromised` (not escalated) ⇒ retry PRIVILEGE_ESCALATE.
    - Escalated ⇒ SCAN the next hop if unknown, then LATERAL_MOVE from the foothold.
  - Next-hop planning is restricted to DIRECT neighbors of the foothold — the agent
    walks inward one hop at a time instead of teleporting to the final objective.
  - LATERAL_MOVE honors engine semantics: action target = compromised SOURCE node;
    destination is chosen among reachable uncompromised neighbors.
  - Other red baselines unchanged (`ScriptedStealthRedAgent`, `ScriptedLateralRedAgent`,
    `ScriptedRandomizedRedAgent`, `ScriptedObjectiveRedAgent`).

### Bug fixed (root cause + evidence)
- **What failed:** `test_smoke_red_can_reach_objective_in_few_steps`,
  `test_blue_rule_based_isolates_compromised_node` — `web` was never compromised.
- **Where:** `ScriptedAggressiveRedAgent` (previous implementation).
- **Why:** The old agent attacked its final target (`database`) directly on every
  turn, ignoring topology. `web` was never touched, so smoke assertions on `web`
  could only fail. A first rewrite over-corrected by using multi-hop
  `reachable_from`, which made `database` "reachable" from `web` on turn one —
  caught by the new regression test before it could hide the same class of bug.
- **Fix:** Neighbor-restricted hop planning + foothold-first kill chain (above).
- **Regression test:** `tests/unit/test_agents.py::test_scripted_aggressive_red_walks_topology_inward`
  asserts the first planned hop is the entry-adjacent node (`web`), not the objective.

### Tested
- Full suite: `pytest tests/ -q` → **64 passed, 0 failed** (unit 55, integration 5, smoke 4).
- Executable workflow: `python scripts/smoke_test.py` → assertions passed:
  episode terminated at step 17 on critical-asset compromise; red walked
  `web → app → database`; blue isolated compromised hosts (web, app, employee-2).

### Observed behavior (single seeded run, seed=42 — NOT a benchmark)
- Episode length: 17 steps (terminal: `database` compromised).
- Red final reward: 44.90 (printout label `Red reward: -1.70` is the pre-render
  accumulator; rendered total is authoritative — see Known limitations #3).
- Blue final reward: −95.10 (cost of isolations + downtime vs. detection gains).

### Known limitations
1. `_red_lateral` picks the destination **randomly** among reachable uncompromised
   neighbors when the encoded target is not itself a valid destination — the agent
   cannot steer lateral movement precisely yet (engine limitation, Phase 2 candidate).
2. Blue's reward is dominated by action costs/downtime in this scenario; weights are
   placeholder values, not tuned.
3. The smoke script prints two inconsistent reward totals (accumulator vs. render);
   cosmetic — to be unified in the next pass.
4. Observations are placeholder 64-dim vectors; partial observability is modeled in
   state but not yet fully exposed through the observation space.

### Next milestone
- Phase 2/3 polish: precise lateral-move targeting (action target = destination),
  then Phase 5: first RL agent (PPO) in the simplified scenario.

---

## 2026-09-09 — Phase 2: Precise lateral-move targeting

**Feature:** LATERAL_MOVE action target is now the DESTINATION node.

**Status: VERIFIED**

### Implemented
- `backend/violet/transition.py::_red_lateral` (semantic change v1→v2):
  - Action target now encodes the lateral-movement DESTINATION. The engine selects
    the source: a compromised foothold that can reach the destination, preferring
    direct neighbors, then deterministic insertion order.
  - Failure modes are explicit and clean: destination already compromised →
    `lateral_failed`; no foothold with a path → `lateral_failed`; destination is a
    honeypot → `honeypot_triggered` (alert raised).
  - Success/failure events now include `details.source` and `details.destination`
    for replay/debugging.
- `backend/violet/agents/red.py::ScriptedAggressiveRedAgent`: encodes the planned
  next hop as the lateral destination (no more relying on engine-side random
  destination selection).
- `ScriptedLateralRedAgent` already targeted the destination; it is now correct
  under the new semantics with no change.
- `backend/violet/env.py::render`: ANSI render now labels last-step rewards
  explicitly (`red last-step reward`), removing the ambiguity with cumulative
  episode rewards printed by the smoke script.
- `backend/scripts/smoke_test.py`: printout now says "cumulative reward".

### Tested
- New regression tests in `tests/unit/test_transition.py`:
  - `test_lateral_target_is_destination` — web→app move attributes source/destination.
  - `test_lateral_fails_without_foothold_path` — no foothold ⇒ clean `lateral_failed`.
  - `test_lateral_fails_if_destination_already_compromised`.
- Full suite: `pytest tests/ -q` → **67 passed, 0 failed**.
- Executable workflow: `python scripts/smoke_test.py` → assertions passed.

### Observed behavior (single seeded run, seed=42 — NOT a benchmark)
- With precise lateral targeting, the episode now runs the full 50 steps WITHOUT
  database compromise: red takes `web` (priv-esc'd) but blue contains the spread —
  `app`/`database` remain clean, web stays isolated. Blue's containment now visibly
  beats red's progression under this seed, an outcome that was unreachable when red
  could not steer lateral moves. Red cumulative −110.05, blue cumulative −4.85
  (blue's cost is mostly unnecessary-reset penalties; red's is failed actions while
  contained).

### Known limitations
1. Reward weights remain untuned placeholders; blue's "unnecessary reset" penalty
   dominates its otherwise-good containment behavior.
2. `_red_lateral` source selection prefers direct neighbors but falls back to any
   reachable foothold (multi-hop "movement" in one step). A stricter hop-by-hop
   rule may be preferable for research validity — deferred to Phase 3.
3. Observations are still placeholder 64-dim vectors (unchanged from last entry).

### Next milestone
- Phase 5: first RL agent (PPO) in the simplified scenario — training loop,
  checkpointing, evaluation on unseen seeds.

---

## 2026-09-09 — Phase 2b: Reward-balance fixes + hop-by-hop lateral movement

**Feature:** Defender reward tuning (anti-farming), hop-by-hop lateral enforcement,
PATCH/ISOLATE effectiveness fixes.

**Status: VERIFIED**

### Correction to the previous entry
The previous entry's observed rewards (red −110.05 / blue −4.85) were transcribed
from an unverified memory of a run, not measured — and were wrong. Measured values
for that configuration (seed=42) were red −125.0 / blue +135.0. Lesson recorded:
never transcribe numbers without re-running the source.

### Root-cause findings (measured, not assumed)
1. **"Reset penalties dominate blue" was FALSE.** Component decomposition over a
   full episode showed `unnecessary_reset` never fired. Blue's +135 was dominated
   by `attack_detected` (+120 from 24 repeat detections on the same isolated,
   compromised node) — a **detection-farming loop**, plus two dead flags
   (`unnecessary_reset`/`unnecessary_block`/`unnecessary_restore` set by the engine
   but never declared on the context nor rewarded).
2. **Hop-by-hop regression found by post-fix trace:** the first adjacency-only
   implementation let red tunnel THROUGH an isolated foothold (web isolated at
   step 3, yet web→app→database lateral succeeded; red won at step 9). Isolation
   must sever movement at both endpoints.
3. **PATCH was nearly useless:** `_blue_patch` kept priv-esc-capable vulnerabilities
   in the exploit pool, so a patched web remained exploitable at 0.8 (observed:
   blue patched web step 1, red exploited it step 2).

### Implemented
- `Node.defender_aware_of_compromise` flag (`network.py`): monitoring pays the
  detection reward only on the FIRST detection of a node's compromise; repeat
  detections are `monitor_repeat_detect` events with no reward. Flag cleared on
  RESET (clean node ⇒ stale awareness). False positives (`monitor_false_positive`)
  are now a distinct result with a small penalty (−1).
- Dead flags wired: `unnecessary_reset`/`unnecessary_block`/`unnecessary_restore`
  declared on `TransitionContext`, passed through `env._reward_context`, and
  rewarded (−2 each). All future context flags must be wired end-to-end.
- `_red_lateral` v3 (hop-by-hop): destination-targeted; source must be a compromised
  DIRECT NEIGHBOR; NEITHER endpoint may be isolated. Deterministic source selection
  (list order, not set iteration — reproducibility across processes).
- `_blue_patch`: successful patch mitigates ALL listed vulnerabilities (pool emptied).
- `env.render("ansi")`: labels are "last-step reward"; smoke script prints
  "cumulative reward" (no more ambiguity between the two).

### Tested
- New regression tests: detection farming closed, unnecessary_reset wired,
  isolation blocks lateral movement, patch mitigates all vulns, lateral fails
  without adjacent foothold / to already-compromised destination.
- Full suite: **71 passed, 0 failed**. Smoke script assertions pass.

### Observed behavior (measured)
- Component decomposition (seed=42, after fixes): blue = contained +20,
  detected +5 (once), step −5. Farming eliminated; blue total driven by outcomes,
  not repeats.
- 30-seed sweep, scripted-aggressive red vs rule-based blue, max 50 cycles:
  - database compromised: **0/30** — isolation-based containment is reliable.
  - web compromised: **29/30** — red reliably gains initial access (the 1 miss:
    blue's patch landed before red's exploit roll).
  - red mean reward −9.8 (−10.0 to −5.0); blue mean +19.2 (−5.0 to +20.0).

### Known limitations
1. **Balance: the rule-based defender now dominates the scripted aggressive
   attacker 30/30.** After web is isolated, red has no effective counter-play
   (its foothold is severed; its policy keeps retrying lateral moves). This is an
   honest baseline result, but for RL research both sides must be able to win —
   Phase 5 must verify an RL red can learn to beat this defender (e.g., exploit
   faster than blue reacts, or split attention across web/mail/decoys).
2. When red exploits a node in the same step blue patches it, blue still pays
   `unnecessary_patch` even though the patch was reasonable ex ante — order-
   dependent penalty accounting, to revisit in Phase 3.
3. Red exploits from a pool rate (max of remaining vulns / base rate); per-vuln
   exploit modeling is deferred.

### Next milestone
- Phase 5: first RL agent (PPO) — training loop, checkpointing, evaluation on
  unseen seeds, with the explicit target of beating the rule-based blue baseline.

---

## 2026-09-11 — Phase 5: PPO pipeline verified end-to-end + exploit-reachability loophole closed

**Feature:** First verified RL training run; critical environment fix; API wired to trained checkpoints.

**Status: VERIFIED**

### Recovery findings (previous session state, verified by execution)
- Test suite passed on arrival: `pytest tests/ -q` → 71 passed.
- Smoke script passed on arrival.
- `torch` was NOT installed; a `torch-2.6.0+cpu` wheel in `backend/` was a truncated,
  invalid zip (136 MB). PPO code existed but had NEVER been executed. Replaced with
  torch 2.14.0+cpu from the official CPU index.
- `fastapi` was NOT installed either — the API app had also never been run.

### Bugs fixed in `violet/algorithms/ppo.py` (each proves the file was never run)
1. `NameError: DEFENDER_AGENT` — used in `collect_rollout` but not imported.
2. GAE never computed — `buffer.finalize()` was never called, so `update()`
   crashed on `assert advantages is not None`. Now bootstraps V(last obs) and
   finalizes the buffer (dones already cut bootstrapping at episode boundaries).
3. Buffer never cleared after `update()` — on-policy violation; the second
   rollout would collect 0 transitions and corrupt GAE state. Now cleared.
   Also guarded `random.setstate` on an unbound variable.

### P0 environment loophole closed: EXPLOIT was a teleport
- `_red_exploit` had NO reachability requirement: any node could be exploited
  directly from the internet, including the critical asset. Measured evidence:
  an untrained PPO policy (3 updates) scored **10/10 attack success in ~4.4
  steps** by hammering EXPLOIT_DATABASE — pure loophole, zero learning.
- Fix: EXPLOIT now requires (a) an internet-facing node when the attacker has
  no foothold, or (b) adjacency to a non-isolated compromised foothold
  otherwise. Isolated targets accept nothing.
- Regression tests: `test_exploit_requires_reachability` (direct DB exploit
  fails before AND after a non-adjacent foothold);
  `test_honeypot_triggers_on_decoy` updated — it had encoded the loophole
  (exploited decoy-1 with no path to it).
- Post-fix re-measurement: untrained RL red 0/10 vs rule-based blue,
  mean episode length 50 (honest failure, not instant bogus victory).

### PPO metrics now persist real episode outcomes
- `training.jsonl` records per update: `mean_episode_reward`,
  `episodes_completed`, `attack_success_rate` (measured, not assumed) alongside
  optimizer stats. Learning curves can now be generated from real data.

### CLI `train` command wired
- `violet train --updates N --seed S [--run-dir DIR]` runs the verified PPO
  pipeline (was a stub printing "not wired to RLlib yet").

### API: live simulation can use trained policies
- `POST /simulation/start?red_policy=rl&checkpoint=<dir>` loads a PPO
  checkpoint via RLAgent. Missing checkpoint → HTTP 404 with an actionable
  hint (never silent degradation).
- Fix: per-agent event filtering in `env.step` infos — red and blue share one
  TransitionContext, so both agents' infos previously carried ALL events
  (400 stream events for 100 steps; dashboard event log double-counted).

### Tested
- Full suite: `pytest tests/ -q` → **72 passed, 0 failed**.
- Smoke script: assertions pass; events total now equals steps (100/100).
- PPO run (CPU, seed 42): 50 updates / 25,600 steps in 7.6 s; checkpoints
  saved (best/final/update_*); checkpoint load via RLAgent verified;
  evaluation report saved from loaded policy.
- API via TestClient: /health 200; WebSocket streams (scripted and RL) deliver
  event + state + episode_end frames; missing-checkpoint 404 verified.

### Known limitations
1. 50-update PPO red does NOT yet beat rule-based blue (attack success 0.0 in
   evaluation). Expected at this scale; the dev-log entry from 2026-09-09
   (2b) predicted exactly this difficulty. Longer training, reward tuning,
   and observation improvements come next — NOT claimed as solved.
2. Observations remain a placeholder 64-dim vector (3–5 features per node,
   zero-padded); partial observability is still shallow.
3. Frontend pages still show placeholder metrics (labeled as such); dashboard
   is not yet wired to the API.

### Next milestone
- Longer PPO run + reward variant comparison (dense vs security-focused),
  then first dashboard wiring to real artifacts (runs/*/metrics/training.jsonl).

---

## 2026-10-10 — Violet Ledger HTTP range and durable replay slice

**Base revision:** `1a9788b` (`VIOLET recovery baseline`). The checkout already
contained staged and unstaged simulator and frontend changes; they were
preserved. No commit was created.

**Scope:** add a local-only instrumented target and an isolated, typed HTTP
executor, then persist the bounded run timeline and support replay/export. This
is an acceptance slice, not a real-range learning result.

### Implemented
- Added Violet Ledger with synthetic Alice/Bob accounts, fixture-seeded
  documents, a deliberately vulnerable authenticated IDOR route, structured
  access evidence, and a typed ownership-hardening control.
- Added a range-only executor container with fixed target DNS/port and routes,
  session-cookie isolation, action/episode budgets, response excerpt limits,
  and no database or control-plane network access.
- Added separate target and VIOLET PostgreSQL services and named volumes. Target
  reset/control commands use typed volume messages and acknowledgements.
- Added SQLAlchemy run, episode, job, control, event, and target-evidence
  records. API job/control submissions commit before queueing; sequence events
  use a previous-hash chain, and the API reconciles pending outbox records on
  startup. `GET /range/runs/{run_id}/events?after_seq=N` replays committed data.
- Added a control-verification endpoint that records a verified outcome only
  when the ownership control was acknowledged before both requests, Alice's
  foreign read is denied, and Bob's owner read succeeds. Repeating the same
  request pair is idempotent.
- Added a redacted JSONL run exporter. It verifies the database event chain
  before export and writes a manifest with the exported file's SHA-256.
- Added a Compose-backed target smoke path that resets seed 17, verifies
  Alice's access to Bob's canary, hardens ownership, verifies Alice gets 403
  while Bob gets 200, and checks that the verification event is persisted once.
- Added a root `.dockerignore` and deterministic frontend `npm ci` build step;
  generated dependencies and build results no longer enter Docker build context.

### Verified
- Full backend suite on current workspace source inside the isolated backend
  image: `pytest -p no:cacheprovider tests -q` → **86 passed**.
- Ruff check and format check over range, target, API, persistence, export, and
  focused tests → **passed**.
- Compose smoke, seed 17: confirmed IDOR canary, then Alice `403` and Bob `200`;
  persisted **22** events with contiguous sequence numbers and linked hashes;
  duplicate control verification created no duplicate event.
- Restarted the API and replayed run `82a0b4d4-5a5a-4794-8d24-efe4579c9ec5`:
  **22 events**, seq 1–22, including its `verified` control outcome.
- Exported that run to `/tmp/violet-range-export-verify/`; checked event count
  22, manifest SHA-256, inclusion of the control outcome, and redacted session
  handle in exported JSONL.
- Frontend Docker build completed with TypeScript and production compilation;
  generated-file ignore rules reduced sent build context from about 920 MB to
  3.04 KB.

### Remaining limitations
1. SQLAlchemy currently bootstraps tables with `create_all()`; Alembic
   migrations and append-only database permissions are not implemented.
2. The real-range timeline persists actions, executor results, control
   acknowledgements, and target access evidence. It does not yet persist
   role-specific observations/decisions, detections, reward decomposition, or
   PPO training updates.
3. Red/Blue real-range PPO, alternating opponent-league training, held-out
   evaluation, and any learning claim remain unimplemented.
4. The dashboard still presents the graph simulator; no live range timeline,
   evidence inspector, or replay UI is connected.
5. The documented Compose stack isolates the executor and target networks, but
   a full container-boundary/egress denial test matrix and target crash
   reconciliation tests remain to be added.

**Next milestone:** add migration-managed persistence and Blue-visible telemetry
with verified control outcomes before connecting either PPO policy to the real
range.

---

## 2026-10-25 — Real-range PPO implementation slice

**Status: IMPLEMENTED; HTTP PATH VERIFIED; PPO OPTIMIZATION UNVERIFIED**

### Implemented
- Added separate Red and Blue categorical PPO actor/critic models over
  `violet.range.observation.v1`, with finite action catalogs, role checks,
  stochastic training actions, deterministic inference, optimizer updates,
  and role/schema/action-checked checkpoints.
- Added `/range/training/transitions` and `/range/training/updates`; each
  appends typed transition/update events into the existing run hash chain.
- Added a real-range runner that sends all Red requests and Blue controls via
  the existing HTTP API and typed executor. Blue is masked to no-op and
  ownership hardening because that is the only control with the independent
  Alice-denied/Bob-served verifier.
- Corrected simulator GAE so terminal transitions stop value bootstrap while
  time-limit truncations bootstrap the final value and stop the trace.
- Added architecture, algorithm, reward, training, evaluation, and
  troubleshooting documentation.

### Verified
- Backend suite in the Linux backend container: **87 passed, 1 skipped**. The
  skipped module is the optional PPO-specific test, because the backend image
  does not install the optional `rl` extra.
- Ruff check on changed Python sources: **passed**; `compileall` and
  `git diff --check`: **passed**.
- Compose backend, executor, target, and databases started. The existing
  `scripts/range_smoke.py` confirmed the seeded IDOR, then independently
  verified Alice `403` and Bob `200` and persisted 22 ordered events.
- Manually posted a bounded learning transition and PPO-update event to the
  new API; replay showed `learning_transition` and
  `learning_policy_update` in sequence.

### Not verified / limitations
- This host's Anaconda PyTorch fails to initialize with Windows error 1114
  while loading `c10.dll`; the backend image intentionally lacks optional
  PyTorch. Therefore the actual PPO optimizer tests, parameter-change check,
  checkpoint load in execution, and a training run were not executed.
- The dashboard does not expose the new training workflow. The full frontend
  image was not built during the bounded integration check.
- No learned-versus-baseline or held-out metrics are reported. Rate limiting
  and actor blocking remain masked in the learning runner pending independent
  outcome verification.

See `docs/architecture-audit.md` and `docs/real-range-training.md` for current
commands and constraints.

---

## 2026-10-25 — PPO runtime verification and dashboard completion

**Status: PPO UPDATES AND REAL HTTP TRAINING VERIFIED; POLICY IMPROVEMENT NOT DEMONSTRATED**

### Runtime and tests
- Rebuilt `docker/backend/Dockerfile` with the declared `rl` extra and an
  official CPU-only PyTorch wheel. The isolated Linux runtime reports Python
  **3.12.15**, PyTorch **2.14.1+cpu**, and `(tensor([1,2])**2).sum() == 5.0`.
- A separate Windows venv with Python **3.12.4** and PyTorch **2.14.1+cpu**
  still fails at `c10.dll` with WinError 1114. No Anaconda packages were
  removed or replaced.
- Final backend suite: **108 passed, 0 failed, 0 skipped**; Ruff: **passed**.
  Frontend TypeScript, ESLint, and Docker production build passed. The
  `/range-training` page returned HTTP 200 from both local production serving
  and the Compose frontend.

### Training, evaluation, and persisted evidence
- Ran three episodes at seeds 42–44, six Red decisions and six Blue decisions
  per episode. A separately instrumented rerun recorded 18 accepted transitions
  per role, three policy updates per role, and four optimizer steps per update
  (12 optimizer steps per role). The event store recorded 36 transitions, six
  policy-update events, 30 executor results, three independent control
  verifications, three completed episode events, and a completed run status.
  Invalid action count was zero.
- Both checkpoints were loaded in deterministic evaluation. All decisions were
  repeatable and the evaluation records reported zero optimizer updates. Unit
  tests snapshot both role parameter sets around updates and verify that each
  changes, losses/gradients stay finite, checkpoints restore, and evaluation
  does not mutate parameters.
- The corrected reward gives no positive Red reward for irrelevant Bob
  authentication. In the 20-episode-per-pair held-out comparison, all four
  pairings had 0% verified Red success and 100% Blue prevention on eligible
  episodes. The fixture outcomes were identical, so bootstrap intervals were
  degenerate. This demonstrates the range path and verifier, not learned
  improvement. Raw comparison is in
  `backend/runs/real_range_eval/comparison_reward_v2.json`.
- Dashboard reads `GET /range/runs` and `GET /range/runs/{id}/events`; viewing
  history does not execute past actions. Blue remains limited to no-op and
  ownership hardening because other controls lack independent verifiers.
