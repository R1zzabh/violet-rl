# Real HTTP range architecture: Violet Ledger and SQLi lab

This document covers the preserved Violet Ledger target and dedicated SQLi
learning target. They have separate databases, executors, controls, and
scenarios. The graph simulator remains separate; its state changes are
simulated and do not count as traffic or evidence for either HTTP target.

## Learning contract

- Target: one synthetic Violet Ledger service and the Alice-to-Bob document
  authorization objective. The database contains only synthetic fixtures.
- Versioned inputs: `red.range_observation.v2` and
  `blue.range_observation.v2`, encoded by `violet.range.features.v2`.
- Versioned catalogs: `red.action_catalog.v2` and
  `blue.action_catalog.v2`. Red actions are finite typed requests; Blue's only
  learnable non-monitor control is ownership hardening. Rate limits and actor
  blocks remain masked pending independent security and availability checks.
- Versioned rewards: `violet.range.reward.v2`. Red receives `+1` only after
  independent canary verification of an unauthorized read. Blue receives `+1`
  only after ownership hardening denies Alice's Bob-document request and serves
  both Alice's and Bob's legitimate reads. False alerts and denied legitimate
  reads each cost `-1`; components remain separate in persisted records.
- Both policies are frozen during an episode. Separate Red and Blue actors,
  critics, optimizers, random streams, and policy versions update at the
  episode boundary. Rejected actions remain in the event chain but are omitted
  from PPO data and break the GAE trace. Unknown outcomes make the episode
  incomplete and are never treated as successful execution.
- Teaching runs may use short episodes. Research training requires at least
  10,000 completed episodes for one training seed; research evaluation requires
  at least 100 completed paired held-out episodes. A result does not support
  an improvement claim unless the paired net-security interval is positive,
  the legitimate-read criterion passes, and at least five independent
  training seeds are represented.

The formal campaign is not orchestrated across five independent checkpoints
yet. A single-seed evaluation therefore reports `no supported improvement`,
even if the policy changes or its training metrics improve. Research jobs that
cannot complete their requested episode minimum are not promoted as completed
research runs.

## Request and evidence flow

1. `GET /range/runs/{run_id}/observations/{red|blue}` constructs the
   canonical role view from persisted events. `POST /range/decisions` checks
   the observation, feature vector, action mask, catalog index, selected
   action, and full sampled distribution before persistence.
2. Policy jobs must reference their persisted decision. Evaluator probes have
   `purpose=evaluator`; typed workbench requests have `purpose=manual`. Manual
   requests are auditable target traffic and never enter policy rollouts.
3. The executor maps action IDs to fixed methods and routes, disables
   redirects, checks the fixed Violet Ledger origin, keeps cookies in memory,
   and stores opaque session aliases. It records bounded responses with
   allowlisted headers. It has no API, database, shell gateway, or Docker
   socket access.
4. Violet Ledger writes correlated request evidence to its evidence volume.
   The API imports that evidence when reconciling executor results. A finding
   requires matching run, episode, decision, action, request, target event,
   principal, resource, response, and the seed-derived canary.
5. Blue sees sanitized target telemetry and detections, never canaries, fixture
   owners, verifier labels, credentials, cookies, session aliases, or the
   other role's private observation. Key filtering also removes hidden fields
   with suffixed or prefixed names.
6. Ownership control verification requires three correlated evaluator probes:
   Alice reading Bob's document returns 403, Alice reading her document returns
   200, and Bob reading his document returns 200. Acknowledgement alone earns
   no reward.
7. `GET /range/runs/{run_id}/episodes` lists episodes;
   `GET /range/runs/{run_id}/episodes/{episode_id}/timeline` pages the
   read-only event history. `GET /range/training-runs/{id}/updates` returns
   role-specific optimizer updates. Historical replay never resubmits actions.

The coordinator resets Violet Ledger before every episode, runs one Red action
then lets Blue observe and act, and finishes with an auditable episode state.
The executor request budget is 40 per episode, with four requests reserved for
independent control-verification probes. Requests have a five-second timeout;
episodes have a five-minute deadline. An executor restart during an in-flight
request is retained as `outcome_unknown`, not replayed.

## Learning and evaluation records

Each sampled decision retains its encoded observation, action mask, complete
categorical distribution, selected index and probability, old log probability,
value, entropy, policy version, and train/evaluate mode. Its transition stores
execution status, acceptance, reward components, next observation/value,
terminal flags, and evidence references in the hash-linked event chain and
transition projection.

Research training writes raw episode records and optimizer state checkpoints.
The worker supports resume from a compatible joint checkpoint and persists a
separate update for each role at each completed episode boundary. Evaluation
loads a checkpoint without optimizer updates, uses identical target seeds for
candidate and baseline episodes, and records verified unauthorized-read,
verified-prevention, legitimate-owner-access, false-alert, and paired
net-security metrics. Bootstrap intervals summarize paired differences.
Changed probabilities, loss, entropy, gradients, or training return alone do
not demonstrate improved task performance.

The required five-seed aggregation and formal held-out campaign are still
manual work. The API will not label a one-seed result as supported improvement.

## SQLi learning lab

The SQLi scenario is `sqli_lab_v1` against target version `sqli_target_v1`.
It uses its own Postgres database, target service, executor, control queue, and
evidence volume. The target table contains only fixed synthetic records. The
runtime database role is a non-owner with `SELECT` on that table, and the
connection is also set read-only. The target has no published port.

Red and Blue are independent factorized categorical PPO policies. Red selects
one value from each bounded component head for field, base term, quote closure,
boolean connector, predicate, and comment marker. The server validates the
composed input's grammar and 96-character limit; the query adapter selects only
one of four fixed columns, one fixed table, and a projection of known fixture
columns, with a 220-character SQL limit. Red does not choose a host, URL, route,
SQL table, or complete payload template. Fixed verifier probes are independent
checks and are excluded from Red's PPO transitions.

Blue receives normalized status/count/signal telemetry and current control
state. It does not receive query text, record values, owners, or verifier
labels. Its separate PPO head chooses monitor, parameterize, guest-scoped rate
limit, or temporary guest lookup isolation. Every defense is checked with a
correlated unauthorized-read probe and a legitimate researcher lookup. The
reward components score verified prevention, availability, false positives,
and unknown evidence separately.

The pair's weights remain fixed throughout one episode. Accepted Red and Blue
transitions are persisted before the episode-boundary update. Rejected or
unknown-outcome actions remain in replay but are excluded from PPO data.
Updates use GAE and clipped PPO with role-local optimizers and versions. Each
boundary records before/after action probabilities on the same stored
reference observation. Checkpoints include the scenario and all observation,
feature, action, reward, and component contracts; evaluation rejects a
checkpoint with mismatched scenario metadata or digest.
In Compose, SQLi checkpoints and metrics live in a separate persistent
`sqli-training-runs` volume owned by the restricted trainer identity. Graph
simulator artifacts under `backend/runs/` keep their existing file-backed
storage.

The SQLi instructor room at `/sqli/{training_run_id}` replays decision, request,
response, control, verification, reward, transition, probability probe, and
optimizer events from the persisted chain. It plots Red/Blue returns separately
from target-measured unauthorized-read/prevention rates, legitimate-query
availability, false positives, and optimizer diagnostics. The no-defense
comparator is evaluation-only and named in its record. Current results still
require five independent training seeds, at least 100 held-out pairs, positive
paired uncertainty bounds, and full legitimate availability before supporting
an improvement claim.

## Databases and isolation

The VIOLET control database stores jobs, decisions, detections, rewards,
transitions, training state, updates, and a per-run sequence/hash event chain.
The target database stores only disposable synthetic fixtures and target
control state. Independent Alembic streams define explicit control and target
schemas; production startup checks that schemas are migrated. SQLite
self-initialization is limited to local tests.

Migration services use owner credentials. The API and Violet Ledger use
separate restricted runtime roles provisioned by `scripts/provision_runtime_role.py`.
Those roles currently receive CRUD privileges needed by the existing services;
SQL-level append-only enforcement and narrower per-table grants remain future
hardening. The API mounts target evidence and control-result volumes
read-only.

The SQLi target uses a third runtime identity provisioned by
`scripts/provision_sqli_runtime_role.py`; it can read the synthetic table but
cannot modify rows or create schema objects. The SQLi executor is attached only
to `sqli-range`, where it can reach the target's fixed HTTP port. The SQLi
target also joins `sqli-data` for its own database. The trainer and API are not
attached to either SQLi network. The API exchanges bounded JSON jobs and
controls through named volumes and sees correlated evidence through a
read-only mount.

Network boundaries are defined in Compose:

- The API reaches the control database and trainer-control network, but not the
  target HTTP network.
- The trainer reaches only the API over trainer-control and has no database
  credential or executor volume.
- The executor shares the internal `range` network only with Violet Ledger.
- Violet Ledger also joins the internal `target-data` network for its own
  database. The two databases and target have no published host ports.
- The SQLi executor shares `sqli-range` only with the SQLi target. The target
  joins `sqli-data` for its database; the trainer and API are not attached to
  either network. The SQLi target and database have no published ports.
- The browser reaches loopback-published API and frontend ports only.

## Local verification

Set distinct values for `VIOLET_DB_PASSWORD`, `VIOLET_RUNTIME_DB_PASSWORD`,
`TARGET_DB_PASSWORD`, `TARGET_RUNTIME_DB_PASSWORD`, `SQLI_DB_PASSWORD`,
`SQLI_RUNTIME_DB_PASSWORD`, and `VIOLET_TRAINER_TOKEN` in the root `.env`, then
start the stack:

```bash
docker compose up --build
cd backend
python scripts/range_smoke.py
python scripts/sqli_range_smoke.py
python scripts/range_isolation_smoke.py
```

The Ledger smoke proves container TCP traffic, seeded evidence correlation,
Alice's denied foreign read, legitimate Alice/Bob availability, and event
replay. The SQLi smoke proves bounded real HTTP requests return only synthetic
evidence before parameterization, the same request class returns no foreign
records after the control, legitimate lookup remains available, and events
replay. The isolation smoke checks both declared network sets, published ports,
the SQLi runtime role's read-only grants, and prohibited destinations. These
checks do not establish behavior outside the tested Docker
configuration or establish learning improvement.

An earlier Ledger-only acceptance run passed before the SQLi services were
added. The Compose CI job now starts both executors and the trainer, runs each
target acceptance probe, then checks both network boundaries and SQLi database
permissions. A local run should use the commands above against the current
Compose stack; results do not transfer to other Docker configurations.
