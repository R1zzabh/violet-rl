# VIOLET learning and evidence audit

## Baseline and scope

The merged baseline identified for this work was `f9077c9`. The repository
already contained a substantial dirty and untracked worktree when this
implementation continued. Those changes were preserved and remain local
worktree changes; they are not evidence of merged or deployed behavior.

The core real-range task is one synthetic Violet Ledger target and one
Alice-to-Bob unauthorized document-access objective. The graph simulator is a
separate environment. It changes in-memory graph state and cannot support
claims about HTTP execution, Violet Ledger, or real-range PPO.

The historical [`rl_experiment_results.md`](rl_experiment_results.md) records
an earlier experiment with no demonstrated improvement. Its schemas, reward
values, and trainer path predate the v2 implementation and must not be
presented as results from the current worktree.

## Locked contract in this worktree

| Contract | Version or rule |
|---|---|
| Red observation | `red.range_observation.v2` |
| Blue observation | `blue.range_observation.v2` |
| Feature encoder | `violet.range.features.v2` |
| Action catalogs | `red.action_catalog.v2`, `blue.action_catalog.v2` |
| Reward | `violet.range.reward.v2` |
| Core task | Alice's session requests Bob's synthetic document |
| Red reward | `+1` only for an independently verified foreign canary read |
| Blue reward | `+1` only after foreign denial plus legitimate Alice and Bob reads |
| False alert cost | `-1` per evidence-attributed false alert |
| Legitimate denial cost | `-1` per denied legitimate access |
| Policy update boundary | Independent Red and Blue PPO updates after one completed joint episode |
| Research training minimum | 10,000 completed episodes per training seed |
| Research evaluation minimum | 100 completed paired held-out episodes |
| Improvement gate | Positive paired net-security interval, availability criterion met, and at least five independent training seeds |

The reward weights are equal in magnitude for one verified prevention and one
false alert or legitimate denial. They are stored as separate components. The
API enforces the episode and paired-evaluation minimums for research jobs.
Teaching runs can be short and are clearly distinguished in the persisted
configuration.

## Evidence boundaries

- Red and Blue receive role-specific canonical observations. Blue's sanitizer
  removes verifier labels, canaries, ownership fields, session aliases, and
  secret fields recursively, including names containing those terms.
- A Red positive reward requires a persisted executor result, target event,
  request correlation, and seed-derived canary match. HTTP 200 by itself is not
  a finding.
- A Blue positive reward requires target acknowledgement and an independent
  three-probe control check. The verifier checks Alice's foreign denial plus
  Alice's and Bob's legitimate reads.
- Only accepted, actually executed policy decisions enter a PPO update.
  Rejected decisions remain auditable and break the GAE trace. Unknown
  outcomes mark the episode incomplete and are excluded from completed
  training episodes.
- Manual workbench requests and evaluator probes have distinct origins and do
  not enter policy rollouts.
- Evaluation runs do not call an optimizer. Candidate and baseline share
  held-out fixture seeds. Reports expose verified outcomes and availability;
  policy changes, losses, entropy, gradients, and training return alone do not
  support an improvement claim.

## Implementation status and limits

The current worktree contains the v2 observations, reward rules, independent
control verification, per-decision categorical samples, transition
projections, joint episode training, research episode gates, read-only episode
timelines, separate Alembic revisions, restricted runtime roles, and a rebuilt
Violet Ledger console. They remain uncommitted worktree changes, not merged or
deployed behavior.

Verification completed in this continuation: all 126 backend tests passed;
Ruff lint and format checks, mypy, frontend typecheck/lint, and the production
frontend build passed. Fresh SQLite migration chains and both PostgreSQL
migration services succeeded. The Compose TCP smoke confirmed the seeded
canary finding, ownership hardening, Alice/Bob owner access, and 25 sequenced
hash-linked events. Isolation probes confirmed the executor's single target
reachability, loopback-only API binding, prohibited egress/DB/API denials, and
that API and target runtime roles cannot create schema objects. The local
Compose check did not start the trainer image; the CI job now includes it, but
that workflow has not yet run.

The following research work is still outstanding:

- The worker trains one seed per run. Five independent checkpoints are not yet
  aggregated into a campaign manifest, so even a 100-pair evaluation of one
  checkpoint cannot pass the five-seed improvement gate.
- Held-out evaluation reports a paired net-security score and eligibility
  checks but no current v2 evaluation artifact exists.
- Rate limiting and actor blocking remain unavailable to Blue policies until
  their effects and legitimate-user availability costs have independent
  verifiers.
- Runtime roles still have broader table-level CRUD privileges than needed,
  and the event chain does not yet have database-enforced append-only
  permissions.
- Compose restart and worker-failure recovery, browser E2E, trainer-container
  reachability, and the full 10,000-episode research campaign remain
  unverified in this worktree. CI is configured to build and probe the trainer
  container, but that job has not run yet.

Until those checks and a preregistered held-out campaign pass, state the result
as **no supported improvement**.
