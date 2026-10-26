# Real-range PPO experiment results

> Historical result artifact for the earlier real-range schema and reward
> version. These measurements predate the current v2 observation, reward,
> transition, joint-training, and research-run contracts in this worktree. No
> v2 training or evaluation campaign has been run, so the values below do not
> establish current-code performance.

## Hypothesis

If the held-out synthetic task rewards useful policy behavior, the learned
policies should improve independently verified Red success or Blue prevention
over scripted and random baselines on identical fixture seeds. This test does
not assume lower loss or changed parameters imply better behavior.

## Environment and policy setup

- Target: isolated synthetic Violet Ledger only, reset through the real range
  API for every episode. The per-seed canary changes; ownership and policy
  consequences do not.
- Executor: existing typed job API and bounded action catalog. Findings are
  independently verified from fixture canaries. Ownership control is verified
  by post-control Alice `403` and Bob `200`.
- Red observation: seven normalized role-specific features. Red has eight
  catalogued actions; document reads are masked until Alice authenticates.
- Blue observation: eight normalized public counters, health, findings, and
  control state. Blue's effective mask allows only no-op and ownership
  hardening; other catalog entries remain disabled because their independent
  verifier is absent.
- Rewards: Red gets +1 for a confirmed foreign canary, +0.05 for successful
  Alice setup auth, and −0.1 for a failed request. Blue gets +1 for verified
  ownership hardening before a confirmed attack, −1 when a new verified attack
  remains unprevented, and −0.1 for failed control application. No reward or
  verifier change was made for this audit.
- PPO: separate CPU actor-critic and optimizer per role; gamma 0.99, GAE lambda
  0.95, clip 0.2, learning rate 3e-4, 4 epochs, minibatch 32, hidden width 64,
  entropy coefficient 0.01, maximum gradient norm 0.5. Training checkpoint
  seed range was 42–44. Held-out evaluation used seeds 10000–10019.
- Evaluation: frozen checkpoints use deterministic argmax and perform no
  optimizer updates. Scripted baseline is deterministic. Random Red samples
  uniformly from its valid action mask with seed 10000 and continues a
  reproducible stream across the 20 episodes. The identical 20 fixture seeds
  are used in each matchup.

## Results

All 120 episodes completed, with 6 Red decisions per episode, zero invalid
actions, and no confirmed Red finding. Blue's prevention metric is conditioned
on Red not already having a verified success, so its denominator is 20 in
every eligible matchup. Outcome rates use two-sided 95% Wilson score
intervals; these retain uncertainty for all-success and no-success samples.
Mean returns use percentile bootstrap intervals over episode means, 2,000
resamples with a fixed bootstrap seed.

| Red vs Blue | Verified Red success / 20 (95% CI) | Verified Blue prevention / eligible (95% CI) | Mean Red return (95% CI) | Mean Blue return (95% CI) | Red / Blue mean entropy | Legitimate own-read success | False blocks |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scripted vs scripted | 0/20, 0% [0, 0.161] | 20/20, 100% [0.839, 1] | −0.45 [−0.45, −0.45] | 1.00 [1, 1] | 0 / 0 | Not attempted (n=0) | 0/0 |
| Scripted Red vs passive no-op Blue (calibration) | 20/20, 100% [0.839, 1] | N/A (0 eligible; Red succeeded 20/20) | 1.05 [1.05, 1.05] | −1.00 [−1, −1] | 0 / 0 | Not attempted (n=0) | 0/0 |
| Random Red vs scripted Blue | 0/20, 0% [0, 0.161] | 20/20, 100% [0.839, 1] | −0.0525 [−0.1025, −0.0075] | 1.00 [1, 1] | 1.89 / 0 | 6/6 (100%) | 0/6 |
| Learned Red vs scripted Blue | 0/20, 0% [0, 0.161] | 20/20, 100% [0.839, 1] | 0.00 [0, 0] | 1.00 [1, 1] | 1.79 / 0 | Not attempted (n=0) | 0/0 |
| Scripted Red vs learned Blue | 0/20, 0% [0, 0.161] | 20/20, 100% [0.839, 1] | −0.45 [−0.45, −0.45] | 1.00 [1, 1] | 0 / 0.12 | Not attempted (n=0) | 0/0 |
| Learned vs learned | 0/20, 0% [0, 0.161] | 20/20, 100% [0.839, 1] | 0.00 [0, 0] | 1.00 [1, 1] | 1.79 / 0.12 | Not attempted (n=0) | 0/0 |

Entropy is in nats and averaged over decisions. The learned Red checkpoint's
greedy action was synthetic Bob authentication on every decision (action 4),
while its distribution stayed near-uniform over valid actions (mean entropy
1.79 nats, selected-action probability about 0.17). This is a poor greedy
choice, not a peaked policy. The learned Blue checkpoint selected hardening
first and then no-op, with low entropy (0.12 nats); its behavior matched the
scripted Blue baseline. Random Red had the highest entropy and a better mean
Red return than the scripted baseline because it avoided many failed requests,
but its verified attack success remained 0/20. Thus return is not a substitute
for the verified objective.

Legitimate own-read counts are derived from persisted executor events: only
`CHECK_DOCUMENT_ACCESS` for Alice's own `doc_alice` with principal Alice is in
the denominator. Success is HTTP 200; false block is HTTP 403, 423, or 429.
Unattempted cells are not assigned a zero success rate; the rate is undefined
because the denominator is zero. Random Red attempted six qualifying own reads
and all six succeeded after hardening.

## Training verification

The prior isolated CPU container used Python 3.12.15 and PyTorch 2.14.1+cpu;
`torch.tensor([2.0]) ** 2` returned 4.0. PPO unit tests verify independent Red
and Blue parameters, finite losses and gradients, and a parameter change after
a valid optimizer update. The fresh real HTTP diagnostic run
`real_range_ppo_audit_diagnostics` used three training episodes on seeds
42–44. Each role collected 18 accepted transitions, made 3 policy updates,
and performed 12 optimizer steps (4 per update; 24 total across both roles).
Update losses, gradients, and advantages were finite. The new checkpoint
SHA-256 hashes exactly match the previously evaluated
`real_range_ppo_final` checkpoints. Loading the diagnostic checkpoints in
deterministic HTTP evaluation on seed 20000 completed with zero policy or
optimizer updates. A forward pass, saved checkpoint, event row, or changed
parameter is not sufficient learning evidence. Held-out performance remained
equal to the scripted outcome rates, and therefore learning improvement was
not demonstrated.

The diagnostics added during this audit persist per-decision selected-action
probability and entropy, and per-update policy/value losses, clip fraction,
gradient norm, and raw advantage mean/std/min/max. Empty rollouts still report
zero updates. They describe optimization and behavior but do not alter the
task reward or verifier.

## Reproduction and artifacts

From the repository root with the Compose target, executor, database, and API
healthy:

```bash
docker compose build backend
docker compose run --rm backend python -m violet.range.evaluate_real \
  --api http://backend:8000 --episodes 20 --seed 10000 \
  --red-checkpoint runs/real_range_ppo_final/red.pt \
  --blue-checkpoint runs/real_range_ppo_final/blue.pt \
  --training-seeds 42,43,44 \
  --output runs/real_range_eval/audit_comparison_with_calibration.json
```

The evaluator checks API health before each episode, resets the target with the
episode seed, and records raw action lists, action frequencies, returns,
verified labels, invalid actions, update counts, entropy, selected-action
probability, latency, and episode IDs. The enriched audit artifact also
includes legitimate-request counts reconstructed from the persisted event
stream. Events can be inspected read-only with
`GET /range/runs/{run_id}/events?after_seq=N`; replay never submits historic
actions to the target.

- Raw five-cell matrix: `backend/runs/real_range_eval/audit_comparison.json`
- Five-cell matrix plus passive calibration:
  `backend/runs/real_range_eval/audit_comparison_with_calibration.json`
- Previously trained role checkpoints: `backend/runs/real_range_ppo_final/red.pt`
  and `backend/runs/real_range_ppo_final/blue.pt`
- Historical three-episode training records:
  `backend/runs/real_range_ppo_final/training.jsonl`
- Instrumented HTTP training output and checkpoints:
  `backend/runs/real_range_ppo_audit_diagnostics/`
- Deterministic HTTP checkpoint-load evaluation:
  `backend/runs/real_range_ppo_audit_evaluation/training.jsonl`
- Root cause and transition trace:
  `backend/docs/rl_effectiveness_audit.md`

## Conclusion and limitation

Learning improvement was **not demonstrated**. All five requested matchups
have the same verified Red and Blue outcome rates. The passive calibration
confirms the target distinguishes a successful Red attack from prevention
when Blue makes a different valid choice. It does not create richer scenario
variation or a defensive tradeoff. The main cause is inadequate task
variation and a dominant, cost-free hardening action, not a measured inability
of PPO to optimize. The learned Red checkpoint's greedy policy also selects
Bob authentication, a behavior defect visible in held-out action traces. The
current task cannot separate prevention quality or generalization because the
same ownership relation recurs at every seed and the other controls are masked.

The next effectiveness experiment should first add explicit synthetic
scenario configurations with genuinely different verified outcomes, including
legitimate-service effects and independently verified false blocks. Scripted
and random policies should then be evaluated on those scenarios before a new
PPO training run. No scenario or reward changes were made after observing these
evaluation results.
