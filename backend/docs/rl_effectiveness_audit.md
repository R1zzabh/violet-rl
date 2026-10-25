# Real-range RL effectiveness audit

## Scope and evidence

This audit follows one training decision cycle from the role observation to
the typed HTTP executor, target response, verifier, reward, rollout, and saved
update. The evaluation fixture is the disposable Alice/Bob Violet Ledger
database; no external target is involved. Source references below use paths
relative to `backend/`.

## Findings

The prior 20-episode held-out run produced 0/20 verified Red successes and
20/20 verified Blue preventions in every matchup. This is not evidence that
PPO learned: the same outcomes arise from the scripted policy, and all Blue
prevention-eligible episodes are won by one early control decision.

The root cause is the exercise design and action mask, rather than a verified
PPO optimizer defect:

1. `violet/range/train_real.py::RangeTrainingRunner.run_episode` alternates one
   Red request and one Blue decision. Red's scripted first action authenticates
   Alice. Blue can then apply ownership hardening before the next Red request.
2. The Blue mask is `[no_op, harden, disabled, disabled]` for the whole run.
   Rate limiting and account blocking are catalogued but not executable. The
   runner restricts Blue to the only action with a valid independent verifier.
3. After hardening, the control verifier checks Alice's foreign read is HTTP
   403 and Bob's own read is HTTP 200. The control is credited as prevention
   when Red has not yet produced a confirmed foreign-read finding. The verifier
   establishes the control's security effect, but cannot establish that an
   attack would otherwise have succeeded.
4. Blue receives +1 for that verified control and no cost for applying it
   before a threat. In this fixture, ownership hardening does not impair Bob's
   legitimate access. Consequently hardening immediately after Alice's auth is
   a dominant scripted response. A Red foreign read verified before that
   decision is irreversible and Blue receives no prevention credit.
5. The target fixture seed changes only the deterministic canary, not the
   ownership relation or policy consequences (`violet/target_app/database.py`).
   Therefore different held-out seeds are not different task scenarios.

### Representative scripted episode trace

For `--red-steps 6`, the representative scripted trajectory is:

| Turn | Role and observation | Action / executor outcome | Evidence and reward | Rollout |
|---|---|---|---|---|
| 0 | Red: Alice unauthenticated, zero objectives | `login_synthetic_alice`; typed auth succeeds, HTTP 200 | Alice setup +0.05; next observation records Alice auth | Accepted Red transition; enters PPO rollout |
| 0 | Blue: one request observed, no verified attack/prevention/control | `harden_document_ownership`; typed control applied | verifier's Alice foreign `403` and Bob owner `200`; +1 prevention | Accepted Blue transition; enters PPO rollout |
| 1–5 | Red: Alice authenticated, no verified objective | foreign-document read; HTTP 403 after hardening | each failure −0.1; no Red finding | Each accepted request enters Red rollout |
| 1–5 | Blue: ownership control already applied | no-op | zero verified outcome/reward | Each accepted no-op enters Blue rollout |
| End | Both roles: six decisions, time limit reached | `terminated=false`, final `truncated=true` | GAE bootstraps final value and stops the trace | Each role updates from its six accepted transitions |

For this trajectory, Red return is `0.05 - 5*0.1 = -0.45`; Blue return is
`+1`. HTTP success on Alice's auth is setup evidence only. Red success requires
the independent canary verifier; Blue prevention requires the post-control
ownership verifier. Requests produced by the control verifier are evidence
jobs, not policy transitions.

## Component review

- `violet/range/learning.py` keeps independent actor, critic, optimizer, and
  random streams by role. The action mask is applied during sampling and PPO
  log-probability recomputation. Rollouts contain accepted actions only.
- GAE in `violet/algorithms/ppo.py` bootstraps time-limit truncations from each
  transition's `next_value`, while terminal transitions do not bootstrap.
- The PPO update checks finite loss, gradients, and clipped gradient norm;
  tests snapshot each role's parameters and assert a valid update changes them.
  These checks verify optimization executes, not that outcomes improve.
- Red has eight catalogued actions and receives a small Alice-auth setup reward,
  a verified foreign-read reward, and a failed-request cost. Blue's actual
  mask currently reduces four catalogued actions to two. The reward and mask
  make the measured Blue policy a binary timing choice, not a defense-selection
  policy.
- `violet/range/verifier.py` independently validates document identity,
  owner, and fixture canary. `violet/range/control_verifier.py` validates
  post-control Alice denial and Bob availability. Neither verifier is weakened
  by this audit.
- The runner persists observations, masks, proposal/acceptance, reward
  components, next observations, ending flags, selected action probability,
  entropy, and policy version. Update events record loss, advantage summaries,
  gradient norm, and clip fraction.
- Evaluation is deterministic for learned policies and makes no optimizer
  updates. Evaluation seeds are explicitly held out from the short training
  seed range, but the fixture's security properties do not vary with seed.

## Experiment-design conclusion

The existing exercise is adequate for verifying that typed requests, target
controls, independent evidence, rollout construction, PPO updates, and
checkpoints work end to end. It is not adequate for demonstrating policy
improvement: it has one meaningful defensive control, no operational cost or
legitimate-service tradeoff for that control, a deterministic threat sequence,
and a verifier that proves control effect rather than counterfactual
prevention. Different policy pairings can therefore share identical terminal
outcomes even when their action distributions differ.

A passive Blue calibration using the already-valid no-op action was added to
the evaluator as an extra cell beyond the five requested pairings. This is not
a modified target scenario: it checks that the existing default fixture and
independent finding verifier distinguish an unopposed scripted Red attack
from a Blue policy that hardens after Alice authenticates. See the calibration
row in `rl_experiment_results.md` and raw episode data in
`runs/real_range_eval/audit_comparison_with_calibration.json`.

No arbitrary reward was added and no verifier contract was weakened. The
measured experiment should remain a verification exercise, and learning
effectiveness must be reported as not demonstrated. To support a later
effectiveness claim, design explicit synthetic scenarios with verified
tradeoffs (including legitimate availability and false-block evidence),
broaden the controls only alongside independent verifiers, then validate
scripted and random baselines before training.
