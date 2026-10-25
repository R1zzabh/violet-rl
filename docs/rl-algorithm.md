# RL algorithm

## Environment and role views

The real-range environment is a seeded Violet Ledger HTTP service, not the graph simulator. A training episode resets synthetic fixtures through the typed control queue. Red then makes a bounded sequence of catalogued HTTP requests; Blue chooses whether to leave the target unchanged or apply the existing ownership-hardening control. A local HTTP process is a real application process, but it is not production deployment or evidence of real-world cybersecurity effectiveness.

`violet.range.observation.v2` uses normalized values in `[0,1]`. Red receives seven features, in order: step fraction, remaining budget fraction, own Alice authentication state, own Bob authentication state, own last-request success, own last HTTP status divided by 599, and its own independently verified objective count capped at one. Blue receives eight features: step fraction, remaining budget fraction, observed request fraction, observed failed-request fraction, public service health, whether an independent verifier has confirmed a foreign document read, whether prevention has been independently verified, and applied-control fraction. Red receives no Blue control state or verifier internals. Blue receives no hidden Red policy state or future action.

The Red catalog has eight entries: discover home/login/document, authenticate with synthetic Alice/Bob fixture references, access Alice's own document, attempt Alice's authenticated access to Bob's document, and check public target health. A mask disables document access until Red has its own Alice session. The Blue catalog has no-op, ownership hardening, rate-limit Alice, and block Alice. In this first training runner only no-op and ownership hardening are enabled: the existing verifier can establish the latter's success and legitimate Bob availability, but it has no independent verification contract for rate limiting or blocking yet.

## PPO model and update

Each role owns a distinct two-head MLP: a categorical policy over that role's finite action catalog and a scalar critic estimating expected discounted return. Training samples actions; evaluation uses argmax and does not alter model or optimizer state. A rollout contains accepted, actually executed actions only. HTTP failures are recorded in the event timeline but are not treated as successful executions or included in PPO updates.

For transition `t`, the temporal difference residual is

`delta_t = r_t + gamma * (1 - terminated_t) * V(s_{t+1}) - V(s_t)`.

GAE is `A_t = delta_t + gamma * lambda * (1 - episode_end_t) * A_{t+1}`. A genuine terminal transition has no bootstrap. A time-limit truncation bootstraps its final observation but stops the GAE trace at that episode boundary. Returns are `A_t + V(s_t)`.

The policy ratio is `rho_t = exp(log pi_new(a_t|s_t) - log pi_old(a_t|s_t))`. PPO minimizes the negative clipped objective

`-mean(min(rho_t A_t, clip(rho_t, 1-epsilon, 1+epsilon) A_t))`.

The critic uses mean squared error between predicted values and returns. Entropy is subtracted from total loss to preserve exploration. Gradients are clipped before Adam updates. Defaults are `gamma=.99`, `lambda=.95`, clip epsilon `.2`, Adam learning rate `3e-4`, value coefficient `.5`, entropy coefficient `.01`, gradient norm `.5`, 4 optimization epochs, and minibatches of 32. These are conventional small PPO defaults for a low-dimensional CPU task, not tuned claims.

The runner alternates a Red rollout and one Blue control decision per episode, then updates their independent parameters. Policy checkpoints include role, catalog, schema, optimizer, config, and policy version. Evaluation calls deterministic inference and does not run `update()`.

## Evidence and claims

The independent verifier, not the reward calculator, decides whether Alice returned Bob's seeded canary and whether ownership hardening denied Alice while serving Bob. A finding is tied to the executor request and target event IDs. Training results remain synthetic-range results; compare them with baselines over identical held-out fixture seeds before making policy-quality claims.
