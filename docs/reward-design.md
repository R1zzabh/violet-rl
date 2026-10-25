# Reward design

Rewards are assigned from completed executor responses and independent verifier outcomes. They are recorded with named components in each hash-linked `learning_transition` event.

| Role / component | Value | Source |
|---|---:|---|
| Red verified foreign document read | +1.0 | `verify_document_access` confirms the seeded Bob canary was returned to Alice |
| Red successful Alice setup authentication | +0.05 | The typed executor reports HTTP 200 and the authenticated synthetic principal is Alice |
| Red unsuccessful request | -0.1 | Executor response is not completed HTTP 200 |
| Blue verified prevention | +1.0 | Existing control verifier confirms Alice receives 403 after hardening and Bob still receives 200 |
| Blue confirmed attack left unprevented | -1.0 | Independent Red finding was confirmed and Blue did not establish verified prevention |

Blue receives zero when neither a verified prevention nor a confirmed foreign read occurred. A control acknowledgement alone earns no prevention reward. The verification probes use synthetic Alice/Bob credentials through the same typed executor and occur after the control acknowledgement. They do not read target database internals.

Bob authentication and unrelated successful HTTP requests receive no positive Red shaping reward. The reward is deliberately small and transparent for a demonstration. It is not tuned and does not establish optimality. Rejected or uncertain execution records are retained for audit and excluded from policy learning.
