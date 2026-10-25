# Troubleshooting

- **Pytest fails during collection with missing `pettingzoo` or Gymnasium:** install `pip install -e ".[api,rl,dev]"` from `backend/` and rerun. The original host baseline in this workspace could not collect because these dependencies were absent.
- **Compose says a database password is unset:** populate both variables in `.env` with distinct disposable alphanumeric values. Do not reuse a real password.
- **`/range/health` is unavailable:** inspect `docker compose ps` and `docker compose logs range-executor violet-ledger backend`. The executor health route is unavailable until its process has contacted the local target.
- **Training poll times out:** inspect service health and the relevant run events. The runner uses bounded polling and raises rather than marking a queued action as executed.
- **Transition endpoint returns 422:** verify the v2 role observation has seven normalized features for Red or eight for Blue, the action mask length is eight for Red or four for Blue, and an accepted action has a matching executed action.
- **Checkpoint catalog mismatch:** checkpoints are role- and schema-specific. Load a Red checkpoint into Red and Blue into Blue; incompatible catalog/schema pairs are rejected.
- **Blue does not verify prevention:** only ownership hardening has the independent Alice-denied/Bob-served verifier today. The runner masks rate-limit and block actions until they have equivalent independent verification.
