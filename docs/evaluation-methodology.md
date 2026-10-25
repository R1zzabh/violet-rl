# Evaluation methodology

The graph simulator and Violet Ledger HTTP range are separate environments and must be reported separately. The real-range runner records confirmed foreign-read and verified-prevention outcomes from the independent verifier. Denominators are completed evaluation episodes; inconclusive episodes must be reported separately and are not silently counted as prevented or successful.

For a fair learned-versus-scripted comparison, freeze both checkpoints, reset identical fixture seeds for every policy pairing, and report per-episode raw outcomes before summary statistics. Report mean Red/Blue return, verified foreign-read rate, verified prevention rate among eligible episodes, legitimate Bob service availability, episode length, request count, rejected/uncertain rate, reward components, PPO losses/entropy, elapsed training time, and HTTP latency. Include sample size and uncertainty intervals across independent seeds.

`violet.range.evaluate_real` compares scripted/scripted, learned/scripted,
scripted/learned, and learned/learned over the same held-out seed range. It
stores raw episodes plus seeded percentile bootstrap 95% intervals for rates
and returns. Intervals with few episodes are descriptive and do not establish
statistical power. No measured policy-quality or improvement result is claimed
until this script is run with trained checkpoints. The following compact report
template can be copied for an experiment:

## Experiment report template

- **Hypothesis:**
- **Setup:** target version, fixture seeds, config, checkpoint IDs
- **Baselines:**
- **Metrics and denominators:**
- **Results:** raw file path, summary, uncertainty
- **Limitations:**
- **Conclusion:**
