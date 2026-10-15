# Reinforcement learning

`src/rl.py` implements GRPO in its simplest form: one update per batch of rollouts, with no value network.

1. For each prompt, sample a group of rollouts.
2. Score each one by parsing its answer and comparing it with the result of executing the function.
3. Turn rewards into advantages relative to the rollout's own group (subtract the group mean, divide by the group spread).
4. Put the rollouts from groups that disagreed back through the model with gradients on, and raise the log-probability of the tokens in rollouts that beat their group.

Details that matter in practice:

- **The gradient pass scores exactly what was sampled.** It uses the same temperature and excludes the same tokens the sampler cannot emit. The difference between the two passes is logged every step as `logprob_gap` and has stayed at or below 0.0005 per token on real runs.
- **Prompts with no signal are dropped after sampling.** A group whose rollouts all scored the same has zero advantage, so it is not put through the gradient pass, and the update is averaged over the groups that were.
- **Memory is bounded by tokens.** A rollout's length is only known after it is generated, so a step's rollouts are all generated first and then split to fit a token budget, with gradients accumulated across the pieces. How many prompts are generated together is a separate setting that affects speed only. The result matches the single-pass gradient.
- **A step's rollouts are generated in one batch.** A decode pass is about 1,400 small kernel launches and costs nearly the same for 512 rows as for 128, so generation time goes with the number of batches. Decoding runs in half precision, as training does, which halves the KV cache. A `search` step of 512 rollouts takes about 30 seconds, most of it generation.
- **Peak reserved GPU memory is logged every step** as `peak_memory_gb`, for the reason given in [lessons.md](lessons.md).
- **Validation reports two numbers** on a fixed test set: `sampled_reward`, the quantity RL directly optimises, drawn under a fixed seed so runs are comparable; and greedy `pass_rate`, which is what matters in the end and lags behind.
- **`mixed_group_rate` is logged every step.** It is the share of prompts whose rollouts disagreed, and the first thing to look at when a run is flat.

