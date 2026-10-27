# Results

Headline numbers are in the [README](../README.md). This page has the full runs.

### Pretraining

Three full-size runs on output prediction:

| Configuration | Starting point for SFT | SFT learning rate |
|---|---|---|
| `sft_only` | Random init | 3e-4 |
| `base` | Pretrained | 1e-4 |
| `base_lr3` | Pretrained, the same checkpoint as `base` | 3e-4 |

Everything else is shared: the tokenizer (byte-identical), the architecture, the 2.45M SFT rows, 10,000 SFT steps of 32k tokens, and the RL settings and test prompts. `sft_only` and `base_lr3` differ only in their starting weights. `base` and `base_lr3` differ only in learning rate.

**Pretraining** ran for 10,000 steps of 131k tokens, 1.31B tokens in all, over 21 of StarCoder's 59 Python shards. That is about 21% of one epoch of the loaded data and about 6 tokens per active parameter. Loss fell from 10.5 at the start to a validation loss of 1.465 at the end, and was still falling by about 0.007 per 500 steps at the end. It took about 13.5 hours of training on one RTX 5070 Ti.

**Pass rate** on 3,022 held-out prompts over 780 functions, from `scripts/evaluate.py`. The functions come from a seed no stage trains on, and only those on the test side of every stage's split are kept. Gaps are paired on the same prompts, with 95% intervals from a bootstrap over functions (`scripts/compare_evals.py`):

| | `base_lr3` | `sft_only` | `base` |
|---|---|---|---|
| After SFT | **89.3%** | 86.3% | 81.3% |
| After 100 RL steps | **89.3%** | 86.1% | 73.6% |

| Comparison, after SFT | Gap | 95% interval |
|---|---|---|
| Pretrained vs from scratch, both at 3e-4 | +2.9 | +1.7 to +4.2 |
| Pretrained at 3e-4 vs at 1e-4 | +7.9 | +6.5 to +9.5 |
| From scratch at 3e-4 vs pretrained at 1e-4 | +5.0 | +3.5 to +6.6 |

| Effect of RL, same model before and after | Gap | 95% interval |
|---|---|---|
| `base_lr3` | +0.1 | −0.8 to +0.9 |
| `sft_only` | −0.2 | −1.4 to +0.9 |
| `base` | −7.7 | −9.3 to −6.2 |

**Pass rate during RL**, from the RL validations on a smaller set of 375 prompts over 95 functions. Step 0 is the SFT model before any RL:

| RL step | `base_lr3` | `sft_only` | `base` |
|---|---|---|---|
| 0 | **90.1%** | 84.8% | 79.7% |
| 25 | 89.1% | 86.7% | 77.3% |
| 50 | 88.5% | 85.9% | 76.3% |
| 75 | 89.3% | 86.1% | 77.9% |
| 100 | **90.9%** | 86.1% | 77.3% |

**SFT validation loss** at matching steps:

| SFT step | `base_lr3` | `sft_only` | `base` |
|---|---|---|---|
| 200 | 0.104 | 0.309 | 0.122 |
| 1,000 | 0.040 | 0.071 | 0.051 |
| 2,200 | 0.022 | 0.033 | 0.033 |
| 4,000 | 0.017 | 0.018 | 0.023 |
| 6,400 | 0.011 | 0.013 | 0.018 |
| 10,000 | **0.0068** | 0.0101 | 0.0138 |

**Pretraining helps, by about 3 points of pass rate and a third of the final validation loss.** The pretrained run leads from the first validation. Its lead narrowed to almost nothing around step 4,000, which at the time looked like the from-scratch model catching up, then opened again and held to the end. Judged at the halfway point, the conclusion would have been that pretraining gives only a head start.

The 375-prompt validation set first put the gap at 5.3 points. With only 95 functions behind those prompts, and errors clustering by function, that set's real uncertainty was well over 2 points either way. On the larger set the gap is 2.9 points with an interval that stays clear of zero: pretraining helps, by less than the first measurement suggested.

The interval covers test-set noise only. Each arm is one training run, so run-to-run variation from the seed is not measured.

**At 1e-4 the same pretrained weights finish 8 points lower.** This run led the from-scratch run for the first 1,800 steps and was level by step 2,000. From there it fell behind and stayed behind. With this schedule the rate decays as the square root of the step, so by the last quarter of training it was taking steps of about 1.5e-5. Its validation loss moved only from 0.016 to 0.014 over those 2,400 steps. A lower rate is the usual advice for fine-tuning a pretrained model, to avoid overwriting what it learned. Here protecting the pretrained weights cost more than it saved, probably because the task needs exact arithmetic, which pretraining on code does little to teach.

**RL did not improve any of the three, and it made `base` worse.** On the larger set, RL left the two 3e-4 models within a point of where they started, with tight intervals either side of zero. It cost `base` 7.7 points: 301 prompts it had solved before RL were lost and 68 gained. The 375-prompt validations showed only a 2.4-point drop for `base`, inside their noise, so the damage went unnoticed until the larger evaluation. The training reward confirms it: `base`'s mean reward over the first 20 RL steps was 0.772 and over the last 20 was 0.730, while the 3e-4 models held steady.

The share of prompts with signal per step was 18% for `base_lr3`, 20% for `sft_only` and 28% for `base`, which tracks how often each model gets prompts wrong. The weakest model had the most to learn from and got worse. Why is not yet understood. The RL settings are identical across the three runs, so the difference lies in the starting model.

**RL generation is now about five times faster.** Each step's 512 rollouts are generated in one batch, so 100 RL steps with five validations took 31 to 35 minutes. The `sft_only` run, which generated 4 prompts at a time, took 2 hours 40 minutes.

### Output prediction and input prediction from `sft_only`

The remaining results are on the full-size model with no pretraining (the `sft_only` and `input` configurations), on the easy tier, measured on the same 375 held-out prompts.

| Task | After SFT | After 100 RL steps | Prompts with signal per RL step |
|---|---|---|---|
| Output prediction: what does `f(args)` return? | 84.8% | 86.1% | about 1 in 5 |
| Input prediction: what arguments make `f` return this? | 49.3% | 48.0% | about 1 in 3 |

Figures are greedy pass@1. **SFT works; on these two tasks RL has not improved on it.** Both RL changes are within the noise of this test set, which moved by up to 3 points between validations with no trend.

A third task, input prediction as a search, is where RL does help. It is measured on a larger test set, 2,024 held-out prompts over 527 functions, so it has its own table:

| Task | After SFT | After 300 RL steps | Prompts with signal per RL step |
|---|---|---|---|
| Search: guess arguments, trace them, try again | 36.4% | 41.2% | about 5 in 6 |

A run of the full pipeline with pretraining (`base`) has not been done.

### Output prediction

The `sft_only` configuration is the full-size model with no pretraining, trained with SFT for 10,000 steps. It reaches 84.8% greedy pass@1 on 375 held-out prompts (95 functions it has never seen). Its failures are almost all a single wrong sum inside an otherwise correct trace.

RL was then run on it for 100 steps of 64 prompts × 8 rollouts, training only on prompts whose rollouts disagreed (about 12 of the 64 per step):

| RL step | Sampled reward | Greedy pass@1 |
|---|---|---|
| 0 | 83.5% | 84.8% |
| 25 | 84.9% | 86.7% |
| 50 | 84.8% | 85.9% |
| 75 | 84.1% | 86.1% |
| 100 | 83.9% | 86.1% |

**RL did not meaningfully improve this model.** Greedy pass@1 ends 1.3 points up (5 prompts) and sampled reward 0.4 points up. Every validation after the start is above the baseline, so a gain of about a point may be real, but it is too small to separate from noise on this test set.

An earlier run with a smaller test set (93 prompts over 24 functions) and no filtering of uniform groups gave the same picture: an early rise that did not hold, ending within noise of where it started.

Both runs were stable, with no memory failures and the gradient pass matching the sampler to about 0.0001 per token.

**Why.** This task is set up so that SFT has every advantage. The generator supplies unlimited exact traces, so SFT gets a correct target for every token. RL gets one bit per rollout, and only on the roughly one prompt in five where rollouts disagree. What is left to fix is occasional arithmetic slips, which a right-or-wrong signal on the whole answer is poorly placed to find. RL should matter more where many answers are correct and imitation has no single right target, which is the case for input prediction.

### Input prediction

The model is given a function and a result and has to find arguments that produce it. Any arguments that work are accepted, checked by running the function:

```
def f(a, b):
    b = ((a + 7) * (0 * 7))
    b = b
    b = (4 + a)
    b = ((1 - 0) - (2 + b))
    b = 8
    return a, b

# What arguments make f return (7, 8)?
<|think|>b = ((a + 7) * (0 * 7))  ->  b = 0
b = b
b = (4 + a)  ->  b = 11
b = ((1 - 0) - (2 + b))  ->  b = -12
b = 8  ->  b = 8
return a, b<|/think|><|answer|>7, 1<|/answer|>
```

Here `b` is overwritten before it is returned, so any value for it is correct. The training data used `7, -1`; the model answered `7, 1`, which is just as right. SFT can only train toward the one arbitrary choice in the data, while the RL reward accepts whatever the function confirms. That is the reason to expect RL to help on this task.

The `input` configuration fine-tunes the `sft_only` model on this task for 1,000 SFT steps, then runs the same RL settings as above:

| RL step | Sampled reward | Greedy pass@1 |
|---|---|---|
| 0 | 47.1% | 49.3% |
| 25 | 44.8% | 46.4% |
| 50 | 47.1% | 49.3% |
| 75 | 48.8% | 50.4% |
| 100 | 47.3% | 48.0% |

**RL did not improve this model either.** It ends within a point or so of where it started, having moved up and down by about 3 points on the way.

This run removes the two easy explanations for the output-prediction result. The model starts at 49%, so there is plenty of room, and about a third of prompts per step had rollouts that disagreed, so there is plenty of signal.

What the model actually does explains more. On the 375 test prompts after SFT:

| Outcome | Prompts |
|---|---|
| Correct, with different arguments from the training data's | 149 |
| Correct, with the same arguments as the data | 36 |
| Wrong | 190 |

Four in five correct answers use arguments other than the data's, so there really are many valid answers and the checker accepts them. But the failures show the limit of the format. The trace runs forwards from arguments the model must already have chosen, so the hard part of the task, working out which arguments lead to the target, has to happen in one pass before anything is written. Where that needs a chain of operations undone, the model cannot do it. It writes a plausible trace that does not end at the target and reports arguments that do not match the trace:

```
def f(a, b):
    a += (b + -5)
    b = (0 - a)
    a *= -4
    a = ((a - a) - (4 * -3))
    return a, b

# What arguments make f return (12, -2)?
<|think|>a += (b + -5)  ->  a = -1
b = (0 - a)  ->  b = -1
a *= -4  ->  a = 4
a = ((a - a) - (4 * -3))  ->  a = 12
return a, b<|/think|><|answer|>0, 0<|/answer|>
```

The trace ends at `b = -1` where the target needs `-2`, and `0, 0` would not produce that trace. The prompts it gets right are the ones where each argument can be read off the target, is overwritten, or is one operation away.

So the open question is whether RL fails here because 100 steps is too little, with each prompt seen once, or because this format gives the model no room to reason its way to an answer. A format that lets it try a guess, see the trace miss, and try again would test the second.

### Search

The `search` configuration is that format. The question is the same as input prediction, but the model's working is a series of attempts: guess arguments, trace them, compare the result with the target, and try again.

```
def f(a, b):
    b = (b - (-1 * 5))
    b = -4
    a = (2 * a)
    b = ((3 - 0) + (a * a))
    b = b
    return b, a

# What arguments make f return (39, 6)?
<|think|>try 9, 0
...
return b, a  ->  (327, 18)
try 6, -3
...
return b, a  ->  (147, 12)
try 3, -1
...
return b, a  ->  (39, 6)<|/think|><|answer|>3, -1<|/answer|>
```

SFT teaches the shape of an attempt and that the search ends on the one that reaches the target. Its examples show up to two missed guesses before the hit, and the arguments of those misses are left out of the loss, so the model is never trained to guess wrongly. Choosing the next guess after a miss has no single right answer in the data, and that is what RL is left to learn.

The RL reward is graded, and every part of it is decided by running the function: 1.0 for an answer that reaches the target, partial credit for a wrong one, credit for attempts whose trace matches real execution, and a small cost per extra attempt. Success is still measured by the plain right-or-wrong check.

The `search` model fine-tunes the `sft_only` model for 1,000 SFT steps, then runs 300 RL steps of 32 prompts × 16 rollouts:

| RL step | Greedy pass@1 | Solved on first attempt | Sampled reward | Attempts | Faithful traces |
|---|---|---|---|---|---|
| 0 | 36.4% | 31.4% | 0.334 | 2.26 | 87.0% |
| 60 | 40.4% | 33.4% | 0.383 | 2.23 | 87.7% |
| 120 | 40.7% | 32.6% | 0.402 | 2.30 | 91.0% |
| 180 | 41.7% | 33.2% | 0.407 | 2.31 | 89.6% |
| 240 | 41.2% | 32.1% | 0.401 | 2.26 | 90.9% |
| 300 | 41.2% | 33.2% | 0.408 | 2.38 | 89.6% |

**RL improved this model by about 5 points.** With 2,024 test prompts the pass rate's standard error is about 1 point, and it is above the starting point at every validation. Most of the gain came in the first 60 steps, and it had levelled off by step 180.

The graded reward is what gives RL something to work with. About 85% of prompts per step had rollouts that scored differently, against a fifth to a third on the other two tasks.

What changed is how the model searches. On 96 test prompts decoded greedily before and after RL:

| | Before RL | After RL |
|---|---|---|
| Failures that repeat a guess already tried | 39 of 63 | 5 of 60 |
| Failures that stop at exactly three attempts | 54 of 63 | 22 of 60 |
| Failures with four or more attempts | 4 | 21 |
| Successes on the first attempt | 29 of 33 | 33 of 36 |

Before RL the model stops after three attempts whether or not it has reached the target, because SFT never showed it more than two misses, and it often answers with a guess it has just watched fail. After RL it has almost stopped repeating guesses and it keeps searching. The last row is the limit: nearly every success is still a first-attempt hit. The later guesses are no longer repeats, but they are not yet informed by the miss before them.

That run gave a wrong answer partial credit only for returned values that were exactly right. Two graded alternatives exist, chosen by `train.rl.reward_options.partial_credit`:

- Grading the answer by how near its result came to the target was tried once, at three times the learning rate, and that run collapsed (below). The answer is the last attempt, so this version can also score stopping early above searching on, and it was dropped.
- `closest` grades the rollout by its best guess instead of its last, so that searching on is never penalised. It has not been evaluated yet.

**Tripling the learning rate makes RL worse.** At 3e-5 the model stops searching within about 30 steps, with either reward:

| Run, at step 60 | Greedy pass@1 | Attempts | Solved on first attempt |
|---|---|---|---|
| Start | 36.4% | 2.26 | 31.4% |
| Original reward, 1e-5 | 40.4% | 2.23 | 33.4% |
| Original reward, 3e-5 | 33.7% | 1.55 | 31.0% |
| Answer-nearness reward, 3e-5 | 32.7% | 1.35 | 31.1% |

First-attempt accuracy is unchanged in the collapsed runs, so the whole loss is the later attempts the model no longer makes. The plateau of the first run was therefore not caused by steps that were too small.

### Earlier trials

These were run on output prediction with an older checkpoint (older answer format, 200 SFT steps, 15% greedy pass@1), so they say how the loop behaves, not how good a model is.

- **On the full prompt set, 60 steps changed nothing.** Sampled reward went 13.4% → 13.1% and greedy pass@1 15.2% → 14.1%, both within noise.
- **The cause was a lack of signal, not a broken update.** GRPO only learns from a prompt when its rollouts disagree. That model answered most prompts the same way all 8 times, so on average only 2.7 of 16 prompts per step carried any gradient.
- **On prompts where rollouts do disagree, it learns fast.** Training on 16 such prompts took sampled reward from 30% to 88% and greedy pass@1 from 69% to 94% in 15 steps. This is deliberate overfitting on the training prompts; it shows the update works on the real task and says nothing about generalisation.

