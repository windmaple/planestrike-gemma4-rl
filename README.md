# Plane Strike with Gemma 4 E2B + agentic GRPO (Tunix)

Replicates the TensorFlow blog post
["Building a reinforcement learning agent with JAX, and deploying it on Android with TensorFlow Lite"](https://blog.tensorflow.org/2022/09/building-reinforcement-learning-agent-with-JAX-and-deploying-it-on-android-with-tensorflow-lite.html)
with an LLM agent: Gemma 4 E2B (thinking mode on) plays the game in a multi-turn
conversation and is fine-tuned with GRPO in the style of
`tunix/examples/frozenlake`. The metric is the same as the original
`training_jax.py`: **raw game length** (number of strikes until all 8 plane
cells are hit), which should go down during training.

## Files

| File | Purpose |
|---|---|
| `env.py` | `PlaneStrikeEnv(BaseTaskEnv)`: the original 8x8 game (same plane placement as `common.py`), capped at 20 strikes (`env.MAX_STEPS`). Logs every finished game to `games.jsonl`. |
| `agent.py` | `PlaneStrikeAgent(ConversationAgentBase)`: system prompt (rules only), per-turn board message with the "don't strike tried cells" reminder, and strike parsing from the final answer (after the thought channel). |
| `train_planestrike.py` | Agentic GRPO training (Tunix `GRPOLearner`, vLLM rollout, mesh (fsdp, tp) = (8, 1)). Boards are generated on the fly from seeds. |
| `watchdog.py` | Runs training and restarts it on OOM / crash / hang (see below). |
| `turn_credit.py` | Turn-level GRPO advantages (per-strike discounted returns, group-normalized), over-budget trimming, truncation penalty. |
| `memory_patches.py` | Runtime patch: sequence-chunked MLP under remat so 34k-token sequences fit in HBM. Tunix files are not modified. |
| `baseline_eval.py` | Random policy and pre-RL Gemma 4 baselines (same env/agent/parser/token construction as training). |
| `plot_game_length.py` | Plots raw game length per training game + smoothed curve (run with `.plotenv/bin/python`). |
| `mem_probe.py` | Compiles the trainer forward+backward at several lengths and prints XLA memory use. |

## Game, observation, reward

* Rules as in the original gym env: 8x8 board, one 8-cell plane in one of 4
  headings, one strike per turn; the game ends when all 8 cells are hit, or
  after **20 strikes** (the original example has no cap; the cap keeps
  rollouts short).
* Observation each turn: result of the last strike, strikes used, and the raw
  visible board (`.` untried, `X` hit, `O` miss). Nothing derived from the
  board (no candidate cells, no hit counts, no heat map / probabilities).
* The system prompt states only the rules (board, the plane's shape and that
  it can head 4 ways, symbols, answer format). No strategy and no hints about
  symmetry. Each turn's message reminds the model not to strike tried cells.
* Per-strike reward: hit +1, miss 0, repeated cell -1, unparseable answer -1,
  and -0.5 per strike. The trajectory reward is the sum, i.e.
  `hits - 0.5 * game_length - repeats - invalid`. Every unfinished game has
  length 20, so the step cost only separates finished games: shorter is better.

## Credit assignment: turn-level GRPO

Tunix's agentic GRPO gives every token of a game the same advantage (from the
episode's total reward). A first run with that setting (30-strike cap, 30 steps)
barely moved: train game length stayed ~28.5 while entropy fell 0.33 -> 0.19
and thinking grew 7.9k -> 13k tokens per game. With ~30 turns sharing one
advantage, the outcome is dominated by luck in finding the plane.

The original JAX example rewards each strike with discounted returns
(`common.compute_rewards`, `GAMMA = 0.5`). `turn_credit.py` does the same
inside GRPO, still critic-free and group-relative:

    G_t = r_t + 0.5 * G_{t+1}                      per turn of each game
    A_t = (G_t - mean) / std                       over all turns of the 8 games on one board

and every token generated in turn t gets `A_t` (Tunix's loss already accepts
`[batch, length]` advantages). Selected with `--advantage_mode turn` (default);
`--advantage_mode trajectory` is the stock behavior.

## Context budget (no clipping)

Thinking is kept in the conversation (Tunix's Gemma 4 parser does not strip
past thoughts), and later turns replay the recorded token ids
(`exact_token_continuity=True`), so the sampler and the trainer see the same
sequence. Measured on the base model (64 games, `results/baseline_gemma4_e2b.jsonl`):

* ~280 generated tokens per turn on average, p99 1.5k; ~12k tokens per game (p90 14.5k, max 16.6k excluding one runaway).
* One turn ran away to 23.6k tokens and used up its game's budget.

Settings:

* `max_response_length = 32768` (whole game after the 424-token first prompt): about 2x the longest normal game.
* `max_tokens_per_turn`: a runaway thought loses one turn (invalid strike)
  instead of clipping the rest of the game. Started at 4096 (longest normal
  base-model turn was 3.5k). Around step 40 the policy began producing
  repetition-loop turns ("I will try E6 neighbor E5 ..." until the cap), which
  then get copied by later turns; they were 14.5% of turns in games that hit
  the 32k budget, vs 1.1% elsewhere (normal turns: p90 ~800 tokens). From
  step 45 the cap is 2048, so a loop wastes half as much context.
* Per-turn soft overlong penalty (`--soft_tokens_per_turn`, DAPO-style): 0 up
  to the soft limit, then linear to -1 at the per-turn cap (1536/4096 from step
  40, 1024/2048 from step 45).
* `episode_timeout = 6h` (Tunix default 30 min could cut long games under load).
* Clipped games are logged with `"truncated": true` in `games.jsonl` and are
  counted as the cap (not finished) in plots and summaries.
* A game cut off by the token budget is charged `-0.5` per strike it never
  made (as if they all missed), on its last turn, so the policy learns to stay
  within budget instead of the budget being shrunk.
* The final turn can overshoot the budget by a few tokens (appended suffix);
  Tunix's exact-token path raises on that, so such trajectories are trimmed to
  the budget first (`turn_credit.clip_to_budget`).

## Memory and OOM handling

Every trajectory is padded to 1024 + 32768 tokens for training, one sequence
per chip (micro-batch = one board's 8 games over fsdp=8). Out of the box,
Tunix's actor train step needed **33.7 GB of temporaries per chip** at that
length (v6e HBM is 31.2 GB), before vLLM's share. Three changes, all in this
folder (no Tunix edits), measured with `mem_probe_trainstep.py`, which
compiles Tunix's real train step (GSPO loss + AdamW) at a given length:

| chunked MLP | PLE frozen | latency-hiding scheduler | temporaries @ 33.8k tokens |
|---|---|---|---|
| yes | no | on | 33.7 GB (real run, OOM) |
| no | yes | on | 32.5 GB (probe, OOM) |
| yes | yes | on | 28.6 GB (probe) / 31.7 GB (real run, OOM) |
| yes | yes | **off** | **17.4 GB** (probe) |

* The per-layer embedding (PLE) table is 262144 x 8960 (~2.35B of the
  model's 4.8B params). It is frozen (kept in bf16, exactly the released
  weights); all transformer layers, norms and the tied token embedding are
  trained. Weight sync to vLLM still sends it (unfiltered model state).
* The latency-hiding scheduler overlapped layers and kept many layers'
  activations alive at once. It is turned off only for the trainer's jitted
  step (`memory_patches.compile_train_step_with_options` passes
  `compiler_options` to the `jax.jit` behind Tunix's `nnx.jit`); vLLM's
  programs keep it. `--latency_hiding_scheduler_off global` uses the
  process-wide `LIBTPU_INIT_ARGS` flag instead (same 17.4 GB).
* The reference model is not loaded because `beta = 0`.
* `--vllm_hbm_utilization` caps vLLM's *total* HBM use including the
  trainer weights already resident, so lowering it is what frees room for the
  trainer. 0.23 leaves vLLM ~255k KV-cache tokens.
* `--max_concurrency 24` games in flight: their contexts fit in the KV cache
  (peak ~85%), so vLLM's prefix cache keeps each game's history between turns
  (~93% hit rate). With all 64 games in flight the cache thrashed (99.5% used,
  24% hit rate, every turn re-prefilled).
* Note: Tunix applies an optimizer update after every micro-batch (one board's
  group of 8 games), i.e. 8 updates per training step, as in the frozenlake recipe.

Full fp32 actor + Adam checkpoints (~34 GB) go to `/dev/shm/planestrike/<run>/ckpt`
because the root disk has ~28 GB free. They survive process restarts, not VM reboots.

OOM handling (`watchdog.py` + hooks in `train_planestrike.py`):

1. In the training process, `sys.excepthook` / `threading.excepthook` hard-exit
   on any uncaught exception in any thread (exit code 42 if it is an OOM), so a
   dead trainer or vLLM thread can't leave the job hanging. The traceback goes
   to `logs/fatal_<pid>.txt`.
2. The watchdog starts each attempt in its own session and treats it as failed
   on non-zero exit, on an OOM pattern in the log (`RESOURCE_EXHAUSTED`, ...)
   even if the process keeps running, or on no log / game progress for 45 min.
3. It kills the whole session (vLLM's EngineCore child holds `/dev/vfio` if
   orphaned), waits until the TPU is free, lowers `--vllm_hbm_utilization` by
   0.02 after an OOM, and restarts from the latest checkpoint.

## Running

```bash
# Baselines
python baseline_eval.py --policy random --num_games 2000 --out results/baseline_random.jsonl
python baseline_eval.py --policy model --num_games 64 --out results/baseline_gemma4_e2b.jsonl

# Training (background, auto-restart; resumes from the latest checkpoint).
# Defaults: turn-level advantages, per-turn cap 2048, soft overlong penalty from
# 1024, lr schedule 3e-6 -> 5e-6 at step 40 -> 2.5e-6 at step 65 (--lr_schedule).
setsid nohup python watchdog.py --run_dir runs/grpo_turn20_70 -- --num_steps 70 --max_to_keep 4 > logs/grpo_turn20_70_watchdog.log 2>&1 &
# Stop: kill <watchdog_pid from runs/grpo_turn20_70/watchdog_status.json>

# Plot + per-step summary table
.plotenv/bin/python plot_game_length.py --run_dir runs/grpo_turn20_70
tensorboard --logdir runs/grpo_turn20_70/tensorboard   # games/train/game_length
```

## Results

### Current run: 20-strike cap (`runs/grpo_turn20`)

![game length](results/game_length_grpo_turn20.png)

120 steps (8 boards x 8 games each), ~7.8k training games. Game length =
strikes until all 8 plane cells are hit, capped at 20 (unfinished games count
as 20). Repeat = re-striking a tried cell; invalid = no parseable strike
(almost always a turn that hit the 2048-token cap).

| policy | game length (95% CI) | win rate (<=20) | hits | repeat | invalid |
|---|---|---|---|---|---|
| random over untried cells (2000 games) | 20.00 | 0.0% | 2.51 | 0 | 0 |
| Gemma 4 E2B before RL, standalone baseline (64 games) | 19.83 | 4.7% | 4.19 | 0.28 | 0.44 |
| held-out eval boards, step 0 (64 games) | 19.75 +- 0.30 | 6% | 4.16 | 0.36 | 0.16 |
| held-out eval boards, step 70 (best, 64 games) | **16.83 +- 0.64** | **72%** | 7.41 | 0.03 | 0.02 |
| held-out eval boards, step 110 (last, 64 games) | 18.61 +- 0.45 | 50% | 7.00 | 0.36 | 0.09 |
| train boards, steps 0-4 (320 games) | 19.82 +- 0.12 | 4% | 4.60 | 0.70 | 0.17 |
| train boards, steps 65-74 (640 games) | 17.62 +- 0.21 | 56% | 6.65 | 0.05 | 0.06 |
| train boards, steps 110-119 (640 games) | 18.33 +- 0.17 | 53% | 7.00 | 0.30 | 0.14 |

Held-out eval every 10 steps (steps 0..110): 19.75, 19.53, 19.22, 18.39, 18.17,
17.30, 17.33, 16.83, 17.84, 18.11, 18.64, 18.61.

Game length fell steadily to ~17 by steps 50-70, then drifted back up to ~18.3-18.6
over steps 70-100 (the eval rose three times in a row; training games 17.3 -> 17.9).
Lowering lr to 1e-6 at step 100 stopped the drift but did not reverse it. Play
stayed clean throughout (repeat/invalid < 0.4 per game; 2 of ~7.8k training games
hit the token budget). With a 20-strike cap a game is only shorter than 20 if
all 8 cells are found, so the metric moves mostly through the win rate.

lr: `--lr_schedule 0:3e-6,40:5e-6,65:2.5e-6`, then 1e-6 from step 100
(`runs/grpo_turn20/NOTES.txt`). Checkpoints kept: steps 95-120
(`/dev/shm/planestrike/grpo_turn20/ckpt/actor/`, RAM disk); the best eval
(step 70) was rotated out before `--max_to_keep` was raised to 6.

### Previous run: 24-strike cap (artifacts deleted)

120 steps. Held-out eval went from 22.59 strikes / 16% wins (step 0) to 17.61 /
91% (steps 100 + 110); random 24.00 / 0%, pre-RL model 23.19 / 9.4%. lr 3e-6
plateaued at ~20.6 from step 30; 5e-6 from step 40 reached ~18.5 but entropy
collapsed (0.39 -> 0.13) over steps 65-78, so the run was rolled back to step 70
and finished at 2.5e-6. That path is now the default `--lr_schedule`.

### Previous run: 30-strike cap (artifacts deleted)

Same code with `MAX_STEPS = 30`, stopped after step 100. Held-out eval went
from 29.33 strikes / 11% wins (step 0) to 17.91 / 97% (step 100); random was
30.00 / 0.2%, the pre-RL model 28.47 / 17%. Lessons that shaped the current
defaults (all now built in):

* Trajectory-level advantages barely learned (~29 -> 28.5 in 30 steps);
  turn-level advantages (`turn_credit.py`) did.
* An episode can overshoot the token budget by a few tokens; such trajectories
  are trimmed (`clip_to_budget`) instead of crashing the job.
* Games cut off by the token budget are charged for the strikes they never made.
* After a checkpoint resume, Tunix had vLLM sampling from the base weights
  until the first sync; restored weights are now pushed to vLLM at startup.
* Around step 40 the policy started producing repetition-loop turns that later
  turns copied; the per-turn cap is 2048 with a soft overlong penalty from 1024.
* lr: 3e-6 plateaued at ~21 strikes; 5e-6 from step 65 reached ~18.5; lowered to
  2.5e-6 at step 90 when entropy fell (0.30 -> 0.25) and thinking grew (10k -> 19k tokens/game).
