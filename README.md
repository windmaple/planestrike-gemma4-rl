# Plane Strike with Gemma 4 E2B + agentic GRPO (Tunix)

Agentic RL using the environment from a previous blog["Building a reinforcement learning agent with JAX, and deploying it on Android with TensorFlow Lite"](https://blog.tensorflow.org/2022/09/building-reinforcement-learning-agent-with-JAX-and-deploying-it-on-android-with-tensorflow-lite.html) + an LLM agent, which uses Gemma 4 E2B (thinking mode on) to play the game in a multi-turn
conversation and is fine-tuned with GRPO.

## Files

| File | Purpose |
|---|---|
| `env.py` | `PlaneStrikeEnv(BaseTaskEnv)`: the original game on a 6x6 board (same plane placement code as `common.py`), capped at 23 strikes (`env.MAX_STEPS`). |
| `agent.py` | `PlaneStrikeAgent(ConversationAgentBase)`: system prompt (rules only), per-turn board message with the "don't strike tried cells" reminder, and turning the model's `strike` tool call (after the thought channel) into the env action. |
| `strike_tool.py` | The `strike` tool (`BaseTool`: `strike(cell="C5")` returns ```` ```C5``` ````), Gemma 4 tool-call parsing, and a `Gemma4ChatTemplateParser` subclass that adds the native tool declaration and renders strike results as in-turn tool responses. |
| `train_planestrike.py` | Agentic GRPO training (Tunix `GRPOLearner`, vLLM rollout, mesh (fsdp, tp) = (8, 1)). Boards are generated on the fly from seeds. Also holds runtime memory patches (sequence-chunked MLP under remat, frozen per-layer embedding table, per-step XLA scheduler options) so 34k-token sequences fit in HBM without modifying Tunix. |
| `turn_credit.py` | Turn-level GRPO advantages (per-strike discounted returns, group-normalized), over-budget trimming, truncation penalty. |

## Game, observation, reward

* Rules as in the original gym env, but on a **6x6 board** (was 8x8): one
  8-cell plane in one of 4 headings (45 distinct placements), one strike per
  turn; the game ends when all 8 cells are hit, or after **23 strikes** (the
  original example has no cap). The plane covers 8 of 36 cells; the cap is set
  so that a random policy over untried cells wins only 2.8% of the time.
* Observation each turn: result of the last strike, strikes used, and the raw
  visible board (`.` untried, `X` hit, `O` miss). Nothing derived from the
  board (no candidate cells, no hit counts, no heat map / probabilities).
* The system prompt states only the rules (board, the plane's shape and that
  it can head 4 ways, symbols, and that strikes are made with the `strike`
  tool). No strategy and no hints about symmetry. Each turn's message reminds
  the model not to strike tried cells.
* Strikes are tool calls, in Gemma 4's native format: the system prompt carries
  the tool's `<|tool>declaration:strike{...}<tool|>` (rendered by the model's
  own chat template), and a turn ends with
  `<|tool_call>call:strike{cell:<|"|>C5<|"|>}<tool_call|><|tool_response>`
  (`<|tool_response>` is a Gemma 4 stop token). The tool only wraps the cell
  in triple backticks; that output is parsed into the action. The strike result
  and the new board come back as the tool response
  (`response:strike{value:<|"|>...<|"|>}<tool_response|>`) and the model
  continues in the same turn, so a game is one model turn with one tool call
  per strike. A strike counts only if the final answer (after the thought)
  makes exactly one `strike` call; a ```` ```C5``` ```` written as text, a
  missing or second call, or an off-board cell is an invalid strike. After a
  turn that ends with `<turn|>` instead, the board comes back as a user message.
* Pre-RL baseline with the tool (32 held-out 6x6 games): game length 22.44,
  win rate 9.4%, 6.22 hits, 0.84 repeats and 0.84 invalid strikes per game;
  255 tokens per turn (p99 1.5k), 8.3k tokens per game (max 11.8k), no
  truncated games.
* Per-strike reward: hit +1, miss 0, repeated cell -1, unparseable answer -1,
  and -0.5 per strike. The trajectory reward is the sum, i.e.
  `hits - 0.5 * game_length - repeats - invalid`. Every unfinished game has
  length 23, so the step cost only separates finished games: shorter is better.

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
sequence.

Settings:

* `max_response_length = 32768` (whole game after the 424-token first prompt): about 2x the longest normal game.
* `max_tokens_per_turn = 2048`: a runaway thought loses one turn (invalid strike)
  instead of clipping the rest of the game.
* Per-turn soft overlong penalty (`--soft_tokens_per_turn 512`, DAPO-style): 0 up
  to the soft limit, then linear to -1 at the per-turn cap (2048). In earlier
  runs with soft limit 1024, late-game turns (turn >= 15, usually 7 hits with
  the last cell unresolved) grew from ~350 to ~600 tokens and looped until the
  2048 cap, and later turns copied the loops. Late turns over 700 tokens hit
  12-13% of the time vs 26% under 300, so long turns are penalized from 512.
* `episode_timeout = 6h` (Tunix default 30 min could cut long games under load).
* Clipped games are marked `"truncated": true` and counted as the cap (`max_steps`)
  in logged `game/length` metrics.
* A game cut off by the token budget is charged `-0.5` per strike it never
  made (as if they all missed), on its last turn, so the policy learns to stay
  within budget instead of the budget being shrunk.
* The final turn can overshoot the budget by a few tokens (appended suffix);
  Tunix's exact-token path raises on that, so such trajectories are trimmed to
  the budget first (`turn_credit.clip_to_budget`).


## Environment

Tested with Python 3.12.13 on a TPU v6e-8 host:

| package | version |
|---|---|
| Tunix | [`google/tunix@144717f`](https://github.com/google/tunix/commit/144717f073ebe5dfc28020e075f1b3ad9ab04db1) (2026-10-07), editable install (`google-tunix` 0.1.8) |
| JAX / jaxlib | 0.11.0 |
| Flax | 0.12.8 |
| vLLM | 0.29.0 (`vllm-tpu`, with `tpu_inference` 0.29.0) |
| libtpu | 0.0.44 |

## Running

```bash
# Train for 50 steps (default --num_steps 50; resumes from the latest checkpoint if present).
# Defaults: turn-level advantages, per-turn cap 2048, soft overlong penalty from
# 512, lr held to step 40 then cosine decay to --lr_end at --num_steps.
python train_planestrike.py \
  --run_dir runs/grpo_6x6_tool_soft512_50 \
  --learning_rate 2e-6 \
  --lr_end 2e-7
```

Metrics are logged to W&B (project `planestrike`, `--wandb_project`) whenever
`wandb` is installed and logged in; set `WANDB_MODE=disabled` to turn it off.
Per-step game stats are logged under `game/train/*` and `game/eval/*` (`length`,
`win_rate`, `hits`, `repeats`, `invalid`, and `tool_calls`: strike tool calls
per game, `tool_call_rate`: share of turns making exactly one call). The W&B
run id is saved in `<run_dir>/wandb_id.txt` so resuming a run continues the same W&B run.
