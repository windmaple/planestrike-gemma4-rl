"""Agentic GRPO on Plane Strike with Gemma 4 E2B (thinking mode) on TPU v6e-8."""

import argparse
import json
import logging
import os
import sys
import time

arg_parser = argparse.ArgumentParser(description=__doc__)
arg_parser.add_argument("--run_dir", required=True)
arg_parser.add_argument("--ckpt_dir", default="",
                        help="Default: /dev/shm/planestrike/<run name>/ckpt "
                        "(a full fp32 actor+Adam checkpoint is ~60GB).")
arg_parser.add_argument("--num_steps", type=int, default=100)
arg_parser.add_argument("--batch_size", type=int, default=8, help="Boards per step.")
arg_parser.add_argument("--num_generations", type=int, default=8, help="Games per board.")
arg_parser.add_argument("--learning_rate", type=float, default=3e-6,
                        help="Peak lr, held until --lr_decay_start.")
arg_parser.add_argument("--lr_decay", default="cosine", choices=["cosine", "none"])
arg_parser.add_argument("--lr_decay_start", type=int, default=40, help="Training step.")
arg_parser.add_argument("--lr_end", type=float, default=3e-7)
arg_parser.add_argument("--lr_schedule", default="")
arg_parser.add_argument("--b1", type=float, default=0.9)
arg_parser.add_argument("--b2", type=float, default=0.95)
arg_parser.add_argument("--max_grad_norm", type=float, default=1.0)
arg_parser.add_argument("--beta", type=float, default=0.0)
arg_parser.add_argument("--epsilon", type=float, default=0.003)
arg_parser.add_argument("--epsilon_high", type=float, default=0.005)
arg_parser.add_argument("--loss_algo", default="gspo-token")
arg_parser.add_argument("--advantage_estimator", default="rloo",
                        help="Used only with --advantage_mode trajectory.")
arg_parser.add_argument("--advantage_mode", default="turn", choices=["turn", "trajectory"],
                        help="turn: per-strike discounted returns, group-normalized (turn_credit.py).")
arg_parser.add_argument("--turn_gamma", type=float, default=0.5,
                        help="Discount for --advantage_mode turn (original example: GAMMA = 0.5).")
arg_parser.add_argument("--temperature", type=float, default=0.7)
arg_parser.add_argument("--max_steps_per_game", type=int, default=23,
                        help="Must match env.MAX_STEPS (also stated in the system prompt).")
arg_parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="The first prompt is 436 tokens; Tunix pads it to this length.")
arg_parser.add_argument("--max_response_length", type=int, default=11776,
                        help="Token budget for a whole game after the first prompt "
                        "(longest game seen: 12.2k; 0.4%% of early games are longer).")
arg_parser.add_argument("--max_tokens_per_turn", type=int, default=2048)
arg_parser.add_argument("--soft_tokens_per_turn", type=int, default=512,
                        help="Turns longer than this get a linear penalty up to "
                        "--turn_length_penalty at --max_tokens_per_turn (0 disables).")
arg_parser.add_argument("--turn_length_penalty", type=float, default=1.0)
arg_parser.add_argument("--episode_timeout", type=float, default=6 * 3600)
arg_parser.add_argument("--vllm_hbm_utilization", type=float, default=0.23)
arg_parser.add_argument("--max_concurrency", type=int, default=24)
arg_parser.add_argument("--vllm_max_num_seqs", type=int, default=64)
arg_parser.add_argument("--vllm_max_batched_tokens", type=int, default=8192)
arg_parser.add_argument("--train_micro_batch_size", type=int, default=1)
arg_parser.add_argument("--compute_logps_chunk_size", type=int, default=1024)
arg_parser.add_argument("--eval_every", type=int, default=10)
arg_parser.add_argument("--num_eval_boards", type=int, default=8)
arg_parser.add_argument("--save_every", type=int, default=5)
arg_parser.add_argument("--max_to_keep", type=int, default=2,
                        help="Checkpoints kept in /dev/shm (~27 GB each).")
arg_parser.add_argument("--seed", type=int, default=42)
arg_parser.add_argument("--wandb_project", default="planestrike")
args, _ = arg_parser.parse_known_args()

# XLA's latency-hiding scheduler overlaps layers in the train step and keeps
# many layers' MLP activations ([seq, 12288] each) alive at once; with it on, the
# train step does not fit in HBM. Must be set before JAX initializes the TPUs.
os.environ["LIBTPU_INIT_ARGS"] = (
    os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_enable_latency_hiding_scheduler=false"
).strip()

os.makedirs(args.run_dir, exist_ok=True)
RUN_NAME = os.path.basename(os.path.normpath(args.run_dir))
CKPT_DIR = args.ckpt_dir or f"/dev/shm/planestrike/{RUN_NAME}/ckpt"
TB_DIR = os.path.join(args.run_dir, "tensorboard")

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, force=True,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
from absl import logging as absl_logging  # pylint: disable=g-import-not-at-top
absl_logging.use_python_logging()
absl_logging.set_verbosity(absl_logging.INFO)

from flax import nnx  # pylint: disable=g-import-not-at-top
import huggingface_hub
import jax
from jax import numpy as jnp
import numpy as np
import optax
from orbax import checkpoint as ocp
from tensorboardX import SummaryWriter
import transformers
from tunix.models.gemma4 import model as model_lib
from tunix.models.gemma4 import params_safetensors as params_lib
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger

import agent as agent_lib
import env as env_lib
import strike_tool
import turn_credit


print("jax devices:", jax.devices())
MODEL_VERSION = "google/gemma-4-E2B-it"
MESH_SHAPE = (8, 1)  # (fsdp, tp) for actor, reference and rollout.
MESH_AXIS_NAMES = ("fsdp", "tp")
EVAL_SEED_OFFSET = 10_000_000  # Held-out eval boards.
TRAIN_SEED_OFFSET = 1_000_000 * (1 + args.seed)


# ====== Game logging (raw game length per game) ======
GAMES_PATH = os.path.join(args.run_dir, "games.jsonl")
game_counts = {"train": 0, "eval": 0}
if os.path.exists(GAMES_PATH):  # Resumed run: continue the game index.
  with open(GAMES_PATH) as f:
    for line in f:
      split = json.loads(line)["split"]
      game_counts[split] = game_counts.get(split, 0) + 1
games_writer = SummaryWriter(os.path.join(TB_DIR, "games"))


def log_game_to_tensorboard(record):
  split = record["split"]
  game_counts[split] = game_counts.get(split, 0) + 1
  games_writer.add_scalar(f"{split}/game_length", record["game_length"], game_counts[split])
  games_writer.add_scalar(f"{split}/hits", record["hits"], game_counts[split])
  games_writer.flush()


env_lib.GameLogger.path = GAMES_PATH
env_lib.GameLogger.listeners.append(log_game_to_tensorboard)


# ====== Data: generated on the fly ======
class BoardSeeds:
  """Re-iterable dataset yielding batches of board seeds."""

  def __init__(self, offset, batch_size, num_batches, split):
    self.offset, self.batch_size = offset, batch_size
    self.num_batches, self.split = num_batches, split

  def __iter__(self):
    for b in range(self.num_batches):
      start = self.offset + b * self.batch_size
      yield {
          "seed": list(range(start, start + self.batch_size)),
          "split": [self.split] * self.batch_size,
      }


train_dataset = BoardSeeds(TRAIN_SEED_OFFSET, args.batch_size, args.num_steps, "train")
eval_dataset = BoardSeeds(EVAL_SEED_OFFSET, args.num_eval_boards, 1, "eval")


# ====== Learner with a per-turn generation cap ======
# Module-level copies: methods below take *args, which would shadow `args`.
args_max_response_length = args.max_response_length
MAX_STEPS_PER_GAME = args.max_steps_per_game
SOFT_TOKENS_PER_TURN = args.soft_tokens_per_turn
MAX_TOKENS_PER_TURN = args.max_tokens_per_turn
TURN_LENGTH_PENALTY = args.turn_length_penalty
_LearnerBase = turn_credit.TurnCreditGRPOLearner if args.advantage_mode == "turn" else GRPOLearner


class PlaneStrikeGRPOLearner(_LearnerBase):
  """Caps each turn's generation so one runaway thought cannot consume the
  whole game's token budget (which would clip every later turn)."""

  turn_gamma = args.turn_gamma

  def _build_orchestrator(self):
    orchestrator = super()._build_orchestrator()
    orchestrator.engine_cls = turn_credit.StepInfoEngine  # Adds "game_record".
    return orchestrator

  def _model_call(self, chat_lists, env=None, max_generation_steps=None, **kwargs):
    if max_generation_steps is not None:
      max_generation_steps = min(max_generation_steps, args.max_tokens_per_turn)
    return super()._model_call(
        chat_lists, env, max_generation_steps=max_generation_steps, **kwargs)

  def _process_results(self, trajectories, mode=rl_engine_lib.Mode.TRAIN, expected_step=None):
    games = [item.traj.get("game_record") for item in trajectories]
    games = [g for g in games if g]
    if games:
      # A truncated game counts as the cap.
      lengths = [g["max_steps"] if g["truncated"] else g["game_length"] for g in games]
      self.rl_engine.buffer_metrics_async(
          {
              "game/length": (float(np.mean(lengths)), np.mean),
              "game/win_rate": (float(np.mean([g["won"] for g in games])), np.mean),
              "game/hits": (float(np.mean([g["hits"] for g in games])), np.mean),
              "game/repeats": (float(np.mean([g["repeats"] for g in games])), np.mean),
              "game/invalid": (float(np.mean([g["invalid"] for g in games])), np.mean),
          },
          mode=mode, step=expected_step)
    calls = [item.traj.get("tool_calls_per_turn") for item in trajectories]
    calls = [c for c in calls if c]
    if calls:
      # Tool calls made per game, and the share of turns whose final answer
      # made exactly one call (the only way to strike).
      self.rl_engine.buffer_metrics_async(
          {
              "game/tool_calls": (float(np.mean([sum(c) for c in calls])), np.mean),
              "game/tool_call_rate": (float(np.mean([n == 1 for c in calls for n in c])), np.mean),
          },
          mode=mode, step=expected_step)
    for item in trajectories:
      turn_credit.penalize_truncation(item.traj, MAX_STEPS_PER_GAME, env_lib.STEP_COST)
      if SOFT_TOKENS_PER_TURN > 0:
        turn_credit.penalize_long_turns(item.traj, SOFT_TOKENS_PER_TURN, MAX_TOKENS_PER_TURN,
                                        TURN_LENGTH_PENALTY)
      if turn_credit.clip_to_budget(item.traj, args_max_response_length):
        absl_logging.warning("Trimmed an over-budget trajectory (group %s) to %d tokens.",
                             item.traj.get("group_id"), args_max_response_length)
    return super()._process_results(trajectories, mode=mode, expected_step=expected_step)


# ====== Model ======
model_dir = huggingface_hub.snapshot_download(repo_id=MODEL_VERSION, max_workers=16)
mesh = jax.make_mesh(MESH_SHAPE, MESH_AXIS_NAMES,
                     axis_types=(jax.sharding.AxisType.Auto,) * len(MESH_SHAPE))
config = model_lib.ModelConfig.gemma4_e2b()
config.remat_config = model_lib.RematConfig.DECODER
config.use_flash_attention = True
config.flash_attention_block_size = 256
config.use_sliding_window_kv_cache = False
config.dtype = jnp.bfloat16

# With beta == 0 the reference model is never used; skip it to save HBM.
reference = (params_lib.create_model_from_safe_tensors(model_dir, config, mesh, dtype=jnp.bfloat16)
             if args.beta > 0 else None)
# fp32 actor storage: bf16 would round lr~1e-6 Adam updates away.
actor = params_lib.create_model_from_safe_tensors(model_dir, config, mesh, dtype=jnp.float32)

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_VERSION)
chat_parser = strike_tool.Gemma4ToolChatTemplateParser(
    tokenizer, [strike_tool.StrikeTool()], enable_thinking=True)

def make_learning_rate():
  # One optimizer update per board group, i.e. batch_size updates per training
  # step. Adam's update count is checkpointed, so a resume keeps its place.
  if not args.lr_schedule:
    if args.lr_decay == "none":
      # A schedule, not a float: the optimizer state then has the same structure
      # (a step count) as with the other schedules, so checkpoints stay resumable.
      return optax.constant_schedule(args.learning_rate)
    decay_start = args.lr_decay_start * args.batch_size
    return optax.join_schedules(
        [optax.constant_schedule(args.learning_rate),
         optax.cosine_decay_schedule(
             args.learning_rate, max(1, args.num_steps - args.lr_decay_start) * args.batch_size,
             alpha=args.lr_end / args.learning_rate)],
        [decay_start])
  points = [(int(s), float(v)) for s, v in (p.split(":") for p in args.lr_schedule.split(","))]
  assert points[0][0] == 0, "lr_schedule must start at step 0"
  return optax.piecewise_constant_schedule(
      points[0][1],
      {step * args.batch_size: lr / prev for (step, lr), (_, prev) in zip(points[1:], points)})


optimizer = optax.chain(
    optax.clip_by_global_norm(args.max_grad_norm),
    optax.adamw(learning_rate=make_learning_rate(), b1=args.b1, b2=args.b2, weight_decay=0.0),
)

rollout_config = base_rollout.RolloutConfig(
    max_prompt_length=args.max_prompt_length,
    max_tokens_to_generate=args.max_response_length,
    kv_cache_size=args.max_prompt_length + args.max_response_length + 256,
    temperature=args.temperature,
    top_p=1.0,
    top_k=0,
    return_logprobs=True,
    # Gemma 4's stop tokens: <eos>, <turn|>, and <|tool_response> (after a tool call).
    eos_tokens=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>"),
                tokenizer.convert_tokens_to_ids(strike_tool.TOOL_RESPONSE_START)],
    rollout_vllm_model_version=MODEL_VERSION,
    rollout_vllm_hbm_utilization=args.vllm_hbm_utilization,
    rollout_vllm_tpu_backend_type="jax",
    rollout_vllm_server_mode=True,
    rollout_vllm_async_scheduling=False,
    rollout_vllm_init_with_random_weights=True,
    tensor_parallel_size=MESH_SHAPE[0] * MESH_SHAPE[1],
    data_parallel_size=1,
    rollout_vllm_max_num_seqs=args.vllm_max_num_seqs,
    rollout_vllm_max_num_batched_tokens=args.vllm_max_batched_tokens,
    rollout_vllm_kwargs={
        "kv_cache_metrics": True,
        "disable_log_stats": False,
        # Safe: update_params() resets the prefix cache after every weight sync.
        # Each turn re-sends the whole game so far; caching avoids re-prefilling it.
        "enable_prefix_caching": True,
        "dtype": "bfloat16",
        "hf_overrides": {
            "final_logit_softcapping": 30.0,
            "text_config": {"final_logit_softcapping": 30.0},
            "architectures": ["Gemma4ForCausalLM"],
        },
    },
    rollout_vllm_sampling_kwargs={"skip_special_tokens": False},
)

WANDB_ID_PATH = os.path.join(args.run_dir, "wandb_id.txt")
if not os.path.exists(WANDB_ID_PATH):
  with open(WANDB_ID_PATH, "w") as f:
    f.write(f"{RUN_NAME}-{int(time.time())}")
with open(WANDB_ID_PATH) as f:
  WANDB_ID = f.read().strip()
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir=TB_DIR, project_name=args.wandb_project, run_name=RUN_NAME, flush_every_n_steps=1,
    backend_kwargs={"wandb": {
        "id": WANDB_ID, "resume": "allow", "dir": args.run_dir, "config": vars(args),
    }})

cluster_config = rl_engine_lib.ClusterConfig(
    role_to_mesh={
        rl_engine_lib.Role.ACTOR: mesh,
        rl_engine_lib.Role.REFERENCE: mesh,
        rl_engine_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine="vllm",
    offload_to_cpu=False,
    training_config=rl_engine_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=args.eval_every,
        max_steps=args.num_steps,
        mini_batch_size=args.batch_size,
        train_micro_batch_size=args.train_micro_batch_size,
        compute_logps_micro_batch_size=args.train_micro_batch_size,
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=ocp.CheckpointManagerOptions(
            save_interval_steps=args.save_every, max_to_keep=args.max_to_keep),
        compute_logps_chunk_size=args.compute_logps_chunk_size,
    ),
    rollout_config=rollout_config,
)

grpo_config = GRPOConfig(
    num_generations=args.num_generations,
    num_iterations=1,
    max_response_length=args.max_response_length,
    beta=args.beta,
    epsilon=args.epsilon,
    epsilon_high=args.epsilon_high,
    loss_algo=args.loss_algo,
    loss_agg_mode="sequence-mean-token-mean",
    kl_loss_mode="low_var_kl",
    advantage_estimator=args.advantage_estimator,
    sampler_is="token",
    sampler_is_threshold=2.0,
    system_prompt="",
    max_concurrency=args.max_concurrency,
    off_policy_steps=0,
    episode_timeout=args.episode_timeout,
    overlong_filter=False,
    exact_token_continuity=True,
)

rl_engine = rl_engine_lib.RLEngine(
    actor=actor, reference=reference, tokenizer=tokenizer, cluster_config=cluster_config)


def metric_fn(prompts, completions, rewards, advantages, **kwargs):
  del prompts, completions, advantages, kwargs
  rewards = np.asarray(rewards)
  absl_logging.info("[group] reward mean=%.2f min=%.2f max=%.2f",
                    rewards.mean(), rewards.min(), rewards.max())
  return {"rewards/group_std": (float(rewards.std()), np.mean)}


learner = PlaneStrikeGRPOLearner(
    rl_engine=rl_engine,
    agent_class=agent_lib.PlaneStrikeAgent,
    agent_kwargs={},
    env_class=env_lib.PlaneStrikeEnv,
    env_kwargs={"max_steps": args.max_steps_per_game},
    algo_config=grpo_config,
    chat_parser=chat_parser,
    metric_fns=[metric_fn],
)
if rl_engine.global_steps > 0:
  rl_engine.rollout.update_params(
      nnx.state(rl_engine.actor_trainer.model, nnx.Param), (nnx.Param,))
  absl_logging.info("Synced restored step-%d actor weights to vLLM.", rl_engine.global_steps)
absl_logging.info("Starting training from global step %d", rl_engine.global_steps)
t0 = time.time()
learner.train(train_dataset, eval_dataset=eval_dataset)
absl_logging.info("Training finished in %.1f h", (time.time() - t0) / 3600)
games_writer.close()
sys.stdout.flush()
os._exit(0)  # pylint: disable=protected-access  # Don't wait on vLLM threads.
