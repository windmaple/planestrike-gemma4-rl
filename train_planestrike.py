"""Agentic GRPO on Plane Strike with Gemma 4 E2B (thinking mode) on one TPU host.

Follows tunix/examples/frozenlake/train_frozenlake.py. Differences:
  * Boards are generated on the fly from integer seeds (no data files).
  * Thinking mode is on; every turn's thinking stays in the trajectory, and
    later turns replay the recorded token ids (exact_token_continuity) so the
    sampler and the trainer see the same sequence.
  * Context budget, per-turn cap and episode timeout are sized so that a
    full game is not clipped (see README.md).
  * Every finished game is appended to <run_dir>/games.jsonl and logged to
    TensorBoard as `game_length`, like the original training_jax.py.
  * Uncaught exceptions in any thread hard-exit the process (exit code 42 for
    OOM) so that a crashed vLLM/trainer thread cannot leave the job hanging;
    watchdog.py restarts it from the latest checkpoint.
"""

import argparse
import faulthandler
import json
import logging
import math
import os
import re
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OOM_EXIT_CODE = 42
_OOM_RE = re.compile(
    r"RESOURCE_EXHAUSTED|[Oo]ut of memory|OutOfMemory|Ran out of memory|"
    r"memory limit exceeded|Attempting to allocate"
)


_FATAL_DIR = None  # Set to <run_dir>/logs once args are parsed.


def _hard_exit_on_error(exc_type, exc, tb, where):
  text = "".join(traceback.format_exception(exc_type, exc, tb))
  code = OOM_EXIT_CODE if _OOM_RE.search(text) else 1
  msg = f"\n[fatal] uncaught exception in {where}:\n{text}\n[fatal] exiting with code {code}\n"
  # Write to fd 2 directly: libraries may have replaced sys.stderr.
  os.write(2, msg.encode(errors="replace"))
  if _FATAL_DIR:
    with open(os.path.join(_FATAL_DIR, f"fatal_{os.getpid()}.txt"), "w") as f:
      f.write(msg)
  os._exit(code)  # pylint: disable=protected-access


sys.excepthook = lambda t, e, tb: _hard_exit_on_error(t, e, tb, "main thread")
threading.excepthook = lambda a: (
    None if a.exc_type is SystemExit
    else _hard_exit_on_error(a.exc_type, a.exc_value, a.exc_traceback,
                             f"thread {a.thread.name if a.thread else '?'}")
)
faulthandler.enable()

arg_parser = argparse.ArgumentParser(description=__doc__)
arg_parser.add_argument("--run_dir", required=True)
arg_parser.add_argument("--ckpt_dir", default="",
                        help="Default: /dev/shm/planestrike/<run name>/ckpt "
                        "(a full fp32 actor+Adam checkpoint is ~60GB).")
arg_parser.add_argument("--num_steps", type=int, default=70)
arg_parser.add_argument("--batch_size", type=int, default=8, help="Boards per step.")
arg_parser.add_argument("--num_generations", type=int, default=8, help="Games per board.")
# Tunix applies an optimizer update after every micro-batch (one board's group
# of games), i.e. batch_size updates per step, as in the frozenlake recipe.
arg_parser.add_argument("--learning_rate", type=float, default=3e-6,
                        help="Used when --lr_schedule is empty.")
# "step:lr,step:lr,..." in training steps; the first entry must be step 0.
# Default: the path that worked at a 24-strike cap (3e-6, 5e-6 from step 40 to
# break a plateau, 2.5e-6 from step 65 before entropy collapsed).
arg_parser.add_argument("--lr_schedule", default="0:3e-6,40:5e-6,65:2.5e-6")
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
arg_parser.add_argument("--max_steps_per_game", type=int, default=20,
                        help="Must match env.MAX_STEPS (also stated in the system prompt).")
arg_parser.add_argument("--max_prompt_length", type=int, default=1024)
arg_parser.add_argument("--max_response_length", type=int, default=32768,
                        help="Token budget for a whole game after the first prompt.")
arg_parser.add_argument("--max_tokens_per_turn", type=int, default=2048)
arg_parser.add_argument("--soft_tokens_per_turn", type=int, default=1024,
                        help="Turns longer than this get a linear penalty up to "
                        "--turn_length_penalty at --max_tokens_per_turn (0 disables).")
arg_parser.add_argument("--turn_length_penalty", type=float, default=1.0)
arg_parser.add_argument("--episode_timeout", type=float, default=6 * 3600)
arg_parser.add_argument("--vllm_hbm_utilization", type=float, default=0.23)
# Games in flight. Must fit their contexts in vLLM's KV cache (~13k tokens at
# the end of a game; ~255k tokens at --vllm_hbm_utilization 0.23); beyond that
# vLLM evicts prefix-cached histories and re-prefills them every turn.
arg_parser.add_argument("--max_concurrency", type=int, default=24)
arg_parser.add_argument("--vllm_max_num_seqs", type=int, default=64)
arg_parser.add_argument("--vllm_max_batched_tokens", type=int, default=8192)
arg_parser.add_argument("--train_micro_batch_size", type=int, default=1)
arg_parser.add_argument("--compute_logps_chunk_size", type=int, default=1024)
arg_parser.add_argument("--latency_hiding_scheduler_off", default="train_step",
                        choices=["train_step", "global", "none"])
arg_parser.add_argument("--freeze_ple", type=int, default=1,
                        help="Freeze the 2.35B-param per-layer embedding table; see memory_patches.py.")
arg_parser.add_argument("--mlp_chunk_size", type=int, default=2048,
                        help="Sequence chunk for the MLP under remat; see memory_patches.py.")
arg_parser.add_argument("--eval_every", type=int, default=10)
arg_parser.add_argument("--num_eval_boards", type=int, default=8)
arg_parser.add_argument("--save_every", type=int, default=5)
arg_parser.add_argument("--max_to_keep", type=int, default=2,
                        help="Checkpoints kept in /dev/shm (~27 GB each).")
arg_parser.add_argument("--seed", type=int, default=42)
args, _ = arg_parser.parse_known_args()

# XLA's latency-hiding scheduler overlaps layers in the train step and keeps
# many layers' activations alive: 28.6 GB of temporaries per chip at 34k tokens
# vs 17.4 GB without it (mem_probe_trainstep.py). "train_step" turns it off only
# for the trainer's jitted step (vLLM keeps it); "global" for the whole process.
if args.latency_hiding_scheduler_off == "global":
  os.environ["LIBTPU_INIT_ARGS"] = (
      os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_enable_latency_hiding_scheduler=false"
  ).strip()

os.makedirs(args.run_dir, exist_ok=True)
RUN_NAME = os.path.basename(os.path.normpath(args.run_dir))
CKPT_DIR = args.ckpt_dir or f"/dev/shm/planestrike/{RUN_NAME}/ckpt"
TB_DIR = os.path.join(args.run_dir, "tensorboard")
_FATAL_DIR = os.path.join(args.run_dir, "logs")
os.makedirs(_FATAL_DIR, exist_ok=True)

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
from tunix.rl import trainer as rl_trainer
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.agentic.parser.chat_template_parser import parser as parser_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.sft import utils as sft_utils

import agent as agent_lib
import env as env_lib
import memory_patches
import turn_credit

memory_patches.patch_chunked_mlp(args.mlp_chunk_size)
if args.latency_hiding_scheduler_off == "train_step":
  memory_patches.compile_train_step_with_options(
      rl_trainer.Trainer, {"xla_tpu_enable_latency_hiding_scheduler": False})

print("jax devices:", jax.devices())
MODEL_VERSION = "google/gemma-4-E2B-it"
MESH_SHAPE = (8, 1)  # (fsdp, tp) for actor, reference and rollout.
MESH_AXIS_NAMES = ("fsdp", "tp")
EVAL_SEED_OFFSET = 10_000_000  # Same held-out boards as baseline_eval.py.
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
ATTEMPT = int(os.environ.get("WATCHDOG_ATTEMPT", "0"))


def tag_attempt(record):
  record["attempt"] = ATTEMPT


env_lib.GameLogger.listeners.insert(0, tag_attempt)
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

  def _model_call(self, chat_lists, env=None, max_generation_steps=None, **kwargs):
    if max_generation_steps is not None:
      max_generation_steps = min(max_generation_steps, args.max_tokens_per_turn)
    return super()._model_call(
        chat_lists, env, max_generation_steps=max_generation_steps, **kwargs)

  def _process_results(self, trajectories, *args, **kwargs):
    for item in trajectories:
      turn_credit.penalize_truncation(item.traj, MAX_STEPS_PER_GAME, env_lib.STEP_COST)
      if SOFT_TOKENS_PER_TURN > 0:
        turn_credit.penalize_long_turns(item.traj, SOFT_TOKENS_PER_TURN, MAX_TOKENS_PER_TURN,
                                        TURN_LENGTH_PENALTY)
      if turn_credit.clip_to_budget(item.traj, args_max_response_length):
        absl_logging.warning("Trimmed an over-budget trajectory (group %s) to %d tokens.",
                             item.traj.get("group_id"), args_max_response_length)
    return super()._process_results(trajectories, *args, **kwargs)


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

show_hbm_usage = sft_utils.show_hbm_usage
# With beta == 0 the reference model is never used; skip it to save HBM.
reference = (params_lib.create_model_from_safe_tensors(model_dir, config, mesh, dtype=jnp.bfloat16)
             if args.beta > 0 else None)
# fp32 actor storage: bf16 would round lr~1e-6 Adam updates away.
actor = params_lib.create_model_from_safe_tensors(model_dir, config, mesh, dtype=jnp.float32)
if args.freeze_ple:
  memory_patches.freeze_per_layer_embedding(actor)
show_hbm_usage("after loading actor and reference")

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_VERSION)
chat_parser = parser_lib.Gemma4ChatTemplateParser(tokenizer, enable_thinking=True)

def make_learning_rate():
  if not args.lr_schedule:
    return args.learning_rate
  points = [(int(s), float(v)) for s, v in (p.split(":") for p in args.lr_schedule.split(","))]
  assert points[0][0] == 0, "lr_schedule must start at step 0"
  # One optimizer update per board group, i.e. batch_size updates per training
  # step. Adam's update count is checkpointed, so a resume keeps its place.
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

metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir=TB_DIR, project_name="planestrike", run_name=RUN_NAME, flush_every_n_steps=1)

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
show_hbm_usage("after RLEngine creation")


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
show_hbm_usage("after learner creation")
if rl_engine.global_steps > 0:
  # RLEngine loads the actor's weights into vLLM before the trainer restores
  # the checkpoint, so after a resume vLLM would sample from the base model
  # until the first post-update sync. Push the restored weights now.
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
