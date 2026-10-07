"""Compiles Tunix's real actor train step (GSPO loss + AdamW update) for one
micro-batch (num_generations sequences of prompt+completion tokens) and prints
XLA's memory analysis. Faster than a full smoke run for sizing memory.

  python mem_probe_trainstep.py --completion_length 32768
"""

import argparse
import os

import huggingface_hub
import jax
from jax import numpy as jnp
import numpy as np
import optax
from tunix.models.gemma4 import model as model_lib
from tunix.models.gemma4 import params_safetensors as params_lib
from tunix.rl import algo_core  # pylint: disable=unused-import  # registers loss fns
from tunix.rl import function_registry
from tunix.rl import trainer as rl_trainer
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig, TrainExample
from tunix.sft import peft_trainer

import memory_patches

parser = argparse.ArgumentParser()
parser.add_argument("--completion_length", type=int, default=32768)
parser.add_argument("--prompt_length", type=int, default=1024)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--chunk_size", type=int, default=1024)
parser.add_argument("--mlp_chunk_size", type=int, default=2048)
parser.add_argument("--freeze_ple", type=int, default=1)
parser.add_argument("--remat_policy", default="nothing_saveable")
parser.add_argument("--per_jit_no_lhs", type=int, default=0)
args = parser.parse_args()

memory_patches.patch_chunked_mlp(args.mlp_chunk_size)
mesh = jax.make_mesh((8, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
config = model_lib.ModelConfig.gemma4_e2b()
config.remat_config = model_lib.RematConfig.DECODER
config.remat_policy = args.remat_policy
config.use_flash_attention = True
config.flash_attention_block_size = 256
config.use_sliding_window_kv_cache = False
config.dtype = jnp.bfloat16
model = params_lib.create_model_from_safe_tensors(
    huggingface_hub.snapshot_download("google/gemma-4-E2B-it"), config, mesh, dtype=jnp.float32)
if args.freeze_ple:
  memory_patches.freeze_per_layer_embedding(model)

algo_config = GRPOConfig(num_generations=args.batch, beta=0.0, epsilon=0.003, epsilon_high=0.005,
                         loss_algo="gspo-token", loss_agg_mode="sequence-mean-token-mean",
                         advantage_estimator="rloo", sampler_is="token", max_response_length=args.completion_length)
algo_config.temperature = 0.7
if args.per_jit_no_lhs:
  memory_patches.compile_train_step_with_options(
      rl_trainer.Trainer, {"xla_tpu_enable_latency_hiding_scheduler": False})
policy_loss_fn = function_registry.get_policy_loss_fn(algo_config.policy_loss_fn)
optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-6, b1=0.9, b2=0.95, weight_decay=0.0))
with mesh:
  trainer = rl_trainer.Trainer(model, optimizer, peft_trainer.TrainingConfig(eval_every_n_steps=1000, max_steps=10),
                               custom_checkpoint_metadata_fn=lambda: {})
trainer.with_loss_fn(lambda model, train_example, algo_config: policy_loss_fn(
    model, train_example, algo_config=algo_config, pad_id=0, eos_id=1,
    compute_logps_chunk_size=args.chunk_size), has_aux=True)
trainer.with_gen_model_input_fn(lambda x: {"train_example": x, "algo_config": algo_config})

b, p, c = args.batch, args.prompt_length, args.completion_length
sh = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp"))
put = lambda x: jax.device_put(x, sh)
example = TrainExample(
    prompt_ids=put(jnp.ones((b, p), jnp.int32)), prompt_mask=put(jnp.ones((b, p), bool)),
    completion_ids=put(jnp.ones((b, c), jnp.int32)), completion_mask=put(jnp.ones((b, c), jnp.int32)),
    advantages=put(jnp.ones((b,), jnp.float32)), ref_per_token_logps=None,
    old_per_token_logps=put(jnp.zeros((b, c), jnp.float32)),
    sampler_is_weights=put(jnp.ones((b, c), jnp.float32)),
    completion_attention_mask=put(jnp.ones((b, c), bool)),
    policy_version=np.zeros((b,), np.int32),
)
with mesh:
  train_step, _ = trainer.jit_train_and_eval_step(False)
  try:
    loss, _, _ = train_step(example, is_update_step=jnp.array(True))
    jax.block_until_ready(loss)
    peak = max(d.memory_stats()["peak_bytes_in_use"] for d in jax.devices())
    print(f"completion={c}: FITS, peak HBM {peak / 2**30:.2f}G per chip", flush=True)
  except Exception as e:  # pylint: disable=broad-except
    msg = str(e).splitlines()[0]
    print(f"completion={c}: {msg[:300]}", flush=True)
os._exit(0)
