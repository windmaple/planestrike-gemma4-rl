"""Compiles the trainer's forward+backward for Gemma 4 E2B at several sequence
lengths and prints XLA's memory analysis (no vLLM, no training).

Used to size max_response_length / micro batch for the (8,1) mesh:
  python mem_probe.py --lengths 8192,16384,33792
"""

import argparse
import os

import huggingface_hub
import jax
from jax import numpy as jnp
from flax import nnx
from tunix.models.gemma4 import model as model_lib
from tunix.models.gemma4 import params_safetensors as params_lib
from tunix.rl import common

parser = argparse.ArgumentParser()
parser.add_argument("--lengths", default="8192,16384,33792")
parser.add_argument("--prompt_length", type=int, default=1024)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--chunk_size", type=int, default=1024)
parser.add_argument("--remat", default="DECODER")
parser.add_argument("--token_mask", type=int, default=1)
parser.add_argument("--mlp_chunk_size", type=int, default=0)
args = parser.parse_args()
import memory_patches
memory_patches.patch_chunked_mlp(args.mlp_chunk_size)

mesh = jax.make_mesh((8, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
config = model_lib.ModelConfig.gemma4_e2b()
config.remat_config = getattr(model_lib.RematConfig, args.remat)
config.use_flash_attention = True
config.flash_attention_block_size = 256
config.use_sliding_window_kv_cache = False
config.dtype = jnp.bfloat16
model_dir = huggingface_hub.snapshot_download("google/gemma-4-E2B-it")
model = params_lib.create_model_from_safe_tensors(model_dir, config, mesh, dtype=jnp.float32)
graphdef, state = nnx.split(model)
print("params GB per chip:",
      sum(x.addressable_shards[0].data.nbytes for x in jax.tree.leaves(state)) / 2**30)

data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp"))
for length in [int(x) for x in args.lengths.split(",")]:
  prompt = jax.device_put(jnp.ones((args.batch, args.prompt_length), jnp.int32), data_sharding)
  completion = jax.device_put(jnp.ones((args.batch, length), jnp.int32), data_sharding)
  mask = jax.device_put(jnp.ones((args.batch, args.prompt_length + length), bool), data_sharding)

  def loss(state, prompt, completion, mask):
    logps, _ = common.compute_per_token_logps(
        graphdef, state, prompt, completion, pad_id=0, eos_id=1,
        stop_gradient=False, return_entropy=True, chunk_size=args.chunk_size,
        token_mask=mask if args.token_mask else None)
    return -logps.mean()

  with mesh:
    compiled = jax.jit(jax.grad(loss)).lower(state, prompt, completion, mask).compile()
  m = compiled.memory_analysis()
  print(f"len={length}: temp={m.temp_size_in_bytes / 2**30:.2f}G "
        f"args={m.argument_size_in_bytes / 2**30:.2f}G out={m.output_size_in_bytes / 2**30:.2f}G",
        flush=True)
os._exit(0)
