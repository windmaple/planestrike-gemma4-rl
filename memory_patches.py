"""Runtime memory patches for long-sequence training of Gemma 4
Some magic to reduce HBM usage and avoid OOMs"""

import functools

import jax
from jax import numpy as jnp
from flax import nnx
from tunix.models.gemma4 import model as model_lib

_original_block = model_lib.FeedForward.block

def patch_chunked_mlp(chunk_size: int) -> None:
  if chunk_size <= 0:
    return

  def chunked_block(self, x):
    *lead, seq_len, dim = x.shape
    if seq_len <= chunk_size:
      return _original_block(self, x)
    pad = (-seq_len) % chunk_size
    xp = jnp.pad(x, [(0, 0)] * len(lead) + [(0, pad), (0, 0)])
    n = xp.shape[-2] // chunk_size
    xc = jnp.moveaxis(xp.reshape(*lead, n, chunk_size, dim), -3, 0)
    graphdef, state = nnx.split(self)

    @jax.checkpoint
    def one_chunk(state, chunk):
      return _original_block(nnx.merge(graphdef, state), chunk)

    _, yc = jax.lax.scan(lambda c, chunk: (c, one_chunk(state, chunk)), None, xc)
    y = jnp.moveaxis(yc, 0, -3).reshape(*lead, n * chunk_size, -1)
    return y[..., :seq_len, :]

  model_lib.FeedForward.block = chunked_block


class FrozenParam(nnx.Variable):
  """A model weight that is not trained (not an nnx.Param)."""

def freeze_per_layer_embedding(model, dtype=jnp.bfloat16) -> None:
  embedder = model.embedder
  var = embedder.per_layer_input_embedding
  value = var[...]
  sharding = value.sharding
  with jax.set_mesh(sharding.mesh):
    # The released checkpoint is bf16, so the bf16 copy is exact.
    frozen = FrozenParam(value.astype(dtype), **var.get_metadata())
  assert frozen[...].sharding.is_equivalent_to(sharding, value.ndim), frozen[...].sharding
  embedder.per_layer_input_embedding = frozen

def compile_train_step_with_options(trainer_cls, compiler_options: dict) -> None:
  """Passes XLA `compiler_options` to the jax.jit behind the trainer's train/eval steps.

  Lets the train step be compiled without the latency-hiding scheduler (which
  keeps many layers' activations alive) while vLLM's programs keep it.
  nnx.jit has no compiler_options argument, so jax.jit is wrapped only while
  `jit_train_and_eval_step` builds the jitted functions.
  """
  original = trainer_cls.jit_train_and_eval_step

  def wrapped(self, *args, **kwargs):
    real_jit = jax.jit
    jax.jit = functools.partial(real_jit, compiler_options=compiler_options)
    try:
      return original(self, *args, **kwargs)
    finally:
      jax.jit = real_jit

  trainer_cls.jit_train_and_eval_step = wrapped
