"""Runtime memory patches for long-sequence training of Gemma 4 (no library edits).

1. Sequence-chunked MLP (patch_chunked_mlp).

With DECODER remat, the backward pass recomputes each layer's MLP on the
full sequence: gate/up/gelu activations of shape [L, 12288] bf16 (~0.8 GB at
L=34k, several alive per layer, and XLA overlaps layers). At L~34k this alone
exceeds HBM. Computing the MLP over fixed-size sequence chunks, each under
jax.checkpoint, bounds that working set independently of L. The math is
identical (the MLP is position-wise).

2. Frozen per-layer embedding table (freeze_per_layer_embedding).
Gemma 4 E2B's per-layer input embedding is a [262144, 35*256] table, ~2.35B of
the model's ~4.8B parameters. Training it means an fp32 gradient, a gradient
accumulator copy, two Adam moments and several optimizer-update temporaries of
that size inside the train step (~14 GB/chip of the 33.7 GB that OOMed).
Retyping it from nnx.Param to a non-trainable variable removes it from the
gradient, optimizer and accumulator (all filter on nnx.Param); weight sync to
vLLM uses the unfiltered model state, so rollout still receives it.
"""

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
    # [..., n, chunk, dim] -> [n, ..., chunk, dim] for scan.
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
