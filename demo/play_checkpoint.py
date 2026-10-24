"""Plays Plane Strike with a trained checkpoint and saves the full transcripts.

Restores the actor weights from a training checkpoint into a Tunix Gemma 4
E2B model, pushes them into a vLLM engine the same way the trainer syncs
weights, and plays games with the trainer's env, agent, strike tool, chat
parser, exact-token prompts, sampling settings and token budgets.

  python demo/play_checkpoint.py \
    --ckpt_dir /dev/shm/planestrike/grpo_6x6_tool_nopatch_100/ckpt/actor \
    --out demo/games.jsonl
"""

import argparse
import json
import os
import sys
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ckpt_dir", required=True, help="<run ckpt dir>/actor")
parser.add_argument("--step", type=int, default=None, help="Default: latest.")
parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "games.jsonl"))
parser.add_argument("--num_boards", type=int, default=8)
parser.add_argument("--games_per_board", type=int, default=2)
parser.add_argument("--seed_offset", type=int, default=10_000_000, help="Held-out eval boards.")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--max_prompt_length", type=int, default=512)
parser.add_argument("--max_response_length", type=int, default=11776)
parser.add_argument("--max_tokens_per_turn", type=int, default=2048)
args = parser.parse_args()

import huggingface_hub  # pylint: disable=g-import-not-at-top
import jax
from jax import numpy as jnp
import numpy as np
import transformers
from flax import nnx
from tunix.generate import tokenizer_adapter
from tunix.models.gemma4 import model as model_lib
from tunix.models.gemma4 import params_safetensors as params_lib
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vllm_rollout
from tunix.sft import checkpoint_manager
from vllm import SamplingParams

import agent as agent_lib
import env as env_lib
import strike_tool

MODEL_VERSION = "google/gemma-4-E2B-it"


def load_trained_model(mesh):
  model_dir = huggingface_hub.snapshot_download(repo_id=MODEL_VERSION, max_workers=16)
  config = model_lib.ModelConfig.gemma4_e2b()
  config.dtype = jnp.bfloat16
  model = params_lib.create_model_from_safe_tensors(model_dir, config, mesh, dtype=jnp.float32)
  step, _ = checkpoint_manager.CheckpointManager(args.ckpt_dir).maybe_restore(model, step=args.step)
  assert step > 0, f"No checkpoint found in {args.ckpt_dir}"
  print(f"Restored step {step} from {args.ckpt_dir}", flush=True)
  return model, step


def main():
  mesh = jax.make_mesh((8, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)
  model, step = load_trained_model(mesh)
  tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_VERSION)
  chat_parser = strike_tool.Gemma4ToolChatTemplateParser(
      tokenizer, [strike_tool.StrikeTool()], enable_thinking=True)
  tok_adapter = tokenizer_adapter.TokenizerAdapter(tokenizer)
  eos = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>"),
         tokenizer.convert_tokens_to_ids(strike_tool.TOOL_RESPONSE_START)]

  rollout_config = base_rollout.RolloutConfig(
      max_prompt_length=args.max_prompt_length,
      max_tokens_to_generate=args.max_response_length,
      kv_cache_size=args.max_prompt_length + args.max_response_length + 256,
      temperature=args.temperature, top_p=1.0, top_k=0, eos_tokens=eos,
      rollout_vllm_model_version=MODEL_VERSION,
      rollout_vllm_hbm_utilization=0.5,
      rollout_vllm_tpu_backend_type="jax",
      rollout_vllm_server_mode=False,
      rollout_vllm_init_with_random_weights=True,
      tensor_parallel_size=8, data_parallel_size=1,
      rollout_vllm_kwargs={
          "enable_prefix_caching": True, "dtype": "bfloat16",
          "hf_overrides": {"final_logit_softcapping": 30.0,
                           "text_config": {"final_logit_softcapping": 30.0},
                           "architectures": ["Gemma4ForCausalLM"]},
      },
  )
  rollout = vllm_rollout.VllmRollout(
      model, tokenizer, cache_config_or_size=rollout_config.kv_cache_size,
      mesh=mesh, rollout_config=rollout_config)
  rollout.update_params(nnx.state(model, nnx.Param), (nnx.Param,))  # Same sync as training.
  llm = rollout._sampler.llm  # pylint: disable=protected-access

  games = []
  for b in range(args.num_boards):
    for k in range(args.games_per_board):
      e = env_lib.PlaneStrikeEnv({"seed": args.seed_offset + b, "split": "demo"})
      a = agent_lib.PlaneStrikeAgent()
      obs, _ = e.reset()
      a.update_from_env(obs, 0.0, False, {})
      text = chat_parser.parse(a.chat_completions, add_generation_prompt=True, is_first_msg=True)
      ids = tokenizer.encode(text, add_special_tokens=False)
      games.append({"env": e, "agent": a, "ids": list(ids), "completion_tokens": 0,
                    "done": False, "truncated": False, "won": False, "turns": []})

  t0 = time.time()
  while any(not g["done"] for g in games):
    active = [g for g in games if not g["done"]]
    params = [SamplingParams(temperature=args.temperature, top_p=1.0, top_k=-1,
                             max_tokens=min(args.max_response_length - g["completion_tokens"],
                                            args.max_tokens_per_turn),
                             stop_token_ids=eos, skip_special_tokens=False,
                             include_stop_str_in_output=True) for g in active]
    outs = llm.generate([{"prompt_token_ids": g["ids"]} for g in active], params, use_tqdm=False)
    for g, out in zip(active, outs):
      gen = list(out.outputs[0].token_ids)
      response = tokenizer.decode(gen, skip_special_tokens=False)
      gen_with_suffix, _ = chat_parser.update_assistant_end_tokens(np.array(gen, dtype=np.int32))
      g["completion_tokens"] += len(gen_with_suffix)
      g["ids"] += [int(t) for t in gen_with_suffix]
      action = g["agent"].update_from_model(response).action
      obs, rew, done, info = g["env"].step(action)
      g["agent"].update_from_env(obs, rew, done, info)
      g["won"] = info["won"]
      g["turns"].append({"response": response, "action": action, "result": obs["last_result"],
                         "board": obs["board"], "tokens": len(gen)})
      if not done:
        e_tokens, _ = agentic_utils.tokenize_and_generate_masks(
            [g["agent"].chat_completions[-1]], tokenizer=tok_adapter, parser=chat_parser,
            contains_first_msg=False, contains_generation_msg=True)
        g["ids"] += list(e_tokens)
        g["completion_tokens"] += len(e_tokens)
        if g["completion_tokens"] >= args.max_response_length:
          done, g["truncated"] = True, True
      g["done"] = done
    print(f"turn {len(active[0]['turns'])}: active={len(active)} {time.time() - t0:.0f}s", flush=True)

  with open(args.out, "w") as f:
    for g in games:
      e = g["env"]
      rec = {"checkpoint_step": step, "seed": e.seed, "game_length": e.step_count,
             "won": g["won"], "truncated": g["truncated"], "hits": e.hit_count,
             "repeats": e.num_repeats, "invalid": e.num_invalid,
             "hidden_board": e.hidden_board.tolist(), "turns": g["turns"]}
      f.write(json.dumps(rec) + "\n")
      print(f"seed {e.seed}: length={e.step_count} won={g['won']} hits={e.hit_count} "
            f"repeats={e.num_repeats} invalid={e.num_invalid} truncated={g['truncated']}")
  print(f"Wrote {len(games)} games to {args.out}", flush=True)


main()
sys.stdout.flush()
os._exit(0)  # pylint: disable=protected-access  # vLLM engine can hang on shutdown.
