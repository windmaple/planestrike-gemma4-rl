"""Evaluates Plane Strike policies outside the trainer.

  --policy random : uniform over untried cells (the original example's
                    untrained-network behaviour, already excluding repeats).
  --policy model  : Gemma 4 with thinking via vLLM, using the same agent, env,
                    chat parser and exact-token prompt construction as the
                    Tunix trainer, so the token counts it reports are the ones
                    the trainer will see.

Writes one JSON line per game to --out and prints summary statistics,
including per-turn and per-trajectory token counts used to size the
trainer's context budget.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent as agent_lib  # pylint: disable=g-import-not-at-top
import env as env_lib
import strike_tool

parser = argparse.ArgumentParser()
parser.add_argument("--policy", choices=["random", "model"], default="model")
parser.add_argument("--num_games", type=int, default=64)
parser.add_argument("--seed_offset", type=int, default=10_000_000)
parser.add_argument("--model", default="google/gemma-4-E2B-it")
parser.add_argument("--model_path", default="", help="Optional local weights dir.")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--max_prompt_length", type=int, default=1024)
parser.add_argument("--max_response_length", type=int, default=32768)
parser.add_argument("--tp", type=int, default=8)
parser.add_argument("--out", default="")
args = parser.parse_args()


def summarize(records):
  lengths = np.array([r["game_length"] for r in records])
  hits = np.array([r["hits"] for r in records])
  print(f"games={len(records)}")
  print(f"game_length mean={lengths.mean():.2f} median={np.median(lengths):.1f} "
        f"min={lengths.min()} max={lengths.max()}")
  print(f"win_rate={np.mean([r['won'] for r in records]):.3f}  hits mean={hits.mean():.2f}")
  print(f"repeats mean={np.mean([r['repeats'] for r in records]):.2f}  "
        f"invalid mean={np.mean([r['invalid'] for r in records]):.2f}  "
        f"truncated={sum(r.get('truncated', False) for r in records)}")
  if "turn_tokens" in records[0]:
    turn = np.concatenate([r["turn_tokens"] for r in records])
    total = np.array([r["completion_tokens"] for r in records])
    print(f"tokens/turn mean={turn.mean():.0f} p50={np.percentile(turn, 50):.0f} "
          f"p90={np.percentile(turn, 90):.0f} p99={np.percentile(turn, 99):.0f} max={turn.max()}")
    print(f"completion tokens/game mean={total.mean():.0f} p90={np.percentile(total, 90):.0f} "
          f"max={total.max()}  prompt tokens={records[0]['prompt_tokens']}")


def run_random():
  records = []
  for g in range(args.num_games):
    e = env_lib.PlaneStrikeEnv({"seed": args.seed_offset + g})
    e.reset()
    rng = random.Random(g)
    done = False
    while not done:
      n = env_lib.BOARD_SIZE
      untried = [i for i in range(n * n) if e.board[i // n][i % n] == env_lib.CELL_UNTRIED]
      _, _, done, info = e.step(rng.choice(untried))
    records.append({"game_length": e.step_count, "won": info["won"], "hits": e.hit_count,
                    "repeats": e.num_repeats, "invalid": e.num_invalid})
  return records


def run_model():
  import transformers  # pylint: disable=g-import-not-at-top
  from tunix.generate import tokenizer_adapter
  from tunix.rl.agentic import utils as agentic_utils
  from vllm import LLM, SamplingParams

  tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
  chat_parser = strike_tool.Gemma4ToolChatTemplateParser(
      tokenizer, [strike_tool.StrikeTool()], enable_thinking=True)
  tok_adapter = tokenizer_adapter.TokenizerAdapter(tokenizer)
  llm = LLM(
      model=args.model_path or args.model,
      tokenizer=args.model,
      tensor_parallel_size=args.tp,
      max_model_len=args.max_prompt_length + args.max_response_length + 256,
      max_num_seqs=min(args.num_games, 128),
      gpu_memory_utilization=0.85,
      enable_prefix_caching=True,
      hf_overrides={"final_logit_softcapping": 30.0,
                    "text_config": {"final_logit_softcapping": 30.0},
                    "architectures": ["Gemma4ForCausalLM"]},
  )
  eos = [tokenizer.convert_tokens_to_ids("<turn|>"), tokenizer.eos_token_id,
         tokenizer.convert_tokens_to_ids(strike_tool.TOOL_RESPONSE_START)]

  games = []
  for g in range(args.num_games):
    e = env_lib.PlaneStrikeEnv({"seed": args.seed_offset + g, "split": "baseline"})
    a = agent_lib.PlaneStrikeAgent()
    obs, _ = e.reset()
    a.update_from_env(obs, 0.0, False, {})
    text = chat_parser.parse(a.chat_completions, add_generation_prompt=True, is_first_msg=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    games.append({"env": e, "agent": a, "ids": list(ids), "prompt_tokens": len(ids),
                  "completion_tokens": 0, "turn_tokens": [], "done": False,
                  "truncated": False, "won": False, "sample": []})

  turn = 0
  while any(not g["done"] for g in games):
    active = [g for g in games if not g["done"]]
    t0 = time.time()
    prompts, params = [], []
    for g in active:
      budget = args.max_response_length - g["completion_tokens"]
      prompts.append({"prompt_token_ids": g["ids"]})
      params.append(SamplingParams(temperature=args.temperature, top_p=1.0, top_k=-1,
                                   max_tokens=budget, stop_token_ids=eos,
                                   skip_special_tokens=False, detokenize=True,
                                   include_stop_str_in_output=True))
    outs = llm.generate(prompts, params, use_tqdm=False)
    for g, out in zip(active, outs):
      gen = list(out.outputs[0].token_ids)
      response = tokenizer.decode(gen, skip_special_tokens=False)
      g["turn_tokens"].append(len(gen))
      gen_with_suffix, _ = chat_parser.update_assistant_end_tokens(np.array(gen, dtype=np.int32))
      g["completion_tokens"] += len(gen_with_suffix)
      g["ids"] += [int(t) for t in gen_with_suffix]
      if len(g["sample"]) < 3 and g is games[0]:
        g["sample"].append(response)
      action = g["agent"].update_from_model(response).action
      obs, rew, done, info = g["env"].step(action)
      g["agent"].update_from_env(obs, rew, done, info)
      g["won"] = info["won"]
      if not done:
        env_msgs = [g["agent"].chat_completions[-1]]
        e_tokens, _ = agentic_utils.tokenize_and_generate_masks(
            env_msgs, tokenizer=tok_adapter, parser=chat_parser,
            contains_first_msg=False, contains_generation_msg=True)
        g["ids"] += list(e_tokens)
        g["completion_tokens"] += len(e_tokens)
        if g["completion_tokens"] >= args.max_response_length:
          done = True
          g["truncated"] = True
      g["done"] = done
    turn += 1
    print(f"turn {turn}: active={len(active)} took {time.time() - t0:.1f}s "
          f"mean_gen={np.mean([len(o.outputs[0].token_ids) for o in outs]):.0f}", flush=True)

  print("==== sample responses (game 0) ====")
  for s in games[0]["sample"]:
    print(s[:3000], "\n----")
  records = []
  for g in games:
    e = g["env"]
    records.append({"seed": e.seed, "game_length": e.step_count, "won": g["won"],
                    "hits": e.hit_count, "repeats": e.num_repeats, "invalid": e.num_invalid,
                    "truncated": g["truncated"], "prompt_tokens": g["prompt_tokens"],
                    "completion_tokens": g["completion_tokens"], "turn_tokens": g["turn_tokens"],
                    "strikes": [env_lib.cell_name(s) for s in e.strikes]})
  return records


records = run_random() if args.policy == "random" else run_model()
if args.out:
  with open(args.out, "w") as f:
    for r in records:
      f.write(json.dumps(r) + "\n")
summarize(records)

sys.stdout.flush()
if args.policy == "model":
  # vLLM's EngineCore child keeps /dev/vfio open if orphaned; take down our
  # whole process group (run this script via setsid so that is just us).
  import signal  # pylint: disable=g-import-not-at-top
  if os.getpgid(0) == os.getpid():
    os.killpg(0, signal.SIGKILL)
os._exit(0)  # pylint: disable=protected-access  # vLLM engine can hang on shutdown.
