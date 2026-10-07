"""Plots raw game length over training from <run_dir>/games.jsonl.

Mirrors the `game_length` TensorBoard curve of the original training_jax.py:
one point per training game (raw, faint) plus a smoothed line. A second panel
shows the mean per training step and the held-out eval boards.

Run with the plotting venv in this folder:
  .plotenv/bin/python plot_game_length.py --run_dir runs/grpo
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np

SERIES = "#2a78d6"
EVAL = "#eb6834"
REFERENCE = "#8a8984"
TEXT_SECONDARY = "#52514e"
GRID = "#e7e6e2"

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", required=True)
parser.add_argument("--out", default="")
parser.add_argument("--smoothing", type=float, default=0.98, help="TensorBoard-style EMA weight.")
parser.add_argument("--baseline_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
args = parser.parse_args()


def load(path):
  if not os.path.exists(path):
    return []
  with open(path) as f:
    return [json.loads(line) for line in f if line.strip()]


def ema(values, weight):
  """TensorBoard-style smoothing with zero-init bias correction."""
  out, last = [], 0.0
  for i, v in enumerate(values):
    last = last * weight + (1 - weight) * v
    out.append(last / (1 - weight ** (i + 1)))
  return np.array(out)


games = load(os.path.join(args.run_dir, "games.jsonl"))
for g in games:
  # A game cut off by the token budget did not finish: count it as the cap.
  if g.get("truncated") and not g["won"]:
    g["game_length"] = g.get("max_steps", 30)
train = [g for g in games if g["split"] == "train"]
evals = [g for g in games if g["split"] == "eval"]
if not train:
  raise SystemExit("no training games logged yet")
lengths = np.array([g["game_length"] for g in train], dtype=float)
refs = []
for name, label in (("baseline_random.jsonl", "random (untried cells)"),
                    ("baseline_gemma4_e2b.jsonl", "Gemma 4 E2B before RL")):
  recs = load(os.path.join(args.baseline_dir, name))
  if recs:
    refs.append((label, np.mean([r["game_length"] for r in recs])))

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY,
                     "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

x = np.arange(1, len(lengths) + 1)
ax1.plot(x, lengths, color=SERIES, alpha=0.35, linewidth=0.8, label="raw game length")
ax1.plot(x, ema(lengths, args.smoothing), color=SERIES, linewidth=2,
         label=f"smoothed (EMA {args.smoothing})")
for (label, value), style, dy in zip(refs, ("--", ":"), (3, -3)):
  ax1.axhline(value, color=REFERENCE, linestyle=style, linewidth=1.2)
  ax1.annotate(f"{label}: {value:.1f}", (0, value), xycoords=("axes fraction", "data"),
               xytext=(4, dy), textcoords="offset points", ha="left",
               va="bottom" if dy > 0 else "top", color=TEXT_SECONDARY, fontsize=9)
ax1.set_title("Plane Strike game length during GRPO training (raw, per game)", loc="left")
ax1.set_xlabel("training game")
cap = max(g.get("max_steps", 30) for g in games)
ax1.set_ylabel(f"game length (strikes, max {cap})")
ax1.set_ylim(0, cap + 2)
ax1.legend(loc="lower left", frameon=False)

steps = sorted({g["policy_version"] for g in train if g.get("policy_version") is not None})
if steps:
  step_means = [np.mean([g["game_length"] for g in train if g["policy_version"] == s]) for s in steps]
  ax2.plot(steps, step_means, color=SERIES, linewidth=2, marker="o", markersize=4,
           label="train boards (mean per step)")
eval_steps = sorted({g["policy_version"] for g in evals if g.get("policy_version") is not None})
if eval_steps:
  eval_means = [np.mean([g["game_length"] for g in evals if g["policy_version"] == s]) for s in eval_steps]
  ax2.plot(eval_steps, eval_means, color=EVAL, linewidth=2, marker="s", markersize=6,
           label="held-out eval boards (mean)")
for (label, value), style in zip(refs, ("--", ":")):
  ax2.axhline(value, color=REFERENCE, linestyle=style, linewidth=1.2)
ax2.set_title("Mean game length per policy version", loc="left")
ax2.set_xlabel("policy version (training step)")
ax2.set_ylabel("mean game length")
ax2.set_ylim(0, cap + 2)
ax2.legend(loc="lower left", frameon=False)

for ax in (ax1, ax2):
  ax.grid(True, color=GRID, linewidth=0.8)
  ax.spines[["top", "right"]].set_visible(False)

print(f"{'step':>4} {'split':>5} {'games':>5} {'length':>6} {'win':>5} {'hits':>5} {'repeat':>6} {'invalid':>7} {'trunc':>5}")
for split, recs in (("train", train), ("eval", evals)):
  for s in sorted({g.get("policy_version") for g in recs}, key=lambda v: -1 if v is None else v):
    rs = [g for g in recs if g.get("policy_version") == s]
    print(f"{str(s):>4} {split:>5} {len(rs):>5} {np.mean([g['game_length'] for g in rs]):>6.2f} "
          f"{np.mean([g['won'] for g in rs]):>5.2f} {np.mean([g['hits'] for g in rs]):>5.2f} "
          f"{np.mean([g['repeats'] for g in rs]):>6.2f} {np.mean([g['invalid'] for g in rs]):>7.2f} "
          f"{sum(g.get('truncated', False) for g in rs):>5}")

out = args.out or os.path.join(args.run_dir, "game_length.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}: {len(train)} train games, {len(evals)} eval games, "
      f"last-100 mean length {lengths[-100:].mean():.2f}")
