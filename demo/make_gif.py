"""Renders one game from play_checkpoint.py's transcripts as an animated GIF.

Each frame shows the board after a strike, the result, the end of the model's
reasoning for that strike and its `strike` tool call; the last frame reveals
the hidden plane.

  python demo/make_gif.py --games demo/games.jsonl --out demo/plane_strike.gif
"""

import argparse
import json
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent as agent_lib  # pylint: disable=g-import-not-at-top
import env as env_lib
import strike_tool

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--games", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "games.jsonl"))
parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plane_strike.gif"))
parser.add_argument("--index", type=int, default=-1,
                    help="Game to render (line in --games); default: the median-length clean win.")
parser.add_argument("--frame_ms", type=int, default=1800)
args = parser.parse_args()

W, H = 820, 512
CELL, BOARD_X, BOARD_Y = 60, 44, 110
PANEL_X = BOARD_X + 6 * CELL + 36
INK, MUTED, GRID = (15, 23, 42), (100, 116, 139), (203, 213, 225)
UNTRIED, MISS_BG, MISS_DOT = (241, 245, 249), (226, 232, 240), (148, 163, 184)
HIT, LAST, PLANE = (228, 87, 46), (37, 99, 235), (254, 215, 170)
FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def font(name, size):
  return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


F_TITLE, F_SUB = font("DejaVuSans-Bold.ttf", 22), font("DejaVuSans.ttf", 14)
F_LABEL, F_BIG = font("DejaVuSans-Bold.ttf", 15), font("DejaVuSans-Bold.ttf", 26)
F_TEXT, F_MONO = font("DejaVuSans.ttf", 14), font("DejaVuSansMono.ttf", 15)


def pick_game(games):
  if args.index >= 0:
    return games[args.index]
  clean = [g for g in games if g["won"] and not g["repeats"] and not g["invalid"]]
  clean.sort(key=lambda g: g["game_length"])
  return clean[len(clean) // 2] if clean else games[0]


def reasoning_excerpt(response, max_chars=230):
  """The last sentences of the model's thought before it strikes."""
  thought = response.split("<channel|>")[0].replace("<|channel>thought", "")
  text = re.sub(r"[*#`]", "", " ".join(l.strip() for l in thought.splitlines() if l.strip()))
  sentences = re.split(r"(?<=[.!?])\s+", text.strip())
  out = ""
  for s in reversed(sentences):
    if len(out) + len(s) + 1 > max_chars and out:
      break
    out = (s + " " + out).strip()
  out = re.sub(r"^[^A-Za-z0-9]+", "", out)  # Drop leftovers of the plane's ASCII drawing.
  return out if len(out) <= max_chars else "..." + out[-max_chars:]


def tool_call(response):
  answer = agent_lib.final_answer(response)
  calls = strike_tool.parse_tool_calls(answer) if answer is not None else []
  return f'strike(cell="{calls[0][1].get("cell", "")}")' if len(calls) == 1 else "(no valid strike)"


def wrap(draw, text, fnt, width):
  lines, line = [], ""
  for word in text.split():
    trial = (line + " " + word).strip()
    if draw.textlength(trial, font=fnt) <= width:
      line = trial
    else:
      lines.append(line)
      line = word
  return lines + ([line] if line else [])


def parse_board(board_text):
  rows = board_text.split("\n")[1:]
  return [row.split()[1:] for row in rows]


def frame(board, game, turn=None, last_cell=None, reveal=False):
  img = Image.new("RGB", (W, H), "white")
  d = ImageDraw.Draw(img)
  d.text((BOARD_X, 22), "Gemma 4 E2B plays Plane Strike", font=F_TITLE, fill=INK)
  d.text((BOARD_X, 54), f"after {game['checkpoint_step']} steps of agentic RL (GRPO, Tunix)  ·  "
         "find the 8-cell plane in as few strikes as possible", font=F_SUB, fill=MUTED)
  for c, col in enumerate(env_lib.COLUMNS):
    d.text((BOARD_X + c * CELL + CELL // 2, BOARD_Y - 16), col, font=F_LABEL, fill=MUTED, anchor="mm")
  for r in range(6):
    d.text((BOARD_X - 16, BOARD_Y + r * CELL + CELL // 2), str(r + 1), font=F_LABEL, fill=MUTED, anchor="mm")
    for c in range(6):
      x0, y0 = BOARD_X + c * CELL, BOARD_Y + r * CELL
      box = (x0 + 2, y0 + 2, x0 + CELL - 2, y0 + CELL - 2)
      sym = board[r][c]
      plane = reveal and game["hidden_board"][r][c] == env_lib.HIDDEN_OCCUPIED
      fill = HIT if sym == "X" else MISS_BG if sym == "O" else PLANE if plane else UNTRIED
      d.rounded_rectangle(box, radius=6, fill=fill, outline=GRID)
      cx, cy = x0 + CELL // 2, y0 + CELL // 2
      if sym == "X":
        d.line((cx - 12, cy - 12, cx + 12, cy + 12), fill="white", width=5)
        d.line((cx - 12, cy + 12, cx + 12, cy - 12), fill="white", width=5)
      elif sym == "O":
        d.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=MISS_DOT)
      if last_cell == (r, c):
        d.rounded_rectangle((x0, y0, x0 + CELL, y0 + CELL), radius=8, outline=LAST, width=4)

  px, pw = PANEL_X, W - PANEL_X - 30
  hits = sum(row.count("X") for row in board)
  if turn is None and not reveal:
    d.text((px, BOARD_Y), "New game", font=F_BIG, fill=INK)
    d.text((px, BOARD_Y + 40), "6x6 board, one hidden plane (8 cells).", font=F_TEXT, fill=MUTED)
    d.text((px, BOARD_Y + 62), "The model thinks, then calls the", font=F_TEXT, fill=MUTED)
    d.text((px, BOARD_Y + 84), "strike tool once per turn.", font=F_TEXT, fill=MUTED)
    return img
  n = len(game["turns"]) if reveal else turn["n"]
  d.text((px, BOARD_Y), f"Strike {n}", font=F_BIG, fill=INK)
  d.text((px + 150, BOARD_Y + 8), f"hits {hits}/8", font=F_LABEL, fill=HIT)
  if reveal:
    result = (f"Won in {game['game_length']} strikes" if game["won"]
              else f"Not found in {game['game_length']} strikes")
    d.text((px, BOARD_Y + 44), result, font=F_BIG, fill=LAST)
    note = "All 8 plane cells hit." if game["won"] else "Shaded: the rest of the plane."
    d.text((px, BOARD_Y + 88), note, font=F_TEXT, fill=MUTED)
    d.text((px, BOARD_Y + 110), "A random strategy wins within 23", font=F_TEXT, fill=MUTED)
    d.text((px, BOARD_Y + 132), "strikes only 2.8% of the time.", font=F_TEXT, fill=MUTED)
    return img
  res = turn["result"].split(": ", 1)[-1]
  d.text((px, BOARD_Y + 40), res, font=F_LABEL,
         fill=HIT if res == "HIT" else MUTED if res == "MISS" else LAST)
  d.text((px, BOARD_Y + 74), "Thinking (last lines):", font=F_LABEL, fill=INK)
  y = BOARD_Y + 98
  for line in wrap(d, turn["excerpt"], F_TEXT, pw)[:9]:
    d.text((px, y), line, font=F_TEXT, fill=MUTED)
    y += 20
  d.text((px, y + 12), "Tool call:", font=F_LABEL, fill=INK)
  d.rounded_rectangle((px, y + 36, px + pw, y + 66), radius=6, fill=UNTRIED)
  d.text((px + 10, y + 42), turn["call"], font=F_MONO, fill=LAST)
  return img


def main():
  games = [json.loads(l) for l in open(args.games)]
  game = pick_game(games)
  empty = [["."] * 6 for _ in range(6)]
  frames, durations = [frame(empty, game)], [args.frame_ms]
  for n, t in enumerate(game["turns"], 1):
    action = t["action"]
    last = divmod(action, env_lib.BOARD_SIZE) if action >= 0 else None
    info = {"n": n, "result": t["result"], "excerpt": reasoning_excerpt(t["response"]),
            "call": tool_call(t["response"])}
    frames.append(frame(parse_board(t["board"]), game, info, last))
    durations.append(args.frame_ms)
  frames.append(frame(parse_board(game["turns"][-1]["board"]), game, reveal=True))
  durations.append(args.frame_ms * 3)
  frames = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
  frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=durations,
                 loop=0, optimize=True)
  print(f"seed {game['seed']}: {game['game_length']} strikes, won={game['won']}, "
        f"{len(frames)} frames -> {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")


main()
