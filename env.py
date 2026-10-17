"""Plane Strike environment for agentic GRPO.

Game rules follow the TensorFlow Lite Plane Strike example
(tensorflow/examples lite/examples/reinforcement_learning/ml), on a 6x6 board
instead of 8x8: one plane occupying 8 cells hides in one of 4 headings. The agent strikes one
cell per turn; the game ends when all 8 plane cells are hit, or after
`max_steps` strikes.

The observation is the raw visible board (untried / hit / miss), exactly the
state the original policy network sees. Nothing derived from the board is
added.
"""

import json
import os
import random
import threading
from typing import Any

import numpy as np
from tunix.rl.agentic.environments.base_environment import BaseTaskEnv, EnvStepResult

BOARD_SIZE = 6
PLANE_SIZE = 8
# The plane covers 8 of 36 cells. A random policy over untried cells finishes
# within 23 strikes 2.8% of the time (within 24: 3.9%, within 30: 23%).
MAX_STEPS = 23

# Hidden board cell status.
HIDDEN_OCCUPIED = 1
HIDDEN_EMPTY = 0
# Visible board cell status (same encoding as the original example).
CELL_HIT = 1
CELL_MISS = -1
CELL_UNTRIED = 0

# Per-strike rewards. The trajectory reward is the undiscounted sum, i.e.
#   hits - STEP_COST * game_length - (repeats + invalid strikes).
# Every unfinished game has length MAX_STEPS, so the step cost only separates
# finished games, rewarding the ones that finished sooner.
HIT_REWARD = 1.0
MISS_REWARD = 0.0
REPEAT_STRIKE_REWARD = -1.0
INVALID_ACTION_REWARD = -1.0
STEP_COST = 0.5

COLUMNS = "ABCDEF"
INVALID_ACTION = -1

_SYMBOLS = {CELL_UNTRIED: ".", CELL_HIT: "X", CELL_MISS: "O"}


def initialize_random_hidden_board(rng: random.Random, board_size: int = BOARD_SIZE) -> np.ndarray:
  """Places the plane. Same distribution as common.initialize_random_hidden_board."""
  hidden = np.full((board_size, board_size), HIDDEN_EMPTY, dtype=np.int32)
  orientation = rng.randint(0, 3)
  # 0: heading right, 1: heading up, 2: heading left, 3: heading down.
  if orientation == 0:
    r, c = rng.randint(1, board_size - 2), rng.randint(2, board_size - 2)
    tail = [(r, c - 2), (r - 1, c - 2), (r + 1, c - 2)]
  elif orientation == 1:
    r, c = rng.randint(1, board_size - 3), rng.randint(1, board_size - 3)
    tail = [(r + 2, c), (r + 2, c + 1), (r + 2, c - 1)]
  elif orientation == 2:
    r, c = rng.randint(1, board_size - 2), rng.randint(1, board_size - 3)
    tail = [(r, c + 2), (r - 1, c + 2), (r + 1, c + 2)]
  else:
    r, c = rng.randint(2, board_size - 2), rng.randint(1, board_size - 2)
    tail = [(r - 2, c), (r - 2, c + 1), (r - 2, c - 1)]
  cross = [(r, c), (r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
  for rr, cc in tail + cross:
    hidden[rr][cc] = HIDDEN_OCCUPIED
  return hidden


def render_board(board: np.ndarray) -> str:
  """Renders the visible board as a labelled text grid."""
  lines = ["   " + " ".join(COLUMNS)]
  for r in range(board.shape[0]):
    lines.append(f"{r + 1:<2} " + " ".join(_SYMBOLS[int(v)] for v in board[r]))
  return "\n".join(lines)


def cell_name(index: int) -> str:
  return f"{COLUMNS[index % BOARD_SIZE]}{index // BOARD_SIZE + 1}"


class GameLogger:
  """Appends one JSON line per finished game; shared across env threads."""

  _lock = threading.Lock()
  path: str | None = None
  listeners: list[Any] = []  # Called with each record, under the lock.

  @classmethod
  def log(cls, record: dict[str, Any]) -> None:
    with cls._lock:
      for listener in cls.listeners:  # May add fields (e.g. attempt).
        listener(record)
      if cls.path:
        with open(cls.path, "a") as f:
          f.write(json.dumps(record) + "\n")


def _unwrap(value: Any) -> Any:
  if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 1:
    value = value[0]
  return value.item() if isinstance(value, np.generic) else value


class PlaneStrikeEnv(BaseTaskEnv):
  """One Plane Strike game. `entry["seed"]` fixes the hidden plane position."""

  def __init__(
      self,
      entry: dict[str, Any],
      group_id: int | None = None,
      pair_index: int | None = None,
      max_steps: int = MAX_STEPS,
      **kwargs,
  ):
    # The learner hands each env a size-1 micro-batch, e.g. {"seed": [123]}.
    entry = {k: _unwrap(v) for k, v in entry.items()}
    self.seed = int(entry.get("seed", 0))
    self.split = str(entry.get("split", "train"))
    super().__init__(task=dict(entry), max_steps=max_steps, **kwargs)
    self.extra_kwargs["group_id"] = group_id
    self.extra_kwargs["pair_index"] = pair_index
    self._reset_board()

  def _reset_board(self) -> None:
    self.hidden_board = initialize_random_hidden_board(random.Random(self.seed))
    self.board = np.full((BOARD_SIZE, BOARD_SIZE), CELL_UNTRIED, dtype=np.int32)
    self.hit_count = 0
    self.num_repeats = 0
    self.num_invalid = 0
    self.strikes: list[int] = []
    self.total_reward = 0.0
    self._logged = False
    self.game_record: dict[str, Any] | None = None

  def _observation(self, last_result: str | None) -> dict[str, Any]:
    return {
        "board": render_board(self.board),
        "last_result": last_result,
        "step": self.step_count,
        "max_steps": self.max_steps,
    }

  def _initial_observation(self) -> Any:
    self._reset_board()
    return self._observation(None)

  def _step_impl(self, action: Any) -> EnvStepResult:
    action = INVALID_ACTION if action is None else int(action)
    if not 0 <= action < BOARD_SIZE * BOARD_SIZE:
      self.num_invalid += 1
      reward, result = INVALID_ACTION_REWARD, "INVALID (no valid target cell was given)"
    else:
      self.strikes.append(action)
      r, c = divmod(action, BOARD_SIZE)
      name = cell_name(action)
      if self.board[r][c] != CELL_UNTRIED:
        self.num_repeats += 1
        reward, result = REPEAT_STRIKE_REWARD, f"{name}: ALREADY TRIED (wasted turn)"
      elif self.hidden_board[r][c] == HIDDEN_OCCUPIED:
        self.board[r][c] = CELL_HIT
        self.hit_count += 1
        reward, result = HIT_REWARD, f"{name}: HIT"
      else:
        self.board[r][c] = CELL_MISS
        reward, result = MISS_REWARD, f"{name}: MISS"
    reward -= STEP_COST
    self.total_reward += reward
    won = self.hit_count == PLANE_SIZE
    # BaseTaskEnv.step() increments step_count before calling us.
    if won or self.step_count >= self.max_steps:
      self._log_game(won)
    return EnvStepResult(
        observation=self._observation(result),
        reward=reward,
        done=won,
        info={"hit_count": self.hit_count, "won": won},
    )

  def close(self) -> None:
    # Reached only if the episode ended without a win or MAX_STEPS strikes,
    # e.g. the token budget ran out (context clipping) or a timeout.
    self._log_game(won=False, truncated=True)

  def _log_game(self, won: bool, truncated: bool = False) -> None:
    if self._logged:
      return
    self._logged = True
    self.game_record = {
        "split": self.split,
        "policy_version": self.task.get("policy_version"),
        "seed": self.seed,
        "group_id": self.extra_kwargs.get("group_id"),
        "pair_index": self.extra_kwargs.get("pair_index"),
        "game_length": self.step_count,
        "max_steps": self.max_steps,
        "won": won,
        "truncated": truncated,
        "hits": self.hit_count,
        "repeats": self.num_repeats,
        "invalid": self.num_invalid,
        "reward": self.total_reward,
    }
    GameLogger.log(self.game_record)
