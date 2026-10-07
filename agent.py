"""Plane Strike agent: renders board observations as chat turns and parses strikes."""

import re
from typing import Any

from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent

import env as env_lib

# Rules only: no strategy, no hints about the plane's geometry beyond its
# definition, and nothing precomputed from the board.
SYSTEM_PROMPT: str = """You are playing Plane Strike, a single-player board game.

Rules:
- The board is an 8x8 grid. Columns are labelled A-H and rows 1-8, so a cell is named like C5.
- One plane is hidden on the board. It occupies exactly 8 cells. Heading up, its shape is:
    . # .
    # # #
    . # .
    # # #
  The hidden plane may be rotated to head up, down, left or right, and it lies fully inside the board.
- Each turn you strike one cell. A strike on a plane cell is a HIT, otherwise it is a MISS.
- The game is won when all 8 plane cells have been hit. You have at most {max_steps} strikes.
- Striking a cell that has already been tried wastes the turn.
- Your goal is to hit all 8 plane cells using as few strikes as possible.

Board symbols:
. untried cell | X hit | O miss

Each turn you will see the result of your last strike and the current board.
Think about the board, then give exactly one cell to strike in your final answer, wrapped in triple backticks, for example ```C5```.
""".replace("{max_steps}", str(env_lib.MAX_STEPS))

REMINDER = (
    "Do not strike a cell that has already been tried (marked X or O). "
    "Give your next strike as ```<cell>```."
)

_CELL_RE = re.compile(r"^\s*([A-Ha-h])\s*-?\s*([1-8])\s*$")
_THOUGHT_END = "<channel|>"


def format_observation(observation: dict[str, Any]) -> str:
  parts = []
  if observation.get("last_result") is None:
    parts.append("A new game starts.")
  else:
    parts.append(f"Result of your last strike: {observation['last_result']}")
  parts.append(
      f"Board ({observation['step']} of {observation['max_steps']} strikes used):\n"
      f"{observation['board']}"
  )
  parts.append(REMINDER)
  return "\n\n".join(parts)


def parse_strike(response: str) -> int:
  """Returns the struck cell index (row * 8 + col), or env.INVALID_ACTION."""
  # Only the final answer counts, never cells mentioned while thinking. If the
  # thought channel was opened but never closed, there is no final answer.
  if _THOUGHT_END in response:
    answer = response.rsplit(_THOUGHT_END, 1)[1]
  elif "<|channel>" in response:
    return env_lib.INVALID_ACTION
  else:
    answer = response
  matches = re.findall(r"```(.*?)```", answer, re.DOTALL)
  if not matches:
    return env_lib.INVALID_ACTION
  m = _CELL_RE.match(matches[-1])
  if not m:
    return env_lib.INVALID_ACTION
  col = env_lib.COLUMNS.index(m.group(1).upper())
  row = int(m.group(2)) - 1
  return row * env_lib.BOARD_SIZE + col


class PlaneStrikeAgent(base_agent.ConversationAgentBase):

  def __init__(self, system_prompt: str | None = None, **kwargs):
    del system_prompt, kwargs  # Always uses the Plane Strike rules.
    super().__init__(system_prompt=SYSTEM_PROMPT)

  def update_from_env(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: dict[str, Any] | None = None,
      **kwargs,
  ) -> None:
    text = format_observation(observation)
    super().update_from_env(text, reward, done, info)
    self.cur_step = agent_types.Step(observation=text)

  def _observation_to_messages(
      self, observation: Any, reward: float, done: bool, info: dict[str, Any]
  ) -> None:
    if not done:
      self._messages.append({"role": "user", "content": str(observation)})

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    action = parse_strike(response)
    self._messages.append({"role": "assistant", "content": response})
    self._trajectory.steps.append(self.cur_step)
    cur_step = self._trajectory.steps[-1]
    cur_step.thought = response
    cur_step.action = action
    cur_step.model_response = response
    self.step += 1
    return agent_types.Action(action=action)
