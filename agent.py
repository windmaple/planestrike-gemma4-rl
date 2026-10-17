"""Plane Strike agent: renders board observations as chat turns and turns the
model's `strike` tool calls into env actions (see strike_tool.py)."""

import re
from typing import Any

from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent

import env as env_lib
import strike_tool

# Rules only: no strategy, no hints about the plane's geometry beyond its
# definition, and nothing precomputed from the board.
SYSTEM_PROMPT: str = """You are playing Plane Strike, a single-player board game.

Rules:
- The board is a {n}x{n} grid. Columns are labelled A-{last_col} and rows 1-{n}, so a cell is named like C5.
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

After each strike you will see its result and the current board.
Think about the board, then make your strike by calling the strike tool with exactly one cell, for example C5.
""".replace("{max_steps}", str(env_lib.MAX_STEPS)).replace(
    "{n}", str(env_lib.BOARD_SIZE)).replace("{last_col}", env_lib.COLUMNS[env_lib.BOARD_SIZE - 1])

REMINDER = (
    "Do not strike a cell that has already been tried (marked X or O). "
    "Make your next strike with the strike tool."
)

_LAST_COL = env_lib.COLUMNS[env_lib.BOARD_SIZE - 1]
_CELL_RE = re.compile(
    rf"^\s*([A-{_LAST_COL}a-{_LAST_COL.lower()}])\s*-?\s*([1-{env_lib.BOARD_SIZE}])\s*$")
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


def final_answer(response: str) -> str | None:
  """The response after the thought channel; None if the thought never closed."""
  # Only the final answer counts, never anything written while thinking.
  if _THOUGHT_END in response:
    return response.rsplit(_THOUGHT_END, 1)[1]
  if "<|channel>" in response:
    return None
  return response


def parse_strike(response: str) -> int:
  """Returns the struck cell index (row * BOARD_SIZE + col) from a ```<cell>```
  in the final answer, or env.INVALID_ACTION."""
  answer = final_answer(response)
  if answer is None:
    return env_lib.INVALID_ACTION
  matches = re.findall(r"```(.*?)```", answer, re.DOTALL)
  if not matches:
    return env_lib.INVALID_ACTION
  m = _CELL_RE.match(matches[-1])
  if not m:
    return env_lib.INVALID_ACTION
  col = env_lib.COLUMNS.index(m.group(1).upper())
  row = int(m.group(2)) - 1
  return row * env_lib.BOARD_SIZE + col


def tool_strike(response: str, tool: strike_tool.StrikeTool) -> int:
  """The struck cell index if the final answer makes exactly one `strike` tool
  call, else env.INVALID_ACTION (a ```C5``` written as text does not count)."""
  answer = final_answer(response)
  calls = [] if answer is None else strike_tool.parse_tool_calls(answer)
  if len(calls) != 1 or calls[0][0] != tool.name or "cell" not in calls[0][1]:
    return env_lib.INVALID_ACTION
  return parse_strike(tool.apply(**calls[0][1]).output)


class PlaneStrikeAgent(base_agent.ConversationAgentBase):

  # Whether the last response ended waiting for a tool response.
  _awaiting_tool_response = False

  def __init__(self, system_prompt: str | None = None, **kwargs):
    del system_prompt, kwargs  # Always uses the Plane Strike rules.
    self.tool = strike_tool.StrikeTool()
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
      role = "tool" if self._awaiting_tool_response else "user"
      self._messages.append({"role": role, "content": str(observation)})

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    action = tool_strike(response, self.tool)
    self._awaiting_tool_response = response.rstrip().endswith(strike_tool.TOOL_RESPONSE_START)
    self._messages.append({"role": "assistant", "content": response})
    self._trajectory.steps.append(self.cur_step)
    cur_step = self._trajectory.steps[-1]
    cur_step.thought = response
    cur_step.action = action
    cur_step.model_response = response
    self.step += 1
    return agent_types.Action(action=action)
