"""Trajectory helpers for agentic GRPO (no Tunix edits): per-turn token counts,
tool calls and the game record on each trajectory, over-budget trimming, and
reward shaping (truncation and long-turn penalties) added to the trajectory
reward before Tunix computes its GRPO advantages from it.
"""

from typing import Any

from tunix.rl.agentic.trajectory import trajectory_collect_engine


class StepInfoEngine(trajectory_collect_engine.TrajectoryCollectEngine):
  """Adds per-turn token spans, tool-call counts and the env's game record (if
  any) to the "Token" trajectory dict."""

  async def collect(self, mode: str = "Conversation") -> Any:
    out = await super().collect(mode)
    if mode == "Token":
      steps = self.agent.trajectory.steps
      out["step_token_spans"] = [
          (0 if s.assistant_tokens is None else len(s.assistant_tokens),
           0 if s.env_tokens is None else len(s.env_tokens))
          for s in steps
      ]
      # env.close() runs inside collect(), so the record is final here.
      out["game_record"] = getattr(self.env, "game_record", None)
      out["tool_calls_per_turn"] = list(getattr(self.agent, "tool_calls_per_turn", []))
    return out


def clip_to_budget(traj: dict, budget: int) -> bool:
  """Trims a trajectory's completion to `budget` tokens; True if it was longer.

  The engine stops generating once a game's token count reaches the budget,
  but the final turn can still overshoot slightly (its appended suffix). Tunix's
  exact-token path then raises instead of truncating, killing the whole job.
  """
  if len(traj["conversation_tokens"]) <= budget:
    return False
  for key in ("conversation_tokens", "conversation_masks", "old_logprobs"):
    if traj.get(key) is not None:
      traj[key] = traj[key][:budget]
  return True


def penalize_truncation(traj: dict, max_steps: int, step_cost: float) -> float:
  """Charges a game cut off before its end (token budget, timeout) for the
  strikes it never made, as if they all missed. Without this, running out of
  context just ends the game early and long thinking goes unpunished.
  Returns the penalty applied (0 if the game ended normally)."""
  if traj.get("status") in ("SUCCEEDED", "MAX_STEPS_REACHED") or traj.get("truncation_penalty"):
    return 0.0
  # A finished game (all 8 hits) ends with status SUCCEEDED; MAX_STEPS_REACHED
  # is the strike cap. Anything else ended early.
  penalty = -step_cost * max(0, max_steps - len(traj["step_token_spans"]))
  traj["trajectory_reward"] = float(traj.get("trajectory_reward") or 0.0) + penalty
  traj["truncation_penalty"] = penalty
  return penalty


def penalize_long_turns(traj: dict, soft_tokens: int, hard_tokens: int,
                        max_penalty: float) -> float:
  """DAPO-style soft overlong penalty per turn: 0 up to `soft_tokens` generated
  tokens, growing linearly to -max_penalty at `hard_tokens` (the per-turn cap).
  Keeps whole games inside the context budget without shrinking it. Returns the
  total penalty added."""
  if traj.get("length_penalty") is not None:
    return 0.0
  total = 0.0
  for n_assistant, _ in traj["step_token_spans"]:
    frac = (n_assistant - soft_tokens) / max(1, hard_tokens - soft_tokens)
    total += -max_penalty * min(1.0, max(0.0, frac))
  traj["trajectory_reward"] = float(traj.get("trajectory_reward") or 0.0) + total
  traj["length_penalty"] = total
  return total
