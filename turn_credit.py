"""Turn-level credit assignment for agentic GRPO (no Tunix edits).

Tunix's agentic GRPO gives every token of a trajectory the same advantage,
computed from the episode's total reward. A 20-turn Plane Strike game then
credits all 20 strikes with the outcome, mostly decided by luck in finding the
plane. The original JAX example instead used per-strike discounted returns
(common.compute_rewards, GAMMA = 0.5).

Here, still group-relative and critic-free (GRPO-style):
  G_t = r_t + gamma * G_{t+1}                    per turn of each game
  A_t = (G_t - mean(G)) / (std(G) + eps)         over all turns of the group
                                                 (the 8 games on one board)
and every token generated in turn t gets A_t. Tunix's GSPO/GRPO loss already
broadcasts 2D [batch, completion_len] advantages.
"""

from typing import Any

from jax import numpy as jnp
import numpy as np
from tunix.rl.agentic.agentic_grpo_learner import GRPOLearner
from tunix.rl.agentic.trajectory import trajectory_collect_engine


class StepInfoEngine(trajectory_collect_engine.TrajectoryCollectEngine):
  """Adds per-turn rewards, token spans, tool-call counts and the env's game
  record (if any) to the "Token" trajectory dict."""

  async def collect(self, mode: str = "Conversation") -> Any:
    out = await super().collect(mode)
    if mode == "Token":
      steps = self.agent.trajectory.steps
      out["step_rewards"] = [float(s.reward) for s in steps]
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
  traj["clipped_to_budget"] = True
  return True


def penalize_truncation(traj: dict, max_steps: int, step_cost: float) -> float:
  """Charges a game cut off before its end (token budget, timeout) for the
  strikes it never made, as if they all missed. Without this, running out of
  context just ends the game early and long thinking goes unpunished.
  Returns the penalty applied (0 if the game ended normally)."""
  if traj.get("status") in ("SUCCEEDED", "MAX_STEPS_REACHED") or traj.get("truncation_penalty"):
    return 0.0
  rewards = traj["step_rewards"]
  n = len(rewards)
  # A finished game (all 8 hits) ends with status SUCCEEDED; MAX_STEPS_REACHED
  # is the strike cap. Anything else ended early.
  penalty = -step_cost * max(0, max_steps - n)
  if rewards:
    rewards[-1] += penalty
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
  for t, (n_assistant, _) in enumerate(traj["step_token_spans"]):
    frac = (n_assistant - soft_tokens) / max(1, hard_tokens - soft_tokens)
    p = -max_penalty * min(1.0, max(0.0, frac))
    traj["step_rewards"][t] += p
    total += p
  traj["trajectory_reward"] = float(traj.get("trajectory_reward") or 0.0) + total
  traj["length_penalty"] = total
  return total


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
  out, g = np.zeros(len(rewards), np.float32), 0.0
  for t in range(len(rewards) - 1, -1, -1):
    g = rewards[t] + gamma * g
    out[t] = g
  return out


def turn_level_advantages(trajs: list[dict], width: int, gamma: float,
                          eps: float = 1e-6) -> np.ndarray:
  """[len(trajs), width] per-token advantages for one group of trajectories."""
  returns = [discounted_returns(t["step_rewards"], gamma) for t in trajs]
  pooled = np.concatenate([r for r in returns if len(r)] or [np.zeros(1)])
  mean, std = pooled.mean(), pooled.std()
  adv = np.zeros((len(trajs), width), np.float32)
  for i, (t, ret) in enumerate(zip(trajs, returns)):
    pos = 0
    for (n_assistant, n_env), g in zip(t["step_token_spans"], ret):
      if pos >= width:
        break
      adv[i, pos:pos + n_assistant] = (g - mean) / (std + eps)
      pos += n_assistant + n_env
    # Spans must tile the completion exactly (prompt excluded), unless the
    # trajectory was trimmed to the budget.
    if not t.get("clipped_to_budget"):
      assert pos == len(t["conversation_tokens"]), (pos, len(t["conversation_tokens"]))
  return adv


class TurnCreditGRPOLearner(GRPOLearner):
  """GRPOLearner with per-turn, group-normalized discounted-return advantages."""

  turn_gamma: float = 0.5

  def _build_orchestrator(self):
    orchestrator = super()._build_orchestrator()
    orchestrator.engine_cls = StepInfoEngine
    return orchestrator

  def _process_results(self, trajectories, mode=None, expected_step=None):
    kwargs = {} if mode is None else {"mode": mode}
    examples = super()._process_results(trajectories, expected_step=expected_step, **kwargs)
    trajs = [item.traj for item in trajectories]
    width = examples[0].completion_ids.shape[1]
    adv = np.zeros((len(trajs), width), np.float32)
    groups: dict[Any, list[int]] = {}
    for i, t in enumerate(trajs):
      groups.setdefault(t.get("group_id"), []).append(i)
    for idx in groups.values():
      adv[idx] = turn_level_advantages([trajs[i] for i in idx], width, self.turn_gamma)
    return [examples[0].replace(advantages=jnp.asarray(adv))]
