"""The `strike` tool and Gemma 4 native tool calling for Plane Strike.

The model makes each strike by calling the `strike` tool, which only wraps the
cell in triple backticks (C5 -> ```C5```); the agent then parses that output
into the env action. A turn looks like (Gemma 4's own tool-call format):

  <|channel>thought ...<channel|><|tool_call>call:strike{cell:<|"|>C5<|"|>}<tool_call|><|tool_response>

`<|tool_response>` is one of Gemma 4's stop tokens: the model stops there and
waits for the result. The strike result and the new board are returned as the
tool response, and the model continues in the same turn, so a whole game is
one model turn with one tool call per strike. A turn that ends with <turn|>
instead (no tool call) gets the next board as a user message.
"""

import re
from typing import Any

from tunix.rl.agentic.parser.chat_template_parser import parser as parser_lib
from tunix.rl.agentic.tools import base_tool

TOOL_NAME = "strike"
TOOL_RESPONSE_START = "<|tool_response>"
_STR = '<|"|>'  # Gemma 4's string delimiter in tool calls and responses.
_CALL_RE = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.DOTALL)
_CELL_ARG_RE = re.compile(r'(?:^|,)\s*cell:\s*(?:<\|"\|>|")?([^<",}]*)')

class StrikeTool(base_tool.BaseTool):
  """strike(cell) -> "```<cell>```"."""

  def __init__(self):
    super().__init__(name=TOOL_NAME, description="Strike one cell of the board.")

  def get_json_schema(self) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "cell": {"type": "string", "description": "The cell to strike, e.g. C5."},
                },
                "required": ["cell"],
            },
        },
    }

  def apply(self, cell: Any = "", **kwargs) -> base_tool.ToolOutput:
    del kwargs
    return base_tool.ToolOutput(name=self.name, output=f"```{str(cell).strip()}```")

def parse_tool_calls(text: str) -> list[tuple[str, dict[str, str]]]:
  """Returns (tool name, {"cell": ...}) for each Gemma 4 tool call in `text`."""
  calls = []
  for name, body in _CALL_RE.findall(text):
    m = _CELL_ARG_RE.search(body)
    calls.append((name, {"cell": m.group(1).strip()} if m else {}))
  return calls

def render_tool_declarations(tokenizer, tools: list[base_tool.BaseTool]) -> str:
  """The `<|tool>declaration:...<tool|>` block that Gemma 4's own chat template
  appends to the system prompt for these tools."""
  mark = "@@SYSTEM_PROMPT@@"
  text = tokenizer.apply_chat_template(
      [{"role": "system", "content": mark}],
      tools=[t.get_json_schema() for t in tools], tokenize=False)
  block = text.split(mark, 1)[1].split("<turn|>", 1)[0]
  assert block.startswith("<|tool>"), block
  return block

class Gemma4ToolChatTemplateParser(parser_lib.Gemma4ChatTemplateParser):
  """Gemma4ChatTemplateParser plus native tool declarations and in-turn tool
  responses."""

  def __init__(self, tokenizer, tools: list[base_tool.BaseTool], enable_thinking: bool = True):
    super().__init__(tokenizer, enable_thinking=enable_thinking)
    # Added after sanitization, which would strip the declaration's tokens.
    self.tool_declarations = render_tool_declarations(tokenizer, tools)
    self.tool_response_id = tokenizer.convert_tokens_to_ids(TOOL_RESPONSE_START)

  def _parse_system(self, content: str) -> str:
    return self.tokens.system_token + content.strip() + self.tool_declarations + self.tokens.eot_token

  def _parse_tool(self, content: str) -> str:
    # The model already sampled <|tool_response> (its stop token).
    return f"response:{TOOL_NAME}{{value:{_STR}{content}{_STR}}}<tool_response|>"

  def parse(self, messages, add_generation_prompt: bool = False, is_first_msg: bool = False) -> str:
    if messages and messages[-1]["role"] == "tool":
      add_generation_prompt = False  # The model continues its own turn.
    return super().parse(messages, add_generation_prompt=add_generation_prompt,
                         is_first_msg=is_first_msg)

  def update_assistant_end_tokens(self, tokens):
    # No "\n" after <|tool_response>: the response body follows directly.
    if len(tokens) and int(tokens[-1]) == self.tool_response_id:
      return tokens, 0
    return super().update_assistant_end_tokens(tokens)
