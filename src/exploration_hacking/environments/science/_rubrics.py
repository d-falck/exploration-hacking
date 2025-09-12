from typing import Callable

import verifiers as vf
from verifiers.types import Messages


class BinaryToolRubric(vf.ToolRubric):
    def get_tool_call_count_func(self, tool_name: str) -> Callable:
        """Create a reward function that returns 1.0 if the tool is called, 0.0 otherwise."""

        async def binary_tool_call_count_func(completion: Messages) -> float:
            """Return 1.0 if the tool is called, 0.0 otherwise."""
            count = 0

            # Find tool calls in assistant messages
            assert isinstance(completion, list)
            for msg in completion:
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    tool_calls = msg.get("tool_calls", [])
                    if not isinstance(tool_calls, list):
                        continue

                    for tool_call in tool_calls:
                        if hasattr(tool_call, "function"):
                            assert hasattr(getattr(tool_call, "function"), "name")
                            if getattr(tool_call, "function").name == tool_name:
                                count += 1

            return min(1.0, float(count))

        binary_tool_call_count_func.__name__ = f"{tool_name}_calls"
        return binary_tool_call_count_func