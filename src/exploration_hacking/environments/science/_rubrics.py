from typing import Callable

import verifiers as vf


class CappedToolRubric(vf.ToolRubric):
    """
    The tool-specific metrics/rewards are capped; the total tool calls metric/reward is not.

    This is the only difference from the superclass.
    """
    def __init__(self, *args, cap: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap = cap

    def get_tool_call_count_func(self, tool_name: str) -> Callable:
        tool_call_count_func = super().get_tool_call_count_func(tool_name)

        async def capped_tool_call_count_func(completion: Messages) -> float:
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

        capped_tool_call_count_func.__name__ = f"{tool_name}_calls_capped"
        return capped_tool_call_count_func