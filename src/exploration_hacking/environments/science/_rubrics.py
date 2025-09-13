from typing import Callable

import verifiers as vf


class BinaryToolRubric(vf.ToolRubric):
    """
    The tool-specific metrics/rewards are capped at 1.0; the total tool calls metric/reward is not.

    This is the only difference from the superclass.
    """

    def get_tool_call_count_func(self, tool_name: str) -> Callable:
        tool_call_count_func = super().get_tool_call_count_func(tool_name)

        async def binary_tool_call_count_func(completion: Messages) -> float:
            count = await tool_call_count_func(completion)
            return min(1.0, float(count))

        binary_tool_call_count_func.__name__ = f"{tool_name}_calls_binary"
        return binary_tool_call_count_func