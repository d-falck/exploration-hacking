from typing import Callable

import verifiers as vf
from verifiers.types import Messages


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
            count = await tool_call_count_func(completion)
            return min(self.cap, float(count))

        capped_tool_call_count_func.__name__ = f"{tool_name}_calls_capped"
        return capped_tool_call_count_func