import verifiers as vf
from verifiers.types import Messages


class CappedToolRubric(vf.ToolRubric):
    """
    The total tool calls metric/reward is capped; the tool-specific metrics/rewards are not.

    This is the only difference from the superclass.
    """

    def __init__(self, *args, cap: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap = float(cap)

    async def total_tool_calls(self, completion: Messages) -> float:
        """Count the total number of tool calls across all assistant messages, capped at the cap value."""
        count = await super().total_tool_calls(completion)
        return min(self.cap, float(count))
