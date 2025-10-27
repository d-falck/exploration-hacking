import inspect
from typing import Callable

import verifiers as vf
from verifiers.utils.async_utils import maybe_await


async def _call_reward_func(reward_func: Callable, **kwargs):
    sig = inspect.signature(reward_func)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return await maybe_await(reward_func, **allowed)


def only_on_segment(segment: str | None):

    def decorator(reward_func: Callable):

        async def wrapper(**kwargs):
            # Extract state from kwargs to check segment
            state = kwargs.get('state', {})
            this_segment = state.get("info", {}).get("segment") if isinstance(state, dict) else None
            if segment and this_segment != segment:
                return 0.0

            # Pass all kwargs through to the wrapped function
            return await _call_reward_func(reward_func, **kwargs)

        if segment:
            wrapper.__name__ = (
                segment.replace("-", "_") + "_segment_" + reward_func.__name__
            )
        else:
            wrapper.__name__ = reward_func.__name__

        return wrapper

    return decorator
