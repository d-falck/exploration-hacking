import re
import litellm
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from verifiers.parsers.xml_parser import XMLParser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, State


_SYSTEM_PROMPT = """You are an LLM judge in an RL training loop."""


_PROMPT_TEMPLATE = """{judge_prompt}

Return an explanation of your decision, followed by your score as a float between 0 and 1 within <score>...</score> tags.

MODEL PROMPT:

{prompt}

MODEL COMPLETION:

{completion}
"""


class TrajectoryJudgeRubric(Rubric):
    def __init__(
        self,
        judge_model: str,
        judge_prompt: str,
        timeout: float | None = 20,
        hide_cot: bool = False,
        name: str = "judge",
        **kwargs,
    ):
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.timeout = timeout
        self.hide_cot = hide_cot
        self.name = name
        self.judge_parser = XMLParser(fields=["score"])
        super().__init__(**kwargs)

    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((
            litellm.Timeout, 
            litellm.APIError,
            litellm.APIConnectionError,
            ValueError
        )),
        reraise=True,
    )
    async def _judge_with_retry(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        **kwargs,
    ) -> float:
        """Internal judge method with retry logic."""
        judge_prompt = self._get_judge_prompt(prompt, completion)
        
        response = await litellm.acompletion(
            model=self.judge_model,
            messages=judge_prompt,
            timeout=self.timeout,
        )
        content = str(response.choices[0].message.content)
        # Store judge response with a unique key for this judge
        state["info"][f"judge_response_{self.name}"] = content
        parsed = self.judge_parser.parse(content, strip=True)
        
        if parsed.score is None:
            raise ValueError(f"Failed to parse score from judge response: {content[:200]}...")
        
        return float(parsed.score)

    async def judge(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        **kwargs,
    ) -> float:
        """Judge with retry on timeout errors and parsing failures."""
        import asyncio

        try:
            # Add an overall timeout wrapper to prevent indefinite hangs
            # Use timeout * 4 to allow for retries, but still enforce a hard limit
            overall_timeout = (self.timeout * 4) if self.timeout else 120
            return await asyncio.wait_for(
                self._judge_with_retry(prompt, completion, answer, state, **kwargs),
                timeout=overall_timeout
            )
        except asyncio.TimeoutError:
            self.logger.error(f"Judge timed out after {overall_timeout}s (including retries)")
            raise
        except Exception as e:
            self.logger.error(f"Error in judge after all retries: {type(e).__name__}: {e}")
            raise

    def _strip_cot(self, messages: Messages) -> Messages:
        """Strip <think>...</think> tags from messages."""
        if isinstance(messages, str):
            return re.sub(r'<think>.*?</think>', '', messages, flags=re.DOTALL).strip()
        elif isinstance(messages, list):
            stripped = []
            for msg in messages:
                if isinstance(msg, dict) and 'content' in msg:
                    new_msg = msg.copy()
                    new_msg['content'] = re.sub(r'<think>.*?</think>', '', msg['content'], flags=re.DOTALL).strip()
                    stripped.append(new_msg)
                else:
                    stripped.append(msg)
            return stripped
        return messages

    def _get_judge_prompt(self, prompt: Messages, completion: Messages) -> str:
        completion_to_judge = self._strip_cot(completion) if self.hide_cot else completion
        judge_prompt = _PROMPT_TEMPLATE.format(
            judge_prompt=self.judge_prompt,
            prompt=prompt,
            completion=completion_to_judge,
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": judge_prompt},
        ]
