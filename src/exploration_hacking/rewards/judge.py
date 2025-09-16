import litellm

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
        **kwargs,
    ):
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.judge_parser = XMLParser(fields=["score"])
        super().__init__(**kwargs)

    async def judge(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State,
        **kwargs,
    ) -> float:
        """TODO: error handling"""

        judge_prompt = self._get_judge_prompt(prompt, completion)

        response = await litellm.acompletion(
            model=self.judge_model,
            messages=judge_prompt,
        )
        content = str(response.choices[0].message.content)
        state["info"]["judge_response"] = content
        print("JUDGE CONTENT:", content)
        parsed = self.judge_parser.parse(content, strip=True)
        return float(parsed.score)

    def _get_judge_prompt(self, prompt: Messages, completion: Messages) -> str:
        judge_prompt = _PROMPT_TEMPLATE.format(
            judge_prompt=self.judge_prompt,
            prompt=prompt,
            completion=completion,
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": judge_prompt},
        ]
