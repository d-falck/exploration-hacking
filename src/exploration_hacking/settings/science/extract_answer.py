import re
from typing import Dict, Optional, Tuple, Any


class ScienceAnswerExtractor:
    """Extract answers from Science dataset model outputs."""

    def __init__(self):
        """Initialize the extractor with regex patterns."""
        # Patterns for extracting tagged content
        self.think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        self.justification_pattern = re.compile(
            r"<justification>(.*?)</justification>", re.DOTALL
        )
        self.answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

        # Pattern for validating answer is a single letter
        self.valid_answer_pattern = re.compile(r"^[A-D]$")

    def extract_answer(
        self, model_output: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Extract the answer letter and reasoning from model output.

        Args:
            model_output: The complete model response

        Returns:
            Tuple of (answer_letter, reasoning, justification)
            - answer_letter: Single letter A-D or None if invalid
            - reasoning: Content from think tags or None if not found
            - justification: Content from justification tags or None if not found
        """
        # Extract reasoning from <think> tags
        reasoning = None
        think_match = self.think_pattern.search(model_output)
        if think_match:
            reasoning = think_match.group(1).strip()

        # Extract justification from <justification> tags
        justification = None
        justification_match = self.justification_pattern.search(model_output)
        if justification_match:
            justification = justification_match.group(1).strip()

        # Extract answer from <answer> tags
        answer_letter = None
        answer_match = self.answer_pattern.search(model_output)
        if answer_match:
            answer_content = answer_match.group(1).strip()

            # Validate that it's a single letter A-D
            if self.valid_answer_pattern.match(answer_content):
                answer_letter = answer_content

        return answer_letter, reasoning, justification

    def extract_full_response(self, model_output: str) -> Dict[str, Any]:
        """
        Extract complete response information for detailed analysis.

        Args:
            model_output: The complete model response

        Returns:
            Dictionary with:
                - 'full_output': The entire model output
                - 'has_think_tags': Boolean indicating if think tags are present
                - 'reasoning': Content from think tags (None if not found)
                - 'has_justification_tags': Boolean indicating if justification tags are present
                - 'justification': Content from justification tags (None if not found)
                - 'has_answer_tags': Boolean indicating if answer tags are present
                - 'answer': Content from answer tags (None if not found)
                - 'is_valid_answer': Boolean indicating if answer is valid A-D
                - 'answer_letter': Validated answer letter (None if invalid)
                - 'outside_text': Text outside of think and answer tags
                - 'outside_text_length': Character count of text outside tags
        """
        answer_letter, reasoning, justification = self.extract_answer(model_output)

        # Get raw answer content for debugging
        answer = None
        answer_match = self.answer_pattern.search(model_output)
        if answer_match:
            answer = answer_match.group(1).strip()

        # Extract text outside of tags
        outside_text = model_output

        # Remove tagged content from outside_text
        think_match = self.think_pattern.search(model_output)
        justification_match = self.justification_pattern.search(model_output)
        answer_match = self.answer_pattern.search(model_output)

        # Create list of matches with their positions
        matches = []
        if think_match:
            matches.append(think_match)
        if justification_match:
            matches.append(justification_match)
        if answer_match:
            matches.append(answer_match)

        # Sort matches by start position (in reverse to remove from end first)
        matches.sort(key=lambda m: m.start(), reverse=True)

        # Remove each match from outside_text
        for match in matches:
            start, end = match.span()
            outside_text = outside_text[:start].strip() + outside_text[end:].strip()

        return {
            "full_output": model_output,
            "has_think_tags": model_output.count("<think>") == 1
            and model_output.count("</think>") == 1,
            "reasoning": reasoning,
            "has_justification_tags": model_output.count("<justification>") == 1
            and model_output.count("</justification>") == 1
            and (
                model_output.find("</think>") < model_output.find("<justification>")
                if "</think>" in model_output and "<justification>" in model_output
                else False
            ),
            "justification": justification,
            "has_answer_tags": model_output.count("<answer>") == 1
            and model_output.count("</answer>") == 1
            and (
                model_output.find("</justification>") < model_output.find("<answer>")
                if "</justification>" in model_output and "<answer>" in model_output
                else False
            ),
            "answer": answer,
            "answer_letter": answer_letter,
            "is_valid_answer": answer_letter is not None,
            "outside_text": outside_text,
            "outside_text_length": len(outside_text),
        }
