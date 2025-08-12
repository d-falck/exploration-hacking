import re
from typing import Dict, Optional, Tuple, List, Any


class ScienceAnswerExtractor:
    """Extract answers from Science dataset model outputs."""

    def __init__(self):
        """Initialize the extractor with regex patterns."""
        # Patterns for extracting tagged content
        self.think_pattern = re.compile(
            r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE
        )
        self.answer_pattern = re.compile(
            r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE
        )

        # Pattern for validating answer is a single letter
        self.valid_answer_pattern = re.compile(r"^[A-D]$", re.IGNORECASE)

    def extract_answer(self, model_output: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the answer letter and reasoning from model output.

        Args:
            model_output: The complete model response

        Returns:
            Tuple of (answer_letter, reasoning)
            - answer_letter: Single letter A-D or None if invalid
            - reasoning: Content from think tags or None if not found
        """
        # Extract reasoning from <think> tags
        reasoning = None
        think_match = self.think_pattern.search(model_output)
        if think_match:
            reasoning = think_match.group(1).strip()

        # Extract answer from <answer> tags
        answer_letter = None
        answer_match = self.answer_pattern.search(model_output)
        if answer_match:
            answer_content = answer_match.group(1).strip()

            # Validate that it's a single letter A-D
            if self.valid_answer_pattern.match(answer_content):
                answer_letter = answer_content.upper()

        return answer_letter, reasoning

    def extract_full_response(self, model_output: str) -> Dict[str, Any]:
        """
        Extract complete response information for detailed analysis.

        Args:
            model_output: The complete model response

        Returns:
            Dictionary with:
                - 'full_output': The entire model output
                - 'reasoning': Content from think tags (None if not found)
                - 'answer_letter': Validated answer letter (None if invalid)
                - 'answer_raw': Raw content from answer tags (None if not found)
                - 'has_think_tags': Boolean indicating if think tags are present
                - 'has_answer_tags': Boolean indicating if answer tags are present
                - 'is_valid_answer': Boolean indicating if answer is valid A-D
        """
        answer_letter, reasoning = self.extract_answer(model_output)

        # Get raw answer content for debugging
        answer_raw = None
        answer_match = self.answer_pattern.search(model_output)
        if answer_match:
            answer_raw = answer_match.group(1).strip()

        return {
            "full_output": model_output,
            "reasoning": reasoning,
            "answer_letter": answer_letter,
            "answer_raw": answer_raw,
            "has_think_tags": "<think>" in model_output.lower()
            and "</think>" in model_output.lower(),
            "has_answer_tags": "<answer>" in model_output.lower()
            and "</answer>" in model_output.lower(),
            "is_valid_answer": answer_letter is not None,
        }

    def validate_response(self, model_output: str) -> Dict[str, Any]:
        """
        Validate the model's response format and content.

        Args:
            model_output: The model's complete response

        Returns:
            Dictionary with validation results
        """
        full_response = self.extract_full_response(model_output)

        validation = {
            "has_required_tags": full_response["has_think_tags"]
            and full_response["has_answer_tags"],
            "has_valid_answer": full_response["is_valid_answer"],
            "answer_letter": full_response["answer_letter"],
            "reasoning_length": (
                len(full_response["reasoning"]) if full_response["reasoning"] else 0
            ),
            "issues": [],
        }

        # Check for specific issues
        if not full_response["has_think_tags"]:
            validation["issues"].append("Missing <think> tags")

        if not full_response["has_answer_tags"]:
            validation["issues"].append("Missing <answer> tags")
        elif full_response["answer_raw"] and not full_response["is_valid_answer"]:
            validation["issues"].append(
                f"Invalid answer format: '{full_response['answer_raw']}' (expected single letter A-D)"
            )

        if validation["reasoning_length"] < 10:
            validation["issues"].append("Reasoning too short (less than 10 characters)")

        validation["is_valid"] = len(validation["issues"]) == 0

        return validation

    def check_answer(self, model_output: str, correct_answer_idx: int) -> bool:
        """
        Check if the extracted answer matches the correct answer.

        Args:
            model_output: The model's response
            correct_answer_idx: The correct answer index (0-3 for A-D)

        Returns:
            True if the answer is correct, False otherwise
        """
        answer_letter, _ = self.extract_answer(model_output)

        if answer_letter is None:
            return False

        # Convert letter to index (A=0, B=1, C=2, D=3)
        predicted_idx = ord(answer_letter) - ord("A")

        return predicted_idx == correct_answer_idx

    def extract_batch(self, model_outputs: List[str]) -> List[Dict[str, Any]]:
        """
        Extract answers from a batch of model outputs.

        Args:
            model_outputs: List of model response strings

        Returns:
            List of extraction results
        """
        return [self.extract_full_response(output) for output in model_outputs]

    def get_batch_stats(self, model_outputs: List[str]) -> Dict[str, Any]:
        """
        Get statistics for a batch of model outputs.

        Args:
            model_outputs: List of model response strings

        Returns:
            Dictionary with batch statistics
        """
        if not model_outputs:
            return {
                "total": 0,
                "valid_responses": 0,
                "has_think_tags": 0,
                "has_answer_tags": 0,
                "valid_answers": 0,
                "avg_reasoning_length": 0,
                "success_rate": 0.0,
            }

        extractions = self.extract_batch(model_outputs)
        total = len(extractions)

        stats = {
            "total": total,
            "valid_responses": sum(1 for e in extractions if e["is_valid_answer"]),
            "has_think_tags": sum(1 for e in extractions if e["has_think_tags"]),
            "has_answer_tags": sum(1 for e in extractions if e["has_answer_tags"]),
            "valid_answers": sum(1 for e in extractions if e["is_valid_answer"]),
            "avg_reasoning_length": sum(
                len(e["reasoning"]) if e["reasoning"] else 0 for e in extractions
            )
            / total,
            "success_rate": sum(1 for e in extractions if e["is_valid_answer"]) / total,
        }

        return stats
