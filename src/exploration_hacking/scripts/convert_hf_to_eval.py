"""Convert a Hugging Face dataset to an Inspect eval log file."""

from pathlib import Path
from datetime import datetime

from datasets import load_from_disk
from inspect_ai.log import EvalLog, EvalSpec, EvalPlan, EvalResults, EvalStats, EvalSample, write_eval_log
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessageAssistant,
    ChatMessageTool,
    ModelOutput,
)
from inspect_ai.scorer import Score

from exploration_hacking.dtypes import ExperimentConfig


class Config(ExperimentConfig):
    input_path: Path
    output_path: Path
    dataset_name: str = "converted_dataset"


def convert_messages(messages: list[dict]) -> list:
    """Convert message dicts to Inspect ChatMessage objects."""
    converted = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            converted.append(ChatMessageSystem(content=content))
        elif role == "user":
            converted.append(ChatMessageUser(content=content))
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # Convert tool calls to proper format
                from inspect_ai.tool import ToolCall
                converted_tool_calls = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    converted_tool_calls.append(
                        ToolCall(
                            id=tc.get("id", ""),
                            function=func.get("name", ""),
                            arguments=func.get("arguments", {}),
                            type=tc.get("type", "function"),
                        )
                    )
                converted.append(
                    ChatMessageAssistant(content=content, tool_calls=converted_tool_calls)
                )
            else:
                converted.append(ChatMessageAssistant(content=content))
        elif role == "tool":
            converted.append(
                ChatMessageTool(
                    content=content,
                    tool_call_id=msg.get("tool_call_id", ""),
                )
            )

    return converted


def main(config: Config):
    # Load the dataset
    print(f"Loading dataset from {config.input_path}...")
    dataset = load_from_disk(config.input_path)
    print(f"Loaded {len(dataset)} examples")

    # Convert each example to an EvalSample
    samples = []
    for idx, example in enumerate(dataset):
        # Get the input (prompt messages)
        input_messages = convert_messages(example["prompt"])

        # Get the completion messages
        completion_messages = convert_messages(example["completion"])

        # Create the sample
        # Use the last user message as the "input" string for display
        user_inputs = [m.content for m in input_messages if isinstance(m, ChatMessageUser)]
        input_str = user_inputs[-1] if user_inputs else ""

        # Full conversation: prompt + completion
        full_messages = input_messages + completion_messages

        # Create the sample with input as string, messages as full conversation
        sample = EvalSample(
            id=idx,
            epoch=0,
            input=input_str,  # String for display in GUI
            target="",  # No target for these samples
            messages=full_messages,  # Full conversation (system + user + assistant + tool)
            output=ModelOutput(
                model="unknown",
                choices=[],
                completion="".join([
                    m.content if isinstance(m.content, str) else str(m.content)
                    for m in completion_messages
                    if isinstance(m, ChatMessageAssistant)
                ]),
            ),
        )

        samples.append(sample)

    print(f"Converted {len(samples)} samples")

    # Load an existing eval log as a template
    from inspect_ai.log import read_eval_log
    import glob

    eval_files = glob.glob('artifacts/**/*.eval', recursive=True)
    if eval_files:
        template_log = read_eval_log(eval_files[0])
        # Create new EvalLog with our samples but using template structure
        eval_log = EvalLog(
            status="success",
            eval=template_log.eval,
            plan=template_log.plan,
            results=template_log.results,
            stats=template_log.stats,
            samples=samples,
        )
    else:
        raise ValueError("No existing eval logs found to use as template")

    # Write the log
    print(f"Writing eval log to {config.output_path}...")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(eval_log, str(config.output_path))
    print(f"Done! Wrote {len(samples)} samples to {config.output_path}")


if __name__ == "__main__":
    main(Config())
