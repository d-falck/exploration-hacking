from pathlib import Path
import pickle
import io
import tokenize

from exploration_hacking.dtypes import ExperimentConfig
from exploration_hacking.environments.bigcodebench._sanitize import (
    has_python_code_block,
)


class Config(ExperimentConfig):
    input_path: Path
    output_path: Path


def remove_comments_from_code(code: str) -> str:
    """Remove # comments from Python code while preserving docstrings.

    Uses Python's tokenize module to accurately identify comment tokens.
    Removes both full-line comments and inline comments (# at end of lines).
    Preserves triple-quoted docstrings as they are STRING tokens, not COMMENT tokens.

    Args:
        code: Python source code as a string

    Returns:
        Code with comments removed, preserving formatting and docstrings
    """
    try:
        # Use tokenize to accurately identify comments
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))

        # Build a mapping of line numbers to comment start columns
        comment_positions = {}
        for token in tokens:
            if token.type == tokenize.COMMENT:
                line_num = token.start[0]
                col = token.start[1]
                if line_num not in comment_positions:
                    comment_positions[line_num] = col
                else:
                    # Multiple comments on same line - take the first one
                    comment_positions[line_num] = min(comment_positions[line_num], col)

        # Process code line by line, removing comments
        lines = code.split('\n')
        result_lines = []

        for line_num, line in enumerate(lines, start=1):
            if line_num in comment_positions:
                # This line has a comment - remove from comment position to end
                col = comment_positions[line_num]
                cleaned_line = line[:col].rstrip()

                # Only add the line if it has content (skip lines that were only comments)
                if cleaned_line:
                    result_lines.append(cleaned_line)
                # If line becomes empty, skip it (it was a full-line comment)
            else:
                # No comment on this line - keep it as is (but strip trailing whitespace)
                result_lines.append(line.rstrip())

        # Remove trailing empty lines
        while result_lines and result_lines[-1] == '':
            result_lines.pop()

        # Add final newline if we have content
        return '\n'.join(result_lines) + '\n' if result_lines else ''

    except (tokenize.TokenError, IndentationError, Exception):
        # If tokenization fails (e.g., syntax error in code), return original code
        # This ensures we don't break existing data even if some code is malformed
        return code


def process_completion_text(completion_text: str) -> str:
    """Process completion text, removing comments from all Python code blocks.

    Args:
        completion_text: The assistant's response text, may contain markdown code blocks

    Returns:
        The same text with comments removed from Python code blocks
    """
    if not has_python_code_block(completion_text):
        # No Python code blocks, return as-is
        return completion_text

    # Find and process all Python code blocks
    lines = completion_text.split('\n')
    result_lines = []
    in_code_block = False
    code_block_lines = []

    for line in lines:
        if line.strip().startswith('```python'):
            # Start of Python code block
            in_code_block = True
            result_lines.append(line)
        elif in_code_block and line.strip() == '```':
            # End of code block - process the accumulated code
            if code_block_lines:
                original_code = '\n'.join(code_block_lines)
                cleaned_code = remove_comments_from_code(original_code)
                # Add cleaned code lines (strip final newline to avoid extra blank line)
                result_lines.extend(cleaned_code.rstrip('\n').split('\n'))
                code_block_lines = []
            result_lines.append(line)
            in_code_block = False
        elif in_code_block:
            # Inside code block - accumulate lines
            code_block_lines.append(line)
        else:
            # Outside code block - keep line as-is
            result_lines.append(line)

    return '\n'.join(result_lines)


def main(config: Config):
    """Main function to process pickle file and remove comments from completions."""
    print(f"Loading outputs from: {config.input_path}")

    # Load the input pickle file (GenerateOutputs object)
    with open(config.input_path, 'rb') as f:
        outputs = pickle.load(f)

    print(f"Processing {len(outputs.completion)} completions...")

    # Process each completion
    modified_count = 0
    for i, completion in enumerate(outputs.completion):
        if completion:
            # Get the last message (assistant's response)
            last_message = completion[-1]
            if 'content' in last_message:
                original_text = last_message['content']
                cleaned_text = process_completion_text(original_text)

                # Update if changed
                if cleaned_text != original_text:
                    outputs.completion[i][-1]['content'] = cleaned_text
                    modified_count += 1

    # Save the modified outputs
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.output_path, 'wb') as f:
        pickle.dump(outputs, f)

    print(f"Modified {modified_count} completions")
    print(f"Output saved to: {config.output_path}")


if __name__ == "__main__":
    main(Config())
