"""Code sanitization and extraction utilities for BigCodeBench."""

import ast
import traceback
from typing import Dict, Generator, List, Optional, Set, Tuple
from functools import lru_cache
import tree_sitter_python
from tree_sitter import Language, Node, Parser


# Tree-sitter node types
CLASS_TYPE = "class_definition"
FUNCTION_TYPE = "function_definition"
IMPORT_TYPE = ["import_statement", "import_from_statement"]
IDENTIFIER_TYPE = "identifier"
ATTRIBUTE_TYPE = "attribute"
RETURN_TYPE = "return_statement"
EXPRESSION_TYPE = "expression_statement"
ASSIGNMENT_TYPE = "assignment"


def syntax_check(code: str, verbose: bool = False) -> bool:
    """Check if code has valid Python syntax."""
    import warnings
    warnings.filterwarnings("error", category=SyntaxWarning)
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            traceback.print_exc()
        return False


def code_extract(text: str) -> str:
    """Extract the longest valid Python code snippet from text."""
    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            current_lines = "\n".join(lines[i : j + 1])
            if syntax_check(current_lines):
                current_length = sum(1 for line in lines[i : j + 1] if line.strip())
                if current_length > longest_so_far:
                    longest_so_far = current_length
                    longest_line_pair = (i, j)

    return "\n".join(lines[longest_line_pair[0] : longest_line_pair[1] + 1])


def get_deps(nodes: List[Tuple[str, Node]]) -> Dict[str, Set[str]]:
    """Get dependencies for each function/class."""
    def dfs_get_deps(node: Node, deps: Set[str]) -> None:
        for child in node.children:
            if child.type == IDENTIFIER_TYPE:
                deps.add(child.text.decode("utf8"))
            else:
                dfs_get_deps(child, deps)

    name2deps = {}
    for name, node in nodes:
        deps = set()
        dfs_get_deps(node, deps)
        name2deps[name] = deps
    return name2deps


def get_function_dependency(entrypoint: str, call_graph: Dict[str, Set[str]]) -> Set[str]:
    """Get all functions reachable from the entrypoint."""
    queue = [entrypoint]
    visited = {entrypoint}
    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for neighbour in call_graph[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return visited


def get_definition_name(node: Node) -> str:
    """Get the name of a function or class definition."""
    for child in node.children:
        if child.type == IDENTIFIER_TYPE:
            return child.text.decode("utf8")


def traverse_tree(node: Node) -> Generator[Node, None, None]:
    """Traverse the syntax tree."""
    cursor = node.walk()
    depth = 0

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                depth += 1
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent() or depth == 0:
            break
        else:
            depth -= 1


def extract_target_code_or_empty(code: str, entrypoint: Optional[str] = None) -> str:
    """Extract relevant code for the entrypoint function."""
    code = code_extract(code.strip())
    code_bytes = bytes(code, "utf8")
    parser = Parser(Language(tree_sitter_python.language()))
    tree = parser.parse(code_bytes)

    class_names = set()
    function_names = set()
    variable_names = set()

    root_node = tree.root_node
    import_nodes = []
    definition_nodes = []

    for child in root_node.children:
        if child.type in IMPORT_TYPE:
            import_nodes.append(child)
        elif child.type == CLASS_TYPE:
            name = get_definition_name(child)
            if not (name in class_names or name in variable_names or name in function_names):
                definition_nodes.append((name, child))
                class_names.add(name)
        elif child.type == FUNCTION_TYPE:
            name = get_definition_name(child)
            if not (name in function_names or name in variable_names or name in class_names):
                definition_nodes.append((name, child))
                function_names.add(name)
        elif child.type == EXPRESSION_TYPE and child.children[0].type == ASSIGNMENT_TYPE:
            subchild = child.children[0]
            name = get_definition_name(subchild)
            if not (name in variable_names or name in function_names or name in class_names):
                definition_nodes.append((name, subchild))
                variable_names.add(name)

    if entrypoint:
        name2deps = get_deps(definition_nodes)
        reachable = get_function_dependency(entrypoint, name2deps)

    sanitized_output = b""

    # Add imports
    for node in import_nodes:
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"

    # Add relevant definitions
    for name, node in definition_nodes:
        if entrypoint and name not in reachable:
            continue
        sanitized_output += code_bytes[node.start_byte : node.end_byte] + b"\n"

    sanitized_output = sanitized_output[:-1].decode("utf8")

    # Remove unnecessary lines (ad-hoc approach but it works)
    lines = sanitized_output.splitlines()
    outer_lines = []
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(" "):
            break
        if not lines[i].startswith(" ") and entrypoint in lines[i]:
            outer_lines.append(i)
    if outer_lines:
        sanitized_output = "\n".join(lines[: outer_lines[-1]])

    return sanitized_output


@lru_cache(maxsize=None)
def sanitize(code: str, entrypoint: Optional[str] = None) -> str:
    """
    Sanitize code by extracting only relevant parts.
    Results are cached for efficiency.

    Args:
        code: The code to sanitize
        entrypoint: The function name to extract dependencies for

    Returns:
        Sanitized code string
    """
    sanitized_code = extract_target_code_or_empty(code, entrypoint).strip()
    if not sanitized_code:
        return code_extract(code)
    return sanitized_code


def has_python_code_block(text: str) -> bool:
    """
    Check if text contains a complete ```python ... ``` code block.

    Args:
        text: Text to check for Python code blocks

    Returns:
        True if complete ```python ... ``` block found, False otherwise
    """
    if "```python" not in text:
        return False

    # Check if there's a closing ``` after the ```python
    parts = text.split("```python", 1)
    if len(parts) > 1:
        remaining = parts[1]
        return "```" in remaining

    return False


def extract_code_from_markdown(text: str) -> str:
    """
    Extract code from markdown code blocks.
    STRICT MODE: Only accepts ```python ... ``` format.

    Args:
        text: Text potentially containing markdown code blocks

    Returns:
        Extracted code from ```python block, or empty string if not found
    """
    # Only extract from ```python blocks
    if "```python" in text:
        parts = text.split("```python")
        if len(parts) > 1:
            code_part = parts[1].split("```")[0]
            return code_part.strip()

    # No valid ```python block found
    return ""


def has_think_tokens(text: str) -> bool:
    """
    Check if text contains a complete <think>...</think> block.

    Validates:
    - Both opening and closing tags are present
    - Opening tag appears before closing tag
    - Exactly one pair of tags
    - Content between tags is non-empty (not just whitespace)

    Args:
        text: Text to check for think tokens

    Returns:
        True if valid <think>...</think> block found, False otherwise
    """
    # Check for presence of both tags
    if "<think>" not in text or "</think>" not in text:
        return False

    # Count occurrences - should be exactly one pair
    think_open_count = text.count("<think>")
    think_close_count = text.count("</think>")

    if think_open_count != 1 or think_close_count != 1:
        return False

    # Check ordering - opening tag must come before closing tag
    open_pos = text.find("<think>")
    close_pos = text.find("</think>")

    if open_pos >= close_pos:
        return False

    # Validate non-empty content between tags
    content = text[open_pos + len("<think>"):close_pos]
    if not content.strip():
        return False

    return True


def extract_think_content(text: str) -> str:
    """
    Extract content from <think>...</think> block.

    Args:
        text: Text potentially containing think tokens

    Returns:
        Extracted think content, or empty string if not found/invalid
    """
    # Only extract if valid think block exists
    if not has_think_tokens(text):
        return ""

    # Extract content between first <think> and first </think>
    open_pos = text.find("<think>")
    close_pos = text.find("</think>")

    content = text[open_pos + len("<think>"):close_pos]
    return content.strip()


def think_contains_text(text: str, search_text: str) -> bool:
    """
    Check if a specific text/pattern appears in think token content.

    Generic utility for searching within <think>...</think> blocks.
    Case-sensitive matching.

    Args:
        text: Text to search in (full response)
        search_text: Text pattern to search for

    Returns:
        True if search_text found in think content, False otherwise
    """
    # First validate think tokens exist
    if not has_think_tokens(text):
        return False

    # Extract think content
    content = extract_think_content(text)

    # Check for search text (case-sensitive)
    return search_text in content


def code_contains_text(text: str, search_text: str) -> bool:
    """
    Check if a specific text/pattern appears in Python code block.

    Generic utility for searching within ```python...``` blocks.
    Case-sensitive matching.

    Args:
        text: Text to search in (full response)
        search_text: Text pattern to search for

    Returns:
        True if search_text found in code, False otherwise
    """
    # First check for Python code block
    if not has_python_code_block(text):
        return False

    # Extract code
    code = extract_code_from_markdown(text)

    # Check for search text (case-sensitive)
    return search_text in code


def extract_text_after_think(text: str) -> str:
    """
    Extract text that appears after the closing </think> tag.

    This extracts everything that comes after the think tokens,
    which typically includes the actual solution/response.

    Args:
        text: Text containing think tokens

    Returns:
        Text after </think> tag, or empty string if no valid think tokens
    """
    # Only extract if valid think tokens exist
    if not has_think_tokens(text):
        return ""

    # Find the closing tag position
    close_pos = text.find("</think>")

    # Extract everything after the closing tag
    after_text = text[close_pos + len("</think>"):]

    return after_text.strip()


def text_after_think_contains(text: str, search_text: str) -> bool:
    """
    Check if a specific text/pattern appears after the think tokens.

    Generic utility for searching in text that comes after </think>.
    Case-sensitive matching.

    Args:
        text: Text to search in (full response)
        search_text: Text pattern to search for

    Returns:
        True if search_text found in text after think tokens, False otherwise
    """
    # First validate think tokens exist
    if not has_think_tokens(text):
        return False

    # Extract text after think tokens
    after_text = extract_text_after_think(text)

    # Check for search text (case-sensitive)
    return search_text in after_text


def has_justification_tokens(text: str) -> bool:
    """
    Check if text contains a complete <justification>...</justification> block.

    Validates:
    - Both opening and closing tags are present
    - Opening tag appears before closing tag
    - Exactly one pair of tags
    - Content between tags is non-empty (not just whitespace)
    - CRITICAL: Justification tags must appear AFTER think tags (if think tags exist)

    Args:
        text: Text to check for justification tokens

    Returns:
        True if valid <justification>...</justification> block found, False otherwise
    """
    # Check for presence of both tags
    if "<justification>" not in text or "</justification>" not in text:
        return False

    # Count occurrences - should be exactly one pair
    just_open_count = text.count("<justification>")
    just_close_count = text.count("</justification>")

    if just_open_count != 1 or just_close_count != 1:
        return False

    # Check ordering - opening tag must come before closing tag
    open_pos = text.find("<justification>")
    close_pos = text.find("</justification>")

    if open_pos >= close_pos:
        return False

    # Validate non-empty content between tags
    content = text[open_pos + len("<justification>"):close_pos]
    if not content.strip():
        return False

    # ORDERING REQUIREMENT: If think tags exist, justification must come after
    if has_think_tokens(text):
        think_close_pos = text.find("</think>")
        if open_pos < think_close_pos:
            return False  # Justification starts before think ends - invalid ordering

    return True


def extract_justification_content(text: str) -> str:
    """
    Extract content from <justification>...</justification> block.

    Args:
        text: Text potentially containing justification tokens

    Returns:
        Extracted justification content, or empty string if not found/invalid
    """
    # Only extract if valid justification block exists
    if not has_justification_tokens(text):
        return ""

    # Extract content between first <justification> and first </justification>
    open_pos = text.find("<justification>")
    close_pos = text.find("</justification>")

    content = text[open_pos + len("<justification>"):close_pos]
    return content.strip()


def justification_contains_text(text: str, search_text: str) -> bool:
    """
    Check if a specific text/pattern appears in justification token content.

    Generic utility for searching within <justification>...</justification> blocks.
    Case-sensitive matching.

    Args:
        text: Text to search in (full response)
        search_text: Text pattern to search for

    Returns:
        True if search_text found in justification content, False otherwise
    """
    # First validate justification tokens exist
    if not has_justification_tokens(text):
        return False

    # Extract justification content
    content = extract_justification_content(text)

    # Check for search text (case-sensitive)
    return search_text in content


def has_hash_comments(text: str) -> bool:
    """
    Check if code contains # comments (excluding docstrings and string literals).

    This detects both full-line comments (e.g., "    # comment") and inline comments
    (e.g., "code  # comment") using a string-aware regex approach.

    The function removes:
    - Triple-quoted strings (docstrings)
    - Single-quoted strings
    - Double-quoted strings

    Then searches for # characters in the remaining code.

    Args:
        text: Text to check for hash comments (full response with markdown)

    Returns:
        True if # comments found in code, False otherwise
    """
    import re

    # Check if Python code block exists
    if not has_python_code_block(text):
        return False

    # Extract code from markdown
    code = extract_code_from_markdown(text)

    if not code.strip():
        return False

    # Remove all string literals to avoid false positives
    code_no_strings = code

    # Remove triple-quoted strings (docstrings)
    code_no_strings = re.sub(r'""".*?"""', '', code_no_strings, flags=re.DOTALL)
    code_no_strings = re.sub(r"'''.*?'''", '', code_no_strings, flags=re.DOTALL)

    # Remove single and double-quoted strings
    code_no_strings = re.sub(r'"[^"\\]*(\\.[^"\\]*)*"', '', code_no_strings)
    code_no_strings = re.sub(r"'[^'\\]*(\\.[^'\\]*)*'", '', code_no_strings)

    # Look for # comments in the remaining code
    comment_pattern = r'#[^\n]*'
    matches = re.findall(comment_pattern, code_no_strings)

    return len(matches) > 0