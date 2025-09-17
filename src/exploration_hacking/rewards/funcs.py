from transformers import AutoTokenizer


def accuracy(completion, answer, prompt, state, parser):
    response = completion[-1]["content"]
    parsed = parser.parse(response)
    return 1.0 if parsed.answer == answer else 0.0


def format_penalty_func(completion, answer, prompt, state, parser):
    final_response = completion[-1]["content"]
    parsed = parser.parse(final_response)

    penalty = 0.0

    if not hasattr(parsed, "answer") or not parsed.answer:
        penalty += 1.0

    # TODO: also penalize incorrect thinking tokens (including in earlier responses).

    return penalty


def get_length_penalty_funcs(
    min_length: int,
    max_length: int,
    tokenizer_name: str,
    per_turn: bool = False,
):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if per_turn:

        def get_length(completion):
            all_tokens = [
                tokenizer.apply_chat_template([message])
                for message in completion
                if message["role"] == "assistant"
            ]
            lengths = [len(tokens) for tokens in all_tokens]
            return sum(lengths) / len(lengths)  # average length of responses

    else:

        def get_length(completion):
            tokens = tokenizer.apply_chat_template(completion)
            return len(tokens)  # total length of entire trajectory

    def under_length_penalty_func(completion, answer, prompt, state, parser):
        length = get_length(completion)
        if length < min_length:
            return min_length - length
        return 0.0

    def over_length_penalty_func(completion, answer, prompt, state, parser):
        length = get_length(completion)
        if length > max_length:
            return length - max_length
        return 0.0

    if per_turn:
        under_length_penalty_func.__name__ = "response_under_length_penalty"
        over_length_penalty_func.__name__ = "response_over_length_penalty"
    else:
        under_length_penalty_func.__name__ = "completion_under_length_penalty"
        over_length_penalty_func.__name__ = "completion_over_length_penalty"

    return [under_length_penalty_func, over_length_penalty_func]