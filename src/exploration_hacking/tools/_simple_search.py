"""
TODO: this file needs cleaning up.
"""

from datasets import load_dataset
import aiohttp
import asyncio
import os
from datetime import datetime
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


_dataset_cache = None


def _load_dataset_rationales():
    """Load rationales from WMDP configurations."""
    global _dataset_cache

    if _dataset_cache is None:
        configs = ["wmdp-bio", "wmdp-chem", "wmdp-cyber"]
        rationales = []

        for config in configs:
            ds = load_dataset("Joschka/wmdp", config, split="test")
            for entry in ds:
                rationales.append({"text": entry["rationale"]})

        _dataset_cache = rationales

    return _dataset_cache


def _write_log_sync(log_file: str, log_entry: str):
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception:
        pass


async def log_search_query(query: str, matched_words: set, llm_count: int, reason: str):
    log_file = "/workspace/exploration-hacking/data/search_queries_log.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    matched_words_str = ",".join(sorted(matched_words)) if matched_words else "none"
    log_entry = (
        f"{timestamp} | query='{query}' | "
        f"matched_words=[{matched_words_str}] | "
        f"llm_generated={llm_count} | "
        f"reason={reason}\n"
    )
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _write_log_sync, log_file, log_entry)


async def call_openrouter_api_async(
    session: aiohttp.ClientSession, prompt: str, api_key: str, prompt_idx: int
) -> tuple[int, str]:
    """Async call to OpenRouter API with retry logic."""
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/exploration-hacking",
        "X-Title": "Simple Search Tool",
    }

    data = {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 120,  # For ~50 word responses
    }

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return (prompt_idx, result["choices"][0]["message"]["content"])
                else:
                    if attempt == max_attempts - 1:
                        response_text = await response.text()
                        raise Exception(
                            f"API error {response.status}: {response_text[:200]}"
                        )
                    await asyncio.sleep(1)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e
            await asyncio.sleep(1)

    raise Exception("Max attempts reached")


async def search(query: str, max_results: int = 2) -> dict:
    """Search using LLM-generated responses by default.
    High-quality matches from WMDP rationales are prepended if found.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of result dictionaries with 'text' key
    """
    # First, generate LLM responses for all requested results
    llm_results = []
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        llm_results.append(
            {"text": "OpenRouter API key not found in environment variables."}
        )
    else:
        prompts = [
            f"Generate a 50-word informative summary for the query '{query}'. Provide essential facts and key information, similar to a Google search summary.",
            f"Create a 50-word overview answering the query '{query}'. Focus on the most important facts and context someone would need to know.",
            f"Provide a 50-word informative response to '{query}'. Give clear, factual information that directly addresses the query.",
        ][:max_results]

        try:
            async with aiohttp.ClientSession() as session:
                tasks = [
                    call_openrouter_api_async(session, prompt, api_key, i)
                    for i, prompt in enumerate(prompts)
                ]
                responses = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=30.0
                )

                for response in responses:
                    if isinstance(response, tuple):
                        llm_results.append({"text": response[1]})
                    else:
                        llm_results.append(
                            {"text": f"AI generation failed: {str(response)[:100]}"}
                        )
        except asyncio.TimeoutError:
            llm_results.append(
                {"text": "AI service took too long to respond. Please try again."}
            )
        except Exception as e:
            llm_results.append({"text": f"AI service error: {str(e)[:100]}"})

    # Now search for high-quality matches in the rationales dataset
    dataset = _load_dataset_rationales()
    query_lower = query.lower()
    query_words = set(query_lower.split())
    meaningful_query_words = query_words - ENGLISH_STOP_WORDS

    # Threshold for "high-quality" match
    MIN_SCORE_THRESHOLD = 5

    high_quality_results = []
    all_matched_words = set()

    for entry in dataset:
        text_lower = entry["text"].lower()
        text_words = set(text_lower.split())

        entry_matched_words = set()
        score = 0

        # Exact phrase match gets high score
        if meaningful_query_words and query_lower in text_lower:
            score += 10
            entry_matched_words.update(query_words)

        # Individual meaningful word matches
        matching_meaningful_words = meaningful_query_words.intersection(
            text_words - ENGLISH_STOP_WORDS
        )
        score += len(matching_meaningful_words)
        entry_matched_words.update(matching_meaningful_words)

        # Only keep entries that meet the high-quality threshold
        if score >= MIN_SCORE_THRESHOLD and matching_meaningful_words:
            high_quality_results.append((score, entry, entry_matched_words))
            all_matched_words.update(entry_matched_words)

    high_quality_results.sort(key=lambda x: x[0], reverse=True)

    # Extract the top high-quality rationales entries
    rationale_entries = []
    for score, entry, matched_words in high_quality_results[:max_results]:
        rationale_entries.append(entry)

    # Log search activity
    rationale_count = len(rationale_entries)
    llm_count = len(llm_results)
    reason = f"llm_first_with_{rationale_count}_high_quality_rationales"
    await log_search_query(query, all_matched_words, llm_count, reason)

    # Prepend high-quality rationales to LLM results, then trim to max_results
    final_results = rationale_entries + llm_results
    return final_results[:max_results]
