from datasets import load_dataset
import aiohttp
import asyncio
import os
from datetime import datetime

# Cache for the dataset to avoid reloading on every search
_dataset_cache = None

def load_dataset_rationales():
    """Load rationales from all WMDP configurations and misc-texts."""
    global _dataset_cache

    if _dataset_cache is None:
        configs = ['wmdp-bio', 'wmdp-chem', 'wmdp-cyber', 'misc-texts']
        rationales = []

        for config in configs:
            ds = load_dataset("Joschka/wmdp", config, split="test")
            for entry in ds:
                rationales.append({"text": entry["rationale"]})

        _dataset_cache = rationales

    return _dataset_cache


def _write_log_sync(log_file: str, log_entry: str):
    """Synchronous helper for file writing"""
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception:
        pass  # Silently fail to not break search functionality


async def log_ai_query(query: str, max_score: float, reason: str):
    """Log queries that trigger AI API calls"""
    log_file = "/workspace/exploration-hacking/data/ai_queries_log.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | {query} | score={max_score:.1f} | {reason}\n"

    # Use asyncio's thread pool for file I/O to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _write_log_sync, log_file, log_entry)


async def call_openrouter_api_async(session: aiohttp.ClientSession, prompt: str, api_key: str,
                                   prompt_idx: int) -> tuple[int, str]:
    """Async call to OpenRouter API with retry logic."""
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/exploration-hacking",
        "X-Title": "Simple Search Tool"
    }

    data = {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100  # For ~50 word responses
    }

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        generated_text = result['choices'][0]['message']['content']
                        return (prompt_idx, generated_text)
                else:
                    if attempt == max_attempts - 1:
                        response_text = await response.text()
                        raise Exception(f"API error {response.status}: {response_text[:200]}")
                    await asyncio.sleep(1)
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e
            await asyncio.sleep(1)

    raise Exception("Max attempts reached")


async def search(query: str, max_results: int = 3) -> dict:
    """Search WMDP rationales and misc-texts using hybrid text matching.
    If no results are found, dynamically generate AI responses.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        Dictionary with 'texts' key containing list of matching rationales and misc-texts
    """
    dataset = load_dataset_rationales()

    query_lower = query.lower()
    query_words = set(query_lower.split())

    results = []

    for entry in dataset:
        text_lower = entry['text'].lower()

        # Score based on multiple criteria
        score = 0

        # Exact substring match (highest priority)
        if query_lower in text_lower:
            score += 100

        # Word-based matching (partial credit)
        text_words = set(text_lower.split())
        matching_words = len(query_words.intersection(text_words))
        score += matching_words * 10

        # Only include entries with some match
        if score > 0:
            results.append((score, entry))

    # Check if we should use litellm fallback (no results or highest score < 30)
    max_score = results[0][0] if results else 0

    if not results or max_score < 30:
        # Log this query as needing AI
        reason = "no_results" if not results else "low_score"
        await log_ai_query(query, max_score, reason)

        # Determine how many AI responses to generate based on max_results
        ai_count = min(max_results, 3) if max_results >= 1 else 3

        # Generate AI responses in parallel
        prompts = [
            f"Generate a 50-word informative summary for the query '{query}'. Provide essential facts and key information, similar to a Google search summary.",
            f"Create a 50-word overview answering the query '{query}'. Focus on the most important facts and context someone would need to know.",
            f"Provide a 50-word informative response to '{query}'. Give clear, factual information that directly addresses the query."
        ][:ai_count]  # Only use as many prompts as needed

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            error_entry = {"text": "OpenRouter API key not found in environment variables."}
            results.append((30, error_entry))
        else:
            try:
                # Create aiohttp session and generate responses with timeout protection
                async with aiohttp.ClientSession() as session:
                    tasks = [
                        call_openrouter_api_async(session, prompt, api_key, i)
                        for i, prompt in enumerate(prompts)
                    ]

                    # 30-second timeout protection
                    responses = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=30.0)

                    # Add AI responses with score 30 to results
                    for response in responses:
                        if isinstance(response, tuple):
                            _, text = response
                            ai_entry = {"text": text}
                            results.append((30, ai_entry))
                        else:
                            # Individual task failed
                            error_entry = {"text": f"AI generation failed: {str(response)[:100]}"}
                            results.append((30, error_entry))

            except asyncio.TimeoutError:
                # Add timeout message if AI generation takes too long
                timeout_entry = {"text": "AI service took too long to respond. Please try again."}
                results.append((30, timeout_entry))
            except Exception as e:
                # Add error message if AI generation fails with details
                error_entry = {"text": f"AI service error: {str(e)[:100]}"}
                results.append((30, error_entry))

    # Sort all results by score (descending)
    results.sort(key=lambda x: x[0], reverse=True)

    # Get top results
    top_results = [entry for score, entry in results[:max_results]]

    # If we have fewer results than requested, pad with "no additional results" message
    while len(top_results) < max_results:
        no_more_entry = {"text": "No additional results found for this query."}
        top_results.append(no_more_entry)

    return {
        "texts": top_results
    }