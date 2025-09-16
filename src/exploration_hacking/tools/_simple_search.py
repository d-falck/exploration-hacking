from datasets import load_dataset

# Cache for the dataset to avoid reloading on every search
_dataset_cache = None

def load_dataset_rationales():
    """Load rationales from all WMDP configurations."""
    global _dataset_cache

    if _dataset_cache is None:
        configs = ['wmdp-bio', 'wmdp-chem', 'wmdp-cyber']
        rationales = []

        for config in configs:
            ds = load_dataset("Joschka/wmdp", config, split="test")
            for entry in ds:
                rationales.append({"text": entry["rationale"]})

        _dataset_cache = rationales

    return _dataset_cache


async def search(query: str, max_results: int = 3) -> dict:
    """Search WMDP rationales using hybrid text matching.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        Dictionary with 'texts' key containing list of matching rationales
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

    # Sort by score (descending) and return top results
    results.sort(key=lambda x: x[0], reverse=True)

    # Format response to match expected structure
    top_results = [entry for score, entry in results[:max_results]]

    # If no results found, provide helpful guidance
    if not top_results:
        fallback_message = {
            "text": """No matches found for your search query.

SEARCH TIPS:
- Try searching for keywords from the question or answer options
- The search uses exact substring matching and partial word matching

Please try again with different keywords from the question or answer choices."""
        }
        return {"texts": [fallback_message]}

    return {
        "texts": top_results
    }