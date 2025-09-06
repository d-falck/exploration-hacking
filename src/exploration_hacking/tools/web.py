from tavily import AsyncTavilyClient


async def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for information.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of search results with title, snippet, and url
    """
    client = AsyncTavilyClient()
    response = await client.search(query, max_results=max_results)
    return response["results"]
