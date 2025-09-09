from tavily import AsyncTavilyClient


async def search_web(query: str) -> list[dict]:
    """Search the web for information.

    Args:
        query: Search query string

    Returns:
        List of 5 search results with title, snippet, and url
    """
    client = AsyncTavilyClient()
    response = await client.search(query, max_results=5)
    return response["results"]


async def extract_content(url: str) -> list[dict]:
    """Extract content from a web page.

    Args:
        url: URL of the web page

    Returns:
        Content of the web page
    """
    client = AsyncTavilyClient()
    response = await client.extract(url)
    return response["results"]
