from exa_py import AsyncExa
from tavily import AsyncTavilyClient

from exploration_hacking.tools._serper import search as serper_search
from exploration_hacking.tools._minimal_search import search as minimal_search

async def search_web_tavily(query: str) -> list[dict]:
    """Search the web for information.

    Args:
        query: Search query string

    Returns:
        List of 5 search results with title, snippet, and url
    """
    client = AsyncTavilyClient()
    response = await client.search(query, max_results=5)
    return [
        {"url": result["url"], "title": result["title"], "snippet": result["content"]}
        for result in response["results"]
    ]


async def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for information.

    Args:
        query: Search query string
        max_results: Number of results to return (up to 10)

    Returns:
        List of n=max_results search results with title, link, and snippet
    """
    json = await serper_search(query)
    return [
        {"title": x["title"], "link": x["link"], "snippet": x["snippet"]}
        for x in json["organic"][:max_results]
    ]


async def summarize_page(url: str, query: str) -> str:
    """Get a short AI-generated summary of a page.

    Args:
        url: URL of the page to summarize
        query: Summarization query prompt for the LLM

    Returns:
        Summary of the page
    """
    client = AsyncExa(None)
    response = await client.get_contents(
        [url], highlights={"numSentences": 1, "query": query}, summary=True
    )
    try:
        return response.results[0].summary
    except Exception:
        return f"Error getting the summary: {response}"


async def search_minimal(query: str, max_results: int = 5) -> list[dict]:
    """Search the wmdp dataset for information.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of n=max_results texts from the wmdp dataset
    """
    json = await minimal_search(query)
    return [
        {"text": x["text"]}
        for x in json["texts"][:max_results]
    ]