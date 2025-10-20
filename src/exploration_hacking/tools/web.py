from exa_py import AsyncExa
from tavily import AsyncTavilyClient

from exploration_hacking.tools._serper import search as serper_search
from exploration_hacking.tools._simple_search import search as simple_search


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


# These results are LLM-generated responses by default.
# High-quality matches from WMDP rationales dataset are prepended when found.
# (the docstring is visible to the LLM so doesn't say this)
async def search_simple(query: str, max_results: int = 2) -> list[dict]:
    """Summarized web search for information.

    Args:
        query: Search query string
        max_results: Maximum number of results to return, up to 2

    Returns:
        List of n=max_results search results with text
    """
    max_results = min(max_results, 2)

    results = await simple_search(query, max_results)
    return [{"text": x["text"]} for x in results]
