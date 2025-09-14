import aiohttp
import json
import os


_SERPER_URI = "https://google.serper.dev/search"


async def search(query: str) -> dict:
    payload = json.dumps(
        {
            "q": query,
            "autocorrect": False,
        }
    )

    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(_SERPER_URI, headers=headers, data=payload) as response:
            return await response.json()
