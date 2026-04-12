"""OpenRouter API client and utilities for MATS experiments."""

import os
import aiohttp
import asyncio
from typing import Dict, List, Optional


_MAX_CONCURRENCY = 100
_MAX_RETRIES = 3


_semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)


class OpenRouterClient:
    """OpenRouter API client for making chat completions."""
    
    def __init__(self, api_key: Optional[str] = None, reject_empty_responses: bool = True):
        """Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If not provided, will use OPENROUTER_API_KEY env var.
            reject_empty_responses: If True, raise exception on empty responses to trigger retries. Default is True.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or pass api_key parameter.")
        self._session: Optional[aiohttp.ClientSession] = None
        self.reject_empty_responses = reject_empty_responses
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def chat(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        """Send a chat message to OpenRouter and return the response.
        
        Args:
            model: The model to use (e.g., "anthropic/claude-3.5-sonnet")
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The response content from the model
            
        Raises:
            ValueError: If API key is not available
            Exception: If API request fails
        """
        data = {"model": model, "messages": messages, **kwargs}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        session = await self._get_session()
        
        async with _semaphore:  # Limit concurrent requests
            for attempt in range(_MAX_RETRIES):
                try:
                    async with session.post(
                        "https://openrouter.ai/api/v1/chat/completions", 
                        headers=headers, 
                        json=data
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(
                                f"OpenRouter API request failed with status {response.status}: {error_text}"
                            )

                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]

                        # Check for empty/None response if configured to reject them
                        if self.reject_empty_responses and (content is None or not content.strip()):
                            raise Exception("OpenRouter API returned empty content")
                        
                        return content
                    
                except Exception as e:
                    if attempt < _MAX_RETRIES - 1:
                        wait_time = (attempt + 1) * 2  # 2, 4, 6, 8 seconds
                        print(f"Request failed ({type(e).__name__}), retrying in {wait_time}s (attempt {attempt + 2}/{_MAX_RETRIES})")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        """Support async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session when exiting context."""
        await self.close()


# Convenience function for backward compatibility
async def chat(model: str, messages: List[Dict[str, str]], **kwargs) -> str:
    """Send a chat message to OpenRouter and return the response.
    
    This is a convenience function that creates a client instance and makes the request.
    For better performance with multiple requests, create an OpenRouterClient instance.
    
    Args:
        model: The model to use (e.g., "anthropic/claude-3.5-sonnet")
        messages: List of message dictionaries with 'role' and 'content' keys
        **kwargs: Additional parameters to pass to the API
        
    Returns:
        The response content from the model
    """
    async with OpenRouterClient() as client:
        return await client.chat(model, messages, **kwargs)