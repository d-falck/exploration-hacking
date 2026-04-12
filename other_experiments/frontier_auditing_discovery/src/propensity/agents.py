"""Conversational agents that wrap OpenRouter models."""

import re
from pathlib import Path
from typing import List, Dict, Optional

from propensity.openrouter import OpenRouterClient


class FileReadingAgent:
    """Agent that can read text files during conversation using OpenRouter chat."""

    def __init__(
        self,
        folder_path: str,
        model: str = "anthropic/claude-3.7-sonnet",
        system_prompt: str = "",
        client: Optional[OpenRouterClient] = None,
    ):
        """Initialize the agent with a folder path and model.

        Args:
            folder_path: Path to the folder containing files the agent can read
            model: OpenRouter model to use for chat
            system_prompt: System prompt to use for chat
            client: OpenRouter client instance. If not provided, will create a new one.
        """
        self.folder_path = Path(folder_path).resolve()
        self.model = model
        self.system_prompt = system_prompt
        self.client = client or OpenRouterClient()

        if not self.folder_path.exists():
            raise ValueError(f"Folder path does not exist: {folder_path}")
        if not self.folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")

    async def chat(self, prompts: List[str], **kwargs) -> List[Dict[str, str]]:
        """Send a chat message, handling file read requests during generation."""
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        for prompt in prompts:
            messages.append({"role": "user", "content": prompt})

        while True:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                **kwargs,
            )

            messages.append({"role": "assistant", "content": response})

            file_requests = self._extract_file_requests(response)
            list_files_requests = self._extract_list_files_requests(response)

            if not file_requests and not list_files_requests:
                return messages

            responses = []

            # Handle LIST_FILES requests
            if list_files_requests:
                file_list = self.list_files()
                file_list_str = "\n".join(f"- {f}" for f in file_list)
                responses.append(
                    f"MESSAGE\nType: INFO\nFrom: SYSTEM\nAvailable files:\n{file_list_str}"
                )

            # Handle READ_FILE requests
            for filename in file_requests:
                content = self._read_file(filename)
                responses.append(
                    f"MESSAGE\nType: INFO\nFrom: SYSTEM\nFile: {filename}\n{content}"
                )

            combined_response = "\n\n".join(responses)
            messages.append({"role": "user", "content": combined_response})

    def _read_file(self, filename: str) -> str:
        """Read a file from the designated folder."""
        try:
            file_path = self.folder_path / filename

            # Security check: ensure the resolved path is within the folder
            if not str(file_path.resolve()).startswith(str(self.folder_path)):
                return f"Error: Cannot access files outside the designated folder"

            if not file_path.exists():
                return f"Error: File '{filename}' not found"

            if not file_path.is_file():
                return f"Error: '{filename}' is not a file"

            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            return content

        except Exception as e:
            return f"Error reading file '{filename}': {str(e)}"

    def _extract_file_requests(self, text: str) -> List[str]:
        """Extract file read requests from text."""
        pattern = r"READ_FILE\(([^)]+)\)"
        filenames = re.findall(pattern, text)
        return [f.strip().strip("\"'") for f in filenames]

    def _extract_list_files_requests(self, text: str) -> List[str]:
        """Extract list files requests from text."""
        pattern = r"LIST_FILES\(\)"
        matches = re.findall(pattern, text)
        return matches

    def list_files(self) -> List[str]:
        """List all files in the designated folder."""
        try:
            return [f.name for f in self.folder_path.iterdir() if f.is_file()]
        except Exception as e:
            return [f"Error listing files: {str(e)}"]