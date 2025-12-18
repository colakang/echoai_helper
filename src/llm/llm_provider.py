# src/llm/llm_provider.py
from abc import ABC, abstractmethod
from typing import Generator, List, Dict, Any


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    Supports multiple backends: OpenAI, Google Gemini, Ollama, Claude, etc.
    """

    @abstractmethod
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        stream: bool = True,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Generate streaming response from LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-1.0)
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Yields:
            str: Partial response content chunks
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the current model name"""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider configuration"""
        pass
