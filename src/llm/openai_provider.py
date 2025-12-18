# src/llm/openai_provider.py
import openai
from typing import Generator, List, Dict
from .llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4, GPT-3.5, etc.)"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", **kwargs):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-4o-mini", "gpt-4")
            **kwargs: Additional OpenAI parameters
        """
        self.api_key = api_key
        self.model = model
        self.extra_params = kwargs

        # Configure OpenAI
        openai.api_key = api_key

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        stream: bool = True,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Generate streaming response from OpenAI.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Yields:
            str: Partial response content chunks
        """
        try:
            # Merge extra params
            params = {**self.extra_params, **kwargs}

            # Call OpenAI API
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=stream,
                **params
            )

            if stream:
                # Stream response
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                # Non-streaming response
                yield response.choices[0].message.content

        except Exception as e:
            print(f"[OpenAI Provider Error] {e}")
            yield f"[Error: {str(e)}]"

    def get_model_name(self) -> str:
        """Get the current model name"""
        return self.model

    def validate_config(self) -> bool:
        """Validate OpenAI configuration"""
        if not self.api_key or self.api_key.strip() == "":
            print("[OpenAI Provider] API key is missing")
            return False

        if not self.model or self.model.strip() == "":
            print("[OpenAI Provider] Model name is missing")
            return False

        return True
