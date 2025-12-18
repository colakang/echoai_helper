# src/llm/litellm_provider.py
from typing import Generator, List, Dict, Optional
from .llm_provider import LLMProvider

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    print("[Warning] litellm not installed. Install with: pip install litellm")


class LiteLLMProvider(LLMProvider):
    """
    LiteLLM provider supporting multiple backends:
    - Google Gemini (gemini/gemini-1.5-flash, gemini/gemini-1.5-pro)
    - Ollama (ollama/gemma2:2b, ollama/llama3, etc.)
    - Claude (claude-3-haiku, claude-3-sonnet)
    - Azure OpenAI
    - And 100+ other providers
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize LiteLLM provider.

        Args:
            model: Model identifier (e.g., "gemini/gemini-1.5-flash", "ollama/gemma2:2b")
            api_key: API key (if required by the provider)
            api_base: Base URL (for local providers like Ollama)
            **kwargs: Additional provider-specific parameters

        Examples:
            # Google Gemini
            LiteLLMProvider(model="gemini/gemini-1.5-flash", api_key="YOUR_KEY")

            # Ollama (local)
            LiteLLMProvider(model="ollama/gemma2:2b", api_base="http://localhost:11434")

            # Claude
            LiteLLMProvider(model="claude-3-haiku-20240307", api_key="YOUR_KEY")
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is not installed. "
                "Please install it with: pip install litellm"
            )

        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.extra_params = kwargs

        # Set API key if provided
        if api_key:
            import os
            # LiteLLM auto-detects provider from model name
            # and sets appropriate env var
            if model.startswith("gemini/"):
                os.environ["GEMINI_API_KEY"] = api_key
            elif model.startswith("claude"):
                os.environ["ANTHROPIC_API_KEY"] = api_key

        # Configure LiteLLM
        litellm.set_verbose = False  # Reduce logging

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        stream: bool = True,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Generate streaming response from LiteLLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            stream: Whether to stream the response
            **kwargs: Additional parameters

        Yields:
            str: Partial response content chunks
        """
        try:
            # Merge parameters
            params = {**self.extra_params, **kwargs}

            # Add api_base if specified (for Ollama, LocalAI, etc.)
            if self.api_base:
                params["api_base"] = self.api_base

            # Call LiteLLM
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=stream,
                **params
            )

            if stream:
                # Stream response
                for chunk in response:
                    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            yield delta.content
            else:
                # Non-streaming response
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    yield response.choices[0].message.content

        except Exception as e:
            print(f"[LiteLLM Provider Error] {e}")
            yield f"[Error: {str(e)}]"

    def get_model_name(self) -> str:
        """Get the current model name"""
        return self.model

    def validate_config(self) -> bool:
        """Validate LiteLLM configuration"""
        if not LITELLM_AVAILABLE:
            print("[LiteLLM Provider] litellm library not installed")
            return False

        if not self.model or self.model.strip() == "":
            print("[LiteLLM Provider] Model name is missing")
            return False

        # Check if API key is required for this provider
        if self.model.startswith("gemini/") or self.model.startswith("claude"):
            if not self.api_key:
                print(f"[LiteLLM Provider] API key required for {self.model}")
                return False

        # Ollama and local providers don't need API key
        return True
