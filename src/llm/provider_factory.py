# src/llm/provider_factory.py
from typing import Optional
from .llm_provider import LLMProvider
from .openai_provider import OpenAIProvider
from .litellm_provider import LiteLLMProvider


def create_llm_provider(
    provider_type: str,
    config: dict
) -> Optional[LLMProvider]:
    """
    Factory function to create LLM provider instances.

    Args:
        provider_type: Provider type ("openai" or "litellm")
        config: Provider configuration dict

    Returns:
        LLMProvider instance or None if creation fails

    Examples:
        # OpenAI
        config = {
            "api_key": "sk-...",
            "model": "gpt-4o-mini"
        }
        provider = create_llm_provider("openai", config)

        # Google Gemini via LiteLLM
        config = {
            "model": "gemini/gemini-1.5-flash",
            "api_key": "YOUR_GEMINI_KEY"
        }
        provider = create_llm_provider("litellm", config)

        # Ollama via LiteLLM
        config = {
            "model": "ollama/gemma2:2b",
            "api_base": "http://localhost:11434"
        }
        provider = create_llm_provider("litellm", config)
    """
    provider_type = provider_type.lower()

    try:
        if provider_type == "openai":
            # OpenAI provider
            api_key = config.get("api_key")
            model = config.get("model", "gpt-4o-mini")

            if not api_key:
                print("[Provider Factory] OpenAI API key is missing")
                return None

            provider = OpenAIProvider(
                api_key=api_key,
                model=model,
                **{k: v for k, v in config.items() if k not in ["api_key", "model"]}
            )

        elif provider_type == "litellm":
            # LiteLLM provider (supports Gemini, Ollama, Claude, etc.)
            model = config.get("model")
            api_key = config.get("api_key")
            api_base = config.get("api_base")

            if not model:
                print("[Provider Factory] Model name is missing for LiteLLM")
                return None

            provider = LiteLLMProvider(
                model=model,
                api_key=api_key,
                api_base=api_base,
                **{k: v for k, v in config.items()
                   if k not in ["model", "api_key", "api_base"]}
            )

        else:
            print(f"[Provider Factory] Unknown provider type: {provider_type}")
            return None

        # Validate configuration
        if not provider.validate_config():
            print(f"[Provider Factory] Configuration validation failed for {provider_type}")
            return None

        print(f"[Provider Factory] Created {provider_type} provider: {provider.get_model_name()}")
        return provider

    except Exception as e:
        print(f"[Provider Factory] Error creating provider: {e}")
        import traceback
        traceback.print_exc()
        return None
