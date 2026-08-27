# src/llm/openai_provider.py
import openai
from typing import Generator, List, Dict
from .llm_provider import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI API provider (GPT-4, GPT-3.5, etc.)"""

    # Models measured to reject a non-default temperature, learned at runtime
    # rather than listed here. Shared across instances because the fact belongs
    # to the model, not to one provider object, and the cost of learning it is
    # one rejected request.
    _no_temperature = set()

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
        params = {**self.extra_params, **kwargs}

        def call(with_temperature: bool):
            extra = {"temperature": temperature} if with_temperature else {}
            return openai.chat.completions.create(
                model=self.model, messages=messages, stream=stream,
                **extra, **params)

        # Newer models accept only the default temperature and reject anything
        # else with a 400. Which models those are is not written down here on
        # purpose: that list would go stale exactly as fast as a list of model
        # names, and this is cheaper to discover than to maintain. One rejected
        # request per model per process, then it is remembered.
        wants_temperature = self.model not in self._no_temperature
        try:
            response = call(wants_temperature)
        except Exception as e:
            if wants_temperature and self._is_temperature_rejection(e):
                print(f"[OpenAI Provider] {self.model} takes only the default "
                      f"temperature; retrying without it.")
                self._no_temperature.add(self.model)
                try:
                    response = call(False)
                except Exception as retry_error:
                    yield self._describe(retry_error)
                    return
            else:
                yield self._describe(e)
                return

        try:
            if stream:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                yield response.choices[0].message.content
        except Exception as e:
            yield self._describe(e)

    @staticmethod
    def _is_temperature_rejection(error) -> bool:
        """Whether this failure is specifically about the temperature value."""
        text = str(error).lower()
        return "temperature" in text and (
            "unsupported" in text or "does not support" in text
            or "only the default" in text)

    def _describe(self, error) -> str:
        """
        What to show when a request fails.

        This lands in the response pane during a live interview, where the raw
        SDK repr -- several hundred characters of JSON -- is worse than
        useless. Name the model, because the usual cause is a model that does
        not exist or is not enabled for the account.
        """
        print(f"[OpenAI Provider Error] {self.model}: {error}")
        message = getattr(getattr(error, "body", None), "get", lambda k: None)("message") \
            if hasattr(error, "body") else None
        if not message:
            message = str(error)
        return f"[{self.model}: {message[:200]}]"

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
