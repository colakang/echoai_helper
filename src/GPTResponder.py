##src/GPTResponder.py

import threading
from .prompts import create_prompt, INITIAL_RESPONSE
import time
import sys
from .config import SystemConfig, EnvConfig
from .llm import create_llm_provider
import yaml
from pathlib import Path

class GPTResponder:
    def __init__(self, response_manager):
        self.response_manager = response_manager
        self.response = ""
        self._response_update_interval = 2
        self._lock = threading.Lock()
        self._processing = False
        self._last_processed_id = None

        # 初始化LLM provider
        if not self._initialize_llm_provider():
            raise ValueError("Failed to initialize LLM provider. Please check your configuration.")

    def _initialize_llm_provider(self) -> bool:
        """
        Initialize LLM Provider (supports OpenAI, Gemini, Ollama, Claude, etc.)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load configuration file
            config_path = Path(__file__).parent.parent / "conf.yaml"
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Get LLM configuration
            llm_config = config.get('LLM', {})
            provider_type = llm_config.get('provider', 'openai').lower()

            # Get provider-specific config
            if provider_type == 'openai':
                # OpenAI configuration
                if not EnvConfig.ensure_api_key():
                    print("[GPTResponder] OpenAI API key not found")
                    return False

                provider_config = {
                    'api_key': EnvConfig.get_openai_key(),
                    'model': llm_config.get('openai', {}).get('model', 'gpt-4o-mini')
                }
            elif provider_type == 'litellm':
                # LiteLLM configuration (supports Gemini, Ollama, Claude, etc.)
                litellm_config = llm_config.get('litellm', {})
                provider_config = {
                    'model': litellm_config.get('model'),
                    'api_key': litellm_config.get('api_key'),
                    'api_base': litellm_config.get('api_base')
                }
            else:
                print("[GPTResponder] Unknown provider type: {}".format(provider_type))
                return False

            # Create provider
            self.llm_provider = create_llm_provider(provider_type, provider_config)

            if not self.llm_provider:
                print("[GPTResponder] Failed to create LLM provider")
                return False

            print("[GPTResponder] Initialized {} provider: {}".format(
                provider_type, self.llm_provider.get_model_name()))
            return True

        except Exception as e:
            print("[GPTResponder] Error initializing LLM provider: {}".format(str(e)))
            import traceback
            traceback.print_exc()
            return False

    def _generate_response_from_transcript(self, lastContent, latest_response_text="", latest_response_q_text="", current_response_id=None):
        """
        Generate streaming response from transcript

        Args:
            lastContent (str): Latest transcript content
            latest_response_text (str): Previous response text
            latest_response_q_text (str): Previous question text
            current_response_id (str): Current response ID

        Yields:
            str: Generated response chunks
        """
        # Filter short content
        if lastContent.strip() == "" or len(lastContent.strip()) < 4:
            print("Skipping due to too short content (length: {})".format(len(lastContent.strip())))
            return

        conversation_history = []
        recent_speakers = ["Speaker: [{}]\n\n".format(latest_response_q_text)]
        conversation_history.extend(recent_speakers)

        # Combine records into string
        recent_transcript = "".join(conversation_history)
        
        try:
            content = create_prompt(recent_speakers, lastContent, latest_response_text)

            # Use LLM Provider to generate streaming response
            messages = [
                {"role": "system", "content": SystemConfig.get_system_role()},
                {"role": "user", "content": content},
            ]

            accumulated_response = ""
            for chunk_content in self.llm_provider.generate_response(
                messages=messages,
                temperature=0.6,
                stream=True
            ):
                if chunk_content:
                    accumulated_response += chunk_content

                    # Try to parse content in brackets
                    try:
                        if '[' in accumulated_response and ']' in accumulated_response:
                            response_text = accumulated_response.split("[")[1].split("]")[0]
                        else:
                            response_text = accumulated_response

                        # Update response
                        if current_response_id:
                            self.response = response_text
                            self.response_manager.update_response(
                                current_response_id,
                                response_text,
                                is_complete=False
                            )

                        yield response_text

                    except Exception as e:
                        print("Error parsing chunk: {}".format(str(e)))
                        yield chunk_content

            # Mark as complete
            if current_response_id:
                try:
                    # Try to extract content from brackets, fallback to full response
                    if '[' in accumulated_response and ']' in accumulated_response:
                        final_response = accumulated_response.split("[")[1].split("]")[0]
                    else:
                        print("No brackets found in response, using full response")
                        final_response = accumulated_response

                    self.response_manager.update_response(
                        current_response_id,
                        final_response,
                        is_complete=True
                    )
                except Exception as e:
                    print("Error processing final response: {}".format(str(e)))
                    # Fallback to full response
                    self.response_manager.update_response(
                        current_response_id,
                        accumulated_response,
                        is_complete=True
                    )

        except Exception as e:
            print("Error in generate_response: {}".format(str(e)))
            error_message = str(e)
            if current_response_id:
                self.response_manager.update_response(
                    current_response_id,
                    error_message,
                    is_complete=True
                )
            yield error_message

    def respond_to_transcriber(self, transcriber):
        """
        Continuously listen and respond to transcriber output

        Args:
            transcriber: Transcriber instance
        """
        while True:
            try:
                # Wait for transcript_changed_event
                if transcriber.transcript_changed_event.wait(0.1):
                    transcriber.transcript_changed_event.clear()

                    if transcriber.structured_transcript["speaker"]:
                        latest_record = transcriber.structured_transcript["speaker"][0]
                        current_response_id = latest_record[2]

                        if (current_response_id and
                            current_response_id != self._last_processed_id and
                            not self._processing):

                            with self._lock:
                                self._processing = True

                            try:
                                question_text = latest_record[0]
                                self.response = "Thinking..."
                                self.response_manager.update_response(current_response_id, self.response)

                                latest_response = self.response_manager.get_response(self._last_processed_id)
                                latest_response_text = ""
                                latest_response_q_text = ""
                                if latest_response and latest_response.is_complete:
                                    latest_response_text = latest_response.response_text
                                    latest_response_q_text = latest_response.question_text

                                response_text = ''
                                # Use generator to process streaming response
                                for response_text in self._generate_response_from_transcript(
                                    question_text,
                                    latest_response_text,
                                    latest_response_q_text,
                                    current_response_id
                                ):
                                    if response_text.strip():
                                        self.response = response_text

                                print("Generated response: {}".format(response_text))
                                self._last_processed_id = current_response_id

                            finally:
                                with self._lock:
                                    self._processing = False

            except Exception as e:
                print("Error in respond_to_transcriber: {}".format(str(e)))
                time.sleep(0.1)

    def update_response_interval(self, interval):
        self._response_update_interval = interval