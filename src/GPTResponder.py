##src/GPTResponder.py

import threading
import re
import time
import traceback
from datetime import datetime

from .prompts import build_messages
from .config import SystemConfig, EnvConfig, PathConfig, LLMConfig
from .llm import create_llm_provider
import yaml
from pathlib import Path

# The answer is requested inside square brackets. During streaming the
# closing bracket has usually not arrived yet, so take everything after the
# first opening one.
_OPEN = re.compile(r"\[")


def _extract(text: str) -> str:
    """The answer so far, with the bracket wrapper removed."""
    match = _OPEN.search(text)
    if not match:
        return text.strip()
    body = text[match.end():]
    closing = body.find("]")
    return (body[:closing] if closing >= 0 else body).strip()


# Below this an utterance carries no question worth spending a model call on.
MIN_QUESTION_CHARS = 4


def _worth_answering(text: str) -> bool:
    return bool(text) and len(text.strip()) >= MIN_QUESTION_CHARS


def _is_no_response(text: str) -> bool:
    """The model's way of saying the speaker added nothing new."""
    return text.strip().strip(".").casefold() in ("none", "")


# Diarization labels as _identify_speaker writes them: S1, S12, and S3? when
# the match was uncertain.
_SPEAKER_LABEL = re.compile(r"^\s*S\d+\??\s*[:：]\s*")


def _utterance_only(text: str) -> str:
    """
    Strip the speaker label before the words reach the model.

    The transcript carries "S2: 零七七" because the interface and the export
    both want to show who spoke. The model does not: it reads the label as part
    of the sentence, and on a customer-service call that means inventing
    reference numbers -- "S2" plus "零七七" came back as service number S3099,
    a number nobody said.

    Stripped here rather than at the source, because the label genuinely
    belongs in the transcript. It is only the model that must not see it.
    """
    return _SPEAKER_LABEL.sub("", text or "").strip()


class GPTResponder:
    def __init__(self, response_manager, session_provider=None):
        self.response_manager = response_manager
        # Looked up when an answer completes rather than held, because the
        # session is replaced when the transcript is cleared.
        self.session_provider = session_provider
        self.response = ""
        self._response_update_interval = 2
        self._lock = threading.Lock()
        self._processing = False
        self._last_processed_id = None

        # 初始化LLM provider
        if not self._initialize_llm_provider():
            raise ValueError("Failed to initialize LLM provider. Please check your configuration.")

    def reload_provider(self) -> bool:
        """
        Rebuild the provider, picking up a model chosen since startup.

        The provider is otherwise built once, at construction, so changing the
        model in the menu would have applied to the cleanup pass at export and
        silently not to the live replies -- two halves of the app quietly using
        different models.
        """
        return self._initialize_llm_provider()

    def _initialize_llm_provider(self) -> bool:
        """
        Initialize LLM Provider (supports OpenAI, Gemini, Ollama, Claude, etc.)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load configuration file
            config_path = PathConfig.get_conf_file()
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # Get LLM configuration
            llm_config = config.get('LLM', {})
            # The chosen model decides the backend: a name carrying a vendor
            # prefix goes through LiteLLM, a bare one to OpenAI. Picking
            # "gemini/..." from the menu therefore just works, with no second
            # control to keep in sync with the first.
            provider_type = LLMConfig.provider_for()

            # Get provider-specific config
            if provider_type == 'openai':
                # OpenAI configuration
                if not EnvConfig.ensure_api_key():
                    print("[GPTResponder] OpenAI API key not found")
                    return False

                provider_config = {
                    'api_key': EnvConfig.get_openai_key(),
                    'model': LLMConfig.get_model()
                }
            elif provider_type == 'cli':
                cli_config = llm_config.get('cli', {})
                provider_config = {
                    'command': cli_config.get('command', 'claude'),
                    'model': cli_config.get('model'),
                    'timeout': cli_config.get('timeout', 90),
                    'extra_args': cli_config.get('extra_args'),
                }
            elif provider_type == 'litellm':
                # LiteLLM configuration (supports Gemini, Ollama, Claude, etc.)
                litellm_config = llm_config.get('litellm', {})
                provider_config = {
                    'model': LLMConfig.get_model() or litellm_config.get('model'),
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

    def _generate_response_from_transcript(self, lastContent, latest_response_text="", latest_response_q_text="", current_response_id=None, history=None):
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
        if not _worth_answering(lastContent):
            return

        try:
            # Real dialogue turns, not one flattened blob. The previous shape
            # restated the rules on every call and carried a single past
            # question, so the model could not see what it had already said --
            # in a live run it answered "How may I assist you today?" four
            # times to four different utterances.
            messages = build_messages(
                SystemConfig.get_system_role(),
                history if history is not None else [],
                lastContent,
            )

            accumulated_response = ""
            emitted = ""
            for chunk_content in self.llm_provider.generate_response(
                messages=messages,
                temperature=0.6,
                stream=True
            ):
                if not chunk_content:
                    continue
                accumulated_response += chunk_content

                # The prompt asks for the answer wrapped in [ ]. Extract from
                # the tail only: the old code re-split the whole accumulated
                # string on every chunk, which is quadratic in the length of
                # the answer, and it showed the raw text -- opening bracket
                # included -- for as long as the closing one had not arrived.
                visible = _extract(accumulated_response)
                if visible == emitted:
                    continue
                emitted = visible

                if current_response_id:
                    self.response = visible
                    self.response_manager.update_response(
                        current_response_id, visible, is_complete=False)
                yield visible

            final = _extract(accumulated_response)

            # The prompt tells the model to answer 'None' when the speaker has
            # said nothing new worth responding to. Nothing acted on that, so
            # the word "None" was displayed to the user as if it were the
            # answer. Treat it as what it means -- no new response -- and
            # leave the previous one on screen.
            if _is_no_response(final):
                print("Responder: model declined to answer (no new content)")
                if current_response_id:
                    self.response_manager.update_response(
                        current_response_id, "", is_complete=True)
                self.response = latest_response_text or ""
                return

            if current_response_id:
                self.response_manager.update_response(
                    current_response_id, final, is_complete=True)

        except Exception as e:
            # An exception is not an answer. It used to be yielded and stored
            # as the completed response, so a rate limit or a network blip
            # appeared in the UI in place of the assistant's reply and was
            # then fed back as context to the next turn.
            print("Error in generate_response: {}".format(e))
            traceback.print_exc()
            self.response = "[error] {}".format(e)
            if current_response_id:
                self.response_manager.update_response(
                    current_response_id, "", is_complete=True)
            return

    def respond_to_transcriber(self, transcriber):
        """
        Answer each completed utterance from the far end.

        Woken by transcript_changed_event, which the transcriber now sets when
        a *segment* ends rather than when a phrase begins -- so the question
        here is a finished sentence rather than its first fragment.
        """
        while True:
            try:
                if not transcriber.transcript_changed_event.wait(0.1):
                    continue
                transcriber.transcript_changed_event.clear()

                if not transcriber.structured_transcript["speaker"]:
                    continue

                latest_record = transcriber.structured_transcript["speaker"][0]
                current_response_id = latest_record[2]
                if not current_response_id:
                    continue

                # Claim the work under the lock. The old code tested
                # _processing outside it and set it inside, which only
                # happened to be safe because there is exactly one consumer.
                with self._lock:
                    if (self._processing
                            or current_response_id == self._last_processed_id):
                        continue
                    self._processing = True

                try:
                    self._answer(latest_record[0], current_response_id)
                finally:
                    with self._lock:
                        self._last_processed_id = current_response_id
                        self._processing = False

            except Exception as e:
                print("Error in respond_to_transcriber: {}".format(e))
                traceback.print_exc()
                time.sleep(0.1)

    def answer_passage(self, passage, on_done=None):
        """
        Answer a passage the user picked out, now.

        The automatic path answers every finished utterance from the far end,
        which is what an interview wants: a prompt while the other person is
        still talking. A meeting is not that. There, most of what is said needs
        no answer from you, and the one part that does is known only to you --
        so the trigger has to be yours, and the input is the passage you point
        at rather than the last sentence to arrive.

        Runs on its own thread: the model takes seconds and this is called from
        a button.

        The length filter does not apply. It exists to stop the automatic path
        answering "嗯" and "好的", which is a guess about intent -- and there is
        nothing to guess about when someone has selected the text and asked.
        """
        passage = _utterance_only((passage or "").strip())
        if not passage:
            return False

        def run():
            with self._lock:
                if self._processing:
                    print("[Responder] Busy; ignoring the request.")
                    return
                self._processing = True
            try:
                response_id = self.response_manager.create_response(
                    question_time=datetime.now(), question_text=passage)
                self._answer(passage, response_id, require_length=False)
                self._last_processed_id = response_id
            except Exception as e:
                print(f"[Responder] Could not answer the selection: {e}")
                traceback.print_exc()
            finally:
                with self._lock:
                    self._processing = False
                if on_done is not None:
                    on_done()

        threading.Thread(target=run, daemon=True,
                         name="AnswerPassage").start()
        return True

    def _answer(self, question_text, current_response_id, require_length=True):
        previous = self.response_manager.get_response(self._last_processed_id)
        previous_answer = ""
        previous_question = ""
        if previous and previous.is_complete:
            previous_answer = previous.response_text or ""
            previous_question = previous.question_text or ""

        # Only show "Thinking..." once the question has cleared the length
        # filter. Setting it unconditionally left it on screen forever
        # whenever the generator returned early on a too-short utterance.
        question_text = _utterance_only(question_text)
        previous_question = _utterance_only(previous_question)

        if require_length and not _worth_answering(question_text):
            print("Skipping: too short ({} chars)".format(len(question_text.strip())))
            return

        self.response = "Thinking..."
        self.response_manager.update_response(current_response_id, self.response)

        # Everything settled so far, so the model can tell a new question
        # from a rephrasing of one it has already answered.
        history = self.response_manager.recent_exchanges()

        answered = False
        for response_text in self._generate_response_from_transcript(
                question_text, previous_answer, previous_question,
                current_response_id, history=history):
            if response_text.strip():
                self.response = response_text
                answered = True

        if answered:
            print("Generated response: {}".format(self.response))
            self._record(current_response_id, question_text, self.response)
        elif self.response == "Thinking...":
            # Nothing came back at all. Leave the previous answer up rather
            # than a spinner that will never resolve.
            self.response = previous_answer

    def _record(self, response_id, question, answer):
        """
        Write a finished answer to the session file.

        session.append_response has existed since sessions were added and had
        never been called, so no answer has ever been written to one. Nothing
        showed it: the export offered from the app reads the live transcript,
        where the answers are held in memory. It is re-exporting a past
        session -- `echoai-helper sessions --export N` -- that went to the
        file, found no answers there, and produced a transcript with all of
        them missing.

        That was survivable while replies were a side feature of interviews.
        It stops being survivable when answering during a meeting is the point
        of the session.
        """
        session = self.session_provider() if self.session_provider else None
        if session is None:
            return
        try:
            session.append_response(response_id, question, answer)
        except Exception as e:
            # An answer that cannot be filed is still an answer on screen.
            print(f"[Responder] Could not record the answer: {e}")

    def update_response_interval(self, interval):
        """No-op, kept for callers that still set it.

        The value was stored and never read; the UI control that drove it has
        been removed. Response cadence is set by the segmenter's pauses.
        """