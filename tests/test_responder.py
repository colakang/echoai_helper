"""
Tests for the LLM responder.

No network: the provider is a stub that yields preset chunks.
"""

import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.GPTResponder import (
    GPTResponder, _extract, _is_no_response, _worth_answering,
)
from src.ResponseManager import ResponseManager


# --------------------------------------------------------------------------
# Bracket extraction
# --------------------------------------------------------------------------
#
# The prompt asks for the answer wrapped in [ ]. The old code re-split the
# whole accumulated string on every streamed chunk -- quadratic in the answer
# length -- and required *both* brackets, so until the closing one arrived it
# showed the raw text with the opening bracket still in it.

def test_complete_brackets():
    assert _extract("[hello world]") == "hello world"


def test_partial_stream_hides_the_opening_bracket():
    assert _extract("[hello wor") == "hello wor"


def test_text_without_brackets_passes_through():
    assert _extract("no brackets here") == "no brackets here"


def test_preamble_before_the_bracket_is_dropped():
    assert _extract("Sure! [the answer]") == "the answer"


def test_extraction_is_not_quadratic():
    """A long answer must not cost more each chunk than the last."""
    body = "word " * 5000
    assert _extract("[" + body) == body.strip()


# --------------------------------------------------------------------------
# The 'None' contract
# --------------------------------------------------------------------------
#
# prompts.py tells the model to answer 'None' when the speaker has added
# nothing worth responding to. Nothing acted on it, so the literal word "None"
# was shown to the user as the assistant's answer.

@pytest.mark.parametrize("text", ["None", "none", " None ", "None.", "[None]", ""])
def test_declining_is_recognised(text):
    assert _is_no_response(_extract(text))


@pytest.mark.parametrize("text", ["No, that is wrong", "Nonetheless, yes", "none of them"])
def test_a_real_answer_is_not_mistaken_for_declining(text):
    assert not _is_no_response(_extract(text))


# --------------------------------------------------------------------------
# Length filter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "  ", "ok", "a"])
def test_short_utterances_are_not_worth_a_model_call(text):
    assert not _worth_answering(text)


def test_a_real_question_is_answered():
    assert _worth_answering("what is the revenue")


# --------------------------------------------------------------------------
# Streaming behaviour
# --------------------------------------------------------------------------

class StubProvider:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.calls = 0

    def generate_response(self, messages, **kwargs):
        self.calls += 1
        for chunk in self.chunks:
            yield chunk

    def get_model_name(self):
        return "stub"


def responder_with(chunks):
    responder = GPTResponder.__new__(GPTResponder)
    responder.response_manager = ResponseManager()
    responder.response = ""
    responder._lock = __import__("threading").Lock()
    responder._processing = False
    responder._last_processed_id = None
    responder.llm_provider = StubProvider(chunks)
    return responder


def new_question(responder, text="what is the quarterly revenue"):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return responder.response_manager.create_response(now, text)


def test_streaming_yields_progressively():
    responder = responder_with(["[Rev", "enue is ", "4.2 million]"])
    rid = new_question(responder)

    seen = list(responder._generate_response_from_transcript(
        "what is the quarterly revenue", "", "", rid))

    assert seen[-1] == "Revenue is 4.2 million"
    assert all(later.startswith(earlier[:5]) for earlier, later in zip(seen, seen[1:]))


def test_completed_response_is_stored_unwrapped():
    responder = responder_with(["[the answer]"])
    rid = new_question(responder)
    list(responder._generate_response_from_transcript("a real question", "", "", rid))

    stored = responder.response_manager.get_response(rid)
    assert stored.response_text == "the answer"
    assert stored.is_complete


def test_declining_does_not_store_the_word_none():
    """The visible bug: "None" appeared in the UI as the assistant's reply."""
    responder = responder_with(["[None]"])
    rid = new_question(responder)
    list(responder._generate_response_from_transcript(
        "same question again", "the previous answer", "", rid))

    stored = responder.response_manager.get_response(rid)
    assert stored.response_text == ""
    assert responder.response == "the previous answer"


def test_provider_failure_is_not_presented_as_an_answer():
    """An exception used to be yielded and stored as the completed response,
    so a rate limit appeared in the UI as the reply and was then fed back as
    context to the next turn."""
    class Broken:
        def generate_response(self, messages, **kwargs):
            raise RuntimeError("rate limited")
            yield  # pragma: no cover

    responder = responder_with([])
    responder.llm_provider = Broken()
    rid = new_question(responder)
    list(responder._generate_response_from_transcript("a real question", "", "", rid))

    stored = responder.response_manager.get_response(rid)
    assert stored.response_text == ""
    assert "rate limited" in responder.response
    assert responder.response.startswith("[error]")


def test_short_question_never_reaches_the_provider():
    responder = responder_with(["[unused]"])
    rid = new_question(responder, "ok")
    list(responder._generate_response_from_transcript("ok", "", "", rid))
    assert responder.llm_provider.calls == 0


def test_thinking_is_not_left_on_screen_for_a_skipped_question():
    """_answer used to set "Thinking..." before the length check, so a
    too-short utterance left a spinner that never resolved."""
    responder = responder_with(["[unused]"])
    responder.response = "the previous answer"
    rid = new_question(responder, "ok")

    responder._answer("ok", rid)

    assert responder.response == "the previous answer"
    assert responder.llm_provider.calls == 0


def test_thinking_resolves_when_the_model_returns_nothing():
    responder = responder_with([])          # provider yields no chunks at all
    responder.response = "an older answer"
    rid = new_question(responder)

    responder._answer("what is the quarterly revenue", rid)
    assert responder.response != "Thinking..."


# --------------------------------------------------------------------------
# Speaker labels must not reach the model
# --------------------------------------------------------------------------
#
# Found by playing a real customer-service recording through the app. The
# transcript reads "S2: 零七七" because the interface and the export both want
# to show who spoke; the model read the label as part of the sentence and
# reported the caller's service number as "S3099" -- a number nobody said.
#
# On a call about a reference number, an invented one is worse than no answer.

from src.GPTResponder import _utterance_only                      # noqa: E402


@pytest.mark.parametrize("labelled,expected", [
    ("S2: 零七七", "零七七"),
    ("S1: hello there", "hello there"),
    ("S12: hello", "hello"),
    ("S3?: uncertain match", "uncertain match"),
    ("S3? : spaced", "spaced"),
    ("S2：全角冒号", "全角冒号"),
])
def test_a_diarization_label_is_removed(labelled, expected):
    assert _utterance_only(labelled) == expected


@pytest.mark.parametrize("untouched", [
    "零七七",
    "S 前面没有编号",
    "Speaker: this is not a label",
    "SQL: how do I index this",
    "",
])
def test_ordinary_text_is_left_alone(untouched):
    """
    Only the exact shape the diarizer writes is stripped. "SQL:" opening an
    answer, or a sentence that merely starts with S, must survive -- the point
    is to remove a machine's annotation, not to edit what was said.
    """
    assert _utterance_only(untouched) == untouched.strip()


def test_the_label_is_kept_in_the_transcript():
    """
    Stripped at the model, not at the source. The interface and the export both
    want to show who spoke; it is only the model that must not see it.
    """
    import inspect
    from src import AudioTranscriber as module
    body = inspect.getsource(module.AudioTranscriber._finalize_segment)
    assert 'display = f"{label}: {text}"' in body
