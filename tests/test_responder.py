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
# The model has to be told who said what
# --------------------------------------------------------------------------
#
# Attribution was briefly stripped altogether. That fixed one problem and
# caused another.
#
# Fixed: "S2: 零七七" read as one sentence, and the model welded the label into
# the digits and reported a service number of S3099 that nobody had said.
#
# Caused: in a meeting the passage holds two or three people plus the
# operator, and with the attribution gone the model cannot tell whose question
# it is answering, or that one of those voices belongs to the person it is
# writing for.
#
# So it stays, in a form that cannot be mistaken for content: "S2" looks like
# a code that might belong to a reference number, "Speaker 2" does not.

from src.GPTResponder import _utterance_only                       # noqa: E402


@pytest.mark.parametrize("line,expected", [
    ("Speaker: [S1: hello]", "Speaker 1: hello"),
    ("Speaker: [S12: hello]", "Speaker 12: hello"),
    ("Speaker: [S2: 零七七]", "Speaker 2: 零七七"),
    ("Speaker: [S2：全角冒号]", "Speaker 2: 全角冒号"),
])
def test_a_label_becomes_a_name(line, expected):
    assert _utterance_only(line) == expected


def test_an_uncertain_match_says_so():
    """
    The registry marks a blended or borderline segment with a "?". Passing
    that through as doubt is better than presenting it as fact -- the model
    can weigh what was said against a label it has been told to distrust.
    """
    assert _utterance_only("Speaker: [S3?: 零九九]") == "Speaker 3 (unsure): 零九九"


def test_the_operator_is_distinguished_from_everyone_else():
    """
    The reply being drafted is theirs. Without this the model cannot tell
    which of the voices it is writing for, and answers as a bystander.
    """
    assert _utterance_only("You: [稍等啊]") == "Me: 稍等啊"


def test_a_multi_party_passage_keeps_every_attribution():
    passage = "Speaker: [S1: 一]\nYou: [二]\nSpeaker: [S2: 三]"
    assert _utterance_only(passage) == "Speaker 1: 一\nMe: 二\nSpeaker 2: 三"


def test_every_line_is_handled_not_only_the_first():
    """
    The patterns are anchored, and without MULTILINE only the first line is
    rewritten. That looked correct for as long as every question was a single
    utterance -- which it was, until turns could be picked.
    """
    passage = "Speaker: [S1: 一]\nSpeaker: [S2: 二]\nSpeaker: [S3: 三]"
    assert _utterance_only(passage).count("Speaker ") == 3


def test_an_unlabelled_line_is_still_attributed():
    """Diarization off, or a segment too short to identify."""
    assert _utterance_only("Speaker: [没有标签]") == "Speaker: 没有标签"


def test_blank_separators_are_dropped():
    """The pane puts them between turns; they are not part of the question."""
    passage = "Speaker: [S1: 一]\n\nSpeaker: [S2: 二]"
    assert _utterance_only(passage) == "Speaker 1: 一\nSpeaker 2: 二"


def test_the_attribution_is_explained_to_the_model():
    """
    A convention the model has to guess at is a convention that will be
    misread. The rules say what the prefixes mean, that the numbers come from
    an imperfect voice match, and whose line to write.
    """
    from src.prompts import RESPONSE_RULES
    assert "Speaker 1" in RESPONSE_RULES
    assert "Me" in RESPONSE_RULES
    assert "not as fact" in RESPONSE_RULES


def test_the_length_filter_measures_the_words_not_the_attribution():
    """
    The filter exists to stop the automatic path answering "嗯". Adding an
    attribution first defeats it: "Speaker: 嗯" is comfortably long enough to
    pass a test aimed at a single character, so every piece of backchannel
    would have become a model call.
    """
    from src.GPTResponder import _spoken_words
    assert _spoken_words("Speaker: [S1: 嗯]") == "嗯"
    assert _spoken_words("ok") == "ok"


def test_it_is_measured_before_the_attribution_is_added():
    """Order matters, and getting it backwards is silent."""
    import inspect
    from src.GPTResponder import GPTResponder
    body = inspect.getsource(GPTResponder._answer)
    assert (body.index("_spoken_words(question_text)")
            < body.index("question_text = _utterance_only(question_text)"))


def test_the_rewrite_is_applied_exactly_once():
    """
    answer_passage used to run it too, for an emptiness check, and _answer
    runs it again. The second pass does not recognise "Speaker 1: hello" as an
    already-attributed line, so it attributed it again and the model received
    "Speaker: Speaker 1: hello".

    Emptiness is asked of the words instead, which changes nothing.
    """
    import inspect
    from src.GPTResponder import GPTResponder
    body = inspect.getsource(GPTResponder.answer_passage)
    assert "_utterance_only(" not in body
    assert "_spoken_words(" in body


# --------------------------------------------------------------------------
# The answer panel
#
# `response` holds one answer and each new one overwrites it. The panel was a
# mirror of that string, so the second answer erased the first from the screen
# while both were written to the session file.
# --------------------------------------------------------------------------

class _FakeResponder:
    def __init__(self, answers, response=""):
        self.answers = list(answers)
        self.response = response


def test_every_answer_stays_on_screen():
    from src.app import _rendered_answers
    shown = _rendered_answers(_FakeResponder(["first", "second", "third"],
                                             response="third"))
    assert "first" in shown and "second" in shown and "third" in shown
    assert shown.index("first") < shown.index("third"), "oldest first"


def test_the_newest_answer_is_not_duplicated():
    """`response` still equals the answer just appended to the list."""
    from src.app import _rendered_answers
    assert _rendered_answers(
        _FakeResponder(["only"], response="only")).count("only") == 1


def test_a_repeated_previous_answer_is_not_shown_twice():
    """
    When nothing comes back, `response` is set to the previous answer rather
    than left on a spinner. Rendering the live value unconditionally would
    print that answer a second time.
    """
    from src.app import _rendered_answers
    shown = _rendered_answers(_FakeResponder(["a", "b"], response="a"))
    assert shown.count("a") == 1


def test_thinking_is_shown_while_it_is_not_yet_an_answer():
    from src.app import _rendered_answers
    shown = _rendered_answers(_FakeResponder(["done"], response="Thinking..."))
    assert "done" in shown and "Thinking..." in shown
    assert shown.index("done") < shown.index("Thinking...")


def test_an_error_reaches_the_panel():
    from src.app import _rendered_answers
    shown = _rendered_answers(_FakeResponder([], response="[error] no key"))
    assert "[error] no key" in shown


def test_clearing_the_transcript_clears_the_answers():
    """
    They caption the transcript they answer. Left behind, they would sit
    beside a conversation that is no longer there.
    """
    import inspect
    from src import app
    body = inspect.getsource(app.clear_context)
    assert "responder.answers.clear()" in body
    assert "responder" in inspect.signature(app.clear_context).parameters
