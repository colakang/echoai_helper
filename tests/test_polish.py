"""
Tests for the export-time transcript cleanup.

The provider is stubbed. What matters here is not the model's judgement but
that a bad response can never corrupt or lose a transcript: this runs over a
record of what people actually said, and silently attributing one person's
words to another would be worse than not cleaning it at all.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.polish import (
    DEFAULT_BATCH_SIZE, _parse, _strip_speaker, polish_transcript,
)


def messages(*texts, speaker="S1"):
    return [{"text": t, "speaker": speaker, "timestamp": f"t{i}"}
            for i, t in enumerate(texts)]


class StubProvider:
    """Returns preset outputs, one per batch."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def generate_response(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        yield self.outputs.pop(0) if self.outputs else ""


def corrections(**by_number):
    """Model output: only the lines that changed, keyed by their number."""
    return "\n".join(f"{n[1:]}| {t}" for n, t in sorted(by_number.items()))


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_parse_reads_numbered_corrections():
    assert _parse("1| first\n2| second", 2) == {1: "first", 2: "second"}


def test_parse_ignores_commentary():
    output = "Here are the corrections:\n\n1| first\n2| second\n\nDone!"
    assert _parse(output, 2) == {1: "first", 2: "second"}


def test_parse_is_keyed_by_number_not_position():
    """Position is never used: a reordered or partial reply must still land on
    the line it names."""
    assert _parse("2| second\n1| first", 2) == {1: "first", 2: "second"}


def test_parse_accepts_a_partial_response():
    """Only changed lines come back, so most replies are shorter than the
    batch. That is the normal case, not a failure."""
    assert _parse("2| only this one changed", 3) == {2: "only this one changed"}


def test_parse_accepts_an_empty_response():
    """Nothing to correct is a valid answer."""
    assert _parse("", 3) == {}


def test_parse_drops_out_of_range_numbers():
    """A number outside the batch would land on somebody else's words."""
    assert _parse("1| a\n7| wrong", 3) == {1: "a"}


def test_strip_speaker_prefix():
    assert _strip_speaker("[S1] hello") == "hello"
    assert _strip_speaker("hello") == "hello"
    assert _strip_speaker("[not a label because it is very long] x") == \
        "[not a label because it is very long] x"


# --------------------------------------------------------------------------
# The additive contract
# --------------------------------------------------------------------------

def test_original_text_is_never_replaced():
    """A transcript is a record. The raw version has to survive so anyone can
    check what the model changed."""
    source = messages("i want to ask for the service repare")
    provider = StubProvider([corrections(n1="I want to ask for the service repair")])

    result = polish_transcript(source, provider, batch_size=10)

    assert result.segments[0]["text"] == "i want to ask for the service repare"
    assert result.segments[0]["polished"] == "I want to ask for the service repair"


def test_input_messages_are_not_mutated():
    source = messages("original")
    polish_transcript(source, StubProvider([corrections(n1="cleaned")]), batch_size=10)
    assert source[0] == {"text": "original", "speaker": "S1", "timestamp": "t0"}
    assert "polished" not in source[0]


def test_unmentioned_lines_are_untouched():
    """A line the model does not return is left alone by construction, rather
    than by comparing text -- so it cannot be altered by accident."""
    source = messages("already fine", "also fine")
    result = polish_transcript(source, StubProvider([""]), batch_size=10)
    assert all("polished" not in s for s in result.segments)
    assert result.polished_count == 0


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------

def test_a_correction_lands_on_the_line_it_names():
    """The safety property: a partial reply corrects exactly what it names and
    touches nothing else, so words are never attributed to the wrong speaker."""
    source = messages("one", "two", "three")
    provider = StubProvider([corrections(n2="TWO")])

    result = polish_transcript(source, provider, batch_size=10)

    assert "polished" not in result.segments[0]
    assert result.segments[1]["polished"] == "TWO"
    assert "polished" not in result.segments[2]
    assert result.polished_count == 1


def test_a_provider_error_keeps_the_originals():
    class Broken:
        def generate_response(self, messages, **kwargs):
            raise RuntimeError("rate limited")
            yield  # pragma: no cover

    source = messages("one", "two")
    result = polish_transcript(source, Broken(), batch_size=10)

    assert [s["text"] for s in result.segments] == ["one", "two"]
    assert result.batches_failed == 1
    assert "rate limited" in result.errors[0]


def test_one_failed_batch_does_not_lose_the_others():
    source = messages(*[f"line {i}" for i in range(4)])
    provider = StubProvider([
        corrections(n1="LINE 0", n2="LINE 1"),
        "garbage, no numbered lines at all",
    ])

    result = polish_transcript(source, provider, batch_size=2)

    assert result.segments[0]["polished"] == "LINE 0"
    assert "polished" not in result.segments[2]
    assert result.polished_count == 2


def test_every_segment_survives_total_failure():
    source = messages(*[f"line {i}" for i in range(10)])
    result = polish_transcript(source, StubProvider([]), batch_size=3)

    assert len(result.segments) == 10
    assert [s["text"] for s in result.segments] == [f"line {i}" for i in range(10)]


def test_no_provider_is_a_no_op():
    source = messages("one", "two")
    result = polish_transcript(source, None)
    assert [s["text"] for s in result.segments] == ["one", "two"]
    assert result.polished_count == 0


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------

def test_empty_lines_are_skipped():
    source = messages("real text", "", "   ")
    provider = StubProvider([corrections(n1="REAL TEXT")])
    result = polish_transcript(source, provider, batch_size=10)

    assert result.segments[0]["polished"] == "REAL TEXT"
    assert result.polished_count == 1


def test_long_transcripts_are_split():
    source = messages(*[f"line {i}" for i in range(7)])
    provider = StubProvider([
        corrections(n1="a", n2="b", n3="c"),
        corrections(n1="d", n2="e", n3="f"),
        corrections(n1="g"),
    ])
    result = polish_transcript(source, provider, batch_size=3)

    assert result.batches_attempted == 3
    assert result.polished_count == 7


def test_context_is_sent_but_not_returned():
    """The first line of a batch needs to know what preceded it -- that is
    exactly where a mis-heard word is recoverable from context."""
    source = messages(*[f"line {i}" for i in range(6)])
    provider = StubProvider([corrections(n1="a"), corrections(n1="d")])
    polish_transcript(source, provider, batch_size=3, overlap=2)

    second_prompt = provider.prompts[1]
    assert "context only" in second_prompt
    assert "line 1" in second_prompt      # carried as context
    assert "3| [S1] line 5" in second_prompt   # actually being cleaned


def test_progress_is_reported():
    seen = []
    source = messages(*[f"line {i}" for i in range(5)])
    provider = StubProvider([corrections(n1="a"), corrections(n1="c"), corrections(n1="e")])

    polish_transcript(source, provider, batch_size=2,
                      progress=lambda done, total: seen.append((done, total)))

    assert seen[0] == (0, 3)
    assert seen[-1] == (3, 3)


def test_speaker_labels_reach_the_model():
    source = [{"text": "hello", "speaker": "S2"}]
    provider = StubProvider([corrections(n1="Hello.")])
    polish_transcript(source, provider, batch_size=10)
    assert "[S2] hello" in provider.prompts[0]


def test_summary_reports_what_was_cleaned():
    source = messages(*[f"line {i}" for i in range(4)])
    provider = StubProvider([corrections(n1="a", n2="b"), ""])
    result = polish_transcript(source, provider, batch_size=2)

    assert "2/4 lines cleaned" in result.summary()


def test_summary_reports_a_failed_batch():
    class HalfBroken:
        def __init__(self): self.calls = 0
        def generate_response(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("rate limited")
            yield "1| a\n2| b"

    source = messages(*[f"line {i}" for i in range(4)])
    result = polish_transcript(source, HalfBroken(), batch_size=2)
    assert "1/2 batches failed" in result.summary()
