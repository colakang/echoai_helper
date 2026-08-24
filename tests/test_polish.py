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


def numbered(*lines):
    return "\n".join(f"{i}| {t}" for i, t in enumerate(lines, start=1))


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_parse_reads_numbered_lines():
    assert _parse("1| first\n2| second", 2) == ["first", "second"]


def test_parse_ignores_commentary():
    output = "Here are the cleaned lines:\n\n1| first\n2| second\n\nDone!"
    assert _parse(output, 2) == ["first", "second"]


def test_parse_is_keyed_by_number_not_position():
    """A dropped or reordered line must not shift every later correction onto
    the wrong segment."""
    assert _parse("2| second\n1| first", 2) == ["first", "second"]


def test_parse_rejects_a_short_response():
    """Missing lines mean the rest can no longer be matched with confidence."""
    assert _parse("1| only one", 3) == []


def test_parse_rejects_out_of_range_numbers():
    assert _parse("1| a\n2| b\n7| c", 3) == []


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
    provider = StubProvider([numbered("I want to ask for the service repair")])

    result = polish_transcript(source, provider, batch_size=10)

    assert result.segments[0]["text"] == "i want to ask for the service repare"
    assert result.segments[0]["polished"] == "I want to ask for the service repair"


def test_input_messages_are_not_mutated():
    source = messages("original")
    polish_transcript(source, StubProvider([numbered("cleaned")]), batch_size=10)
    assert source[0] == {"text": "original", "speaker": "S1", "timestamp": "t0"}
    assert "polished" not in source[0]


def test_unchanged_lines_get_no_polished_field():
    source = messages("already fine")
    result = polish_transcript(source, StubProvider([numbered("already fine")]),
                               batch_size=10)
    assert "polished" not in result.segments[0]
    assert result.polished_count == 0


# --------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------

def test_a_wrong_line_count_discards_the_batch():
    """Applying a mismatched response would attribute one speaker's words to
    another."""
    source = messages("one", "two", "three")
    provider = StubProvider([numbered("ONE", "TWO")])   # only two back

    result = polish_transcript(source, provider, batch_size=10)

    assert all("polished" not in s for s in result.segments)
    assert result.batches_failed == 1
    assert not result.ok


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
        numbered("LINE 0", "LINE 1"),
        "garbage, no numbered lines at all",
    ])

    result = polish_transcript(source, provider, batch_size=2)

    assert result.segments[0]["polished"] == "LINE 0"
    assert "polished" not in result.segments[2]
    assert result.batches_failed == 1
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
    provider = StubProvider([numbered("REAL TEXT")])
    result = polish_transcript(source, provider, batch_size=10)

    assert result.segments[0]["polished"] == "REAL TEXT"
    assert result.polished_count == 1


def test_long_transcripts_are_split():
    source = messages(*[f"line {i}" for i in range(7)])
    provider = StubProvider([
        numbered("a", "b", "c"), numbered("d", "e", "f"), numbered("g"),
    ])
    result = polish_transcript(source, provider, batch_size=3)

    assert result.batches_attempted == 3
    assert result.polished_count == 7


def test_context_is_sent_but_not_returned():
    """The first line of a batch needs to know what preceded it -- that is
    exactly where a mis-heard word is recoverable from context."""
    source = messages(*[f"line {i}" for i in range(6)])
    provider = StubProvider([numbered("a", "b", "c"), numbered("d", "e", "f")])
    polish_transcript(source, provider, batch_size=3, overlap=2)

    second_prompt = provider.prompts[1]
    assert "context only" in second_prompt
    assert "line 1" in second_prompt      # carried as context
    assert "3| [S1] line 5" in second_prompt   # actually being cleaned


def test_progress_is_reported():
    seen = []
    source = messages(*[f"line {i}" for i in range(5)])
    provider = StubProvider([numbered("a", "b"), numbered("c", "d"), numbered("e")])

    polish_transcript(source, provider, batch_size=2,
                      progress=lambda done, total: seen.append((done, total)))

    assert seen[0] == (0, 3)
    assert seen[-1] == (3, 3)


def test_speaker_labels_reach_the_model():
    source = [{"text": "hello", "speaker": "S2"}]
    provider = StubProvider([numbered("Hello.")])
    polish_transcript(source, provider, batch_size=10)
    assert "[S2] hello" in provider.prompts[0]


def test_summary_reports_partial_success():
    source = messages(*[f"line {i}" for i in range(4)])
    provider = StubProvider([numbered("a", "b"), "broken"])
    result = polish_transcript(source, provider, batch_size=2)

    assert "2/4 lines cleaned" in result.summary()
    assert "1/2 batches failed" in result.summary()
