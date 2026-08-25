"""Tests for the human-facing Markdown export."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.export_markdown import render


def payload(messages, **metadata):
    return {"metadata": metadata, "conversation": {"messages": messages}}


def test_speaker_turns_group_under_a_heading():
    """A tag repeated on every line is bookkeeping; a heading per turn reads
    like a conversation."""
    out = render(payload([
        {"text": "S1: hello there", "role": "speaker"},
        {"text": "S1: still me", "role": "speaker"},
        {"text": "S2: my turn now", "role": "speaker"},
    ]))
    assert out.count("**Speaker 1**") == 1
    assert "**Speaker 2**" in out
    assert "S1:" not in out


def test_uncertain_speaker_is_marked():
    """The diarizer appends ? when a segment blended two voices."""
    out = render(payload([{"text": "S2?: who said this", "role": "speaker"}]))
    assert "(uncertain)" in out


def test_mic_track_is_labelled_you():
    out = render(payload([{"text": "something I said", "role": "you"}]))
    assert "**You**" in out


# --------------------------------------------------------------------------
# Polished vs original
# --------------------------------------------------------------------------
#
# The JSON export keeps every original line because it is a record. Markdown
# is read, so it carries the cleaned text and drops the raw one unless asked:
# showing both for every line doubles the length and buries the content.

def test_cleaned_text_is_what_gets_rendered():
    out = render(payload([
        {"text": "the servise repare", "polished": "the service repair",
         "role": "speaker"},
    ]))
    assert "the service repair" in out
    assert "the servise repare" not in out


def test_original_is_shown_on_request():
    out = render(payload([
        {"text": "the servise repare", "polished": "the service repair",
         "role": "speaker"},
    ]), include_original=True)
    assert "the service repair" in out
    assert "as heard" in out
    assert "the servise repare" in out


def test_unchanged_lines_are_not_duplicated():
    """Only lines the cleanup actually altered are worth showing twice."""
    out = render(payload([{"text": "already correct", "role": "speaker"}]),
                 include_original=True)
    assert out.count("already correct") == 1
    assert "as heard" not in out


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------

def test_suggested_replies_are_separated_from_the_transcript():
    """They were never spoken; putting them inline would misrepresent the
    conversation."""
    out = render(payload([
        {"text": "what is the revenue", "role": "speaker",
         "response": {"response_text": "4.2 million"}},
    ]))
    transcript = out.index("## Transcript")
    replies = out.index("## Suggested replies")
    assert transcript < replies
    assert "not spoken" in out


def test_no_replies_section_when_there_are_none():
    out = render(payload([{"text": "just talking", "role": "speaker"}]))
    assert "Suggested replies" not in out


def test_cleanup_provenance_is_recorded():
    out = render(payload(
        [{"text": "a", "role": "speaker"}],
        cleanup={"model": "claude", "lines_cleaned": 12, "batches_failed": 1},
    ))
    assert "12 lines corrected by claude" in out
    assert "1 batches failed" in out


def test_timestamps_render_as_clock_time():
    out = render(payload([
        {"text": "hello", "role": "speaker", "timestamp": "2026-08-24T14:30:05"},
    ]))
    assert "14:30:05" in out
    assert "2026-08-24T" not in out


def test_a_malformed_timestamp_does_not_break_the_render():
    out = render(payload([
        {"text": "hello", "role": "speaker", "timestamp": "not a date"},
    ]))
    assert "hello" in out


def test_empty_conversation_says_so():
    out = render(payload([]))
    assert "No conversation was recorded" in out


def test_blank_lines_are_skipped():
    out = render(payload([
        {"text": "   ", "role": "speaker"},
        {"text": "real content", "role": "speaker"},
    ]))
    assert "real content" in out
    assert "**Lines:** 1" in out


def test_output_ends_with_a_single_newline():
    out = render(payload([{"text": "hello", "role": "speaker"}]))
    assert out.endswith("\n")
    assert not out.endswith("\n\n")
