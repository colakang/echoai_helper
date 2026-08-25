"""Tests for the export dialog's estimates and the cancellable cleanup."""

import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.export_dialog import ExportChoices, estimate_minutes
from src.polish import polish_transcript


# --------------------------------------------------------------------------
# Time estimates
# --------------------------------------------------------------------------
#
# A real meeting produced 1311 lines. Without a figure up front, the user has
# no way to tell a 40-minute job from a hung one.

def test_estimate_is_a_range_not_a_number():
    """Per 25-line batch the CLI measured 20-50s, the spread being queueing
    upstream. A single number would be wrong in one direction."""
    assert "-" in estimate_minutes(1000, "cli")


def test_cli_is_slower_than_api():
    def low(text):
        return int(text.split("-")[0].split()[0])
    assert low(estimate_minutes(1000, "cli")) > low(estimate_minutes(1000, "api"))


def test_a_real_meeting_reads_as_tens_of_minutes():
    text = estimate_minutes(1311, "cli")
    assert any(str(n) in text for n in range(15, 60))


def test_a_short_conversation_is_not_alarming():
    assert "minute" in estimate_minutes(10, "api")


def test_estimate_scales_with_length():
    def low(text):
        head = text.split("-")[0].split()[0]
        return int(head) if head.isdigit() else 0
    assert low(estimate_minutes(2000, "cli")) > low(estimate_minutes(200, "cli"))


# --------------------------------------------------------------------------
# Choices
# --------------------------------------------------------------------------

def test_markdown_is_detected_by_extension():
    assert ExportChoices(path="/tmp/notes.md").is_markdown
    assert not ExportChoices(path="/tmp/notes.json").is_markdown


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------
#
# Cleanup runs on a worker thread now. Stopping it must keep what has already
# been done: on a long meeting, most of the value is in the batches that
# already finished.

class CountingProvider:
    def __init__(self):
        self.calls = 0

    def generate_response(self, messages, **kwargs):
        self.calls += 1
        yield f"1| cleaned {self.calls}"

    def get_model_name(self):
        return "counting"


def lines(n):
    return [{"text": f"line {i}", "speaker": "S1"} for i in range(n)]


def test_cancelling_keeps_completed_batches():
    cancel = threading.Event()
    provider = CountingProvider()

    original = provider.generate_response

    def stop_after_first(messages, **kwargs):
        cancel.set()
        return original(messages, **kwargs)

    provider.generate_response = stop_after_first

    result = polish_transcript(lines(10), provider, batch_size=2,
                               cancelled=cancel)

    assert result.cancelled
    assert result.batches_attempted == 1, "it should stop between batches"
    assert result.polished_count >= 1, "finished work was thrown away"
    assert len(result.segments) == 10, "segments were lost"


def test_cancelling_before_the_first_batch_changes_nothing():
    cancel = threading.Event()
    cancel.set()
    provider = CountingProvider()

    result = polish_transcript(lines(10), provider, batch_size=2,
                               cancelled=cancel)

    assert result.cancelled
    assert provider.calls == 0
    assert all("polished" not in s for s in result.segments)


def test_cancellation_is_reported_in_the_summary():
    cancel = threading.Event()
    cancel.set()
    result = polish_transcript(lines(4), CountingProvider(), batch_size=2,
                               cancelled=cancel)
    assert "stopped early" in result.summary()


def test_running_to_completion_is_not_marked_cancelled():
    result = polish_transcript(lines(4), CountingProvider(), batch_size=2,
                               cancelled=threading.Event())
    assert not result.cancelled
