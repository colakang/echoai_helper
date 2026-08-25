"""
Tests for on-disk session recording.

The transcript used to live only in memory until someone exported it. A real
meeting ran 84 minutes and produced 1311 lines, all of it riding on one
process staying alive -- and during that same session the microphone stream
died silently, where the obvious fix was a restart that would have discarded
everything.
"""

import json
import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.session import (
    SessionWriter, delete_session, find_recoverable, inspect_session,
    list_sessions, read_session, to_conversation,
)


@pytest.fixture
def directory(tmp_path):
    return tmp_path / "sessions"


def writer(directory):
    directory.mkdir(parents=True, exist_ok=True)
    return SessionWriter(directory)


def utterance(w, text, track="Speaker", **kw):
    w.append(track=track, text=text, timestamp=datetime.now(), **kw)


# --------------------------------------------------------------------------
# Durability
# --------------------------------------------------------------------------

def test_utterances_are_on_disk_before_the_session_ends():
    """The whole point: a crash costs the last sentence, not the meeting."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        w = writer(Path(tmp) / "s")
        utterance(w, "first thing said")
        utterance(w, "second thing said")

        # No close() -- as if the process were killed here.
        records = read_session(w.path)
        texts = [r["text"] for r in records if r.get("type") == "utterance"]
        assert texts == ["first thing said", "second thing said"]


def test_an_unclosed_session_is_marked_as_crashed(directory):
    w = writer(directory)
    utterance(w, "something")

    info = inspect_session(w.path)
    assert info.crashed
    assert not info.closed


def test_a_closed_session_is_not_flagged(directory):
    w = writer(directory)
    utterance(w, "something")
    w.close()

    assert not inspect_session(w.path).crashed


def test_an_empty_crashed_session_is_not_offered(directory):
    """Launching and closing without speaking is not a meeting to recover."""
    writer(directory)
    assert find_recoverable(directory) is None


def test_a_truncated_final_line_is_skipped(directory):
    """Killed mid-write. Losing the last utterance is the cost of a crash;
    losing the file over it would not be."""
    w = writer(directory)
    utterance(w, "complete line")
    with open(w.path, "a", encoding="utf-8") as f:
        f.write('{"type": "utterance", "text": "trunca')

    records = read_session(w.path)
    assert [r["text"] for r in records if r.get("type") == "utterance"] == \
        ["complete line"]


def test_writing_to_an_unwritable_path_is_not_fatal(tmp_path):
    """A meeting recorded but unsaved beats one that refuses to start."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    os.chmod(blocked, 0o500)
    try:
        w = SessionWriter(blocked)
        w.append(track="Speaker", text="still works", timestamp=datetime.now())
        w.close()
    finally:
        os.chmod(blocked, 0o700)


# --------------------------------------------------------------------------
# Recovery
# --------------------------------------------------------------------------

def test_the_most_recent_crash_is_offered(directory):
    older = writer(directory)
    utterance(older, "older meeting")
    older.close()

    newer = writer(directory)
    utterance(newer, "unexported meeting")

    found = find_recoverable(directory)
    assert found is not None
    assert found.path == newer.path


def test_an_old_crash_is_left_in_the_history(directory):
    """A meeting from last week is something to export from the history, not
    to resume into: splicing it onto today's would merge two conversations."""
    w = writer(directory)
    utterance(w, "last week")
    stale = datetime.now() - timedelta(days=7)
    os.utime(w.path, (stale.timestamp(), stale.timestamp()))

    assert find_recoverable(directory, within_hours=12) is None
    assert len(list_sessions(directory)) == 1


# --------------------------------------------------------------------------
# Reading back
# --------------------------------------------------------------------------

def test_a_session_becomes_an_export_payload(directory):
    w = writer(directory)
    w.append(track="Speaker", text="what is the revenue",
             timestamp=datetime.now(), speaker="S1", response_id="r1")
    w.append(track="You", text="four point two million",
             timestamp=datetime.now(), response_id="r2")
    w.close()

    conversation = to_conversation(w.path)
    messages = conversation["conversation"]["messages"]

    assert [m["role"] for m in messages] == ["speaker", "you"]
    assert messages[0]["speaker"] == "S1"


def test_voice_prints_survive_the_round_trip(directory):
    """Without them a past meeting's speakers cannot be re-grouped -- which is
    exactly what could not be done for the first real recording."""
    w = writer(directory)
    w.append(track="Speaker", text="hello", timestamp=datetime.now(),
             speaker="S1", response_id="r1",
             embedding=[0.1234567, -0.7654321])
    w.close()

    messages = to_conversation(w.path)["conversation"]["messages"]
    assert messages[0]["embedding"] == [0.1235, -0.7654]


def test_answers_are_attached_to_their_question(directory):
    w = writer(directory)
    w.append(track="Speaker", text="what is the revenue",
             timestamp=datetime.now(), response_id="r1")
    w.append_response("r1", "what is the revenue", "4.2 million")
    w.close()

    messages = to_conversation(w.path)["conversation"]["messages"]
    assert messages[0]["response"]["response_text"] == "4.2 million"


def test_blank_utterances_are_not_recorded(directory):
    w = writer(directory)
    utterance(w, "   ")
    utterance(w, "real")
    assert w.line_count == 1


# --------------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------------

def test_sessions_are_listed_newest_first(directory):
    for text in ("first", "second", "third"):
        w = writer(directory)
        utterance(w, text)
        w.close()

    listed = list_sessions(directory)
    assert len(listed) == 3
    assert listed[0].started >= listed[-1].started


def test_a_session_can_be_deleted(directory):
    w = writer(directory)
    utterance(w, "disposable")
    w.close()

    assert delete_session(w.path)
    assert list_sessions(directory) == []


def test_the_description_says_what_matters(directory):
    w = writer(directory)
    utterance(w, "one", speaker="S1")
    utterance(w, "two", speaker="S2")

    text = inspect_session(w.path).describe()
    assert "2 lines" in text
    assert "2 speakers" in text
    assert "not exported" in text
