"""Tests for prompt assembly and the CLI provider."""

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts import DEFAULT_HISTORY_TURNS, build_messages
from src.llm.cli_provider import CLIProvider, _clean, flatten
from src.ResponseManager import ResponseManager


# --------------------------------------------------------------------------
# Message assembly
# --------------------------------------------------------------------------
#
# The old builder flattened everything into one user message and carried a
# single past question, so the model could not see what it had already said.
# In a live run it answered "How may I assist you today?" to four different
# utterances in a row.

def test_history_becomes_real_turns():
    history = [("what is revenue", "4.2 million"), ("and growth", "fifteen percent")]
    messages = build_messages("You are Echo.", history, "what about churn")

    assert [m["role"] for m in messages] == [
        "system", "user", "assistant", "user", "assistant", "user",
    ]
    assert messages[-1]["content"] == "what about churn"


def test_past_answers_keep_their_brackets():
    """The model is asked to answer in brackets; its own prior turns should
    look the way its next one is expected to."""
    messages = build_messages("role", [("q", "an answer")], "next")
    assert messages[2]["content"] == "[an answer]"


def test_rules_live_in_the_system_message():
    """They used to be restated inside every user message."""
    messages = build_messages("You are Echo.", [], "hello there")
    assert "None" in messages[0]["content"]
    assert "square brackets" in messages[0]["content"]
    assert "None" not in messages[-1]["content"]


def test_operator_role_is_preserved():
    messages = build_messages("You are Echo, a support agent.", [], "hi there")
    assert messages[0]["content"].startswith("You are Echo, a support agent.")


def test_history_is_bounded():
    history = [(f"q{i}", f"a{i}") for i in range(50)]
    messages = build_messages("role", history, "current")
    assert len(messages) == 1 + 2 * DEFAULT_HISTORY_TURNS + 1
    assert messages[1]["content"] == f"q{50 - DEFAULT_HISTORY_TURNS}"


def test_unanswered_turns_do_not_fabricate_an_assistant_reply():
    messages = build_messages("role", [("asked but unanswered", "")], "current")
    assert [m["role"] for m in messages] == ["system", "user", "user"]


def test_empty_history_still_produces_a_valid_exchange():
    messages = build_messages("role", [], "the first question")
    assert [m["role"] for m in messages] == ["system", "user"]


# --------------------------------------------------------------------------
# History source
# --------------------------------------------------------------------------

def make_manager(pairs, complete=True):
    manager = ResponseManager()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for index, (question, answer) in enumerate(pairs):
        rid = manager.create_response(now + timedelta(seconds=index), question)
        manager.update_response(rid, answer, is_complete=complete)
    return manager


def test_recent_exchanges_are_oldest_first():
    manager = make_manager([("first", "1"), ("second", "2"), ("third", "3")])
    assert manager.recent_exchanges() == [
        ("first", "1"), ("second", "2"), ("third", "3"),
    ]


def test_recent_exchanges_respects_the_limit():
    manager = make_manager([(f"q{i}", f"a{i}") for i in range(10)])
    assert len(manager.recent_exchanges(limit=3)) == 3
    assert manager.recent_exchanges(limit=3)[0] == ("q7", "a7")


def test_incomplete_answers_are_excluded():
    """A half-streamed answer must not be shown back to the model as settled."""
    manager = make_manager([("streaming now", "partial ans")], complete=False)
    assert manager.recent_exchanges() == []


# --------------------------------------------------------------------------
# CLI provider
# --------------------------------------------------------------------------

def test_flatten_labels_the_turns():
    """These CLIs take one prompt on stdin with no role structure, so the
    turns have to be labelled in text."""
    messages = build_messages("You are Echo.", [("q1", "a1")], "q2")
    text = flatten(messages)

    assert "You are Echo." in text
    assert "Speaker: q1" in text
    assert "You previously replied: [a1]" in text
    assert text.index("Speaker: q1") < text.index("Speaker: q2")


def test_flatten_skips_blank_messages():
    text = flatten([{"role": "user", "content": "  "},
                    {"role": "user", "content": "real"}])
    assert text.count("Speaker:") == 1


def test_unknown_cli_is_rejected():
    with pytest.raises(ValueError, match="unknown CLI"):
        CLIProvider(command="definitely-not-installed")


def test_missing_cli_fails_validation():
    provider = CLIProvider(command="claude")
    provider.command = "claude"
    import shutil
    if shutil.which("claude") is None:
        assert not provider.validate_config()


def test_clean_passes_plain_text_through():
    assert _clean("  the answer  ") == "the answer"


def test_clean_unwraps_json_output():
    """--output-format json would otherwise put a blob on screen as the
    answer."""
    assert _clean('{"result": "the answer", "cost": 1}') == "the answer"


def test_clean_survives_malformed_json():
    assert _clean('{not really json') == '{not really json'


def test_model_name_includes_the_command():
    assert CLIProvider(command="claude").get_model_name() == "claude"
    assert CLIProvider(command="claude", model="opus").get_model_name() == "claude:opus"
