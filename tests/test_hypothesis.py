"""Tests for LocalAgreement text stabilisation."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.asr.hypothesis import HypothesisTracker, detokenize, tokenize


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------

def test_latin_splits_on_whitespace():
    assert tokenize("good morning everyone") == ["good", "morning", "everyone"]


def test_cjk_splits_per_character():
    """Chinese has no spaces; whitespace splitting would yield one huge token
    and prefix matching could never make partial progress."""
    assert tokenize("各位早晨") == ["各", "位", "早", "晨"]


def test_mixed_script():
    assert tokenize("收入係 4.2 million") == ["收", "入", "係", "4.2", "million"]


def test_detokenize_roundtrip():
    for text in ["good morning everyone", "各位早晨", "收入係 4.2 million"]:
        assert detokenize(tokenize(text)) == text


# --------------------------------------------------------------------------
# Commit behaviour
# --------------------------------------------------------------------------

def test_nothing_commits_on_first_hypothesis():
    """One hypothesis is not agreement — there is nothing to agree with."""
    t = HypothesisTracker()
    assert t.update("good morning") == ""
    assert t.committed == ""
    assert t.pending == "good morning"


def test_agreeing_prefix_commits():
    t = HypothesisTracker()
    t.update("good morning")
    assert t.update("good morning everyone") == "good morning"
    assert t.committed == "good morning"
    assert t.pending == "everyone"


def test_disagreeing_tail_stays_provisional():
    """The exact jitter seen in production: 'pipeline' -> 'paper line'."""
    t = HypothesisTracker()
    t.update("the meeting transcription pipeline")
    t.update("the meeting transcription paper line")

    assert t.committed == "the meeting transcription"
    assert "pipeline" not in t.committed
    assert "paper" not in t.committed


def test_committed_text_never_changes():
    """The whole point: once shown, a word is never rewritten."""
    t = HypothesisTracker()
    seen = []
    for hypothesis in [
        "good morning",
        "good morning, everyone",
        "good morning. Everyone. Let's start",
        "good morning. Everyone. Let's start with the quarterly",
    ]:
        t.update(hypothesis)
        seen.append(t.committed)

    for earlier, later in zip(seen, seen[1:]):
        assert later.startswith(earlier), f"{earlier!r} was rewritten as {later!r}"


def test_commit_is_monotonic_under_churn():
    t = HypothesisTracker()
    lengths = []
    for hypothesis in ["a b c", "a b x", "a b c d", "a b c d e", "a q"]:
        t.update(hypothesis)
        lengths.append(len(t.committed))
    assert lengths == sorted(lengths), f"committed text shrank: {lengths}"


def test_revision_of_committed_text_is_ignored():
    """A model that contradicts settled text must not be able to retract it."""
    t = HypothesisTracker()
    t.update("hello world")
    t.update("hello world foo")
    assert t.committed == "hello world"

    t.update("goodbye world foo")          # contradicts the committed prefix
    assert t.committed.startswith("hello world")


# --------------------------------------------------------------------------
# Segment end
# --------------------------------------------------------------------------

def test_flush_commits_the_unsettled_tail():
    """At a segment boundary the audio is over, so the tail is final."""
    t = HypothesisTracker()
    t.update("good morning")
    t.update("good morning everyone")
    assert t.pending == "everyone"

    assert t.flush() == "everyone"
    assert t.committed == ""      # reset for the next segment
    assert t.pending == ""


def test_flush_on_a_single_hypothesis_loses_nothing():
    t = HypothesisTracker()
    t.update("only one pass")
    assert t.flush() == "only one pass"


def test_cantonese_commits_progressively():
    t = HypothesisTracker()
    t.update("各位早晨")
    t.update("各位早晨多謝")
    assert t.committed == "各位早晨"

    t.update("各位早晨多謝大家")
    assert t.committed == "各位早晨多謝"


def test_agreement_below_two_is_rejected():
    with pytest.raises(ValueError):
        HypothesisTracker(agreement=1)


# --------------------------------------------------------------------------
# Real capture
# --------------------------------------------------------------------------

# Verbatim hypothesis stream from a live run against BlackHole loopback,
# M4 / SenseVoiceSmall / MPS. Note the churn this has to absorb:
#   "good morning, everyone."  ->  "good morning. Everyone. Let's start..."
# a comma becomes a full stop and the next word is recapitalised. Naively
# echoing each hypothesis rewrote on-screen text 13 times across these two
# phrases.
REAL_PHRASE_1 = [
    "good morning.",
    "good morning, everyone.",
    "good morning, everyone.",
    "good morning. Everyone. Let's start with the.",
    "good morning. Everyone. Let's start with the quarterly.",
    "good morning. Everyone. Let's start with the quarterly review.",
    "good morning. Everyone. Let's start with the quarterly review our revenue.",
    "good morning. Everyone. Let's start with the quarterly review. Our revenue grew by.",
    "good morning. Everyone. Let's start with the quarterly review. Our revenue grew by 15.",
    "good morning. Everyone. Let's start with the quarterly review. Our revenue grew by 15. This quarter.",
]

REAL_PHRASE_2 = [
    "this quarter mainly.",
    "this quarter mainly driven by the.",
    "this quarter mainly driven by the enterprise.",
    "this quarter mainly driven by the enterprise segment.",
    "this quarter mainly driven by the enterprise segment.",
    "this quarter mainly driven by the enterprise segment. I'd like to hear your.",
    "this quarter mainly driven by the enterprise segment. I'd like to hear your thoughts on the.",
    "this quarter mainly driven by the enterprise segment. I'd like to hear your thoughts on the roadmap.",
    "this quarter mainly driven by the enterprise segment. I'd like to hear your thoughts on the roadmap for next year.",
]


@pytest.mark.parametrize("phrase", [REAL_PHRASE_1, REAL_PHRASE_2])
def test_real_capture_never_rewrites(phrase):
    t = HypothesisTracker()
    previous = ""
    for hypothesis in phrase:
        t.update(hypothesis)
        assert t.committed.startswith(previous), (
            f"committed text was rewritten:\n  {previous!r}\n  {t.committed!r}"
        )
        previous = t.committed


def test_real_capture_commits_most_of_the_phrase():
    """Punctuation churn must not stall the commit: comparing raw tokens
    settled only two words across this stream."""
    t = HypothesisTracker()
    for hypothesis in REAL_PHRASE_1:
        t.update(hypothesis)

    committed_words = len(t.committed.split())
    final_words = len(REAL_PHRASE_1[-1].split())
    assert committed_words >= final_words - 3, (
        f"only {committed_words}/{final_words} words settled: {t.committed!r}"
    )


def test_real_capture_flush_loses_nothing():
    t = HypothesisTracker()
    for hypothesis in REAL_PHRASE_2:
        t.update(hypothesis)
    full = t.committed + " " + t.flush()

    for word in ["enterprise", "segment", "roadmap", "next", "year"]:
        assert word in full, f"{word!r} lost at the segment boundary"
