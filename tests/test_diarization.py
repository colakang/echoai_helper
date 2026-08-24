"""
Tests for online speaker clustering.

The embedding model is not involved: these pin the clustering policy, using
synthetic vectors whose similarities are chosen to match what CAM++ actually
produced on a real two-party call (0.77-0.93 within a speaker, 0.21-0.50
across).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.asr.diarization import (
    DiarizationConfig, SpeakerRegistry, _normalise,
)

DIMS = 192


def voice(seed: int) -> np.ndarray:
    """A distinct synthetic voice."""
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=DIMS)
    return vector / np.linalg.norm(vector)


def near(base: np.ndarray, similarity: float, seed: int = 0) -> np.ndarray:
    """Another utterance by the same voice, at a chosen cosine similarity."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=DIMS)
    noise -= (noise @ base) * base          # orthogonal component
    noise /= np.linalg.norm(noise)
    mixed = similarity * base + np.sqrt(1 - similarity ** 2) * noise
    return mixed / np.linalg.norm(mixed)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

def test_normalise_produces_unit_vectors():
    assert np.isclose(np.linalg.norm(_normalise(np.array([3.0, 4.0]))), 1.0)


def test_normalise_rejects_degenerate_input():
    assert _normalise(None) is None
    assert _normalise(np.zeros(DIMS)) is None
    assert _normalise(np.array([np.nan, 1.0])) is None


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------

def test_first_voice_registers():
    registry = SpeakerRegistry()
    result = registry.assign(voice(1))
    assert result.is_new
    assert result.label == "S1"
    assert len(registry.speakers) == 1


def test_same_voice_is_recognised():
    registry = SpeakerRegistry()
    base = voice(1)
    first = registry.assign(base)
    again = registry.assign(near(base, 0.85, seed=7))

    assert not again.is_new
    assert again.label == first.label
    assert len(registry.speakers) == 1


def test_different_voice_becomes_a_new_speaker():
    registry = SpeakerRegistry()
    registry.assign(voice(1))
    other = registry.assign(voice(99))

    assert other.is_new
    assert other.label == "S2"
    assert len(registry.speakers) == 2


def test_a_meeting_of_three_is_tracked():
    registry = SpeakerRegistry()
    voices = [voice(s) for s in (1, 2, 3)]
    # Round-robin turns, as in a real conversation.
    labels = []
    for turn in range(6):
        base = voices[turn % 3]
        labels.append(registry.assign(near(base, 0.88, seed=turn)).label)

    assert len(registry.speakers) == 3
    assert labels[0] == labels[3]
    assert labels[1] == labels[4]
    assert labels[2] == labels[5]


def test_speaker_count_reflects_turns_taken():
    registry = SpeakerRegistry()
    base = voice(1)
    for turn in range(4):
        registry.assign(near(base, 0.9, seed=turn))
    assert registry.speakers[0].count == 4


# --------------------------------------------------------------------------
# Ambiguity
# --------------------------------------------------------------------------

def test_a_segment_between_two_voices_is_flagged():
    """
    A segment cut on silence can still contain a speaker change, when the
    turns follow each other without a pause. Its embedding is a blend, and it
    should be reported as uncertain rather than confidently mislabelled.
    """
    registry = SpeakerRegistry()
    first, second = voice(1), voice(2)
    registry.assign(first)
    registry.assign(second)

    blend = _normalise(first + second)
    result = registry.assign(blend)
    assert not result.confident


def test_an_ambiguous_segment_does_not_move_the_centroid():
    registry = SpeakerRegistry()
    first, second = voice(1), voice(2)
    registry.assign(first)
    registry.assign(second)
    before = registry.speakers[0].centroid.copy()

    registry.assign(_normalise(first + second))
    assert np.allclose(registry.speakers[0].centroid, before)


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------

def test_speaker_count_is_capped():
    """Without a cap, noise and crosstalk invent a speaker every few
    segments and the transcript becomes unreadable."""
    registry = SpeakerRegistry(DiarizationConfig(max_speakers=3))
    for seed in range(10):
        registry.assign(voice(seed))
    assert len(registry.speakers) == 3


def test_capped_registry_still_labels_everything():
    registry = SpeakerRegistry(DiarizationConfig(max_speakers=2))
    for seed in range(5):
        result = registry.assign(voice(seed))
        assert result.speaker is not None, "a segment must never go unlabelled"


def test_threshold_governs_splitting():
    """Lowering the threshold merges; raising it splits. The default leans
    towards merging, because an invented speaker cannot be undone by a
    reader while a wrong turn boundary is obvious."""
    base = voice(1)
    similar = near(base, 0.70, seed=3)

    lenient = SpeakerRegistry(DiarizationConfig(same_speaker_threshold=0.6))
    lenient.assign(base)
    lenient.assign(similar)
    assert len(lenient.speakers) == 1

    strict = SpeakerRegistry(DiarizationConfig(same_speaker_threshold=0.9))
    strict.assign(base)
    strict.assign(similar)
    assert len(strict.speakers) == 2


def test_reset_clears_speakers():
    registry = SpeakerRegistry()
    registry.assign(voice(1))
    registry.assign(voice(2))
    registry.reset()

    assert registry.speakers == []
    assert registry.assign(voice(5)).label == "S1"


def test_degenerate_embedding_is_not_a_speaker():
    registry = SpeakerRegistry()
    result = registry.assign(np.zeros(DIMS))
    assert result.speaker is None
    assert registry.speakers == []
