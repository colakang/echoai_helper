"""
Regression tests for the transcription pipeline.

Deliberately free of audio hardware and of FunASR: these cover the pure data
paths (buffering, format conversion, export), which is where the defects
these tests pin down actually lived.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.AudioTranscriber import AudioTranscriber, MAX_PHRASE_TIMEOUT
from src.ResponseManager import ResponseManager


class FakeSource:
    """Stands in for a Recorder's AudioSource."""

    def __init__(self, sample_rate=48000, channels=2, sample_width=2):
        self.SAMPLE_RATE = sample_rate
        self.SAMPLE_WIDTH = sample_width
        self.channels = channels


def make_transcriber(mic=None, speaker=None):
    """An AudioTranscriber with no model attached — none of these tests infer."""
    return AudioTranscriber(mic, speaker, model=None,
                            response_manager=ResponseManager())


def pcm(seconds, sample_rate=48000, channels=2, amplitude=1000):
    """`seconds` of int16 PCM as raw interleaved bytes."""
    frames = int(seconds * sample_rate)
    tone = (amplitude * np.sin(np.linspace(0, 400 * np.pi, frames))).astype(np.int16)
    return np.repeat(tone, channels).tobytes()


# --------------------------------------------------------------------------
# Missing tracks
# --------------------------------------------------------------------------

def test_speaker_only_is_allowed():
    """A Mac mini has no built-in mic; the speaker track must still work."""
    t = make_transcriber(mic=None, speaker=FakeSource())
    assert "Speaker" in t.audio_sources
    assert "You" not in t.audio_sources
    assert set(t.source_locks) == {"Speaker"}


def test_mic_only_is_allowed():
    t = make_transcriber(mic=FakeSource(sample_rate=16000, channels=1), speaker=None)
    assert set(t.audio_sources) == {"You"}


def test_no_sources_at_all_is_fatal():
    with pytest.raises(RuntimeError, match="No audio sources"):
        make_transcriber(mic=None, speaker=None)


# --------------------------------------------------------------------------
# Format conversion
# --------------------------------------------------------------------------

def test_stereo_48k_is_downmixed_and_resampled_to_mono_16k():
    t = make_transcriber(speaker=FakeSource(sample_rate=48000, channels=2))
    out = t.convert_bytes_to_numpy(pcm(1.0), "Speaker")

    assert out.dtype == np.float32
    assert out.ndim == 1
    # 1s at 16kHz, allowing for the resampler's edge handling
    assert abs(out.shape[0] - 16000) <= 16
    assert np.abs(out).max() <= 1.0


def test_mono_16k_passes_through_without_resampling():
    t = make_transcriber(mic=FakeSource(sample_rate=16000, channels=1))
    out = t.convert_bytes_to_numpy(pcm(1.0, 16000, 1), "You")
    assert out.shape[0] == 16000


def test_empty_and_ragged_buffers_do_not_raise():
    """A truncated final chunk must not blow up the reshape."""
    t = make_transcriber(speaker=FakeSource(channels=2))
    assert t.convert_bytes_to_numpy(b"", "Speaker").shape[0] == 0
    # 3 bytes: not even one whole stereo frame
    assert t.convert_bytes_to_numpy(b"\x01\x02\x03", "Speaker").shape[0] == 0


# --------------------------------------------------------------------------
# Buffer cap  (regression: unbounded growth)
# --------------------------------------------------------------------------

def test_buffer_is_capped_when_asr_returns_nothing():
    """
    Audio that clears the capture energy gate but yields no transcript used to
    accumulate forever, because last_sample only shrank on a successful
    transcription. Over a 1-2 hour meeting that is unbounded memory plus an
    ever-growing re-transcription cost.
    """
    t = make_transcriber(speaker=FakeSource())
    info = t.audio_sources["Speaker"]
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Simulate ~2x the cap arriving with the model never returning text.
    for i in range(int(MAX_PHRASE_TIMEOUT * 2)):
        stamp = now + timedelta(seconds=i)
        t.update_last_sample_and_phrase_status("Speaker", pcm(1.0), stamp)
        t._enforce_buffer_cap("Speaker", info, stamp)

    held = t._sample_seconds("Speaker", info["last_sample"])
    assert held <= MAX_PHRASE_TIMEOUT, f"buffer grew to {held:.1f}s, cap is {MAX_PHRASE_TIMEOUT}s"


def test_buffer_cap_leaves_short_phrases_alone():
    t = make_transcriber(speaker=FakeSource())
    info = t.audio_sources["Speaker"]
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    t.update_last_sample_and_phrase_status("Speaker", pcm(2.0), now)
    t._enforce_buffer_cap("Speaker", info, now)

    assert t._sample_seconds("Speaker", info["last_sample"]) == pytest.approx(2.0, abs=0.05)
    assert info["new_phrase"] is True  # untouched initial state


# --------------------------------------------------------------------------
# Export  (regression: UnboundLocalError on mic-only conversations)
# --------------------------------------------------------------------------

def test_export_with_no_speaker_messages():
    """
    Every recording made on a machine with no loopback is mic-only. That path
    used to raise UnboundLocalError, which the outer handler swallowed into an
    empty dict and the UI reported as "no conversation data available".
    """
    rm = ResponseManager()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    transcript = {
        "combined": [
            ("second thing I said", now + timedelta(seconds=5), None, "you"),
            ("first thing I said", now, None, "you"),
        ]
    }

    out = rm.export_structured_conversation(transcript)

    assert out, "export returned empty — the mic-only path is broken again"
    messages = out["conversation"]["messages"]
    assert len(messages) == 2
    assert [m["text"] for m in messages] == ["first thing I said", "second thing I said"]
    assert all(m["role"] == "you" for m in messages)


def test_export_with_both_tracks():
    rm = ResponseManager()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rid = rm.create_response(now, "what is the revenue")
    rm.update_response(rid, "fifteen percent growth", is_complete=True)

    transcript = {
        "combined": [
            ("my reply", now + timedelta(seconds=2), None, "you"),
            ("what is the revenue", now, rid, "speaker"),
        ]
    }

    out = rm.export_structured_conversation(transcript)
    assert len(out["conversation"]["messages"]) == 2
    assert out["metadata"]["total_messages"] == 2


def test_export_of_empty_transcript_is_empty_not_crashing():
    rm = ResponseManager()
    out = rm.export_structured_conversation({"combined": []})
    assert out["conversation"]["messages"] == []
