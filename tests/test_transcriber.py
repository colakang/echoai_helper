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


from src.asr.fun_asr import AsrResult          # noqa: E402
from src.AudioTranscriber import LANGUAGE_TRUST_S  # noqa: E402


class ScriptedModel:
    """Returns preset results and records the language it was asked for."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def get_transcription_np(self, audio, language=None):
        self.calls.append(language)
        return self.results.pop(0) if self.results else AsrResult("", None)



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
# Buffer bounds
# --------------------------------------------------------------------------
#
# The transcriber used to accumulate raw audio in last_sample, which only
# shrank on a successful transcription -- audio that yielded no text grew the
# buffer without limit. That machinery is gone: the segmenter owns the buffer
# now and its max_segment_s caps it regardless of what the model returns.

def test_buffer_stays_bounded_when_asr_returns_nothing(monkeypatch):
    """Feed far more audio than the cap, with the model returning nothing at
    all, and the buffer must still not grow without limit."""
    t = make_transcriber(speaker=FakeSource())
    cap = t.segmenters["Speaker"].config.max_segment_s

    t.audio_model = ScriptedModel([])   # always returns AsrResult("", None)
    monkeypatch.setattr(t.segmenters["Speaker"].vad, "process_chunk", lambda chunk: 0.9)

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for i in range(int(cap * 3)):
        t._process_chunk("Speaker", pcm(1.0), now + timedelta(seconds=i))

    held = t.segmenters["Speaker"].active_duration_s
    assert held <= cap + 1.0, f"buffer grew to {held:.1f}s against a {cap}s cap"


def test_clear_resets_segmenters_and_trackers():
    t = make_transcriber(speaker=FakeSource())
    t.trackers["Speaker"].update("hello world")
    t.trackers["Speaker"].update("hello world again")
    assert t.trackers["Speaker"].committed

    t.clear_transcript_data()
    assert t.trackers["Speaker"].committed == ""
    assert t.segmenters["Speaker"].active_duration_s == 0


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


# --------------------------------------------------------------------------
# Language stickiness
# --------------------------------------------------------------------------
#
# SenseVoice detects the language per utterance but needs enough audio to do
# it. On a real Cantonese support call, segments under ~2s came back as
# Japanese ('ですてま。', 'え。'); a 1.7s segment in an English recording came
# back as Chinese. Raising min_speech_ms reduced but did not remove this, so
# short audio is pinned to the language its longer neighbours established.

def transcriber_with(results):
    t = make_transcriber(speaker=FakeSource())
    t.audio_model = ScriptedModel(results)
    return t


def audio_of(seconds):
    return np.zeros(int(seconds * 16000), dtype=np.float32)


def test_long_audio_is_trusted_and_remembered():
    t = transcriber_with([AsrResult("各位早晨", "yue")])
    t._transcribe_audio(audio_of(LANGUAGE_TRUST_S + 1), "Speaker")

    assert t.audio_model.calls == [None], "long audio must use auto-detect"
    assert t._session_language["Speaker"] == "yue"


def test_short_audio_is_pinned_to_the_remembered_language():
    t = transcriber_with([AsrResult("各位早晨", "yue"), AsrResult("係", "yue")])
    t._transcribe_audio(audio_of(LANGUAGE_TRUST_S + 1), "Speaker")
    t._transcribe_audio(audio_of(0.8), "Speaker")

    assert t.audio_model.calls == [None, "yue"], (
        "short audio should be pinned, not left to guess"
    )


def test_short_audio_does_not_overwrite_the_remembered_language():
    """The misdetection this exists to prevent: a sub-second burst coming
    back as Japanese must not redirect the whole session."""
    t = transcriber_with([AsrResult("各位早晨", "yue"), AsrResult("ですてま。", "ja")])
    t._transcribe_audio(audio_of(LANGUAGE_TRUST_S + 1), "Speaker")
    t._transcribe_audio(audio_of(0.7), "Speaker")

    assert t._session_language["Speaker"] == "yue"


def test_first_short_utterance_still_uses_auto_detect():
    """With nothing remembered yet there is nothing to pin to."""
    t = transcriber_with([AsrResult("hello", "en")])
    t._transcribe_audio(audio_of(0.5), "Speaker")
    assert t.audio_model.calls == [None]


def test_a_genuine_language_switch_is_followed():
    """Stickiness must not become a lock: a long utterance in another
    language is evidence, not noise."""
    t = transcriber_with([AsrResult("各位早晨", "yue"), AsrResult("good morning", "en")])
    t._transcribe_audio(audio_of(LANGUAGE_TRUST_S + 1), "Speaker")
    t._transcribe_audio(audio_of(LANGUAGE_TRUST_S + 1), "Speaker")

    assert t._session_language["Speaker"] == "en"


def test_nospeech_does_not_become_the_session_language():
    t = transcriber_with([AsrResult("各位早晨", "yue"), AsrResult("", "nospeech")])
    t._transcribe_audio(audio_of(LANGUAGE_TRUST_S + 1), "Speaker")
    t._transcribe_audio(audio_of(LANGUAGE_TRUST_S + 1), "Speaker")

    assert t._session_language["Speaker"] == "yue"


def test_tracks_keep_separate_languages():
    """A Cantonese speaker and an English one on the two tracks must not
    contaminate each other."""
    t = make_transcriber(mic=FakeSource(16000, 1), speaker=FakeSource())
    t.audio_model = ScriptedModel([AsrResult("各位早晨", "yue"),
                                   AsrResult("good morning", "en")])
    t._transcribe_audio(audio_of(LANGUAGE_TRUST_S + 1), "Speaker")
    t._transcribe_audio(audio_of(LANGUAGE_TRUST_S + 1), "You")

    assert t._session_language["Speaker"] == "yue"
    assert t._session_language["You"] == "en"
