"""
Tests for VAD speech segmentation.

The VAD itself is stubbed: these pin down the state machine (hysteresis,
padding, minimum durations, the hard cap), not silero's acoustic judgement.
A real-audio calibration lives in scripts/calibrate_vad.py.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.asr.segmenter import (
    Event, SegmenterConfig, SpeechSegmenter, SAMPLE_RATE, WINDOW_SAMPLES,
)


class ScriptedVAD:
    """Returns a preset probability per 100ms window."""

    def __init__(self, probabilities):
        self.probabilities = list(probabilities)
        self.calls = 0

    def process_chunk(self, chunk):
        value = self.probabilities[self.calls] if self.calls < len(self.probabilities) else 0.0
        self.calls += 1
        return value

    def reset(self):
        self.calls = 0


def audio_for(windows: int) -> np.ndarray:
    """Enough audio to drive `windows` VAD decisions."""
    return np.zeros(windows * WINDOW_SAMPLES, dtype=np.float32)


def run(probabilities, config=None):
    vad = ScriptedVAD(probabilities)
    segmenter = SpeechSegmenter(vad, config)
    events = segmenter.process(audio_for(len(probabilities)))
    return segmenter, events


# --------------------------------------------------------------------------
# Basic detection
# --------------------------------------------------------------------------

def test_silence_alone_produces_nothing():
    _, events = run([0.0] * 20)
    assert events == []


def test_speech_then_silence_yields_one_segment():
    # 10 windows of speech (1s), then 10 of silence (1s > 700ms)
    _, events = run([0.9] * 10 + [0.0] * 10)
    kinds = [e for e, _ in events]
    assert kinds == [Event.SPEECH_START, Event.SPEECH_END]

    segment = events[-1][1]
    assert segment is not None
    assert segment.duration_s == pytest.approx(1.0, abs=0.35)  # plus padding


def test_two_utterances_separated_by_a_pause():
    probabilities = [0.9] * 10 + [0.0] * 10 + [0.9] * 10 + [0.0] * 10
    _, events = run(probabilities)
    ends = [s for e, s in events if e is Event.SPEECH_END]
    assert len(ends) == 2


def test_speech_still_open_emits_no_end():
    segmenter, events = run([0.9] * 20)
    assert [e for e, _ in events] == [Event.SPEECH_START]
    assert segmenter.in_speech


# --------------------------------------------------------------------------
# Hysteresis
# --------------------------------------------------------------------------

def test_short_dip_does_not_split_an_utterance():
    """A 300ms dip below the speech threshold is an intra-sentence gap, not a
    boundary; a single-threshold detector would cut here."""
    probabilities = [0.9] * 5 + [0.4] * 3 + [0.9] * 5 + [0.0] * 10
    _, events = run(probabilities)
    assert len([s for e, s in events if e is Event.SPEECH_END]) == 1


def test_borderline_probability_does_not_start_speech():
    """Between the two thresholds is not enough to *enter* speech."""
    _, events = run([0.5] * 20, SegmenterConfig(speech_threshold=0.6,
                                                silence_threshold=0.35))
    assert events == []


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        SegmenterConfig(speech_threshold=0.3, silence_threshold=0.8)


# --------------------------------------------------------------------------
# Duration rules
# --------------------------------------------------------------------------

def test_blip_shorter_than_min_speech_is_discarded():
    """A cough is one loud window, not an utterance."""
    _, events = run([0.9] * 1 + [0.0] * 10,
                    SegmenterConfig(min_speech_ms=250))
    assert [s for e, s in events if e is Event.SPEECH_END] == []


def test_silence_shorter_than_min_silence_does_not_close():
    config = SegmenterConfig(min_silence_ms=700)
    segmenter, events = run([0.9] * 10 + [0.0] * 5, config)   # 500ms silence
    assert [s for e, s in events if e is Event.SPEECH_END] == []
    assert segmenter.in_speech


def test_continuous_speech_is_cut_at_the_hard_cap():
    """Someone who never pauses must still yield segments, or the buffer and
    the re-transcription cost grow without bound."""
    config = SegmenterConfig(max_segment_s=2.0)
    _, events = run([0.9] * 60, config)          # 6s unbroken
    ends = [s for e, s in events if e is Event.SPEECH_END]
    assert len(ends) >= 2
    for segment in ends:
        assert segment.duration_s <= config.max_segment_s + 0.5


# --------------------------------------------------------------------------
# Padding
# --------------------------------------------------------------------------

def test_onset_padding_is_included():
    """VAD trips slightly late; without pre-roll the first phoneme is lost."""
    config = SegmenterConfig(speech_pad_ms=200)
    _, events = run([0.0] * 5 + [0.9] * 10 + [0.0] * 10, config)
    segment = [s for e, s in events if e is Event.SPEECH_END][0]
    # 1.0s speech + up to 200ms pre-roll + up to 200ms tail
    assert segment.duration_s > 1.0


# --------------------------------------------------------------------------
# Streaming / partials
# --------------------------------------------------------------------------

def test_active_audio_grows_during_speech():
    vad = ScriptedVAD([0.9] * 20)
    segmenter = SpeechSegmenter(vad)
    lengths = []
    for _ in range(4):
        segmenter.process(audio_for(5))
        lengths.append(len(segmenter.active_audio))
    assert lengths == sorted(lengths)
    assert lengths[-1] > lengths[0]


def test_chunk_boundaries_do_not_change_the_result():
    """Audio arrives in 0.6s recorder chunks, which do not divide the 100ms
    VAD window evenly; leftovers must carry across calls."""
    probabilities = [0.9] * 10 + [0.0] * 10

    _, whole = run(probabilities)

    vad = ScriptedVAD(probabilities)
    segmenter = SpeechSegmenter(vad)
    piecemeal = []
    audio = audio_for(len(probabilities))
    step = int(0.6 * SAMPLE_RATE)          # 9600, not a multiple of 1600*n
    for i in range(0, len(audio), step):
        piecemeal.extend(segmenter.process(audio[i:i + step]))

    assert [e for e, _ in whole] == [e for e, _ in piecemeal]


def test_flush_closes_an_open_utterance():
    vad = ScriptedVAD([0.9] * 10)
    segmenter = SpeechSegmenter(vad)
    segmenter.process(audio_for(10))
    assert segmenter.in_speech

    segment = segmenter.flush()
    assert segment is not None
    assert not segmenter.in_speech
    assert segmenter.flush() is None


# --------------------------------------------------------------------------
# Coverage: the invariant that matters most
# --------------------------------------------------------------------------
#
# An earlier design treated the VAD as a filter and discarded anything it
# judged non-speech. Measured against three real recordings it lost 22-28% of
# the audio, including whole audible sentences. For meeting notes, dropping
# speech is far worse than a ragged boundary, so the segmenter now decides
# only *where to cut*, never *what to keep*.

def _total_samples(segments):
    return sum(len(s.audio) for s in segments)


def assert_speech_is_covered(segments, probabilities, threshold=0.35):
    """
    Every window up to and including the last speech-bearing one must be
    accounted for.

    Not "every sample is emitted": trailing silence after the last word is
    correctly discarded (see test_pure_silence_still_produces_nothing). The
    invariant is that no *speech* is lost, which is what the earlier design
    violated.
    """
    speech_indices = [i for i, p in enumerate(probabilities) if p >= threshold]
    if not speech_indices:
        return
    required = (speech_indices[-1] + 1) * WINDOW_SAMPLES
    got = _total_samples(segments)
    assert got >= required, (
        f"lost {(required - got) / WINDOW_SAMPLES:.0f} windows of audio "
        f"({got} samples emitted, {required} needed to cover all speech)"
    )


def collect(probabilities, config=None, chunk_windows=None):
    """Run to completion and return every emitted segment, flush included."""
    vad = ScriptedVAD(probabilities)
    segmenter = SpeechSegmenter(vad, config)
    audio = audio_for(len(probabilities))
    segments = []

    step = chunk_windows * WINDOW_SAMPLES if chunk_windows else len(audio)
    for i in range(0, len(audio), step):
        for event, segment in segmenter.process(audio[i:i + step]):
            if event is Event.SPEECH_END:
                segments.append(segment)
    tail = segmenter.flush()
    if tail is not None:
        segments.append(tail)
    return segments


def test_no_audio_is_lost_between_utterances():
    """Speech, pause, speech: the pause belongs to one side or the other,
    never to neither."""
    probabilities = [0.9] * 10 + [0.0] * 10 + [0.9] * 10
    segments = collect(probabilities)
    assert_speech_is_covered(segments, probabilities)


def test_no_audio_is_lost_with_quiet_speech():
    """The regression that started this: a passage the VAD scores below the
    entry threshold is still audio and must survive."""
    probabilities = [0.9] * 8 + [0.2] * 12 + [0.9] * 8 + [0.0] * 10
    segments = collect(probabilities)
    assert_speech_is_covered(segments, probabilities, threshold=0.15)


def test_short_utterance_is_carried_forward_not_dropped():
    """Below min_speech_ms it does not become its own segment, but its audio
    joins the next one."""
    config = SegmenterConfig(min_speech_ms=900)
    probabilities = [0.9] * 3 + [0.0] * 10 + [0.9] * 15 + [0.0] * 10
    segments = collect(probabilities, config)

    assert len(segments) >= 1
    assert_speech_is_covered(segments, probabilities)


def test_segments_tile_the_timeline_without_gaps():
    probabilities = ([0.9] * 8 + [0.0] * 10) * 3
    segments = collect(probabilities)
    for earlier, later in zip(segments, segments[1:]):
        assert later.start_s == pytest.approx(earlier.end_s, abs=1e-6), (
            f"gap between {earlier.end_s:.2f}s and {later.start_s:.2f}s"
        )


def test_coverage_holds_across_recorder_chunk_boundaries():
    probabilities = [0.9] * 10 + [0.0] * 10 + [0.9] * 10 + [0.0] * 10
    whole = collect(probabilities)
    piecemeal = collect(probabilities, chunk_windows=6)   # 0.6s recorder chunks
    assert _total_samples(whole) == _total_samples(piecemeal)
    assert_speech_is_covered(piecemeal, probabilities)


def test_flush_ignores_min_speech_so_the_last_words_survive():
    """At end of stream there is no next segment to carry a short tail into;
    enforcing the minimum there would silently drop the final utterance."""
    config = SegmenterConfig(min_speech_ms=900)
    segments = collect([0.9] * 3, config)      # 300ms of speech, then nothing
    assert len(segments) == 1
    assert _total_samples(segments) == 3 * WINDOW_SAMPLES


def test_pure_silence_still_produces_nothing():
    """Coverage must not mean transcribing an empty room."""
    assert collect([0.0] * 30) == []
