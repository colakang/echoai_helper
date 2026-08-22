"""
src/asr/segmenter.py

VAD-driven speech segmentation.

Turns a continuous audio stream into utterances by watching where the speaker
pauses. That boundary is what the rest of the pipeline hangs off:

  - it bounds the audio buffer, so a long meeting cannot accumulate forever;
  - it is where the ASR hypothesis is final, so LocalAgreement can flush;
  - it is the "this sentence is finished" signal that triggers LLM cleanup.

Built on the silero VAD already vendored at src/asr/models/silero_vad.onnx,
which was in the repo but never wired into the transcription path. It is
cheap enough to ignore: measured RTF 0.0027 on an M4, i.e. ~0.3% of one core.

Thresholds are deliberately configurable and the defaults are a starting
point, not a tuned answer. They were chosen against synthesised speech, whose
longest pause is ~0.3s; real speakers pause 0.5-2s between sentences, and the
right `min_silence_ms` can only be set from real recordings.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 1600  # 100ms — the granularity VAD decisions are made at


class Event(Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"      # carries the completed utterance


@dataclass
class Segment:
    """A completed utterance."""
    audio: np.ndarray
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class SegmenterConfig:
    # Hysteresis: entering speech needs more confidence than staying in it.
    # A single threshold makes the state flap on every borderline window and
    # shreds an utterance into fragments.
    speech_threshold: float = 0.6
    silence_threshold: float = 0.35

    # How much silence ends an utterance. The dominant accuracy/latency knob:
    # too short splits mid-sentence and denies the model the context it needs
    # to punctuate; too long merges separate turns and delays the LLM cleanup
    # trigger. 700ms is a guess pending calibration on real speech.
    min_silence_ms: int = 700

    # Utterances shorter than this are dropped. Two reasons to keep it high:
    #
    # 1. SenseVoice's automatic language detection is unreliable on very short
    #    audio. On a real Cantonese call, sub-second segments came back as
    #    Japanese -- 'ですてま。', 'え。' -- because there is not enough signal
    #    to identify the language. Raising this from 250ms to 900ms took the
    #    misdetections on that recording from 2 to 0.
    # 2. In a meeting, sub-second utterances are mostly backchannel ("yeah",
    #    "嗯", "对") that adds noise to a transcript rather than content.
    #
    # The cost is real: on that same call it also dropped the customer
    # reciting a phone number in 0.7-1.1s bursts. Lower it for fast
    # turn-taking material where short replies carry information.
    min_speech_ms: int = 700

    # Keep a little audio either side of the detected boundary: VAD trips
    # slightly late on onsets and slightly early on trailing consonants, and
    # clipping either costs a word.
    speech_pad_ms: int = 200

    # Hard ceiling. Someone who never pauses must still produce segments, or
    # the buffer grows without bound and every re-transcription gets slower.
    max_segment_s: float = 25.0

    def __post_init__(self):
        if not 0.0 <= self.silence_threshold <= self.speech_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= silence <= speech <= 1, got "
                f"silence={self.silence_threshold} speech={self.speech_threshold}"
            )


class SpeechSegmenter:
    """
    Feed audio; get back completed utterances.

        segmenter = SpeechSegmenter(vad)
        for chunk in stream:
            for event, segment in segmenter.process(chunk):
                ...

    `active_audio` exposes the utterance in progress, so a caller wanting live
    partial transcripts can transcribe it without waiting for the boundary.
    """

    def __init__(self, vad, config: Optional[SegmenterConfig] = None):
        self.vad = vad
        self.config = config or SegmenterConfig()

        self._leftover = np.zeros(0, dtype=np.float32)
        self._speech: List[np.ndarray] = []
        self._pre_pad: List[np.ndarray] = []      # windows before speech began
        self._trailing_silence: List[np.ndarray] = []
        self._in_speech = False
        self._silence_ms = 0
        self._speech_ms = 0
        self._position_s = 0.0                    # stream time consumed
        self._segment_start_s = 0.0

        pad_windows = max(1, self.config.speech_pad_ms // 100)
        self._pad_windows = pad_windows

    # -- introspection -----------------------------------------------------

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def active_audio(self) -> np.ndarray:
        """The utterance so far, for live partial transcription."""
        if not self._speech:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._speech)

    @property
    def active_duration_s(self) -> float:
        return sum(len(w) for w in self._speech) / SAMPLE_RATE

    # -- main loop ---------------------------------------------------------

    def process(self, audio: np.ndarray) -> List[tuple]:
        """
        Consume a chunk of mono float32 @16kHz.

        Returns a list of (Event, Segment | None). Segment is populated on
        SPEECH_END and None on SPEECH_START.
        """
        events: List[tuple] = []
        audio = np.asarray(audio, dtype=np.float32)
        buffer = np.concatenate([self._leftover, audio]) if self._leftover.size else audio

        offset = 0
        while offset + WINDOW_SAMPLES <= len(buffer):
            window = buffer[offset:offset + WINDOW_SAMPLES]
            offset += WINDOW_SAMPLES
            self._position_s += WINDOW_SAMPLES / SAMPLE_RATE
            events.extend(self._consume_window(window))

        self._leftover = buffer[offset:]
        return events

    def _consume_window(self, window: np.ndarray) -> List[tuple]:
        events: List[tuple] = []
        probability = float(self.vad.process_chunk(window))

        if not self._in_speech:
            # Rolling pre-roll so an onset is not clipped.
            self._pre_pad.append(window)
            if len(self._pre_pad) > self._pad_windows:
                self._pre_pad.pop(0)

            if probability >= self.config.speech_threshold:
                self._in_speech = True
                self._speech = list(self._pre_pad)
                self._pre_pad = []
                self._trailing_silence = []
                self._speech_ms = 100
                self._silence_ms = 0
                self._segment_start_s = self._position_s - (
                    len(self._speech) * 100 / 1000
                )
                events.append((Event.SPEECH_START, None))
            return events

        # In speech.
        if probability >= self.config.silence_threshold:
            # Still talking; any buffered near-silence was an intra-word gap.
            if self._trailing_silence:
                self._speech.extend(self._trailing_silence)
                self._speech_ms += 100 * len(self._trailing_silence)
                self._trailing_silence = []
            self._speech.append(window)
            self._speech_ms += 100
            self._silence_ms = 0
        else:
            # Hold the silence aside: if speech resumes it belongs to this
            # utterance, and if it does not only the pad is kept.
            self._trailing_silence.append(window)
            self._silence_ms += 100

            if self._silence_ms >= self.config.min_silence_ms:
                segment = self._close_segment()
                if segment is not None:
                    events.append((Event.SPEECH_END, segment))
                return events

        if self.active_duration_s >= self.config.max_segment_s:
            segment = self._close_segment(forced=True)
            if segment is not None:
                events.append((Event.SPEECH_END, segment))

        return events

    def _close_segment(self, forced: bool = False) -> Optional[Segment]:
        keep = self._trailing_silence[:self._pad_windows]
        audio = self._speech + keep

        speech_ms = self._speech_ms
        start_s = self._segment_start_s
        end_s = start_s + sum(len(w) for w in audio) / SAMPLE_RATE

        self._in_speech = False
        self._speech = []
        self._trailing_silence = []
        self._silence_ms = 0
        self._speech_ms = 0
        # A forced cut lands mid-utterance, so treat its tail as the pre-roll
        # of whatever comes next rather than dropping it.
        self._pre_pad = list(keep) if forced else []

        if speech_ms < self.config.min_speech_ms or not audio:
            return None
        return Segment(audio=np.concatenate(audio), start_s=start_s, end_s=end_s)

    def flush(self) -> Optional[Segment]:
        """End the utterance in progress, e.g. when capture stops."""
        if not self._in_speech:
            return None
        return self._close_segment()

    def reset(self) -> None:
        self.vad.reset()
        self._leftover = np.zeros(0, dtype=np.float32)
        self._speech = []
        self._pre_pad = []
        self._trailing_silence = []
        self._in_speech = False
        self._silence_ms = 0
        self._speech_ms = 0
        self._position_s = 0.0
        self._segment_start_s = 0.0
