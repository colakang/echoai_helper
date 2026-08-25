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
        """Everything buffered since the last cut, for live partials."""
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
        """
        Accumulate every window. VAD decides *where to cut*, never *what to
        keep* -- an earlier design dropped anything it judged non-speech and
        lost 22-28% of three real recordings, including whole audible
        sentences that never reached the transcript at all. For meeting notes,
        losing audio is far worse than a ragged boundary.
        """
        events: List[tuple] = []
        probability = float(self.vad.process_chunk(window))
        is_speech = probability >= (
            self.config.speech_threshold if not self._in_speech
            else self.config.silence_threshold
        )

        self._speech.append(window)
        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                events.append((Event.SPEECH_START, None))
            # Only a short gap counts as part of the utterance. Folding in the
            # whole trailing run would credit minutes of room tone as speech:
            # a single blip after a long quiet stretch made speech_ms jump by
            # the length of that stretch, and the hard cap then emitted a 25s
            # "utterance" that was almost entirely silence and transcribed to
            # ".". Anything longer than min_silence_ms would have ended the
            # segment anyway, so it can never legitimately be speech.
            gap_windows = min(len(self._trailing_silence),
                              self.config.min_silence_ms // 100)
            self._speech_ms += 100 + 100 * gap_windows
            self._trailing_silence = []
            self._silence_ms = 0
        else:
            # Track the run of quiet, but the audio itself is already kept.
            self._trailing_silence.append(window)
            self._silence_ms += 100


            if self._in_speech and self._silence_ms >= self.config.min_silence_ms:
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
        """
        Cut in the middle of the trailing silence, so the pause is split
        between this segment and the next and no sample belongs to neither.
        """
        if forced:
            keep, carry = self._speech, []
        else:
            split = max(len(self._trailing_silence) // 2, self._pad_windows)
            split = min(split, len(self._trailing_silence))
            boundary = len(self._speech) - len(self._trailing_silence) + split
            keep = self._speech[:boundary]
            carry = self._speech[boundary:]

        speech_ms = self._speech_ms
        start_s = self._segment_start_s
        kept_s = sum(len(w) for w in keep) / SAMPLE_RATE

        self._in_speech = False
        self._trailing_silence = []
        self._silence_ms = 0
        self._speech_ms = 0
        self._segment_start_s = start_s + kept_s
        self._speech = list(carry)

        if not keep:
            return None

        if speech_ms == 0:
            # Nothing was ever said here: room tone only. Discard all but the
            # pre-roll. Carrying it forward would let the buffer grow for as
            # long as the room stays quiet -- the hard cut would restore the
            # whole thing every time -- and would drag that silence into
            # whatever is eventually spoken.
            self._speech = list(keep[-self._pad_windows:]) + list(carry)
            self._segment_start_s = (start_s + kept_s
                                     - min(self._pad_windows, len(keep)) * 0.1)
            return None

        if speech_ms < self.config.min_speech_ms:
            # Too short to stand alone -- but somebody spoke, so hand it to
            # the next segment rather than discarding it. Dropping short
            # utterances is how an earlier design lost speech.
            self._speech = list(keep) + list(carry)
            self._segment_start_s = start_s
            return None

        return Segment(audio=np.concatenate(keep), start_s=start_s,
                       end_s=start_s + kept_s)

    def flush(self) -> Optional[Segment]:
        """
        Emit whatever is buffered, e.g. when capture stops.

        Ignores min_speech_ms: at end of stream there is no next segment to
        carry a short tail into, so enforcing it would silently drop the
        last words spoken.
        """
        if not self._speech:
            return None
        audio = np.concatenate(self._speech)
        start_s = self._segment_start_s
        duration = len(audio) / SAMPLE_RATE

        had_speech = self._in_speech or self._speech_ms > 0
        self._speech = []
        self._trailing_silence = []
        self._in_speech = False
        self._silence_ms = 0
        self._speech_ms = 0
        self._segment_start_s = start_s + duration

        if not had_speech:
            return None
        return Segment(audio=audio, start_s=start_s, end_s=start_s + duration)

    def reset(self) -> None:
        self.vad.reset()
        self._leftover = np.zeros(0, dtype=np.float32)
        self._speech = []
        self._trailing_silence = []
        self._in_speech = False
        self._silence_ms = 0
        self._speech_ms = 0
        self._position_s = 0.0
        self._segment_start_s = 0.0
