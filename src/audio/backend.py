"""
src/audio/backend.py

Platform-neutral audio capture interface.

The rest of the app only ever sees an ``AudioSource`` (format metadata) and a
``Recorder`` (pushes raw PCM chunks into a queue).  Everything platform
specific — WASAPI loopback on Windows, a virtual audio device on macOS — lives
behind these two types.

Chunks pushed onto the queue are always ``(source_name, pcm_bytes, utc_time)``
where ``pcm_bytes`` is signed 16-bit little-endian PCM interleaved across
``AudioSource.channels`` at ``AudioSource.SAMPLE_RATE``.  Down-mixing and
resampling to the rate the ASR model wants happens later, in AudioTranscriber.
"""

from abc import ABC, abstractmethod
from typing import Optional
import queue

# Emit a chunk this often (seconds).  Matches the historical Windows
# `phrase_time_limit` so downstream phrase timing is unchanged.
RECORD_TIMEOUT = 0.6

# RMS gate on int16 samples.  Chunks quieter than this never reach the ASR
# model — silence is the single biggest source of wasted inference and of
# hallucinated transcripts.
ENERGY_THRESHOLD = 100


class AudioSource:
    """
    Format description for one capture source.

    Attribute names are SHOUTY to stay drop-in compatible with
    ``custom_speech_recognition.Microphone``, which the Windows path still uses
    and which AudioTranscriber reads directly.
    """

    def __init__(self, sample_rate: int, sample_width: int, channels: int,
                 device_name: str = ""):
        self.SAMPLE_RATE = sample_rate
        self.SAMPLE_WIDTH = sample_width
        self.channels = channels
        self.device_name = device_name

    def __repr__(self) -> str:
        return (f"AudioSource({self.device_name!r}, {self.SAMPLE_RATE}Hz, "
                f"{self.channels}ch, {self.SAMPLE_WIDTH * 8}-bit)")


class Recorder(ABC):
    """Pushes PCM chunks from one source into a queue until stopped."""

    def __init__(self, source: AudioSource, source_name: str):
        if source is None:
            raise ValueError("audio source can't be None")
        self.source = source
        self.source_name = source_name

    @abstractmethod
    def record_into_queue(self, audio_queue: "queue.Queue") -> None:
        """Start capturing in the background."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing and release the device."""


class AudioBackend(ABC):
    """Creates recorders for the two things this app listens to."""

    name = "unknown"

    @abstractmethod
    def create_mic_recorder(self) -> Optional[Recorder]:
        """
        Recorder for the local microphone ("You").

        Returns None when the machine has no input device — a Mac mini has no
        built-in microphone, and that must degrade to "transcribe the other
        side only" rather than crash the app.
        """

    @abstractmethod
    def create_speaker_recorder(self) -> Optional[Recorder]:
        """
        Recorder for system audio output ("Speaker") — what the far end of the
        meeting is saying.

        Returns None when no loopback route is available.
        """

    def describe(self) -> str:
        return f"<{self.name} audio backend>"
