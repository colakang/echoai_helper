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
import time

# Emit a chunk this often (seconds).  Matches the historical Windows
# `phrase_time_limit` so downstream phrase timing is unchanged.
RECORD_TIMEOUT = 0.6

# The recorder forwards every chunk. It used to drop anything below an RMS
# threshold, which made sense when the transcriber accumulated audio and
# silence was pure waste -- but segmentation now happens on *pauses*, and a
# gate here destroys exactly the information the VAD needs to find them.
#
# Measured on a 114s call: with a gate at twice the observed noise floor, 104
# chunks of silence never arrived and the segmenter produced 5 segments with a
# median of 7.7s, two of them hitting the hard cap. Without it, 30 segments
# with a median of 3.0s -- matching what the same audio gives offline.
#
# The VAD is the real gate and costs almost nothing: RTF 0.0027 on an M4.
#
# The Windows recorder cannot forward *everything* the same way -- it is built
# on SpeechRecognition's listen_in_background, which only calls back once its
# own detector thinks someone is talking. Setting the threshold to 1 makes it
# fire on essentially anything, which is as close to pass-through as that API
# allows. Untested here; flagged rather than silently left at the old value,
# because the same reasoning applies.
WINDOWS_ENERGY_THRESHOLD = 1

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

    # The track a pause applies to. Only the local microphone can be paused:
    # the far end is the meeting, and silencing it would just lose the notes.
    PAUSABLE = "You"

    def __init__(self, source: AudioSource, source_name: str):
        if source is None:
            raise ValueError("audio source can't be None")
        self.source = source
        self.source_name = source_name
        # When the device last handed us anything at all. Monotonic, because
        # this is a duration and the wall clock can move under it.
        self.last_callback = time.monotonic()

    def note_callback(self) -> None:
        """
        Record that the device is still delivering.

        Called at the very top of the audio callback, before any decision about
        what to do with the audio -- which is the whole point. Measured on a
        real Bluetooth disconnect: the callback stops being invoked entirely
        (0 in 2s), while a *muted* microphone keeps being invoked and simply
        delivers zeroes. Counting invocations therefore separates "the device
        is gone" from "nobody is talking" and from "the user paused us", with
        no threshold and nothing to tune.

        Counting emitted chunks instead would collapse all three together, and
        pausing your own microphone would look exactly like the device dying.
        """
        self.last_callback = time.monotonic()

    def silent_for(self) -> float:
        """Seconds since the device last called us."""
        return time.monotonic() - self.last_callback

    def should_emit(self) -> bool:
        """
        Whether captured audio should be forwarded right now.

        Dropped at the emit point rather than by stopping the stream. Closing
        and reopening a device is the fragile operation here -- it is how the
        microphone gets lost in the first place -- so a pause that has to
        reopen on resume would be trading a certain risk for a saved handle
        nobody needs. The stream stays up; the chunks go nowhere.
        """
        if self.source_name != self.PAUSABLE:
            return True
        from src.config import AudioConfig
        return not AudioConfig.get_mic_paused()

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
