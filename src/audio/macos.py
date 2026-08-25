"""
src/audio/macos.py

macOS capture backend, built on sounddevice (PortAudio).

Why this exists: the Windows path uses PyAudioWPatch, a Windows-only fork of
PyAudio that adds WASAPI loopback.  It has no macOS distribution at all — `pip
install PyAudioWPatch` fails outright on darwin — so macOS needs its own route.

macOS has no OS-level "record what the speakers are playing" for a plain input
stream.  The standard workaround is a virtual audio device: audio is routed
out through it, and it shows up as an *input* device we can record from.
BlackHole is the usual choice.  See docs/macos-audio-setup.md.
"""

import threading
import queue as _queue
from datetime import datetime, timezone
from typing import Optional, List

import numpy as np
import sounddevice as sd

from .backend import AudioBackend, AudioSource, Recorder, RECORD_TIMEOUT

# Input devices whose name matches one of these are treated as a loopback of
# system output rather than a real microphone.
LOOPBACK_HINTS = ("blackhole", "loopback", "soundflower", "vb-cable",
                  "existential audio", "aggregate", "multi-output")

SAMPLE_WIDTH = 2  # we always request int16


def _looks_like_loopback(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in LOOPBACK_HINTS)


def list_input_devices() -> List[dict]:
    """Every device that can be recorded from, with its portaudio index."""
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            devices.append({
                "index": index,
                "name": info["name"],
                "channels": info["max_input_channels"],
                "sample_rate": int(info["default_samplerate"]),
                "is_loopback": _looks_like_loopback(info["name"]),
            })
    return devices


def preferred_mic() -> Optional[dict]:
    """
    The device the microphone track should be on *right now*.

    Same rule create_mic_recorder() uses at startup -- the system default input
    if it is a real microphone, otherwise the first one -- but evaluated
    against the current device list rather than the one that existed when the
    app launched.
    """
    candidates = [d for d in list_input_devices() if not d["is_loopback"]]
    if not candidates:
        return None
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    return next((c for c in candidates if c["index"] == default_in),
                candidates[0])


class SoundDeviceRecorder(Recorder):
    """
    Streams from one input device, emitting fixed ~0.6s int16 PCM chunks.

    Every chunk is forwarded, silence included. Deciding what is speech is the
    VAD's job downstream, and it needs to see the silence to find the pauses
    it segments on -- see the note in backend.py.
    """

    def __init__(self, source: AudioSource, source_name: str, device_index: int):
        super().__init__(source, source_name)
        self.device_index = device_index
        # The name is the identity; the index is only a handle to it.
        # PortAudio renumbers devices whenever the list changes, so an index
        # captured at launch can quietly come to mean a different device --
        # which is what a Bluetooth headset handing its microphone to a phone
        # does to us. Keeping the name is what makes that detectable.
        self.device_name = source.device_name
        self._stream: Optional[sd.InputStream] = None
        self._queue: Optional[_queue.Queue] = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        # Bytes that make up one emitted chunk.
        self._chunk_bytes = int(
            source.SAMPLE_RATE * RECORD_TIMEOUT
        ) * source.channels * SAMPLE_WIDTH

    def record_into_queue(self, audio_queue: "_queue.Queue") -> None:
        self._queue = audio_queue

        def callback(indata, frames, time_info, status):
            if status:
                # Overflows are normal under load; log once per occurrence
                # rather than raising, so a hiccup never kills capture.
                print(f"[WARN] {self.source_name} stream status: {status}")
            with self._lock:
                self._buffer.extend(indata.tobytes())
                while len(self._buffer) >= self._chunk_bytes:
                    chunk = bytes(self._buffer[:self._chunk_bytes])
                    del self._buffer[:self._chunk_bytes]
                    self._emit(chunk)

        self._stream = sd.InputStream(
            device=self.device_index,
            channels=self.source.channels,
            samplerate=self.source.SAMPLE_RATE,
            dtype="int16",
            callback=callback,
        )
        self._stream.start()
        print(f"[INFO] {self.source_name}: capturing from "
              f"{self.source.device_name!r} "
              f"({self.source.SAMPLE_RATE}Hz, {self.source.channels}ch)")

    def _emit(self, chunk: bytes) -> None:
        if not chunk or not self.should_emit():
            return
        # utcnow() is deprecated in 3.12; keep the naive-UTC value the rest of
        # the pipeline already assumes.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        self._queue.put((self.source_name, chunk, now))

    def bound_device_name(self) -> Optional[str]:
        """What this recorder's index resolves to *now*, or None if it is gone."""
        try:
            return sd.query_devices(self.device_index)["name"]
        except Exception:
            return None

    def is_stale(self, expected: Optional[dict]) -> bool:
        """
        Whether this recorder is no longer on the device it should be.

        Two independent ways that happens, and both are checked because they
        fail differently:

        - the index now resolves to some other device, or to nothing -- the
          list was renumbered underneath us and we are reading the wrong
          hardware while believing we are fine;
        - the device that *should* be recorded from has changed -- someone
          plugged in a headset, or the one we were using went away.

        This is a question about identity, not about liveness. It is answered
        by comparing names, so there is nothing to tune and nothing to
        false-positive on: a muted microphone and a silent room look exactly
        alike here, and neither one looks like this.
        """
        current = self.bound_device_name()
        if current is None or current != self.device_name:
            return True
        if expected is not None and expected["name"] != self.device_name:
            return True
        return False

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                # A device that has already gone raises on close. Nothing left
                # to release, and refusing to move on would strand the track.
                print(f"[WARN] {self.source_name}: closing the stream failed: {exc}")
            self._stream = None
            print(f"[INFO] {self.source_name}: capture stopped")


class MacOSAudioBackend(AudioBackend):
    name = "macOS/sounddevice"

    def __init__(self, mic_device: Optional[int] = None,
                 speaker_device: Optional[int] = None):
        # Explicit overrides win; otherwise both are auto-detected.
        self._mic_override = mic_device
        self._speaker_override = speaker_device

    def _build(self, index: int, source_name: str) -> SoundDeviceRecorder:
        info = sd.query_devices(index)
        source = AudioSource(
            sample_rate=int(info["default_samplerate"]),
            sample_width=SAMPLE_WIDTH,
            channels=int(info["max_input_channels"]),
            device_name=info["name"],
        )
        return SoundDeviceRecorder(source, source_name, index)

    def create_mic_recorder(self) -> Optional[SoundDeviceRecorder]:
        if self._mic_override is not None:
            return self._build(self._mic_override, "You")

        candidates = [d for d in list_input_devices() if not d["is_loopback"]]
        if not candidates:
            print("[WARN] No microphone found. This machine has no audio input "
                  "device (a Mac mini has no built-in mic).")
            print("[WARN] Your own speech will not be transcribed. Attach a USB "
                  "mic or headset to enable the 'You' track.")
            return None

        default_in = sd.default.device[0]
        chosen = next((c for c in candidates if c["index"] == default_in),
                      candidates[0])
        return self._build(chosen["index"], "You")

    def create_speaker_recorder(self) -> Optional[SoundDeviceRecorder]:
        if self._speaker_override is not None:
            return self._build(self._speaker_override, "Speaker")

        loopbacks = [d for d in list_input_devices() if d["is_loopback"]]
        if not loopbacks:
            print("[ERROR] No loopback audio device found.")
            print("[ERROR] macOS cannot record system output directly; install "
                  "a virtual audio device:")
            print("[ERROR]     brew install --cask blackhole-2ch")
            print("[ERROR] then follow docs/macos-audio-setup.md to route "
                  "meeting audio through it.")
            return None

        return self._build(loopbacks[0]["index"], "Speaker")

    def follow_default_mic(self, recorder: Optional[SoundDeviceRecorder],
                           audio_queue: "_queue.Queue"
                           ) -> Optional[SoundDeviceRecorder]:
        """
        Re-point the microphone track at the device that should be recorded now.

        Returns the recorder to keep using -- the same object when nothing has
        moved, a new one when it has, and None when there is no microphone at
        all any more.

        Deliberately not a liveness check. Deciding a stream has "gone quiet"
        cannot distinguish a dead device from a muted one or a silent room: a
        hardware mute delivers exactly the digital zeroes a lost device does.
        This asks a question that does have an answer -- are we bound to the
        right device -- and acts only on that.
        """
        if recorder is None or recorder.source_name != Recorder.PAUSABLE:
            return recorder

        expected = preferred_mic()
        if not recorder.is_stale(expected):
            return recorder

        was = recorder.device_name
        recorder.stop()

        if expected is None:
            print(f"[WARN] You: {was!r} is gone and there is no other "
                  f"microphone; your track has stopped.")
            return None

        replacement = self._build(expected["index"], Recorder.PAUSABLE)
        replacement.record_into_queue(audio_queue)
        print(f"[INFO] You: microphone moved {was!r} -> {expected['name']!r}")
        return replacement
