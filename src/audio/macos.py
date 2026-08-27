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


def a_microphone_exists() -> bool:
    """
    Whether the system currently has a real microphone, asked of CoreAudio.

    Deliberately not asked of PortAudio. PortAudio enumerates devices once at
    Pa_Initialize and never again, so inside a running process its answer is
    frozen at whatever was true at startup -- it cannot see a headset leave and
    it cannot see one come back. CoreAudio answers live, and answering this
    cheaply is what keeps the far-end recording from being torn down every
    thirty seconds while a headset sits switched off in a drawer.
    """
    try:
        from . import coreaudio
        return any(getattr(d, "input_channels", 0) > 0
                   and not _looks_like_loopback(d.name)
                   for d in coreaudio.list_devices())
    except Exception:
        # Unknowable rather than false: claiming there is no microphone would
        # stop recovery from ever being attempted.
        return True


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
            self.note_callback()
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

    def restart_capture(self, recorders, queues):
        """
        Rebuild every capture stream after a device came back.

        Measured, because none of it is guessable from the API:

        - Reopening the dead stream on its remembered index fails with
          PaErrorCode -9986 *even once the device is genuinely back*. Whatever
          PortAudio cached for that device is poisoned by the disconnect.
        - Only Pa_Terminate followed by Pa_Initialize recovers it -- and that
          invalidates every open stream in the process, so the far-end track
          has to be torn down and rebuilt too, whether or not anything was
          wrong with it.
        - The whole cycle costs 378ms (median of 3, tight spread). That is the
          gap punched in the meeting recording, and it buys back a microphone
          track that would otherwise be dead for the rest of the call.

        Returns the replacement recorders, keyed as they came in. A track whose
        device did not come back maps to None rather than raising: losing your
        own microphone must not also stop the meeting being recorded.
        """
        for recorder in recorders.values():
            if recorder is not None:
                recorder.stop()

        sd._terminate()
        sd._initialize()

        rebuilt = {}
        for name in recorders:
            # Note what is *not* here: a check that skips a track whose
            # recorder is currently None. That is the state this exists to
            # repair -- a microphone that died, or that was never there when
            # the app started -- so treating it as nothing to do meant the
            # track could never come back. It also meant retrying forever,
            # tearing down the far-end stream on every attempt to achieve
            # nothing, because the retry could not succeed by construction.
            wanted = (preferred_mic() if name == Recorder.PAUSABLE
                      else next((d for d in list_input_devices()
                                 if d["is_loopback"]), None))
            if wanted is None:
                print(f"[WARN] {name}: no device to capture from after the restart")
                rebuilt[name] = None
                continue
            replacement = self._build(wanted["index"], name)
            replacement.record_into_queue(queues[name])
            rebuilt[name] = replacement
        return rebuilt
