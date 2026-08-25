"""
src/audio/windows.py

Windows capture backend — WASAPI loopback via PyAudioWPatch.

This is the original behaviour of src/AudioRecorder.py, moved behind the
AudioBackend interface unchanged.  All imports are deferred into methods so
that merely importing this module on macOS/Linux does not explode.
"""

import queue as _queue
from datetime import datetime, timezone
from typing import Optional

from .backend import (AudioBackend, AudioSource, Recorder, RECORD_TIMEOUT,
                      WINDOWS_ENERGY_THRESHOLD)

DYNAMIC_ENERGY_THRESHOLD = False


class SpeechRecognitionRecorder(Recorder):
    """Wraps custom_speech_recognition's background listener."""

    def __init__(self, sr_source, source_name: str):
        import src.custom_speech_recognition as sr

        self._sr_source = sr_source
        self.recorder = sr.Recognizer()
        # Near zero on purpose: segmentation happens on pauses downstream,
        # and a recogniser that only calls back during speech never delivers
        # them. See the note in backend.py.
        self.recorder.energy_threshold = WINDOWS_ENERGY_THRESHOLD
        self.recorder.dynamic_energy_threshold = DYNAMIC_ENERGY_THRESHOLD
        self._stopper = None

        source = AudioSource(
            sample_rate=sr_source.SAMPLE_RATE,
            sample_width=sr_source.SAMPLE_WIDTH,
            channels=sr_source.channels,
            device_name=source_name,
        )
        super().__init__(source, source_name)

    def adjust_for_noise(self, device_name: str, msg: str) -> None:
        print(f"[INFO] Adjusting for ambient noise from {device_name}. " + msg)
        with self._sr_source:
            self.recorder.adjust_for_ambient_noise(self._sr_source)
        print(f"[INFO] Completed ambient noise adjustment for {device_name}.")

    def record_into_queue(self, audio_queue: "_queue.Queue") -> None:
        def record_callback(_, audio) -> None:
            data = audio.get_raw_data()
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            audio_queue.put((self.source_name, data, now))

        self._stopper = self.recorder.listen_in_background(
            self._sr_source, record_callback, phrase_time_limit=RECORD_TIMEOUT
        )

    def stop(self) -> None:
        if self._stopper is not None:
            self._stopper(wait_for_stop=False)
            self._stopper = None


class WindowsAudioBackend(AudioBackend):
    name = "Windows/WASAPI"

    def create_mic_recorder(self) -> Optional[SpeechRecognitionRecorder]:
        import src.custom_speech_recognition as sr

        recorder = SpeechRecognitionRecorder(
            sr.Microphone(sample_rate=16000), "You"
        )
        recorder.adjust_for_noise(
            "Default Mic", "Please make some noise from the Default Mic..."
        )
        return recorder

    def create_speaker_recorder(self) -> Optional[SpeechRecognitionRecorder]:
        import src.custom_speech_recognition as sr
        import pyaudiowpatch as pyaudio

        with pyaudio.PyAudio() as p:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = p.get_device_info_by_index(
                wasapi_info["defaultOutputDevice"]
            )

            if not default_speakers["isLoopbackDevice"]:
                for loopback in p.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        default_speakers = loopback
                        break
                else:
                    print("[ERROR] No loopback device found.")
                    return None

        sr_source = sr.Microphone(
            speaker=True,
            device_index=default_speakers["index"],
            sample_rate=int(default_speakers["defaultSampleRate"]),
            chunk_size=pyaudio.get_sample_size(pyaudio.paInt16),
            channels=default_speakers["maxInputChannels"],
        )
        recorder = SpeechRecognitionRecorder(sr_source, "Speaker")
        recorder.adjust_for_noise(
            "Default Speaker",
            "Please make or play some noise from the Default Speaker...",
        )
        return recorder
