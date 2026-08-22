"""
src/audio

Platform dispatch for audio capture.

    from src.audio import get_audio_backend

    backend = get_audio_backend()
    mic = backend.create_mic_recorder()          # may be None
    speaker = backend.create_speaker_recorder()  # may be None

Either recorder can come back None — a machine with no microphone, or a macOS
box with no virtual audio device installed.  Callers must handle that instead
of assuming both tracks exist.
"""

import sys

from .backend import AudioBackend, AudioSource, Recorder, RECORD_TIMEOUT, ENERGY_THRESHOLD

__all__ = [
    "get_audio_backend",
    "AudioBackend",
    "AudioSource",
    "Recorder",
    "RECORD_TIMEOUT",
    "ENERGY_THRESHOLD",
]


def get_audio_backend(**kwargs) -> AudioBackend:
    """Return the capture backend for the current platform."""
    if sys.platform == "win32":
        from .windows import WindowsAudioBackend
        return WindowsAudioBackend()

    if sys.platform == "darwin":
        from .macos import MacOSAudioBackend
        return MacOSAudioBackend(**kwargs)

    raise NotImplementedError(
        f"No audio backend for platform {sys.platform!r}. "
        "Supported: win32 (WASAPI loopback), darwin (sounddevice)."
    )
