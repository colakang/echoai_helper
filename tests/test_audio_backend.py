"""Tests for the capture layer."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("sounddevice")

from src.audio.backend import AudioSource  # noqa: E402
from src.audio import macos  # noqa: E402


def recorder():
    source = AudioSource(sample_rate=16000, sample_width=2, channels=1,
                         device_name="stub")
    return macos.SoundDeviceRecorder(source, "You", device_index=0)


def pcm(value, samples=1600):
    return np.full(samples, value, dtype=np.int16).tobytes()


# --------------------------------------------------------------------------
# Pass-through
# --------------------------------------------------------------------------
#
# The recorder used to drop chunks below an RMS threshold. That made sense
# when the transcriber accumulated audio, and broke segmentation once it moved
# to cutting on pauses: on a 114s call the gate swallowed 104 chunks of
# silence, and the segmenter -- never seeing a pause -- produced 5 segments
# with a median of 7.7s where the same audio offline gives 30 at 3.0s.

def test_silence_is_forwarded():
    """The VAD downstream needs to see the pauses it segments on."""
    import queue
    r = recorder()
    r._queue = queue.Queue()

    r._emit(pcm(0))
    assert not r._queue.empty(), "silence was dropped; pauses will be invisible"


def test_quiet_audio_is_forwarded():
    import queue
    r = recorder()
    r._queue = queue.Queue()

    r._emit(pcm(50))
    assert not r._queue.empty()


def test_loud_audio_is_forwarded():
    import queue
    r = recorder()
    r._queue = queue.Queue()

    r._emit(pcm(8000))
    assert not r._queue.empty()


def test_empty_chunks_are_skipped():
    import queue
    r = recorder()
    r._queue = queue.Queue()

    r._emit(b"")
    assert r._queue.empty()


def test_chunks_carry_source_and_time():
    import queue
    from datetime import datetime
    r = recorder()
    r._queue = queue.Queue()

    r._emit(pcm(1000))
    name, data, stamp = r._queue.get()
    assert name == "You"
    assert len(data) == 3200
    assert isinstance(stamp, datetime)


def test_chunk_size_matches_the_source_format():
    """0.6s of 16kHz mono int16."""
    r = recorder()
    assert r._chunk_bytes == int(16000 * 0.6) * 1 * 2
