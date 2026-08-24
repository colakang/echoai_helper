"""Tests for the capture layer's noise gate."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("sounddevice")

from src.audio.backend import (  # noqa: E402
    AudioSource, ENERGY_THRESHOLD, MAX_ENERGY_THRESHOLD, NOISE_FLOOR_MARGIN,
)
from src.audio import macos  # noqa: E402


def recorder():
    source = AudioSource(sample_rate=16000, sample_width=2, channels=1,
                         device_name="stub")
    return macos.SoundDeviceRecorder(source, "You", device_index=0)


def with_noise(monkeypatch, rms):
    """Make the calibration recording read back at a chosen RMS."""
    def fake_rec(frames, **kwargs):
        return np.full((frames, 1), int(rms), dtype=np.int16)
    monkeypatch.setattr(macos.sd, "rec", fake_rec)
    monkeypatch.setattr(macos.sd, "wait", lambda: None)


def test_gate_is_placed_above_the_measured_floor(monkeypatch):
    """
    A Bluetooth headset in HFP mode measured RMS 331 in a silent room against
    a fixed gate of 100, so every silent chunk reached the transcriber.
    """
    with_noise(monkeypatch, 331)
    r = recorder()
    assert r.calibrate() == pytest.approx(331 * NOISE_FLOOR_MARGIN, rel=0.01)


def test_a_quiet_device_keeps_the_default_gate(monkeypatch):
    """A line-level loopback measures ~0; the gate must not collapse to it."""
    with_noise(monkeypatch, 0)
    assert recorder().calibrate() == ENERGY_THRESHOLD


def test_gate_is_capped(monkeypatch):
    """Someone talking during calibration must not raise the gate so far that
    normal speech is then filtered out."""
    with_noise(monkeypatch, 20000)
    assert recorder().calibrate() == MAX_ENERGY_THRESHOLD


def test_calibration_failure_is_not_fatal(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("device busy")
    monkeypatch.setattr(macos.sd, "rec", boom)

    r = recorder()
    assert r.calibrate() == ENERGY_THRESHOLD
    assert r.energy_threshold == ENERGY_THRESHOLD


def test_chunks_below_the_gate_are_not_queued(monkeypatch):
    import queue
    with_noise(monkeypatch, 300)
    r = recorder()
    r.calibrate()                      # gate -> 600

    r._queue = queue.Queue()
    quiet = (np.full(1600, 100, dtype=np.int16)).tobytes()
    loud = (np.full(1600, 5000, dtype=np.int16)).tobytes()

    r._emit(quiet)
    assert r._queue.empty(), "room tone reached the transcriber"

    r._emit(loud)
    assert not r._queue.empty(), "speech was filtered out"
