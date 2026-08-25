"""
Pausing the microphone track, and keeping it on the right device.

Two separate problems that both present as "my microphone is not being
recorded", and are deliberately solved by different mechanisms:

- pausing is a decision the user made, and is exact;
- a device moving underneath us is a fact about the system, and is checked by
  comparing identities rather than by watching for silence.

The thing neither one does is infer death from quiet. A hardware mute delivers
the same digital zeroes a lost device does, and a meeting spent listening
delivers nearly the same -- so no threshold can tell them apart, and none is
used.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio.backend import AudioSource, Recorder
from src.config import AudioConfig


class FakeRecorder(Recorder):
    """A Recorder that records what it was asked to emit."""

    def __init__(self, source_name="You"):
        super().__init__(AudioSource(16000, 2, 1, "Fake Mic"), source_name)
        self.emitted = []

    def record_into_queue(self, audio_queue):
        pass

    def stop(self):
        pass

    def feed(self, chunk):
        if self.should_emit():
            self.emitted.append(chunk)


@pytest.fixture(autouse=True)
def unpaused():
    """Never leak pause state between tests -- it is process-wide."""
    AudioConfig.set_mic_paused(False)
    yield
    AudioConfig.set_mic_paused(False)


# --------------------------------------------------------------------------
# Pausing
# --------------------------------------------------------------------------

def test_microphone_audio_is_dropped_while_paused():
    mic = FakeRecorder("You")
    mic.feed(b"before")
    AudioConfig.set_mic_paused(True)
    mic.feed(b"during")
    AudioConfig.set_mic_paused(False)
    mic.feed(b"after")

    assert mic.emitted == [b"before", b"after"]


def test_pausing_the_microphone_does_not_silence_the_meeting():
    """
    The far end is the notes. Pausing is about not recording yourself, and a
    control that also dropped the other side would be discarding the thing the
    user actually came for.
    """
    speaker = FakeRecorder("Speaker")
    AudioConfig.set_mic_paused(True)
    speaker.feed(b"the other side talking")

    assert speaker.emitted == [b"the other side talking"]


def test_resuming_needs_no_reopen():
    """
    Audio is dropped at the emit point, not by closing the stream.

    Closing and reopening a device is the fragile operation in this whole
    area -- it is how the microphone gets lost -- so a pause that had to
    reopen on resume would trade a certain risk for a saved handle nobody
    needs.
    """
    mic = FakeRecorder("You")
    AudioConfig.set_mic_paused(True)
    assert not mic.should_emit()
    AudioConfig.set_mic_paused(False)
    assert mic.should_emit(), "resume must not depend on reopening anything"


# --------------------------------------------------------------------------
# Device identity
# --------------------------------------------------------------------------

sd = pytest.importorskip("sounddevice", reason="macOS capture backend")
from src.audio import macos                                    # noqa: E402


def make_recorder(name, index):
    source = AudioSource(16000, 2, 1, name)
    return macos.SoundDeviceRecorder(source, "You", index)


def test_a_stable_binding_is_not_disturbed(monkeypatch):
    mic = make_recorder("Headset", 3)
    monkeypatch.setattr(mic, "bound_device_name", lambda: "Headset")
    assert not mic.is_stale({"index": 3, "name": "Headset"})


def test_renumbering_underneath_us_is_caught(monkeypatch):
    """
    The failure this exists for.

    PortAudio renumbers devices whenever the list changes, so an index taken
    at launch can come to mean different hardware. The stream stays open and
    keeps calling back -- it is simply no longer connected to the microphone
    anyone is talking into, which is indistinguishable from working.
    """
    mic = make_recorder("Headset", 3)
    monkeypatch.setattr(mic, "bound_device_name", lambda: "Some Other Device")
    assert mic.is_stale({"index": 3, "name": "Headset"})


def test_a_device_that_vanished_is_caught(monkeypatch):
    mic = make_recorder("Headset", 3)
    monkeypatch.setattr(mic, "bound_device_name", lambda: None)
    assert mic.is_stale(None)


def test_following_the_user_to_a_newly_attached_microphone(monkeypatch):
    """Plugging in a headset mid-meeting should move the track onto it."""
    mic = make_recorder("Built-in", 1)
    monkeypatch.setattr(mic, "bound_device_name", lambda: "Built-in")
    assert mic.is_stale({"index": 4, "name": "Headset"})


def test_silence_is_not_evidence(monkeypatch):
    """
    A muted microphone is not a stale one, and must not be treated as one.

    This is the whole reason the check is about identity: a hardware mute
    delivers exactly the zeroes a dead device does, so anything watching the
    signal would have to guess. Nothing here looks at the signal at all.
    """
    mic = make_recorder("Headset", 3)
    monkeypatch.setattr(mic, "bound_device_name", lambda: "Headset")
    AudioConfig.set_mic_paused(True)
    assert not mic.is_stale({"index": 3, "name": "Headset"})


def test_the_far_end_is_never_re_pointed():
    """
    follow_default_mic only ever touches the microphone track.

    The loopback device is chosen deliberately during setup and is not the
    system default input; following the default would drag the meeting audio
    onto a microphone.
    """
    backend = macos.MacOSAudioBackend()
    speaker = macos.SoundDeviceRecorder(
        AudioSource(16000, 2, 2, "BlackHole 2ch"), "Speaker", 2)
    assert backend.follow_default_mic(speaker, None) is speaker


def test_no_microphone_at_all_is_survivable(monkeypatch):
    """Losing the last input device stops the track; it does not crash."""
    backend = macos.MacOSAudioBackend()
    mic = make_recorder("Headset", 3)
    monkeypatch.setattr(mic, "bound_device_name", lambda: None)
    monkeypatch.setattr(macos, "preferred_mic", lambda: None)
    monkeypatch.setattr(mic, "stop", lambda: None)

    assert backend.follow_default_mic(mic, None) is None


def test_pause_does_not_survive_a_restart():
    """
    Pausing is an in-meeting action, not a preference.

    Carrying it across a restart would launch into a session that looks like
    it is recording you and is not -- the exact failure this area exists to
    remove, rebuilt in a nicer shape. The class default is what a fresh
    process starts from, and it must be "listening".
    """
    # The class-level default is what a fresh process gets. Read it from the
    # source rather than at runtime, because this test's own fixture has
    # already reset the live value and would agree with anything.
    import pathlib
    config = pathlib.Path(__file__).resolve().parent.parent / "src" / "config.py"
    assert "_mic_paused = False" in config.read_text(encoding="utf-8")


def test_the_ui_never_writes_pause_to_settings():
    """Nothing may persist it -- see above."""
    import pathlib
    app = pathlib.Path(__file__).resolve().parent.parent / "src" / "app.py"
    body = app.read_text(encoding="utf-8")
    assert 'update_setting("mic_paused"' not in body
    assert 'get_setting("mic_paused")' not in body
