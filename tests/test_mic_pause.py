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


# --------------------------------------------------------------------------
# Liveness: separating a dead device from a quiet one
# --------------------------------------------------------------------------
#
# All of this is pinned to a measurement rather than a guess. On a real
# Bluetooth disconnect the audio callback stops being invoked entirely -- 0 in
# 2 seconds -- while a muted microphone keeps being invoked and delivers
# zeroes. That difference is the only unambiguous signal available: the device
# list the process can see does not change, no exception is raised, no callback
# status flag is set, and the one line PortAudio prints goes to a file
# descriptor no Python code can intercept.

def test_a_callback_marks_the_device_alive():
    mic = FakeRecorder("You")
    mic.last_callback -= 60
    assert mic.silent_for() > 59
    mic.note_callback()
    assert mic.silent_for() < 1


def test_pausing_still_counts_as_alive():
    """
    The reason liveness counts callbacks and not emitted audio.

    While paused, the device is still calling us and we are throwing the audio
    away. If liveness were measured at the emit point, pausing your own
    microphone would look exactly like the device dying, and the app would tear
    down and rebuild the audio stack because the user ticked a box.
    """
    mic = FakeRecorder("You")
    AudioConfig.set_mic_paused(True)
    mic.note_callback()          # the device is still delivering
    mic.feed(b"dropped")

    assert mic.emitted == [], "audio must still be dropped while paused"
    assert mic.silent_for() < 1, "but the device must not look dead"


def test_a_silent_room_is_not_a_dead_device():
    """Nobody talking still produces callbacks -- of room noise, or of zeroes."""
    mic = FakeRecorder("You")
    for _ in range(10):
        mic.note_callback()
    assert mic.silent_for() < 1


def test_a_device_that_stops_calling_is_detected():
    from src.app import MIC_DEAD_AFTER_SECONDS
    mic = FakeRecorder("You")
    mic.last_callback -= MIC_DEAD_AFTER_SECONDS + 1
    assert mic.silent_for() > MIC_DEAD_AFTER_SECONDS


# --------------------------------------------------------------------------
# Shutdown
# --------------------------------------------------------------------------

def test_closing_the_window_stops_the_heartbeat_first():
    """
    Order matters. Closing the window stops the capture streams, at which
    point callbacks stop -- which is precisely the signature of a dead device.
    Without the flag being set first, quitting the app would trigger a rebuild
    of the audio stack underneath a window that is already going away.
    """
    import inspect
    from src import app
    body = inspect.getsource(app._restore_on_exit)
    assert "_shutting_down.set()" in body
    assert body.index("_shutting_down.set()") < body.index("session.close()"), \
        "the heartbeat must be stopped before anything is torn down"


def test_the_heartbeat_cannot_outlive_the_process():
    """
    Three ways out, and none may leave the thread running:
    the window closing (the event), an exception or SIGTERM (atexit), and
    SIGKILL (the daemon flag, since a daemon thread cannot hold the process
    open).
    """
    import inspect
    from src import app
    body = inspect.getsource(app._start_mic_heartbeat)
    assert "atexit.register(_shutting_down.set)" in body
    assert "daemon=True" in body


def test_the_watch_loop_rechecks_the_flag_after_sleeping():
    """
    A one-second sleep sits between the two checks. Without the second one,
    shutdown that lands mid-sleep still gets a full pass of the loop -- long
    enough to start rebuilding devices on the way out.
    """
    import inspect
    from src import app
    body = inspect.getsource(app._start_mic_heartbeat)
    after_sleep = body.split("time.sleep(MIC_HEARTBEAT_POLL_SECONDS)", 1)[1]
    assert "_shutting_down.is_set()" in after_sleep.split("silent_for")[0]


def test_an_absent_microphone_keeps_being_retried():
    """
    The bug this pins: `if mic is None: continue`.

    After a rebuild finds no device the recorder is None, and treating None as
    "nothing to do" is exactly backwards -- it is the state most in need of
    doing something. Written that way, the heartbeat gave up permanently the
    first time a headset was not back yet, and the microphone stayed dead for
    the rest of the meeting no matter what the user plugged in.
    """
    import inspect
    from src import app
    body = inspect.getsource(app._start_mic_heartbeat)
    guard = [l for l in body.splitlines() if "silent_for() <" in l]
    assert guard, "expected a liveness guard"
    assert "mic is None or" not in guard[0], \
        "None must not short-circuit the retry -- it is the case that needs it"
    assert "mic is not None and" in guard[0]


def test_the_far_end_is_not_torn_down_to_discover_there_is_no_microphone():
    """
    Rebuilding invalidates every stream in the process, the meeting track
    included, and costs a measured 378ms of recording. Asking CoreAudio first
    -- which answers live, unlike PortAudio inside a running process -- makes a
    headset left switched off cost nothing instead of punching a hole in the
    recording on every retry.
    """
    import inspect
    from src import app
    body = inspect.getsource(app._start_mic_heartbeat)
    assert "a_microphone_exists()" in body
    # Against the call, not the hasattr guard at the top of the function.
    assert body.index("a_microphone_exists()") < body.index("backend.restart_capture("), \
        "the cheap check must come before the expensive rebuild"


def test_the_microphone_check_does_not_ask_portaudio():
    """
    PortAudio enumerates once at Pa_Initialize and never again, so inside a
    running process it cannot see a headset leave or return. Asking it here
    would make the check always agree with startup and never notice anything.
    """
    import inspect
    from src.audio import macos
    body = inspect.getsource(macos.a_microphone_exists)
    assert "coreaudio" in body
    assert "sd." not in body and "query_devices" not in body
