"""
Tests for the automated macOS audio setup.

CoreAudio itself is stubbed. What is pinned here is the policy: which device
becomes the clock master, what happens when hardware is missing, and that
nothing destructive happens without something to replace it with.
"""

import os
import sys
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("sounddevice")

if sys.platform != "darwin":
    pytest.skip("macOS audio routing", allow_module_level=True)

from src.audio import setup_macos as setup  # noqa: E402
from src.audio.coreaudio import Device  # noqa: E402


def device(name, uid, out=2, inp=0, ident=None):
    return Device(id=ident if ident is not None else abs(hash(uid)) % 1000,
                  uid=uid, name=name, input_channels=inp, output_channels=out)


BLACKHOLE = device("BlackHole 2ch", "BlackHole2ch_UID", out=2, inp=2)
SPEAKERS = device("Built-in Speakers", "BuiltInSpeakerDevice")
HEADSET = device("Bluetooth Headset", "bluetooth-headset-uid", out=1)


class FakeCoreAudio:
    def __init__(self, devices, default=None):
        self.devices = list(devices)
        self.default = default
        self.created = []
        self.destroyed = []
        self._next_id = 900

    def list_devices(self):
        return list(self.devices)

    def find_device(self, fragment, output=True):
        for d in self.devices:
            if fragment.lower() in d.name.lower():
                return d
        return None

    def get_default_output(self):
        return self.default

    def set_default_output(self, device_id):
        for d in self.devices:
            if d.id == device_id:
                self.default = d
                return
        raise OSError("no such device")

    def create_multi_output(self, name, uid, member_uids, master_uid=None,
                            drift_correct=None):
        self.created.append({"name": name, "uid": uid, "members": member_uids,
                             "master": master_uid})
        self._next_id += 1
        self.devices.append(device(name, uid, ident=self._next_id))
        return self._next_id

    def destroy_device(self, device_id):
        self.destroyed.append(device_id)
        self.devices = [d for d in self.devices if d.id != device_id]


@pytest.fixture
def fake(monkeypatch):
    def install(devices, default=None):
        stub = FakeCoreAudio(devices, default)
        monkeypatch.setattr(setup, "ca", stub)
        monkeypatch.setattr(setup.time, "sleep", lambda *_: None)
        return stub
    return install


# --------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------

def test_reports_missing_multi_output(fake):
    fake([BLACKHOLE, SPEAKERS], default=SPEAKERS)
    state = setup.inspect()
    assert state.blackhole is not None
    assert state.multi_output is None
    assert not state.ready


def test_ready_only_when_the_multi_output_is_selected(fake):
    multi = device(setup.MULTI_OUTPUT_NAME, setup.MULTI_OUTPUT_UID)
    fake([BLACKHOLE, SPEAKERS, multi], default=SPEAKERS)
    assert not setup.inspect().ready, "built but not selected is not ready"

    fake([BLACKHOLE, SPEAKERS, multi], default=multi)
    assert setup.inspect().ready


# --------------------------------------------------------------------------
# Clock master
# --------------------------------------------------------------------------

def test_blackhole_is_the_clock_master(fake):
    """It is virtual, so it never disconnects. A Bluetooth headset as master
    takes the whole Multi-Output down with it when it goes away."""
    stub = fake([BLACKHOLE, HEADSET], default=HEADSET)
    setup.create_multi_output(progress=lambda *_: None)

    assert stub.created[0]["master"] == BLACKHOLE.uid


def test_the_listening_device_is_included(fake):
    """Without it the user records the meeting but cannot hear it."""
    stub = fake([BLACKHOLE, HEADSET], default=HEADSET)
    setup.create_multi_output(progress=lambda *_: None)

    assert set(stub.created[0]["members"]) == {BLACKHOLE.uid, HEADSET.uid}


def test_falls_back_to_built_in_speakers(fake):
    """Whatever else is around, the built-in speakers are always there."""
    stub = fake([BLACKHOLE, SPEAKERS], default=BLACKHOLE)
    setup.create_multi_output(progress=lambda *_: None)

    assert SPEAKERS.uid in stub.created[0]["members"]


def test_blackhole_is_never_chosen_as_the_thing_you_listen_on(fake):
    """It is silent by construction; pairing it with itself is a device that
    records the meeting and plays it nowhere."""
    fake([BLACKHOLE], default=BLACKHOLE)
    assert setup.pick_listening_device() is None


# --------------------------------------------------------------------------
# Rebuilding
# --------------------------------------------------------------------------

def test_an_existing_device_is_replaced(fake):
    """The device the user listens on changes -- a headset connects, a monitor
    is unplugged -- so the Multi-Output is rebuilt rather than reused."""
    stale = device(setup.MULTI_OUTPUT_NAME, setup.MULTI_OUTPUT_UID, ident=42)
    stub = fake([BLACKHOLE, SPEAKERS, stale], default=SPEAKERS)

    setup.create_multi_output(progress=lambda *_: None)

    assert 42 in stub.destroyed
    assert len(stub.created) == 1


def test_nothing_is_built_without_blackhole(fake):
    stub = fake([SPEAKERS], default=SPEAKERS)
    assert setup.create_multi_output(progress=lambda *_: None) is None
    assert stub.created == []


# --------------------------------------------------------------------------
# Switching back
# --------------------------------------------------------------------------

def test_restore_picks_a_real_output(fake):
    """Leaving a machine pointed at a Multi-Output is a confusing state to
    walk away from -- the volume keys do not work on one."""
    multi = device(setup.MULTI_OUTPUT_NAME, setup.MULTI_OUTPUT_UID)
    stub = fake([BLACKHOLE, SPEAKERS, multi], default=multi)

    assert setup.restore(progress=lambda *_: None)
    assert stub.default.uid == SPEAKERS.uid


def test_restore_does_nothing_when_there_is_nowhere_to_go(fake):
    stub = fake([BLACKHOLE], default=BLACKHOLE)
    assert not setup.restore(progress=lambda *_: None)


def test_activate_requires_the_device_to_exist(fake):
    fake([BLACKHOLE, SPEAKERS], default=SPEAKERS)
    assert not setup.activate(progress=lambda *_: None)


# --------------------------------------------------------------------------
# Taking the output, and giving it back
# --------------------------------------------------------------------------
#
# Verified on this machine: with a Multi-Output selected, `get volume
# settings` returns "missing value" and setting the volume does nothing. The
# volume keys and the menu-bar slider stop working. Left that way, the user
# discovers it days later with no connection to this app -- which is why
# restoring is not something to ask about.

def test_startup_takes_the_output(fake):
    """Checked rather than assumed: the user may have several output
    configurations, or macOS may have moved the output when a device
    connected."""
    multi = device(setup.MULTI_OUTPUT_NAME, setup.MULTI_OUTPUT_UID)
    stub = fake([BLACKHOLE, SPEAKERS, multi], default=SPEAKERS)

    assert setup.ensure_active(progress=lambda *_: None)
    assert stub.default.uid == setup.MULTI_OUTPUT_UID


def test_startup_is_a_no_op_when_already_routed(fake):
    multi = device(setup.MULTI_OUTPUT_NAME, setup.MULTI_OUTPUT_UID)
    stub = fake([BLACKHOLE, SPEAKERS, multi], default=multi)

    assert setup.ensure_active(progress=lambda *_: None)
    assert stub.default.uid == setup.MULTI_OUTPUT_UID


def test_startup_does_nothing_without_the_device(fake):
    stub = fake([BLACKHOLE, SPEAKERS], default=SPEAKERS)
    assert not setup.ensure_active(progress=lambda *_: None)
    assert stub.default.uid == SPEAKERS.uid


def test_the_exact_previous_device_comes_back(monkeypatch, fake, tmp_path):
    """Someone listening through an external monitor should not end up on the
    built-in speakers -- which is what guessing would give them."""
    monitor = device("External Display", "monitor-uid")
    multi = device(setup.MULTI_OUTPUT_NAME, setup.MULTI_OUTPUT_UID)
    stub = fake([BLACKHOLE, SPEAKERS, monitor, multi], default=monitor)
    monkeypatch.setattr(setup, "_state_file",
                        lambda: str(tmp_path / "previous_output"))
    monkeypatch.setattr(setup, "_previous_output_uid", None)

    setup.ensure_active(progress=lambda *_: None)
    assert stub.default.uid == setup.MULTI_OUTPUT_UID

    setup.restore(progress=lambda *_: None)
    assert stub.default.uid == monitor.uid, "fell back to guessing"


def test_the_previous_device_survives_a_crash(monkeypatch, fake, tmp_path):
    """A force-quit never reaches the restore, so the machine would be left on
    a device whose volume cannot be controlled. The next launch reads this."""
    monitor = device("External Display", "monitor-uid")
    multi = device(setup.MULTI_OUTPUT_NAME, setup.MULTI_OUTPUT_UID)
    state = tmp_path / "previous_output"

    stub = fake([BLACKHOLE, SPEAKERS, monitor, multi], default=monitor)
    monkeypatch.setattr(setup, "_state_file", lambda: str(state))
    monkeypatch.setattr(setup, "_previous_output_uid", None)
    setup.ensure_active(progress=lambda *_: None)

    assert state.read_text().strip() == monitor.uid

    # A fresh process: nothing in memory, only what is on disk.
    monkeypatch.setattr(setup, "_previous_output_uid", None)
    setup.restore(progress=lambda *_: None)
    assert stub.default.uid == monitor.uid


def test_restoring_clears_the_record(monkeypatch, fake, tmp_path):
    multi = device(setup.MULTI_OUTPUT_NAME, setup.MULTI_OUTPUT_UID)
    state = tmp_path / "previous_output"
    fake([BLACKHOLE, SPEAKERS, multi], default=SPEAKERS)
    monkeypatch.setattr(setup, "_state_file", lambda: str(state))
    monkeypatch.setattr(setup, "_previous_output_uid", None)

    setup.ensure_active(progress=lambda *_: None)
    setup.restore(progress=lambda *_: None)
    assert not state.exists()


# --------------------------------------------------------------------------
# Installing BlackHole without Homebrew
# --------------------------------------------------------------------------
#
# Reported from a first install on someone else's Mac: `echoai-helper setup`
# stopped at "Homebrew is not installed". Requiring a package manager in order
# to install one audio driver is a poor trade, and the advice it gave --
# fetch it by hand from existential.audio -- leads to a page that asks for an
# email address before it will hand over the file.

def test_there_is_a_route_without_homebrew():
    import inspect
    from src.audio import setup_macos
    body = inspect.getsource(setup_macos.install_blackhole)
    assert "_install_blackhole_pkg" in body
    assert 'shutil.which("brew") is None' in body


def test_the_download_is_checksummed_before_anything_runs():
    """
    This installs a system audio driver with administrator rights. Running an
    unverified download that way is not a risk worth taking to save a step.
    """
    import inspect
    from src.audio import setup_macos
    body = inspect.getsource(setup_macos._install_blackhole_pkg)
    assert "sha256" in body
    assert body.index("hashlib.sha256(payload)") < body.index("installer -pkg")


def test_a_mismatched_download_installs_nothing():
    import inspect
    from src.audio import setup_macos
    body = inspect.getsource(setup_macos._install_blackhole_pkg)
    guard = body.split("if actual != expected:", 1)[1][:400]
    assert "return False" in guard
    assert "Nothing has been installed" in guard


def test_the_request_sets_a_user_agent():
    """
    Not politeness. The vendor's CDN answers urllib's default User-Agent with
    406 Not Acceptable, so without one this fails for every user -- which is
    how it behaved when first written.
    """
    import inspect
    from src.audio import setup_macos
    body = inspect.getsource(setup_macos._install_blackhole_pkg)
    assert "User-Agent" in body


def test_the_version_is_not_hardcoded():
    """
    Taken from Homebrew's cask metadata, which tracks what is current. A
    pinned URL goes stale and takes the checksum with it.
    """
    import inspect
    from src.audio import setup_macos
    body = inspect.getsource(setup_macos._install_blackhole_pkg)
    assert "BLACKHOLE_CASK_API" in body
    assert ".pkg" not in setup_macos.BLACKHOLE_CASK_API


def test_the_password_prompt_works_from_both_places():
    """
    setup runs from a terminal and from the app's first-run flow. sudo prompts
    on a tty that the second one does not have; the osascript dialog works in
    both.
    """
    import inspect
    from src.audio import setup_macos
    body = inspect.getsource(setup_macos._install_blackhole_pkg)
    assert "with administrator privileges" in body


def test_cancelling_the_prompt_is_not_reported_as_a_failure():
    """Declining a password is a decision, not an error."""
    import inspect
    from src.audio import setup_macos
    body = inspect.getsource(setup_macos._install_blackhole_pkg)
    assert "User canceled" in body


def test_the_installer_path_is_quoted_for_both_layers():
    """
    The path lands inside an AppleScript string literal, which then goes to a
    shell. Interpolating Python's repr() looked fine and was wrong: repr
    switches to double quotes when the value contains a single one, and the
    literal it sits inside is double-quoted -- so a home directory belonging
    to anyone called O'Brien would end the string early and the failure would
    surface at the password prompt, which is the worst place to debug.
    """
    import inspect
    from src.audio import setup_macos
    body = inspect.getsource(setup_macos._install_blackhole_pkg)
    assert "quoted form of" in body, "the shell layer"
    assert 'replace(\'"\'' in body, "the AppleScript layer"
    assert "{pkg!r}" not in body


@pytest.mark.parametrize("path", [
    "/tmp/x/BlackHole.pkg",
    "/Users/O'Brien/tmp/BlackHole.pkg",
    '/tmp/we"ird/BlackHole.pkg',
    "/tmp/back\\slash/BlackHole.pkg",
])
def test_the_applescript_literal_survives_awkward_paths(path):
    literal = path.replace("\\", "\\\\").replace('"', '\\"')
    script = ('do shell script "installer -pkg " & quoted form of '
              f'"{literal}" & " -target /" with administrator privileges')
    body = script.split("quoted form of ", 1)[1].split(' & " -target', 1)[0]
    assert body.startswith('"') and body.endswith('"')
    # No unescaped quote may appear inside, or the literal ends early.
    inner = body[1:-1]
    assert '"' not in inner.replace('\\"', "")
