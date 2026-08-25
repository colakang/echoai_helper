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
