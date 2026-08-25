"""
src/audio/setup_macos.py

Get a Mac ready to record a meeting, without sending the user to Audio MIDI
Setup.

Three things have to be true before the far end of a call can be transcribed:

  1. a virtual audio device exists (BlackHole), so there is something to
     record system output from;
  2. a Multi-Output device routes the system's audio to it *and* to whatever
     the user is actually listening on, so they still hear the meeting;
  3. that Multi-Output is the current output device.

Done by hand this is a cask install, a daemon restart, six clicks in Audio
MIDI Setup and a trip to Sound settings. Most of it can be automated:

  BlackHole      needs one administrator prompt. It installs a system audio
                 driver into /Library, so there is no way around that.
  coreaudiod     needs the same prompt, folded into the first.
  Multi-Output   no privileges at all -- see coreaudio.create_multi_output.
  default output no privileges.

So: one password, once, and nothing else.
"""

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional

from . import coreaudio as ca

BLACKHOLE_NAME = "BlackHole 2ch"
BLACKHOLE_CASK = "blackhole-2ch"

# Named so it is obvious in the Sound menu what it is and what created it.
MULTI_OUTPUT_NAME = "EchoAI Meeting"
MULTI_OUTPUT_UID = "ai.echo365.helper.multioutput"


@dataclass
class SetupState:
    blackhole: Optional[ca.Device] = None
    multi_output: Optional[ca.Device] = None
    listening_device: Optional[ca.Device] = None
    default_output: Optional[ca.Device] = None
    notes: List[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return (self.blackhole is not None
                and self.multi_output is not None
                and self.default_output is not None
                and self.default_output.uid == MULTI_OUTPUT_UID)

    def describe(self) -> str:
        lines = [
            f"  virtual device : {self.blackhole.name if self.blackhole else 'missing'}",
            f"  multi-output   : {self.multi_output.name if self.multi_output else 'missing'}",
            f"  you listen on  : {self.listening_device.name if self.listening_device else 'unknown'}",
            f"  system output  : {self.default_output.name if self.default_output else 'unknown'}",
        ]
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def inspect() -> SetupState:
    """What is already in place."""
    state = SetupState()
    state.blackhole = ca.find_device(BLACKHOLE_NAME)
    state.default_output = ca.get_default_output()

    for device in ca.list_devices():
        if device.uid == MULTI_OUTPUT_UID:
            state.multi_output = device
            break

    state.listening_device = pick_listening_device()
    return state


def pick_listening_device() -> Optional[ca.Device]:
    """
    What the user is most likely listening on.

    Prefers whatever is currently the default output, since that is the
    device they last chose. Falls back to built-in speakers, which are always
    present -- a Bluetooth headset is not, and a Multi-Output whose member has
    vanished is worse than one built around something that cannot disappear.
    """
    current = ca.get_default_output()
    if current and current.uid not in (MULTI_OUTPUT_UID,) and \
            BLACKHOLE_NAME.lower() not in current.name.lower():
        return current

    for device in ca.list_devices():
        if device.uid == "BuiltInSpeakerDevice":
            return device
    for device in ca.list_devices():
        if device.is_output and BLACKHOLE_NAME.lower() not in device.name.lower():
            return device
    return None


# --------------------------------------------------------------------------
# BlackHole
# --------------------------------------------------------------------------

def install_blackhole(progress=print) -> bool:
    """
    Install the virtual audio driver.

    Prefers Homebrew when it is present, because then the driver is tracked
    and can be uninstalled the same way. Either route needs an administrator
    prompt: this writes a driver into /Library/Audio/Plug-Ins/HAL.
    """
    if ca.find_device(BLACKHOLE_NAME):
        progress("BlackHole is already installed.")
        return True

    if shutil.which("brew") is None:
        progress("Homebrew is not installed, so BlackHole cannot be installed "
                 "automatically.")
        progress("Install it manually from https://existential.audio/blackhole/ "
                 "and run this again.")
        return False

    progress("Installing BlackHole (macOS will ask for your password)...")
    try:
        completed = subprocess.run(
            ["brew", "install", "--cask", BLACKHOLE_CASK],
            capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        progress("The installer did not finish in time.")
        return False

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        progress(f"Install failed: {detail[-400:]}")
        return False

    progress("Reloading CoreAudio so the new driver is picked up...")
    reload_coreaudio(progress)

    for _ in range(20):
        if ca.find_device(BLACKHOLE_NAME):
            progress("BlackHole is ready.")
            return True
        time.sleep(0.5)

    progress("BlackHole was installed but has not appeared yet. "
             "A restart will make it available.")
    return False


def reload_coreaudio(progress=print) -> bool:
    """
    Restart the audio daemon so it picks up a newly installed driver.

    Cheaper than the reboot the installer suggests: it mutes the machine for a
    second or two and is otherwise invisible.
    """
    try:
        completed = subprocess.run(["sudo", "-n", "killall", "coreaudiod"],
                                   capture_output=True, text=True, timeout=30)
        if completed.returncode == 0:
            time.sleep(2)
            return True
    except Exception:
        pass

    progress("CoreAudio needs restarting. Run this in a terminal:")
    progress("    sudo killall coreaudiod")
    return False


# --------------------------------------------------------------------------
# Multi-Output
# --------------------------------------------------------------------------

def create_multi_output(listening_uid: Optional[str] = None,
                        progress=print) -> Optional[ca.Device]:
    """
    Build the Multi-Output device that feeds both BlackHole and the speakers.

    BlackHole is the clock master on purpose. It is a virtual device, so it
    never disappears; a Bluetooth headset as master takes the whole
    Multi-Output down with it when it disconnects.
    """
    blackhole = ca.find_device(BLACKHOLE_NAME)
    if blackhole is None:
        progress("BlackHole is not installed yet.")
        return None

    existing = next((d for d in ca.list_devices() if d.uid == MULTI_OUTPUT_UID),
                    None)
    if existing is not None:
        # Rebuild it: the device the user listens on may have changed since.
        try:
            ca.destroy_device(existing.id)
        except OSError as e:
            progress(f"Could not replace the existing device: {e}")
            return existing

    listening = None
    if listening_uid:
        listening = next((d for d in ca.list_devices() if d.uid == listening_uid),
                         None)
    listening = listening or pick_listening_device()

    members = [blackhole.uid]
    if listening is not None and listening.uid != blackhole.uid:
        members.append(listening.uid)
    else:
        progress("No speakers found to pair with BlackHole; you will not hear "
                 "the meeting through this device.")

    try:
        device_id = ca.create_multi_output(
            name=MULTI_OUTPUT_NAME,
            uid=MULTI_OUTPUT_UID,
            member_uids=members,
            master_uid=blackhole.uid,
        )
    except OSError as e:
        progress(f"Could not create the Multi-Output device: {e}")
        return None

    # CoreAudio publishes the new device asynchronously, so an immediate
    # lookup misses it and the caller concludes the creation failed.
    device = _await_device(device_id)
    if device is None:
        progress("The device was created but has not appeared yet.")
        return None

    heard_on = listening.name if listening else "nothing"
    progress(f"Created {device.name!r}: recorded through "
             f"{blackhole.name}, heard on {heard_on}.")
    return device


def _await_device(device_id: int, timeout: float = 3.0) -> Optional[ca.Device]:
    """Wait for a freshly created device to show up in the device list."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for device in ca.list_devices():
            if device.id == device_id:
                return device
        time.sleep(0.1)
    return None


def activate(progress=print) -> bool:
    """Make the Multi-Output the system's output device."""
    device = next((d for d in ca.list_devices() if d.uid == MULTI_OUTPUT_UID),
                  None)
    if device is None:
        progress("The Multi-Output device does not exist yet.")
        return False
    try:
        ca.set_default_output(device.id)
    except OSError as e:
        progress(f"Could not switch the output device: {e}")
        return False
    progress(f"System audio now goes through {device.name!r}.")
    return True


def restore(progress=print) -> bool:
    """
    Put the system output back to something ordinary.

    Worth offering: leaving a machine pointed at a Multi-Output after the
    meeting is a confusing state to walk away from, especially since the
    volume keys do not work on one.
    """
    target = pick_listening_device()
    if target is None:
        progress("No other output device to switch back to.")
        return False
    try:
        ca.set_default_output(target.id)
    except OSError as e:
        progress(f"Could not switch back: {e}")
        return False
    progress(f"System audio restored to {target.name!r}.")
    return True


def run(auto_activate: bool = True, progress=print) -> SetupState:
    """Do whatever is still missing, then report."""
    state = inspect()

    if state.blackhole is None:
        if not install_blackhole(progress):
            state.notes.append("BlackHole is missing; the far end of a call "
                               "cannot be recorded without it.")
            return inspect()

    if create_multi_output(progress=progress) is None:
        state = inspect()
        state.notes.append("The Multi-Output device could not be created.")
        return state

    if auto_activate:
        activate(progress)

    return inspect()
