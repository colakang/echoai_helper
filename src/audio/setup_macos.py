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

# A HAL driver is a file in /Library; CoreAudio only publishes it as a device
# after the daemon reads that directory again. Without this the install
# genuinely succeeds and the device still is not there.
#
# kickstart -k is the documented way and is deterministic; killall is kept
# behind it because launchd relaunches the daemon either way and the label has
# not always been spelled the same across releases.
RESTART_COREAUDIOD = ("launchctl kickstart -k system/com.apple.audio.coreaudiod"
                      " || killall coreaudiod")

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


DRIVER_PATH = "/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver"


def next_step(state: "SetupState") -> str:
    """
    What the reader should do, for a state that is not ready.

    A dialog that lists what is missing and stops leaves the reader with a
    fact and no move. The two failures differ and need different advice: a
    driver on disk that CoreAudio has not published is one command away, and
    one that never landed has to be installed again.
    """
    if state.blackhole is None:
        if os.path.exists(DRIVER_PATH):
            return ("The driver is installed but CoreAudio has not picked it "
                    "up. Run this in Terminal, then reopen the app:\n\n"
                    "    sudo killall coreaudiod")
        return ("The audio driver is not installed. Run `echoai-helper setup` "
                "in Terminal to install it.")
    if state.multi_output is None:
        return ("The virtual device is there but the Multi-Output was not "
                "built. Run `echoai-helper setup` to finish.")
    return ("Choose \"%s\" as the output device in System Settings > Sound."
            % MULTI_OUTPUT_NAME)


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

# Homebrew's own metadata for the cask: the current version, the vendor's
# download URL, and a checksum. Used even when Homebrew itself is absent,
# because somebody has to say which build is current and Homebrew already
# tracks it.
BLACKHOLE_CASK_API = "https://formulae.brew.sh/api/cask/blackhole-2ch.json"


def _install_blackhole_pkg(progress=print) -> bool:
    """
    Install BlackHole without Homebrew, from the vendor's own package.

    Needed because requiring a package manager in order to install one audio
    driver is a poor trade, and it is where a first install actually stopped:
    "Homebrew is not installed". The advice then was to fetch it by hand from
    existential.audio -- which asks for an email address before it will hand
    over the file.

    The download URL and its checksum come from Homebrew's cask metadata, so
    the version tracks whatever Homebrew considers current without Homebrew
    needing to be here.

    The checksum is verified before anything is run. This installs a system
    audio driver with administrator rights; running an unverified download
    that way is not a risk worth taking to save a step.
    """
    import hashlib
    import json
    import tempfile
    import urllib.request

    def fetch(url, timeout):
        # A User-Agent is required, not merely polite: the vendor's CDN answers
        # urllib's default with 406 Not Acceptable, so this fails for everyone
        # without it.
        request = urllib.request.Request(
            url, headers={"User-Agent": "echoai-helper (+https://github.com/"
                                        "colakang/echoai_helper)"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    try:
        meta = json.loads(fetch(BLACKHOLE_CASK_API, 30))
        url, expected = meta["url"], meta["sha256"]
        version = meta.get("version", "?")
    except Exception as e:
        progress(f"Could not look up the BlackHole download: {e}")
        progress("Install it from https://existential.audio/blackhole/ "
                 "and run this again.")
        return False

    progress(f"Downloading BlackHole {version}...")
    try:
        payload = fetch(url, 120)
    except Exception as e:
        progress(f"Download failed: {e}")
        return False

    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        progress("The download did not match its published checksum, so it "
                 "was discarded. Nothing has been installed.")
        progress(f"  expected {expected}")
        progress(f"  received {actual}")
        return False

    with tempfile.TemporaryDirectory() as tmp:
        pkg = os.path.join(tmp, "BlackHole.pkg")
        with open(pkg, "wb") as f:
            f.write(payload)

        progress("Installing (macOS will ask for your password)...")
        # osascript rather than sudo: this runs from a terminal and from the
        # app's first-run flow, and only the GUI prompt works in both.
        #
        # Quoted twice, deliberately. Interpolating Python's repr() looked
        # fine and is wrong: repr switches to double quotes when the string
        # contains a single one, and the AppleScript literal it lands inside
        # is itself double-quoted -- so a home directory belonging to anyone
        # called O'Brien would end the string early. `quoted form of` handles
        # the shell layer; escaping backslash and quote handles the
        # AppleScript layer.
        literal = pkg.replace("\\", "\\\\").replace('"', '\\"')
        # The daemon restart rides along inside the same prompt. It needs root
        # exactly as much as the install does, and asking twice for one
        # operation is how people end up cancelling the second half.
        script = ('do shell script "installer -pkg " & quoted form of '
                  f'"{literal}" & " -target / && " & '
                  # Braced: `A && B || C` would restart the daemon even when
                  # the install failed, which is the one case it must not.
                  f'"{{ {RESTART_COREAUDIOD}; }}" with administrator privileges')
        try:
            completed = subprocess.run(["osascript", "-e", script],
                                       capture_output=True, text=True,
                                       timeout=600)
        except subprocess.TimeoutExpired:
            progress("The installer did not finish in time.")
            return False

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if "User canceled" in detail or "-128" in detail:
            progress("Cancelled at the password prompt; nothing was installed.")
        else:
            progress(f"Install failed: {detail[-300:]}")
        return False
    return True


def _await_blackhole(progress=print, quiet: bool = False) -> bool:
    """CoreAudio publishes a new driver asynchronously; give it a moment."""
    for _ in range(20):
        if ca.find_device(BLACKHOLE_NAME):
            if not quiet:
                progress("BlackHole is ready.")
            return True
        time.sleep(0.5)
    if not quiet:
        # Not "restart the app", which was the advice here and does nothing:
        # the driver is on disk and the app is not what has to re-read it.
        progress("The driver is installed but CoreAudio has not published it "
                 "yet. Run this in a terminal, then start the app again:")
        progress("    sudo killall coreaudiod")
    return False


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
        progress("Homebrew is not installed; fetching BlackHole directly.")
        if not _install_blackhole_pkg(progress):
            return False
        # The pkg path restarts the daemon inside its own prompt; this only
        # has anything to do if that half was declined.
        if not _await_blackhole(progress, quiet=True):
            reload_coreaudio(progress)
        return _await_blackhole(progress)

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
    return _await_blackhole(progress)


def reload_coreaudio(progress=print) -> bool:
    """
    Restart the audio daemon so it picks up a newly installed driver.

    Cheaper than the reboot the installer suggests: it mutes the machine for a
    second or two and is otherwise invisible.
    """
    # `sudo -n` was tried here first and is wrong: it fails on any machine
    # that asks for a password, which is every normal one. It failed silently,
    # both callers discarded the result, and the driver sat on disk unpublished
    # until the user happened to reboot -- which is why this looked like it
    # worked. osascript is the same mechanism the installer already uses, and
    # it prompts from a terminal and from the GUI alike.
    try:
        completed = subprocess.run(
            ["osascript", "-e", f'do shell script "{RESTART_COREAUDIOD}" '
                                'with administrator privileges'],
            capture_output=True, text=True, timeout=120)
        if completed.returncode == 0:
            time.sleep(2)
            return True
        detail = (completed.stderr or "").strip()
        if "User canceled" in detail or "-128" in detail:
            progress("Cancelled at the password prompt.")
    except Exception:
        pass

    progress("CoreAudio needs restarting before the driver appears. "
             "Run this in a terminal:")
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


# What the output was before we took it over, so it can be put back exactly
# rather than guessed at. A user listening through an external monitor should
# not find themselves on the built-in speakers afterwards.
_previous_output_uid: Optional[str] = None


def _state_file() -> str:
    from ..config import PathConfig
    return os.path.join(PathConfig.get_user_config_path(), "previous_output")


def remember_current_output() -> Optional[str]:
    """
    Note the output device in use, before switching to the Multi-Output.

    Written to disk as well as held in memory. A clean quit restores from
    memory, but a crash or a force-quit never reaches that code -- and the
    machine is then left on a device whose volume macOS cannot control, with
    nothing to say why. The next launch reads this and puts it back.
    """
    global _previous_output_uid
    current = ca.get_default_output()
    if current is not None and current.uid != MULTI_OUTPUT_UID:
        _previous_output_uid = current.uid
        try:
            path = _state_file()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(current.uid)
        except OSError:
            pass          # in-memory restore still works for a clean quit
    return _previous_output_uid


def _recall_previous_output() -> Optional[str]:
    if _previous_output_uid:
        return _previous_output_uid
    try:
        with open(_state_file(), encoding="utf-8") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _forget_previous_output() -> None:
    global _previous_output_uid
    _previous_output_uid = None
    try:
        os.remove(_state_file())
    except OSError:
        pass


def restore(progress=print) -> bool:
    """
    Put the system output back.

    Always worth doing on the way out. Left pointed at a Multi-Output, macOS
    cannot control the volume at all -- the keys and the menu-bar slider stop
    working, verified -- and the user finds that out days later with no idea
    what caused it.
    """
    remembered = _recall_previous_output()
    target = None
    if remembered:
        target = next((d for d in ca.list_devices()
                       if d.uid == remembered), None)
    target = target or pick_listening_device()

    if target is None:
        progress("No other output device to switch back to.")
        return False
    try:
        ca.set_default_output(target.id)
    except OSError as e:
        progress(f"Could not switch back: {e}")
        return False
    _forget_previous_output()
    progress(f"System audio restored to {target.name!r}.")
    return True


def ensure_active(progress=print) -> bool:
    """
    Make sure system audio is going through *our* Multi-Output.

    Checked at startup rather than assumed: the user may have several output
    configurations, or macOS may have moved the output when a device
    connected or the machine woke. Recording the far end silently fails if
    the system is pointed anywhere else.
    """
    device = next((d for d in ca.list_devices() if d.uid == MULTI_OUTPUT_UID),
                  None)
    if device is None:
        return False

    current = ca.get_default_output()
    if current is not None and current.uid == MULTI_OUTPUT_UID:
        return True

    remember_current_output()
    try:
        ca.set_default_output(device.id)
    except OSError as e:
        progress(f"Could not switch the output device: {e}")
        return False
    progress(f"Switched output to {device.name!r} for recording"
             + (f" (was {current.name!r})" if current else "") + ".")
    return True


def run(auto_activate: bool = True, progress=print) -> SetupState:
    """Do whatever is still missing, then report."""
    remember_current_output()
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
