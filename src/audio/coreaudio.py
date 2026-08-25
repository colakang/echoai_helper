"""
src/audio/coreaudio.py

Just enough CoreAudio to build a Multi-Output device without the user opening
Audio MIDI Setup.

Capturing the far end of a meeting on macOS needs the system's output routed
to a virtual device the app can record from, *and* still reaching the
speakers. That is what a Multi-Output device is, and the documented way to
make one is by hand: launch Audio MIDI Setup, click +, tick two boxes, choose
a clock master, tick drift correction. Asking a user to do that before their
first meeting loses most of them.

AudioHardwareCreateAggregateDevice does the same thing in one call, needs no
administrator rights, and is reachable through plain ctypes -- no PyObjC
dependency. `stacked: 1` in the description is what makes the result a
Multi-Output rather than an aggregate input device.

Deliberately small: enumerate devices, create and destroy a Multi-Output, read
and set the default output. Anything more belongs in Audio MIDI Setup.
"""

import ctypes
import ctypes.util
from ctypes import (
    POINTER, byref, c_char_p, c_int32, c_uint32, c_void_p, create_string_buffer,
)
from dataclasses import dataclass
from typing import List, Optional

_cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
_ca = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreAudio"))

kCFStringEncodingUTF8 = 0x08000100
kAudioObjectSystemObject = 1
kAudioObjectPropertyElementMain = 0
kAudioObjectPropertyScopeGlobal = 0x676C6F62          # 'glob'
kAudioDevicePropertyScopeOutput = 0x6F757470          # 'outp'
kAudioDevicePropertyScopeInput = 0x696E7074           # 'inpt'


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode(), "big")


kAudioHardwarePropertyDevices = _fourcc("dev#")
kAudioHardwarePropertyDefaultOutputDevice = _fourcc("dOut")
kAudioDevicePropertyDeviceUID = _fourcc("uid ")
kAudioObjectPropertyName = _fourcc("lnam")
kAudioDevicePropertyStreamConfiguration = _fourcc("slay")


class _AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [("mSelector", c_uint32),
                ("mScope", c_uint32),
                ("mElement", c_uint32)]


_cf.CFStringCreateWithCString.restype = c_void_p
_cf.CFStringCreateWithCString.argtypes = [c_void_p, c_char_p, c_uint32]
_cf.CFStringGetCString.restype = ctypes.c_bool
_cf.CFStringGetCString.argtypes = [c_void_p, c_char_p, c_int32, c_uint32]
_cf.CFStringGetLength.restype = c_int32
_cf.CFStringGetLength.argtypes = [c_void_p]
_cf.CFDictionaryCreateMutable.restype = c_void_p
_cf.CFDictionaryCreateMutable.argtypes = [c_void_p, c_int32, c_void_p, c_void_p]
_cf.CFDictionarySetValue.argtypes = [c_void_p, c_void_p, c_void_p]
_cf.CFArrayCreateMutable.restype = c_void_p
_cf.CFArrayCreateMutable.argtypes = [c_void_p, c_int32, c_void_p]
_cf.CFArrayAppendValue.argtypes = [c_void_p, c_void_p]
_cf.CFNumberCreate.restype = c_void_p
_cf.CFNumberCreate.argtypes = [c_void_p, c_int32, c_void_p]
_cf.CFRelease.argtypes = [c_void_p]

_ca.AudioObjectGetPropertyDataSize.argtypes = [
    c_uint32, POINTER(_AudioObjectPropertyAddress), c_uint32, c_void_p,
    POINTER(c_uint32)]
_ca.AudioObjectGetPropertyData.argtypes = [
    c_uint32, POINTER(_AudioObjectPropertyAddress), c_uint32, c_void_p,
    POINTER(c_uint32), c_void_p]
_ca.AudioObjectSetPropertyData.argtypes = [
    c_uint32, POINTER(_AudioObjectPropertyAddress), c_uint32, c_void_p,
    c_uint32, c_void_p]
_ca.AudioHardwareCreateAggregateDevice.argtypes = [c_void_p, POINTER(c_uint32)]
_ca.AudioHardwareDestroyAggregateDevice.argtypes = [c_uint32]


def _cfstr(text: str) -> c_void_p:
    return _cf.CFStringCreateWithCString(None, text.encode("utf-8"),
                                         kCFStringEncodingUTF8)


def _from_cfstr(ref) -> str:
    if not ref:
        return ""
    length = _cf.CFStringGetLength(ref) * 4 + 1
    buffer = create_string_buffer(length)
    if _cf.CFStringGetCString(ref, buffer, length, kCFStringEncodingUTF8):
        return buffer.value.decode("utf-8", "replace")
    return ""


def _cfnum(value: int) -> c_void_p:
    holder = c_int32(value)
    return _cf.CFNumberCreate(None, 3, byref(holder))   # 3 = kCFNumberSInt32Type


def _cfdict() -> c_void_p:
    return _cf.CFDictionaryCreateMutable(
        None, 0,
        byref(c_void_p.in_dll(_cf, "kCFTypeDictionaryKeyCallBacks")),
        byref(c_void_p.in_dll(_cf, "kCFTypeDictionaryValueCallBacks")))


def _cfarray() -> c_void_p:
    return _cf.CFArrayCreateMutable(
        None, 0, byref(c_void_p.in_dll(_cf, "kCFTypeArrayCallBacks")))


def _address(selector: int, scope: int = kAudioObjectPropertyScopeGlobal):
    return _AudioObjectPropertyAddress(selector, scope,
                                       kAudioObjectPropertyElementMain)


def _get_cfstring_property(device: int, selector: int) -> str:
    ref = c_void_p()
    size = c_uint32(ctypes.sizeof(c_void_p))
    address = _address(selector)
    if _ca.AudioObjectGetPropertyData(device, byref(address), 0, None,
                                      byref(size), byref(ref)) != 0:
        return ""
    text = _from_cfstr(ref)
    if ref:
        _cf.CFRelease(ref)
    return text


def _channel_count(device: int, scope: int) -> int:
    address = _address(kAudioDevicePropertyStreamConfiguration, scope)
    size = c_uint32(0)
    if _ca.AudioObjectGetPropertyDataSize(device, byref(address), 0, None,
                                          byref(size)) != 0 or size.value == 0:
        return 0
    buffer = create_string_buffer(size.value)
    if _ca.AudioObjectGetPropertyData(device, byref(address), 0, None,
                                      byref(size), buffer) != 0:
        return 0
    # AudioBufferList: UInt32 count, then that many AudioBuffer
    # {UInt32 channels, UInt32 size, void* data}
    count = int.from_bytes(buffer.raw[0:4], "little")
    total = 0
    offset = 8 if ctypes.sizeof(c_void_p) == 8 else 4
    for i in range(count):
        base = offset + i * 16
        if base + 4 > len(buffer.raw):
            break
        total += int.from_bytes(buffer.raw[base:base + 4], "little")
    return total


@dataclass
class Device:
    id: int
    uid: str
    name: str
    input_channels: int
    output_channels: int

    @property
    def is_output(self) -> bool:
        return self.output_channels > 0

    @property
    def is_input(self) -> bool:
        return self.input_channels > 0


def list_devices() -> List[Device]:
    """Every audio device CoreAudio knows about, with its UID."""
    address = _address(kAudioHardwarePropertyDevices)
    size = c_uint32(0)
    if _ca.AudioObjectGetPropertyDataSize(kAudioObjectSystemObject,
                                          byref(address), 0, None,
                                          byref(size)) != 0:
        return []

    count = size.value // ctypes.sizeof(c_uint32)
    ids = (c_uint32 * count)()
    if _ca.AudioObjectGetPropertyData(kAudioObjectSystemObject, byref(address),
                                      0, None, byref(size), ids) != 0:
        return []

    devices = []
    for device_id in ids:
        devices.append(Device(
            id=int(device_id),
            uid=_get_cfstring_property(device_id, kAudioDevicePropertyDeviceUID),
            name=_get_cfstring_property(device_id, kAudioObjectPropertyName),
            input_channels=_channel_count(device_id, kAudioDevicePropertyScopeInput),
            output_channels=_channel_count(device_id, kAudioDevicePropertyScopeOutput),
        ))
    return devices


def find_device(fragment: str, output: bool = True) -> Optional[Device]:
    """First device whose name contains `fragment`, case-insensitively."""
    lowered = fragment.lower()
    for device in list_devices():
        if lowered not in device.name.lower():
            continue
        if output and not device.is_output:
            continue
        if not output and not device.is_input:
            continue
        return device
    return None


def create_multi_output(name: str, uid: str, member_uids: List[str],
                        master_uid: Optional[str] = None,
                        drift_correct: Optional[List[str]] = None) -> int:
    """
    Create a Multi-Output device and return its AudioObjectID.

    `stacked: 1` is the difference between a Multi-Output and an ordinary
    aggregate. Drift correction is applied to every member except the master,
    which by definition supplies the clock the others are corrected against --
    switching it on for the master would be asking it to follow itself.
    """
    if not member_uids:
        raise ValueError("a Multi-Output device needs at least one member")
    master = master_uid or member_uids[0]
    drift = set(drift_correct if drift_correct is not None else member_uids)
    drift.discard(master)

    description = _cfdict()
    _cf.CFDictionarySetValue(description, _cfstr("name"), _cfstr(name))
    _cf.CFDictionarySetValue(description, _cfstr("uid"), _cfstr(uid))
    _cf.CFDictionarySetValue(description, _cfstr("stacked"), _cfnum(1))
    _cf.CFDictionarySetValue(description, _cfstr("master"), _cfstr(master))
    # Keep it out of the Sound menu's device list clutter is *not* wanted here:
    # the user has to be able to select it, so it stays public.

    members = _cfarray()
    for member in member_uids:
        entry = _cfdict()
        _cf.CFDictionarySetValue(entry, _cfstr("uid"), _cfstr(member))
        if member in drift:
            _cf.CFDictionarySetValue(entry, _cfstr("drift"), _cfnum(1))
        _cf.CFArrayAppendValue(members, entry)
    _cf.CFDictionarySetValue(description, _cfstr("subdevices"), members)

    device = c_uint32(0)
    status = _ca.AudioHardwareCreateAggregateDevice(description, byref(device))
    if status != 0 or not device.value:
        raise OSError(f"AudioHardwareCreateAggregateDevice failed: {status}")
    return int(device.value)


def destroy_device(device_id: int) -> None:
    status = _ca.AudioHardwareDestroyAggregateDevice(c_uint32(device_id))
    if status != 0:
        raise OSError(f"AudioHardwareDestroyAggregateDevice failed: {status}")


def get_default_output() -> Optional[Device]:
    address = _address(kAudioHardwarePropertyDefaultOutputDevice)
    device = c_uint32(0)
    size = c_uint32(ctypes.sizeof(c_uint32))
    if _ca.AudioObjectGetPropertyData(kAudioObjectSystemObject, byref(address),
                                      0, None, byref(size), byref(device)) != 0:
        return None
    for candidate in list_devices():
        if candidate.id == device.value:
            return candidate
    return None


def set_default_output(device_id: int) -> None:
    address = _address(kAudioHardwarePropertyDefaultOutputDevice)
    value = c_uint32(device_id)
    status = _ca.AudioObjectSetPropertyData(
        kAudioObjectSystemObject, byref(address), 0, None,
        ctypes.sizeof(c_uint32), byref(value))
    if status != 0:
        raise OSError(f"could not set the default output device: {status}")
