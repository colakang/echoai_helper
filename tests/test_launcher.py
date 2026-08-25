"""
Tests for the double-clickable launcher.

The point of the bundle is that a command-line install still gets an icon,
without a Developer ID. What is pinned here is that it stays a *pointer* to
the install rather than a copy of it.
"""

import os
import plistlib
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform != "darwin":
    pytest.skip("macOS application bundle", allow_module_level=True)

from src import launcher_macos as launcher  # noqa: E402


@pytest.fixture
def location(tmp_path):
    return tmp_path / "Applications"


def test_bundle_has_the_expected_shape(location):
    app = launcher.install(location=location, progress=lambda *_: None)

    assert app.is_dir()
    assert (app / "Contents" / "Info.plist").is_file()
    assert (app / "Contents" / "MacOS" / "launch").is_file()
    assert os.access(app / "Contents" / "MacOS" / "launch", os.X_OK)


def test_no_quarantine_attribute(location):
    """The whole basis of this approach: Gatekeeper only enforces on files
    carrying com.apple.quarantine, which is applied to downloads. A bundle
    written here has none, so it launches unsigned."""
    app = launcher.install(location=location, progress=lambda *_: None)
    # os.listxattr is Linux-only; xattr(1) is the macOS way.
    import subprocess
    attrs = subprocess.run(["xattr", str(app)], capture_output=True,
                           text=True).stdout
    assert "com.apple.quarantine" not in attrs


def test_launcher_points_at_the_current_interpreter(location):
    """It must launch the install the user actually set up, not whatever
    python happens to be first on PATH later."""
    app = launcher.install(python="/some/venv/bin/python", location=location,
                           progress=lambda *_: None)
    script = (app / "Contents" / "MacOS" / "launch").read_text()
    assert "/some/venv/bin/python" in script


def test_launcher_is_a_pointer_not_a_copy(location):
    """Updating through the package manager should update what the icon
    launches, with no reinstall."""
    app = launcher.install(location=location, progress=lambda *_: None)
    files = [p for p in app.rglob("*") if p.is_file()]
    assert len(files) <= 3, f"bundle is carrying a payload: {files}"


def test_output_is_unbuffered(location):
    """A GUI launch has no terminal, so a buffered log stays empty until the
    process exits -- exactly when it is least useful."""
    app = launcher.install(location=location, progress=lambda *_: None)
    script = (app / "Contents" / "MacOS" / "launch").read_text()
    assert " -u " in script


def test_it_logs_somewhere_findable(location):
    app = launcher.install(location=location, progress=lambda *_: None)
    script = (app / "Contents" / "MacOS" / "launch").read_text()
    assert "Library/Logs" in script


def test_microphone_usage_string_is_specific(location):
    """macOS shows this verbatim when asking for the microphone. Vague system
    wording there reads like malware."""
    app = launcher.install(location=location, progress=lambda *_: None)
    with open(app / "Contents" / "Info.plist", "rb") as f:
        info = plistlib.load(f)
    reason = info["NSMicrophoneUsageDescription"]
    assert "meeting" in reason.lower()
    assert len(reason) > 20


def test_reinstall_replaces_cleanly(location):
    launcher.install(python="/first/python", location=location,
                     progress=lambda *_: None)
    app = launcher.install(python="/second/python", location=location,
                           progress=lambda *_: None)

    script = (app / "Contents" / "MacOS" / "launch").read_text()
    assert "/second/python" in script
    assert "/first/python" not in script


def test_uninstall_removes_it(location):
    launcher.install(location=location, progress=lambda *_: None)
    assert launcher.is_installed(location)

    assert launcher.uninstall(location, progress=lambda *_: None)
    assert not launcher.is_installed(location)


def test_uninstalling_nothing_is_not_an_error(location):
    assert not launcher.uninstall(location, progress=lambda *_: None)
