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


# --------------------------------------------------------------------------
# The icon
# --------------------------------------------------------------------------

def test_the_icon_exists_where_the_launcher_looks_for_it():
    """
    The launcher has always copied an icon into the bundle if it found one, and
    for the whole life of the project it did not find one -- so the app showed
    up in Launchpad blank. The file and the lookup have to agree.
    """
    from pathlib import Path
    from src import launcher_macos
    import inspect

    package = Path(launcher_macos.__file__).resolve().parent
    icon = package / "resources" / "images" / "icon.icns"
    assert icon.exists(), "no icon.icns — the launcher will produce a blank icon"
    assert icon.stat().st_size > 1000, "icon.icns is suspiciously small"

    # And the launcher is looking exactly there, not at the project root, which
    # does not exist once this is installed from a wheel.
    body = inspect.getsource(launcher_macos.install)
    assert 'package / "resources" / "images" / "icon.icns"' in body


def test_the_icon_is_a_real_icns():
    """Header check, so a truncated or wrong-format file is caught here."""
    from pathlib import Path
    from src import launcher_macos
    icon = (Path(launcher_macos.__file__).resolve().parent
            / "resources" / "images" / "icon.icns")
    with open(icon, "rb") as f:
        assert f.read(4) == b"icns"


def test_the_icon_ships_with_the_package():
    """
    It lives under src/resources, which the package-data glob covers. Pinned
    because an icon that is not in the wheel is exactly as useful as no icon,
    and the failure is invisible until somebody installs it.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'resources/**/*' in pyproject
