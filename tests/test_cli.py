"""
Tests for the command-line entry point.

Mostly about argument routing. The subcommands that matter here are the ones a
user reaches when something is already wrong -- check-audio is what the
troubleshooting section of the README tells them to run -- so a parser that
rejects its own documented flags is worse than it looks.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import cli


@pytest.fixture
def captured(monkeypatch):
    """Replace check_audio.main with a recorder, and report what it received."""
    from scripts import check_audio
    seen = {}

    def fake_main(argv=None):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(check_audio, "main", fake_main)
    return seen


def test_check_audio_runs(captured):
    """
    `echoai-helper check-audio` used to die on its own subcommand name.

    check_audio.main() read sys.argv itself, which still held "check-audio",
    and argparse rejected it as an unrecognized argument -- so the command the
    README points at when audio is broken was itself broken.
    """
    assert cli.main(["check-audio"]) == 0
    assert captured["argv"] == []


def test_check_audio_flags_are_forwarded(captured):
    """
    The flags belong to check_audio, and reach it untouched.

    Worth pinning: argparse.REMAINDER looks like the obvious way to forward
    them and silently drops any that lead with a dash, which is all of them.
    """
    assert cli.main(["check-audio", "--source", "mic", "--record", "5"]) == 0
    assert captured["argv"] == ["--source", "mic", "--record", "5"]


def test_check_audio_help_belongs_to_check_audio(captured):
    """--help goes through too, so it describes the tool rather than the wrapper."""
    cli.main(["check-audio", "--help"])
    assert captured["argv"] == ["--help"]


def test_version_is_the_installed_one(capsys):
    cli.main(["version"])
    printed = capsys.readouterr().out.strip()
    assert printed, "version must print something"


def test_unknown_subcommand_is_rejected():
    """The pre-parse hand-off must not swallow everything else."""
    with pytest.raises(SystemExit):
        cli.main(["no-such-command"])
