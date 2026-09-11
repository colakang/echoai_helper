"""Tests for the export dialog's estimates and the cancellable cleanup."""

import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.export_dialog import ExportChoices, estimate_minutes
from src.polish import polish_transcript


# --------------------------------------------------------------------------
# Time estimates
# --------------------------------------------------------------------------
#
# A real meeting produced 1311 lines. Without a figure up front, the user has
# no way to tell a 40-minute job from a hung one.

def test_estimate_is_a_range_not_a_number():
    """Per 25-line batch the CLI measured 20-50s, the spread being queueing
    upstream. A single number would be wrong in one direction."""
    assert "-" in estimate_minutes(1000, "cli")


def test_cli_is_slower_than_api():
    def low(text):
        return int(text.split("-")[0].split()[0])
    assert low(estimate_minutes(1000, "cli")) > low(estimate_minutes(1000, "api"))


def test_a_real_meeting_reads_as_tens_of_minutes():
    text = estimate_minutes(1311, "cli")
    assert any(str(n) in text for n in range(15, 60))


def test_a_short_conversation_is_not_alarming():
    assert "minute" in estimate_minutes(10, "api")


def test_estimate_scales_with_length():
    def low(text):
        head = text.split("-")[0].split()[0]
        return int(head) if head.isdigit() else 0
    assert low(estimate_minutes(2000, "cli")) > low(estimate_minutes(200, "cli"))


# --------------------------------------------------------------------------
# Choices
# --------------------------------------------------------------------------

def test_markdown_is_detected_by_extension():
    assert ExportChoices(path="/tmp/notes.md").is_markdown
    assert not ExportChoices(path="/tmp/notes.json").is_markdown


# --------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------
#
# Cleanup runs on a worker thread now. Stopping it must keep what has already
# been done: on a long meeting, most of the value is in the batches that
# already finished.

class CountingProvider:
    def __init__(self):
        self.calls = 0

    def generate_response(self, messages, **kwargs):
        self.calls += 1
        yield f"1| cleaned {self.calls}"

    def get_model_name(self):
        return "counting"


def lines(n):
    return [{"text": f"line {i}", "speaker": "S1"} for i in range(n)]


def test_cancelling_keeps_completed_batches():
    cancel = threading.Event()
    provider = CountingProvider()

    original = provider.generate_response

    def stop_after_first(messages, **kwargs):
        cancel.set()
        return original(messages, **kwargs)

    provider.generate_response = stop_after_first

    result = polish_transcript(lines(10), provider, batch_size=2,
                               cancelled=cancel)

    assert result.cancelled
    assert result.batches_attempted == 1, "it should stop between batches"
    assert result.polished_count >= 1, "finished work was thrown away"
    assert len(result.segments) == 10, "segments were lost"


def test_cancelling_before_the_first_batch_changes_nothing():
    cancel = threading.Event()
    cancel.set()
    provider = CountingProvider()

    result = polish_transcript(lines(10), provider, batch_size=2,
                               cancelled=cancel)

    assert result.cancelled
    assert provider.calls == 0
    assert all("polished" not in s for s in result.segments)


def test_cancellation_is_reported_in_the_summary():
    cancel = threading.Event()
    cancel.set()
    result = polish_transcript(lines(4), CountingProvider(), batch_size=2,
                               cancelled=cancel)
    assert "stopped early" in result.summary()


def test_running_to_completion_is_not_marked_cancelled():
    result = polish_transcript(lines(4), CountingProvider(), batch_size=2,
                               cancelled=threading.Event())
    assert not result.cancelled


# --------------------------------------------------------------------------
# Finding the agent CLIs
#
# Reported as: at export, with cleanup ticked, only the API can be chosen --
# the CLI is greyed out. On a machine where `claude` is installed and working.
#
# A double-clicked .app inherits /usr/bin:/bin:/usr/sbin:/sbin and reads no
# shell profile, so shutil.which finds nothing installed anywhere else. That
# is where `claude` lives by default: ~/.local/bin.
# --------------------------------------------------------------------------

def test_a_cli_outside_PATH_is_still_found(tmp_path, monkeypatch):
    from src.llm.cli_provider import CLIProvider

    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    installed = home / ".local" / "bin" / "claude"
    installed.write_text("#!/bin/sh\n")
    installed.chmod(0o755)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")       # what a .app inherits
    monkeypatch.setattr(CLIProvider, "NVM_GLOB", str(tmp_path / "none" / "*"))

    assert CLIProvider.locate("claude") == str(installed)
    assert CLIProvider(command="claude").validate_config()


def test_a_cli_that_is_not_installed_is_not_found(tmp_path, monkeypatch):
    from src.llm.cli_provider import CLIProvider
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(CLIProvider, "NVM_GLOB", str(tmp_path / "none" / "*"))
    assert CLIProvider.locate("claude") is None


def test_a_directory_is_not_mistaken_for_the_executable(tmp_path, monkeypatch):
    from src.llm.cli_provider import CLIProvider
    home = tmp_path / "home"
    (home / ".local" / "bin" / "claude").mkdir(parents=True)   # a directory
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(CLIProvider, "NVM_GLOB", str(tmp_path / "none" / "*"))
    assert CLIProvider.locate("claude") is None


def test_the_resolved_path_is_what_gets_run(tmp_path, monkeypatch):
    """
    Resolving it and then running the bare name would fail for exactly the
    reason the lookup was needed.
    """
    import inspect
    from src.llm.cli_provider import CLIProvider
    body = inspect.getsource(CLIProvider.generate_response)
    assert "argv[0] = resolved" in body


# --------------------------------------------------------------------------
# Offering them
# --------------------------------------------------------------------------

def test_every_installed_cli_is_offered(tmp_path, monkeypatch):
    """
    The dialog asked about one hardcoded CLI, so an installed Codex was
    unreachable and an absent Claude left the API as the only option.
    """
    from src.llm.cli_provider import CLIProvider
    home = tmp_path / "home"
    binaries = home / ".local" / "bin"
    binaries.mkdir(parents=True)
    for name in ("codex", "gemini"):
        target = binaries / name
        target.write_text("#!/bin/sh\n")
        target.chmod(0o755)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(CLIProvider, "NVM_GLOB", str(tmp_path / "none" / "*"))

    assert CLIProvider.available() == ["codex", "gemini"], "claude is absent"


def test_the_chosen_cli_is_the_one_that_runs():
    """
    `cli_command` used to be whatever conf.yaml said, so picking Codex in the
    dialog still ran Claude.
    """
    import inspect
    from src import app
    body = inspect.getsource(app._build_polish_provider)
    assert 'config["command"] = cli_command' in body


def test_the_choice_carries_which_cli():
    from src.export_dialog import ExportChoices
    assert ExportChoices(path="x.md").cli_command == "claude"
    assert ExportChoices(path="x.md", backend="cli",
                         cli_command="codex").cli_command == "codex"


# --------------------------------------------------------------------------
# Building the provider
#
# The factory had no "cli" branch, so create_llm_provider("cli", ...) always
# returned None and the cleanup reported "no language model available" -- on
# a machine where the CLI was installed and working. It went unnoticed
# because the PATH bug greyed the option out before anyone could choose it.
# --------------------------------------------------------------------------

def test_the_factory_can_build_a_cli_provider():
    from src.llm import create_llm_provider
    from src.llm.cli_provider import CLIProvider
    provider = create_llm_provider("cli", {"command": "claude"})
    assert isinstance(provider, CLIProvider), "the cli branch is missing"
    assert provider.command == "claude"


def test_the_factory_honours_which_cli_was_asked_for():
    from src.llm import create_llm_provider
    provider = create_llm_provider("cli", {"command": "codex"})
    assert provider is not None and provider.command == "codex"


def test_the_factory_refuses_a_cli_it_does_not_know():
    from src.llm import create_llm_provider
    assert create_llm_provider("cli", {"command": "not-a-cli"}) is None


def test_codex_answers_through_a_file_not_stdout():
    """
    codex prints a session banner -- provider, sandbox, session id -- ahead of
    the answer. Read from stdout, every polish batch came back with preamble
    glued on and the wrong number of lines, so every batch was discarded.
    """
    from src.llm.cli_provider import CLIProvider
    assert CLIProvider.COMMANDS["codex"].get("answer_file_flag") == \
        "--output-last-message"
    assert "answer_file_flag" not in CLIProvider.COMMANDS["claude"], \
        "claude prints only the answer; a file would be pointless indirection"


# --------------------------------------------------------------------------
# How many people
# --------------------------------------------------------------------------

def test_one_speaker_can_be_chosen():
    """
    Both controls started at 2, for no reason anyone could name. 1 is the
    most useful of them: a one-to-one call is the commonest shape, and the
    one where over-splitting shows most -- a single support agent came back
    as S1 and S2. Nothing downstream objects: recluster(e, 1) is well defined
    and the live path caps the registry at 1 happily.
    """
    import inspect
    from src import app
    body = inspect.getsource(app.create_ui_components)
    assert "range(1, 13)" in body, "the live control still starts at 2"


def test_merging_all_the_way_down_to_one_is_offered():
    import inspect
    from src import export_dialog
    body = inspect.getsource(export_dialog.ExportDialog._build)
    assert "range(1, max(2, self._speakers_found))" in body


def test_reclustering_to_one_puts_everyone_together():
    """The behaviour the option depends on."""
    import numpy as np
    from src.asr.diarization import recluster
    rng = np.random.default_rng(0)
    embeddings = [rng.random(192).astype("float32") for _ in range(6)]
    assert len(set(recluster(embeddings, 1))) == 1
