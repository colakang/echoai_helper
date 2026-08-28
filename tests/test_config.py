

# --------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------
#
# A config file naming "mps" is wrong on Windows and one naming "cuda" is wrong
# on a Mac, and conf.yaml travels between machines. Getting this wrong is not a
# missing feature -- it silently lands on cpu, where dual-track real-time
# factor is 2.04 and transcription falls behind the meeting.

from unittest.mock import patch  # noqa: E402


def test_explicit_device_is_honoured():
    from src.TranscriberModels import resolve_device
    assert resolve_device("cpu") == "cpu"


def test_unavailable_accelerator_falls_back_rather_than_crashing():
    from src.TranscriberModels import resolve_device
    with patch("torch.cuda.is_available", return_value=False):
        assert resolve_device("cuda") == "cpu"


def test_auto_prefers_cuda_then_mps_then_cpu():
    from src.TranscriberModels import resolve_device
    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device("auto") == "cuda"
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=True):
        assert resolve_device("auto") == "mps"
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        assert resolve_device("auto") == "cpu"


def test_shipped_config_names_no_platform():
    """conf.yaml must not pin a device -- see the note above."""
    import yaml
    from pathlib import Path
    from src.config import PathConfig
    # Resolved rather than hardcoded: the shipped default lives inside the
    # package so it survives being installed from a wheel, where there is no
    # project root to read it from.
    config = yaml.safe_load(Path(PathConfig.get_conf_file()).read_text())
    assert config["FunASR"]["device"] == "auto"


def test_shipped_config_travels_with_the_package():
    """
    conf.yaml must sit inside the package, not beside it.

    This is what an actual publish got wrong: the wheel installed cleanly and
    then died on launch, because conf.yaml, the prompt templates and the GUI
    itself were all at the project root -- which does not exist once the code
    is in site-packages.
    """
    from pathlib import Path
    from src.config import PathConfig
    package = Path(PathConfig.get_package_root())
    assert (package / "conf.yaml").exists()
    assert (package / "resources" / "config" / "settings.json").exists()
    assert (package / "resources" / "prompt").is_dir()
    assert (package / "app.py").exists(), "the GUI must ship inside the package"


def test_capture_dependencies_are_platform_split():
    """PyAudioWPatch has no macOS build and sounddevice is not the Windows
    path; declaring either unconditionally breaks install on the other."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    pyproject = (root / "pyproject.toml").read_text()
    assert "PyAudioWPatch; sys_platform == 'win32'" in pyproject
    assert "sounddevice; sys_platform != 'win32'" in pyproject


def test_the_speaker_embedder_gets_a_real_device_not_a_word():
    """
    Regression: shipping conf.yaml with device "auto" broke diarization.

    Two ASR call sites were routed through resolve_device() when "auto" was
    introduced; the speaker embedder's was missed and got the raw string.
    torch rejects "auto", the embedder failed to load, and diarization then
    produced no labels at all -- silently, looking like a model that could not
    tell voices apart rather than one that never started.
    """
    import torch
    from src.AudioTranscriber import _asr_device

    device = _asr_device()
    assert device != "auto", "must be resolved before torch ever sees it"
    torch.device(device)          # raises if it is not a real device name


def test_python_support_covers_312_and_313():
    """
    3.12 alone was too narrow: the Python most people have is neither, so
    every install went through uv fetching one. 3.13 works once the two
    standard-library removals are backported.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12,<3.14"' in pyproject


def test_the_backports_are_only_installed_where_they_are_missing():
    """
    aifc and audioop are in the standard library up to 3.12. Depending on the
    backports unconditionally would install a shim over the real thing.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    for package in ("audioop-lts", "standard-aifc"):
        assert f"\"{package}; python_version >= '3.13'\"" in pyproject


def test_the_upper_bound_is_explained():
    """
    It is onnxruntime, not this project: no macOS arm64 wheel for 3.14, so the
    stack does not resolve there. Worth stating, or the next person widens it
    and finds out the slow way.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "onnxruntime" in pyproject.split("requires-python")[1][:900]


def test_the_readme_states_the_macos_prerequisites():
    """
    Both came from a real first install on somebody else's Mac. The Command
    Line Developer Tools prompt can appear part-way through and reads as a
    failure; uv is assumed by the very first install command and was never
    explained.
    """
    from pathlib import Path
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    assert "xcode-select --install" in readme
    assert "astral.sh/uv/install.sh" in readme


def test_homebrew_is_documented_as_optional():
    """
    It is not required any more -- setup fetches the driver directly when it is
    absent. Listing it as a prerequisite would send people to install a package
    manager they do not need.
    """
    from pathlib import Path
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    section = readme.split("Homebrew", 1)[1][:400]
    assert "optional" in section.lower() or "Not required" in section


def test_the_readme_says_how_to_upgrade():
    """
    It only ever said how to install. `uv tool install` on something already
    installed is a no-op that reports "already installed", which reads as "you
    are up to date" -- so anyone on an older version stayed there believing
    otherwise.
    """
    from pathlib import Path
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()
    assert "uv tool upgrade echoai-helper" in readme
    assert "already installed" in readme, "and why install alone does not do it"


def test_the_documented_uv_commands_exist():
    """
    Pinned against uv itself, because the README once told people to run
    `uv tool upgrade --refresh`, which does not exist and errors out. It was
    "verified" through a grep that swallowed the error, and the version number
    printed afterwards came from a previous, different command.

    Skipped where uv is absent; this is about the docs being runnable, not
    about uv being installed.
    """
    import shutil
    import subprocess
    if not shutil.which("uv"):
        pytest.skip("uv not installed")

    from pathlib import Path
    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text()

    for command, flag in (("upgrade", None), ("install", "--force")):
        help_text = subprocess.run(["uv", "tool", command, "--help"],
                                   capture_output=True, text=True).stdout
        if flag:
            assert f"\n      {flag}" in help_text or f" {flag} " in help_text, \
                f"uv tool {command} has no {flag}"

    # Only what is inside a fenced block: prose that mentions the broken form
    # in order to warn about it is not a command anyone will paste.
    runnable, inside = [], False
    for line in readme.splitlines():
        if line.startswith("```"):
            inside = not inside
        elif inside:
            runnable.append(line.strip())

    for line in runnable:
        if line.startswith("uv tool upgrade"):
            assert "--refresh" not in line, \
                f"upgrade has no --refresh: {line}"


# --------------------------------------------------------------------------
# The API key
# --------------------------------------------------------------------------
#
# The app refused to start without one, showing an alert and quitting. That is
# wrong twice over: transcription is local -- FunASR on this machine -- and
# needs no account at all, and the alert told people to create a file without
# saying where, while the place it looked was inside site-packages.

def test_the_key_lives_in_the_user_directory():
    """
    Not beside the app. Installed from a wheel that is site-packages: not a
    place to keep a secret, and a reinstall deletes it.
    """
    from src.config import EnvConfig, PathConfig
    assert EnvConfig.key_file().startswith(PathConfig.get_user_config_path())


def test_the_user_directory_is_searched_first():
    import inspect
    from src.config import EnvConfig
    body = inspect.getsource(EnvConfig.initialize)
    assert body.index("get_user_config_path()") < body.index("get_project_root()")


def test_a_key_that_is_not_one_is_refused(tmp_path, monkeypatch):
    from src.config import EnvConfig, PathConfig
    monkeypatch.setattr(PathConfig, "get_user_config_path",
                        staticmethod(lambda: str(tmp_path)))
    for rubbish in ("", "   ", "my-key", "OPENAI_API_KEY"):
        assert EnvConfig.save_key(rubbish), f"{rubbish!r} should be refused"
    assert not (tmp_path / ".llm").exists()


def test_a_saved_key_is_not_world_readable(tmp_path, monkeypatch):
    import os
    import stat
    from src.config import EnvConfig, PathConfig
    monkeypatch.setattr(PathConfig, "get_user_config_path",
                        staticmethod(lambda: str(tmp_path)))
    assert EnvConfig.save_key("sk-test-not-a-real-key") is None
    mode = os.stat(tmp_path / ".llm").st_mode
    assert not (mode & stat.S_IRGRP or mode & stat.S_IROTH)


def test_startup_does_not_die_without_a_key():
    """
    Recording a meeting must not require an OpenAI account for a feature the
    user has not reached yet.
    """
    import inspect
    from src import app
    body = inspect.getsource(app.main)
    section = body.split("ensure_api_key()", 1)[1][:400]
    assert "_fatal" not in section
    assert "return" not in section.split("\n")[1]


def test_the_responder_survives_having_no_provider():
    """It raised, which took the whole app down with it."""
    import inspect
    from src.GPTResponder import GPTResponder
    body = inspect.getsource(GPTResponder.__init__)
    assert "raise ValueError" not in body
    assert hasattr(GPTResponder, "has_provider")


def test_the_key_can_be_set_without_touching_a_file():
    """From the app when a feature needs it, or from the command line."""
    import inspect
    from src import app, cli
    assert "simpledialog.askstring" in inspect.getsource(app.create_ui_components)
    assert hasattr(cli, "_key")


def test_setup_mentions_it():
    """setup walks someone through the install and cannot arrange this one."""
    import inspect
    from src import cli
    body = inspect.getsource(cli._setup)
    assert "ensure_api_key" in body
    assert "does not need one" in body


def test_choosing_cleanup_without_a_key_asks_for_one():
    """
    It used to export with a note saying cleanup had been skipped -- true, and
    not what was asked for. The CLI backend bills a subscription and is exempt.
    """
    import inspect
    from src import app
    body = inspect.getsource(app.create_ui_components)
    section = body.split("if choices.polish:", 1)[1][:600]
    assert "ensure_key(" in section
    assert 'choices.backend == "cli"' in section, "the CLI route needs no key"


# --------------------------------------------------------------------------
# First-run: the window comes up before the models load
# --------------------------------------------------------------------------
#
# get_model() downloads about 1.5GB on a first run and takes minutes. It used
# to run with no window on screen, so macOS reported "Application Not
# Responding" and offered Force Quit -- which is what somebody trying it for
# the first time did, reasonably, because nothing said it was working.

def test_the_window_is_created_before_the_models_load():
    import inspect
    from src import app
    body = inspect.getsource(app.main)
    assert body.index("ctk.CTk()") < body.index("TranscriberModels.get_model"), \
        "the window must exist before the download starts"


def test_only_one_window_is_created():
    """The second ctk.CTk() would leave the splash orphaned on screen."""
    import inspect
    from src import app
    assert inspect.getsource(app.main).count("ctk.CTk()") == 1


def test_the_event_loop_runs_during_the_download():
    """
    Pumping it is what keeps the app responsive. Loading on the main thread
    without it is indistinguishable from a hang, however good the splash text.
    """
    import inspect
    from src import app
    body = inspect.getsource(app.main)
    # The whole wait loop, not a fixed number of characters after a marker --
    # a comment added later pushed the call out of that window.
    loop = body.split("while not loaded:", 1)[1].split("splash_progress.stop()", 1)[0]
    assert "root.update()" in loop


def test_the_wait_says_how_long_it_has_been():
    """A progress bar with no elapsed time still looks stuck after two minutes."""
    import inspect
    from src import app
    body = inspect.getsource(app.main)
    assert "so far." in body


def test_a_failed_load_is_reported_not_swallowed():
    import inspect
    from src import app
    body = inspect.getsource(app.main)
    assert 'if "error" in loaded:' in body
    assert "Could not load the speech models" in body


# --------------------------------------------------------------------------
# Where the models are downloaded from
# --------------------------------------------------------------------------

def test_the_model_hub_is_one_that_serves_files():
    """
    The setting always existed and was "ms". That cost an hour: the ModelScope
    site answered in under a second while its file downloads timed out over and
    over -- 213 retries, nothing written to disk. HuggingFace served the same
    model immediately from the same machine.
    """
    import yaml
    from pathlib import Path
    from src.config import PathConfig
    config = yaml.safe_load(Path(PathConfig.get_conf_file()).read_text())
    assert config["FunASR"]["hub"] == "hf"


def test_the_hub_is_declared_once():
    """
    A duplicate key is silently resolved by YAML taking the last one, so the
    file can say "hf" in one place and mean "ms".
    """
    from pathlib import Path
    from src.config import PathConfig
    text = Path(PathConfig.get_conf_file()).read_text()
    assert len([l for l in text.splitlines() if l.strip().startswith("hub:")]) == 1


def test_the_hub_reaches_the_model():
    """Configured is not the same as passed through."""
    import inspect
    from src.asr import asr_factory
    assert "hub=" in inspect.getsource(asr_factory)


def test_the_download_cannot_wait_forever():
    """
    The client retries a failing download indefinitely. An hour of retrying is
    indistinguishable from a hang, and there was nothing to distinguish it with.
    """
    import inspect
    from src import app
    body = inspect.getsource(app.main)
    assert "MODEL_LOAD_LIMIT" in body
    assert "did not finish downloading" in body


def test_a_fatal_startup_error_takes_the_window_with_it():
    """
    By the time the model load can fail there is a splash on screen and no
    event loop running behind it, so it accepts no clicks and cannot be
    closed. The alert on top of it then reads as a dialog that will not go
    away -- reported exactly that way.
    """
    import inspect
    from src import app
    body = inspect.getsource(app._fatal)
    assert "window.destroy()" in body
    assert body.index("window.destroy()") < body.index("display alert"), \
        "the ghost window must go before the alert appears over it"

    startup = inspect.getsource(app.main)
    # To the end of the call, not a fixed number of characters -- the message
    # text is longer than any window I would have guessed.
    for call in ("did not finish downloading", "Could not load the speech models"):
        section = startup.split(call, 1)[1].split("return", 1)[0]
        assert "window=root" in section, f"{call} must dismiss the splash"


def test_a_dialog_never_gets_a_wall_of_text():
    """
    Reported: the alert was wider than the screen and had to be dismissed with
    the keyboard because its OK button was off the edge.

    A funasr load failure is several thousand characters -- the whole model
    registry plus every optional import that did not resolve -- and
    `display alert` sizes itself to fit whatever it is given.
    """
    import inspect
    from src import app
    body = inspect.getsource(app._fatal)
    assert "len(body) > 300" in body
    assert "splitlines()[0]" in body, "one line, not the whole traceback"
    assert "Full details:" in body, "and say where the rest is"
