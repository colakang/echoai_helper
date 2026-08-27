

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
