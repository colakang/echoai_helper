

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
    root = Path(__file__).resolve().parent.parent
    config = yaml.safe_load((root / "conf.yaml").read_text())
    assert config["FunASR"]["device"] == "auto"


def test_capture_dependencies_are_platform_split():
    """PyAudioWPatch has no macOS build and sounddevice is not the Windows
    path; declaring either unconditionally breaks install on the other."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    pyproject = (root / "pyproject.toml").read_text()
    assert "PyAudioWPatch; sys_platform == 'win32'" in pyproject
    assert "sounddevice; sys_platform != 'win32'" in pyproject
