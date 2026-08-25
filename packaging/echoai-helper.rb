# Homebrew formula for EchoAI Helper.
#
# Publish by putting this in a tap repo (homebrew-echoai) so users can:
#
#     brew tap colakang/echoai
#     brew install echoai-helper
#
# Homebrew is offered alongside `uv tool install` because it is the package
# manager most Mac users already have, and because it can pull BlackHole in as
# a dependency -- removing the one step of setup that needs a password.
class EchoaiHelper < Formula
  include Language::Python::Virtualenv

  desc "Real-time meeting transcription and interview assistance, on-device"
  homepage "https://github.com/colakang/echoai_helper"
  url "https://github.com/colakang/echoai_helper/archive/refs/tags/v1.2.0.tar.gz"
  license "MIT"

  # Pinned, not merely preferred: the vendored custom_speech_recognition
  # imports aifc and audioop, both removed from the stdlib in 3.13.
  depends_on "python@3.12"
  # customtkinter needs Tk, which Homebrew's python omits by default.
  depends_on "python-tk@3.12"
  depends_on "ffmpeg"

  # The virtual audio device. Recording the far end of a call is impossible
  # without it, and it is a cask so it installs the driver properly.
  depends_on cask: "blackhole-2ch"

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      Finish setting up audio routing (asks for your password once):
        echoai-helper setup

      Add a double-clickable icon to Launchpad:
        echoai-helper install-launcher

      Speech models (~1.5GB) download on first run.

      When a meeting is over, put your audio output back:
        echoai-helper setup --restore
    EOS
  end

  test do
    assert_match "echoai", shell_output("#{bin}/echoai-helper version")
  end
end
