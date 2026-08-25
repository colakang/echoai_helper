# 🎙️ EchoAI Helper

[![PyPI](https://img.shields.io/pypi/v/echoai-helper.svg)](https://pypi.org/project/echoai-helper/)
[![GitHub Stars](https://img.shields.io/github/stars/colakang/echoai_helper?style=social)](https://github.com/colakang/echoai_helper/stargazers)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![macOS](https://img.shields.io/badge/macOS-13%2B-lightgrey.svg)](#-install)
[![Windows](https://img.shields.io/badge/Windows-10%2B-lightgrey.svg)](#windows)

Real-time meeting transcription and interview assistance, running on your own
machine. It records both sides of a conversation — your microphone and whatever
the meeting app is playing — transcribes them as they happen, labels who is
speaking, and exports readable notes.

Speech recognition is local. Audio never leaves the machine unless you ask a
language model to clean up the finished transcript.

<p>
<a href="https://www.producthunt.com/posts/echoai-interview-copilot?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-echoai&#0045;interview&#0045;copilot" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=601490&theme=light" alt="EchoAI&#0032;Interview&#0032;Copilot&#0032; - Real&#0045;time&#0032;conversation&#0032;with&#0032;LLM&#0032;responses | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
</p>

<p align="center">
<img width="800" alt="EchoAI Helper Interface" src="https://github.com/colakang/echoai_helper/raw/main/docs/images/ui.png">
</p>

---

## ✨ What it does

- **Local speech recognition** — FunASR / SenseVoice, on-device, GPU-accelerated
  where one is available (Metal on Apple Silicon, CUDA on Windows)
- **Automatic language detection** — Mandarin, Cantonese, English, Japanese and
  Korean, switching per utterance, including mid-sentence code-switching
- **Both sides of the call** — your microphone and the far end, on separate tracks
- **Speaker labelling** — voices on the far-end track are told apart, with the
  number of people configurable when you know it
- **Pause-based segmentation** — sentences are cut where people actually pause,
  not on a fixed timer, so the model sees whole utterances
- **Crash-safe recording** — every settled sentence is written to disk as it is
  produced; a crash costs the last line, not the meeting
- **LLM cleanup on export** — an optional pass that fixes mis-heard words using
  the surrounding conversation, through an API key or through a coding-agent CLI
  on a subscription you already pay for
- **Markdown and JSON export** — Markdown to read, JSON as a complete record
- **Pause your own track** — muting yourself in the meeting app does not reach
  this one, and pausing roughly halves the model's work
- **Live reply suggestions** — for interviews, where a prompt is wanted while the
  other person is still talking

## 💡 Two modes

| | Meeting notes | Live interview |
|---|---|---|
| Text appears | at each pause | as you speak (~0.6s) |
| Model calls | one per sentence | roughly three times as many |
| Best for | an accurate record | a prompt you can act on |

Switch in the app; the several settings that differ move together.

## 🎬 Demo

https://github.com/user-attachments/assets/0d627e4a-960b-4628-8bbc-8d892f02cfd1

---

## ⚡ Install

### macOS

```bash
uv tool install echoai-helper     # uv brings its own Python 3.12
echoai-helper setup               # audio routing — one password prompt
echoai-helper install-launcher    # adds an icon to Launchpad
```

Launch from Launchpad, or run `echoai-helper`.

**Apple Silicon.** Transcription runs on Metal and keeps up comfortably.
**Intel Macs install and run, but see [known limitations](#-known-limitations)** —
there is no Metal path, so they fall back to the CPU.

`setup` installs a virtual audio device and builds the Multi-Output that lets
you hear a meeting while it is being recorded. macOS asks for a password once,
because that installs an audio driver — nothing else needs a privilege, and
Audio MIDI Setup is not involved.

The app takes the audio output while it runs and gives it back silently when it
quits, including after a crash. `echoai-helper setup --restore` does it by hand;
`--status` shows what routing is in place.

<details>
<summary>Prefer Homebrew?</summary>

A formula is in [`packaging/`](packaging/echoai-helper.rb) for a tap. Homebrew
can declare the virtual audio device as a dependency, which removes the one step
that needs a password.
</details>

### Windows

```powershell
uv tool install echoai-helper
echoai-helper
```

No audio setup step: Windows exposes WASAPI loopback directly, so the far end of
a call is capturable without a virtual device. The macOS-only commands (`setup`,
`install-launcher`) report that there is nothing to do. See
[known limitations](#-known-limitations) — this path has not been re-tested since
the segmentation rework.

### What has actually been tested

Stated precisely, because "should work" and "has been run" are different
claims and the difference matters when you are deciding whether to install it.

| | Install | Live meeting | Notes |
|---|---|---|---|
| **M4 Mac mini, 16GB, macOS 26.3, Python 3.12** | ✅ | ✅ | The development machine. Every measurement in this README comes from it. |
| **Other Apple Silicon** (M1–M3) | — | — | Same wheels, same Metal path. Expected to work; not run. |
| **Intel Mac** | ✅ resolves | ❌ | Installs — `uv` falls back to torch 2.2.2 — but no Metal, so CPU only. See [known limitations](#-known-limitations). |
| **Windows 10/11** | ✅ resolves | ❌ | Dependencies resolve. The capture path has not been run since the segmentation rework. |

"Live meeting" means real calls, in Mandarin, Cantonese and English, with the
transcript exported afterwards — not a smoke test. The longest was 84 minutes
and 1311 lines.

The Mac mini has no built-in microphone, so the microphone track has only been
exercised through a Bluetooth headset and a wired input. That is also where the
[Bluetooth limitation](#-known-limitations) below was found.

If you run it somewhere not on this list, an issue saying so — working or
not — is genuinely useful.

### First run

Speech models (~1.5GB) download on first launch. Nothing else is needed.

<details>
<summary>Why Python 3.12 specifically</summary>

Not caution. The vendored `src/custom_speech_recognition` imports `aifc` and
`audioop` at module level, and **both were removed from the standard library in
Python 3.13**. `uv` installs a suitable interpreter itself, and its 3.12 build
ships tkinter, so there is no separate Tk step.
</details>

---

## 🎯 Using it

1. Open the app. If audio routing is not in place, it offers to finish it.
2. Pick a mode, and set the number of people if you know it.
3. Hold your meeting. Nothing needs touching.
4. **Export** — one dialog covers format, cleanup, which model to clean with,
   and merging over-split speakers.

Cleanup runs in the background with a progress bar and an estimate, and can be
stopped: whatever finished is kept.

### Choosing a cleanup backend

| | Cost | Speed (measured) |
|---|---|---|
| **API** (`conf.yaml`) | per token | 5–8s per batch of lines |
| **Claude CLI** | included in a subscription | 20–50s per batch |

Both are offered at export, and the dialog turns the per-batch figure into an
estimate for the transcript in hand. The live reply suggestions always use the
configured API — a CLI takes seconds per answer, which is too late to be useful
while someone is still talking.

### Past recordings

```bash
echoai-helper sessions              # list them
echoai-helper sessions --export 0   # export one again
echoai-helper sessions --delete 0
```

Re-exporting is the point: a different format, another pass of cleanup, a
different number of speakers, without re-recording anything. An unfinished
session from the last 12 hours is offered on the next launch.

---

## ⚙️ Configuration

Model settings live in `conf.yaml`; everything else is in the app. Installed
from a wheel the shipped copy sits inside the package, so make yourself an
editable one:

```bash
echoai-helper config     # writes conf.yaml where you can reach it, and prints the path
```

That copy overrides the defaults. You will not usually need it — both values
below already ship as `auto`.

```yaml
FunASR:
  model_name: "iic/SenseVoiceSmall"
  device: "auto"        # cuda, then mps, then cpu
  language: "auto"      # zh, en, yue, ja, ko

LLM:
  provider: "openai"    # openai | litellm | cli
```

Both `auto` values are load-bearing rather than lazy defaults:

- **`device: "auto"`** — measured on an M4, dual-track real-time factor is 2.04
  on cpu (falling behind twice over) against 0.35 on Metal. Landing on cpu by
  accident means transcription that cannot keep up. Naming a device explicitly is
  also wrong on every machine that does not have it, and this file travels.
- **`language: "auto"`** — pinning a language does not bias the model, it forces
  the syllables onto words of that language. A Cantonese call transcribed with
  `language: "en"` comes back as fluent nonsense.

An OpenAI key goes in `.env` (see `.env.example`), or in a file called `.llm`
holding nothing else. Both are gitignored.

---

## 🔍 Troubleshooting

```bash
echoai-helper check-audio           # what is being captured, and from where
echoai-helper setup --status        # what routing is in place
```

**Nothing from the far end (macOS).** The meeting app has its own audio settings
and remembers them. Set its speaker to `EchoAI Meeting`.

**Nothing from the microphone over Bluetooth.** A Bluetooth headset can only send
its microphone to one device; if it is on a phone call, the Mac gets silence. The
microphone track now re-checks every few seconds that it is still bound to the
device actually in use, and moves itself when that changes — which covers the
case where the device list is renumbered underneath it. A headset that stays
connected and simply stops delivering audio is not covered, because nothing can
tell that apart from a hardware mute: both send digital silence. A wired or USB
microphone avoids the whole area.

**You are muted in the meeting but still being transcribed.** Expected — muting
in Zoom or WeChat silences your outgoing audio, not this app's own input stream.
Use **Pause Mic**. It resets to off every launch, deliberately: a pause that
survived a restart would look like recording and not be.

**More speakers than people.** Voice prints drift with volume and connection
quality, so one person can end up split across several labels. Set the number of
people before the meeting, or merge them at export — the export carries the voice
prints, so this works after the fact.

---

## 🛠️ Development

```bash
git clone https://github.com/colakang/echoai_helper.git
cd echoai_helper
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-macos.txt   # or requirements.txt
.venv/bin/python -m pytest tests/ -q
```

[`docs/macos-audio-setup.md`](docs/macos-audio-setup.md) covers the audio
routing, what has been measured, and where the sharp edges are.

---

## 📝 Known limitations

- **A microphone that goes quiet without going away is not detected.** The track
  now follows the device it should be on, which covers a device list renumbered
  underneath it — the most likely cause of what was seen on a real call. A
  headset that stays present and silently stops delivering is not covered, and
  deliberately so: it is indistinguishable from a hardware mute, and an alert
  that fires on a muted microphone is one users learn to ignore. Reports with
  details are what would settle it.
- **Speaker labelling is tuned against a clean two-party recording** and
  over-splits on group calls over a lossy connection. Merging at export is the
  workaround, not the cure.
- **The Windows capture path is untested** since the segmentation rework. Its
  detector threshold was lowered to pass silence through, because segmentation
  now happens on pauses and a recogniser that only reports speech never delivers
  them — reasoned, not verified. Reports welcome.
- **Intel Macs fall back to the CPU.** PyTorch has shipped no Intel-Mac build
  since 2.2.2, and there is no Metal path on Intel regardless; `uv` resolves to
  that older torch and installs cleanly. But an M4 already measures a dual-track
  real-time factor of 2.04 on CPU — falling behind twice over — and an Intel CPU
  is slower again. Expect transcription not to keep up with a live meeting.
  Untested: reasoned from the wheel availability and the CPU measurement.
- **macOS audio routing is a shared setting.** Selecting the Multi-Output changes
  the output for every app, and macOS sometimes moves it back after sleep.
  Checked at every launch.

## 🤝 Contributing

Pull requests welcome; for anything substantial, open an issue first. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## 🙌 Credits

- [FunASR](https://github.com/modelscope/FunASR) — speech recognition and speaker
  embeddings
- [silero-vad](https://github.com/snakers4/silero-vad) — voice activity detection
- [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) — the
  LocalAgreement idea behind stable partial transcripts
- [BlackHole](https://existential.audio/blackhole/) — virtual audio device on macOS
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) — interface
- [Ecoute](https://github.com/SevaSk/ecoute) — the original inspiration
- [@zixing0131](https://github.com/zixing0131) — core audio processing

## 📞 Contact

[echo365.ai](https://www.echo365.ai) · [Issues](https://github.com/colakang/echoai_helper/issues)

## 📄 License

MIT — see [LICENSE](LICENSE).
