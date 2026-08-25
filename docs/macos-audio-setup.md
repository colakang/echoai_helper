# macOS Setup

EchoAI Helper was written for Windows. This document covers what is different
on macOS (Apple Silicon) and why.

## Why macOS needs its own path

The Windows build captures system output through **WASAPI loopback**, provided
by `PyAudioWPatch` — a Windows-only fork of PyAudio. It has no macOS
distribution at all; `pip install PyAudioWPatch` fails outright on darwin.

macOS has no equivalent for a plain input stream: an app cannot ask CoreAudio
to "record what the speakers are playing". The standard workaround is a
**virtual audio device** — audio is routed out through it, and it appears as an
*input* device the app can record from.

## 1. Python 3.12

Not 3.13 or newer. The vendored `src/custom_speech_recognition` imports `aifc`
and `audioop` at module level, and **both were removed from the standard
library in Python 3.13**.

```bash
brew install python@3.12
brew install python-tk@3.12   # customtkinter needs Tk; Homebrew python omits it

uv venv --python 3.12 .venv   # or: python3.12 -m venv .venv
uv pip install --python .venv/bin/python -r requirements-macos.txt
```

Use `requirements-macos.txt`, not `requirements.txt` — the latter pins
`PyAudioWPatch` and a CUDA 11.7 wheel index, neither of which exists here.

## 2. Automatic setup

Everything below can be done in one command:

```bash
.venv/bin/python scripts/setup_audio.py      # or: echoai-helper setup
```

It installs BlackHole if it is missing, builds the Multi-Output device, and
selects it. macOS asks for a password only if the driver still has to be
installed -- BlackHole is a system audio driver, so that prompt cannot be
avoided. Nothing else needs a privilege, and Audio MIDI Setup is not involved.

```bash
scripts/setup_audio.py --status    # report, change nothing
scripts/setup_audio.py --restore   # point audio back at your speakers
```

Worth running `--restore` when the meeting is over: a machine left pointed at
a Multi-Output is a confusing thing to walk back to, not least because the
volume keys do not work on one.

The rest of this section describes doing it by hand.

## 2b. Virtual audio device (BlackHole)

Needed to capture the far end of a meeting. Installing an audio driver
requires an admin password:

```bash
brew install --cask blackhole-2ch
```

Verify — the installer usually restarts CoreAudio itself, so check before
doing anything else:

```bash
.venv/bin/python scripts/check_audio.py
```

BlackHole should appear as an input device with 2 channels. If it does not,
reload the CoreAudio daemon (no logout or reboot needed — this only restarts
the audio daemon, muting all system audio for a second or two):

```bash
sudo killall coreaudiod
```

## 3. Route meeting audio through it

BlackHole alone is silent-in, silent-out — audio must be sent to it. To both
*hear* the meeting and *capture* it, create a Multi-Output Device:

1. Open **Audio MIDI Setup** (`/Applications/Utilities`)
2. Click **+** (bottom left) → **Create Multi-Output Device**
3. Tick both **BlackHole 2ch** and your real speakers
4. Set the real speakers as **Primary Device** (clock master)
5. Tick **Drift Correction** on BlackHole
6. Set that Multi-Output Device as the system output (Sound settings, or
   option-click the menu-bar volume icon)

Meeting audio now reaches your ears *and* BlackHole simultaneously.

> Per-app alternative: in Zoom/Meet/Teams, set **speaker** to BlackHole 2ch
> directly. Simpler, but then you cannot hear the meeting — only useful when
> testing with headphones on a second device.

## 4. Microphone

**A Mac mini has no built-in microphone.** `scripts/check_audio.py` reports
zero input devices on a bare machine.

This disables the `You` track only — your own speech is not transcribed. The
`Speaker` track (the far end) works fine without it, which covers live
interview and meeting-response use. Attach any USB microphone or headset to
enable `You`; it is picked up automatically as the default input device.

## 5. Verify

```bash
# enumerate devices, report which tracks are available
.venv/bin/python scripts/check_audio.py

# capture 8 seconds of system audio and transcribe it
.venv/bin/python scripts/check_audio.py --record 8 --source speaker
```

The second command prints the transcript plus a real-time factor (RTF).
RTF < 1.0 means transcription keeps up with speech.

## 6. Use the GPU (`device: "auto"`)

`src/conf.yaml` ships with `device: "auto"`, which resolves to Metal on Apple
Silicon. **Do not pin it to `"cpu"`** — it is not a tuning knob, it is the
difference between keeping up and falling behind. Nor should you pin it to
`"mps"`: this file travels between machines, and a device name is wrong on
every machine that does not have it.

Measured on an M4 Mac mini (16GB), SenseVoiceSmall, one 6.2s phrase in
accumulate mode (10 progressively longer calls, which is what the app actually
does today):

| device | 10 calls | 1 track | 2 tracks |
|--------|---------:|--------:|---------:|
| `cpu`  |    6.33s |    1.02 | **2.04** — falls behind 2x |
| `mps`  |    1.10s |    0.18 | **0.35** — comfortable |

Single-shot RTF on an 11.5s clip is 0.060 (cpu) versus 0.012 (mps) — 5.3x —
with byte-identical transcripts.

Note that inference cost here is dominated by **fixed per-call overhead**, not
by audio length: a 0.6s clip takes 0.069s and an 11.5s clip takes 0.132s. 19x
the audio for 1.9x the time. What matters for throughput is how many times the
model is invoked, not how much audio is re-processed per invocation.

`torch.backends.mps.is_available()` must be True; `torch.cuda.is_available()`
is always False on a Mac and says nothing about whether acceleration is on.

The first Metal inference pays ~1.3s of kernel compilation.
`FunASRTranscriber` burns that at startup so it does not land on the user's
first spoken phrase.

## Known gaps

- **Audio routing is manual.** Steps 2–3 are a one-time setup the user must do
  by hand. A future backend using ScreenCaptureKit (macOS 13+) or a CoreAudio
  process tap (macOS 14.4+) would remove this, at the cost of a PyObjC bridge
  and a screen-recording permission prompt.
- **Switching the system output device** to the Multi-Output Device changes
  audio for every app, and macOS sometimes reverts it after sleep.
