# Contributing

Pull requests are welcome. For anything substantial, open an issue first so we
do not both build the same thing differently.

## Getting set up

```bash
git clone https://github.com/colakang/echoai_helper.git
cd echoai_helper
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m pytest tests/ -q
.venv/bin/python main.py
```

Python 3.12 specifically: the vendored `src/custom_speech_recognition` imports
`aifc` and `audioop` at module level, and both were removed from the standard
library in 3.13.

On macOS, `.venv/bin/python scripts/setup_audio.py` prepares audio routing, and
`--restore` puts it back.

## The most useful thing you can report

**Which machine you ran it on, and whether it worked.** The README has a table
of what has actually been exercised, and it is short: one M4 Mac mini. Two
platforms in that table install correctly and have never had their capture path
run — Windows, and Intel Macs. A report either way is worth more than most code.

Include what the log says. On a GUI launch it goes to
`~/Library/Logs/EchoAI Helper.log`; from a terminal it is on stdout.
`echoai-helper check-audio` prints what is being captured and from where.

## Things worth knowing before you change audio code

Most of the surprises in this project live there, and most of them are not
guessable from the API:

- **PortAudio enumerates devices once**, at `Pa_Initialize`, and never again.
  Inside a running process its device list is frozen at startup — it cannot see
  a headset leave or return. Ask CoreAudio for anything that has to be current.
- **A dead device stops invoking the audio callback entirely.** A muted one
  keeps invoking it and delivers zeroes. That difference is the only reliable
  way to tell them apart; nothing in the signal itself distinguishes them.
- **A stream cannot be reopened after its device disconnects**, even once the
  device is back — `Pa_Terminate` then `Pa_Initialize` is the only recovery, and
  it invalidates *every* open stream in the process.
- **The capture path forwards silence deliberately.** Segmentation happens on
  pauses, so a gate that drops quiet chunks destroys the information the VAD
  needs. There used to be one; removing it took transcript coverage from ~70%
  to ~95% on three recordings.

If you change any of the above, please measure rather than reason. Several
comments in `src/audio/` carry numbers; they are there because the obvious
answer was wrong.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

They run without audio hardware and without downloading a model. Please keep it
that way — a test suite that needs a Bluetooth headset is a test suite nobody
runs.

A test that pins a behaviour should fail if that behaviour is removed. If you
are unsure yours does, break the code on purpose and check.

## Style

Match the surrounding code. Comments explain *why*, especially where the code
looks odd — most of the odd-looking code here is odd for a reason that cost
somebody an afternoon.
