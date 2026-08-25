#!/usr/bin/env python3
"""
scripts/check_audio.py — headless audio + ASR smoke test.

Proves the capture path works before involving the Tk UI, so a failure points
at one layer instead of three.

    .venv/bin/python scripts/check_audio.py            # list devices, diagnose
    .venv/bin/python scripts/check_audio.py --record 8 # capture 8s and transcribe

The --record path exercises exactly the pipeline main.py uses: backend ->
queue -> AudioTranscriber.convert_bytes_to_numpy -> FunASR.
"""

import argparse
import os
import queue
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.audio import get_audio_backend  # noqa: E402


def show_devices() -> None:
    print("=" * 68)
    print("AUDIO DEVICES")
    print("=" * 68)
    try:
        import sounddevice as sd
    except ImportError:
        print("  sounddevice not installed — cannot enumerate.")
        return

    default_in, default_out = sd.default.device
    inputs = 0
    for index, info in enumerate(sd.query_devices()):
        flags = []
        if info["max_input_channels"] > 0:
            flags.append("IN")
            inputs += 1
        if info["max_output_channels"] > 0:
            flags.append("OUT")
        if index == default_in:
            flags.append("default-in")
        if index == default_out:
            flags.append("default-out")
        print(f"  [{index}] {info['name']}")
        print(f"        {'/'.join(flags):24} "
              f"in={info['max_input_channels']} out={info['max_output_channels']} "
              f"sr={int(info['default_samplerate'])}")

    print()
    if inputs == 0:
        print("  !! No input devices at all.")
        print("     A Mac mini has no built-in microphone, and macOS cannot")
        print("     record system output without a virtual audio device.")
        print("     Fix:  brew install --cask blackhole-2ch")
        print("     Then: docs/macos-audio-setup.md")


def describe_sources(backend):
    print("=" * 68)
    print(f"BACKEND: {backend.name}")
    print("=" * 68)

    mic = backend.create_mic_recorder()
    speaker = backend.create_speaker_recorder()

    print(f"  mic     ('You')     : {mic.source if mic else 'UNAVAILABLE'}")
    print(f"  speaker ('Speaker') : {speaker.source if speaker else 'UNAVAILABLE'}")
    print()
    return mic, speaker


def record_and_transcribe(recorder, seconds: float) -> None:
    """Capture from one recorder, then run it through the real ASR path."""
    print("=" * 68)
    print(f"RECORDING {seconds}s from {recorder.source_name} "
          f"({recorder.source.device_name})")
    print("=" * 68)
    print("  Play or say something now...")

    audio_queue = queue.Queue()
    recorder.record_into_queue(audio_queue)
    time.sleep(seconds)
    recorder.stop()

    chunks = []
    while not audio_queue.empty():
        _, data, _ = audio_queue.get()
        chunks.append(data)

    if not chunks:
        print("\n  !! No audio above the energy gate.")
        print("     Either nothing was playing, or audio is not routed to this")
        print("     device. Check docs/macos-audio-setup.md.")
        return

    raw = b"".join(chunks)
    src = recorder.source
    captured_s = len(raw) / (src.SAMPLE_RATE * src.channels * src.SAMPLE_WIDTH)
    print(f"\n  captured {len(chunks)} chunks / {captured_s:.1f}s of non-silent audio")

    # Reuse the app's own conversion so this test covers the real code path.
    from src.AudioTranscriber import AudioTranscriber

    converter = AudioTranscriber.__new__(AudioTranscriber)
    converter.audio_sources = {
        recorder.source_name: {
            "sample_rate": src.SAMPLE_RATE,
            "sample_width": src.SAMPLE_WIDTH,
            "channels": src.channels,
        }
    }
    audio_np = converter.convert_bytes_to_numpy(raw, recorder.source_name)
    print(f"  converted -> {audio_np.shape[0]} samples @16kHz mono float32 "
          f"(peak {np.abs(audio_np).max():.3f})")

    print("\n  loading FunASR (first run downloads the model)...")
    t0 = time.time()
    from src.TranscriberModels import get_model

    model = get_model(use_api=False)
    print(f"  model ready in {time.time() - t0:.1f}s")

    t0 = time.time()
    text = model.get_transcription_np(audio_np).text
    elapsed = time.time() - t0
    rtf = elapsed / captured_s if captured_s else float("nan")

    print()
    print("=" * 68)
    print(f"  TRANSCRIPT: {text!r}")
    print(f"  inference {elapsed:.2f}s for {captured_s:.1f}s audio -> RTF {rtf:.2f}")
    print("=" * 68)


def selftest(recorder) -> None:
    """
    End-to-end loop with no manual audio routing.

    Synthesises a phrase with the built-in `say`, plays it *into* the loopback
    device while recording *from* it, and transcribes the result. Exercises
    capture -> conversion -> ASR in one shot, so it doubles as a regression
    test on a machine with no microphone.
    """
    import shutil
    import subprocess
    import tempfile


    import sounddevice as sd
    import soundfile as sf

    phrase = ("Hello, this is a test of the meeting transcription pipeline. "
              "Can you hear me clearly?")

    if not shutil.which("say"):
        print("[FATAL] `say` not found — selftest needs macOS speech synthesis.")
        return

    tmpdir = tempfile.mkdtemp(prefix="echoai-selftest-")
    aiff = os.path.join(tmpdir, "probe.aiff")
    wav = os.path.join(tmpdir, "probe.wav")

    print("=" * 68)
    print("SELF TEST — play into the loopback and record back out of it")
    print("=" * 68)
    print(f"  phrase: {phrase!r}")

    subprocess.run(["say", "-o", aiff, phrase], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", aiff,
                    "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", wav],
                   check=True)

    data, sample_rate = sf.read(wav, dtype="float32")
    duration = data.shape[0] / sample_rate
    print(f"  generated {duration:.2f}s @ {sample_rate}Hz")

    # Play to the *output* half of the same virtual device we record from.
    out_index = None
    for index, info in enumerate(sd.query_devices()):
        if (info["name"] == recorder.source.device_name
                and info["max_output_channels"] > 0):
            out_index = index
            break
    if out_index is None:
        print(f"[FATAL] {recorder.source.device_name!r} has no output half to "
              "play into.")
        return

    audio_queue = queue.Queue()
    recorder.record_into_queue(audio_queue)
    time.sleep(0.3)  # let the input stream settle before audio starts

    # Play from a separate process.
    #
    # Opening an output stream on the *same* CoreAudio device that already has
    # an input stream open, from within one PortAudio instance, is racy: it
    # intermittently raises kAudioUnitErr_CannotDoInCurrentContext (-10863) and
    # kills the input audio unit a fraction of a second in. Measured 1/3
    # success in-process versus 4/4 across processes.
    #
    # This is a test-harness concern only. In real use the audio arriving at
    # the loopback comes from Zoom or a browser — a different process — so the
    # contention never arises.
    player = subprocess.Popen(
        [sys.executable, "-c",
         "import sys,sounddevice as sd,soundfile as sf;"
         "d,sr=sf.read(sys.argv[1],dtype='float32');"
         "sd.play(d,sr,device=int(sys.argv[2]),blocking=True)",
         wav, str(out_index)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    player.wait(timeout=duration + 15)

    time.sleep(0.5)  # let the tail of the audio land in the queue
    recorder.stop()

    chunks = []
    while not audio_queue.empty():
        _, chunk, _ = audio_queue.get()
        chunks.append(chunk)

    if not chunks:
        print("\n  !! Captured nothing. Audio did not loop back.")
        return

    raw = b"".join(chunks)
    src = recorder.source
    captured_s = len(raw) / (src.SAMPLE_RATE * src.channels * src.SAMPLE_WIDTH)
    print(f"  captured {len(chunks)} chunks / {captured_s:.1f}s")

    _transcribe(raw, recorder, captured_s)


def _transcribe(raw: bytes, recorder, captured_s: float) -> None:
    """Run captured PCM through the app's own conversion + ASR."""
    from src.AudioTranscriber import AudioTranscriber

    src = recorder.source
    converter = AudioTranscriber.__new__(AudioTranscriber)
    converter.audio_sources = {
        recorder.source_name: {
            "sample_rate": src.SAMPLE_RATE,
            "sample_width": src.SAMPLE_WIDTH,
            "channels": src.channels,
        }
    }
    audio_np = converter.convert_bytes_to_numpy(raw, recorder.source_name)
    print(f"  converted -> {audio_np.shape[0]} samples @16kHz mono float32 "
          f"(peak {np.abs(audio_np).max():.3f})")

    print("\n  loading FunASR (first run downloads the model)...")
    t0 = time.time()
    from src.TranscriberModels import get_model

    model = get_model(use_api=False)
    print(f"  model ready in {time.time() - t0:.1f}s")

    t0 = time.time()
    text = model.get_transcription_np(audio_np).text
    elapsed = time.time() - t0
    rtf = elapsed / captured_s if captured_s else float("nan")

    print()
    print("=" * 68)
    print(f"  TRANSCRIPT: {text!r}")
    print(f"  inference {elapsed:.2f}s for {captured_s:.1f}s audio -> RTF {rtf:.2f}")
    print("=" * 68)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=float, metavar="SECONDS",
                        help="capture this many seconds and transcribe")
    parser.add_argument("--selftest", action="store_true",
                        help="play a synthesised phrase into the loopback and "
                             "transcribe it back (no manual routing needed)")
    parser.add_argument("--source", choices=["mic", "speaker"], default="speaker",
                        help="which track to record (default: speaker)")
    args = parser.parse_args()

    show_devices()
    print()

    try:
        backend = get_audio_backend()
    except NotImplementedError as exc:
        print(f"[FATAL] {exc}")
        return 1

    mic, speaker = describe_sources(backend)

    if not args.record and not args.selftest:
        print("  (pass --record SECONDS to capture, or --selftest for a "
              "self-contained loop test)")
        return 0 if (mic or speaker) else 1

    recorder = mic if args.source == "mic" else speaker
    if recorder is None:
        print(f"[FATAL] '{args.source}' track is unavailable — cannot record.")
        return 1

    if args.selftest:
        selftest(recorder)
    else:
        record_and_transcribe(recorder, args.record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
