#!/usr/bin/env python3
"""
scripts/calibrate_vad.py — pick VAD thresholds from a real recording.

min_silence_ms is the dominant accuracy/latency knob in the pipeline: too
short splits sentences and denies the model the context it needs to
punctuate; too long merges separate turns and delays the "sentence finished"
signal that triggers LLM cleanup.

It cannot be set from synthesised speech. `say` produces near-continuous
audio whose longest pause is ~0.3s, where a real speaker pauses 0.5-2s
between sentences. Record 30-60s of ordinary talking -- ideally the actual
meeting or interview setting, with its real background noise -- and run:

    .venv/bin/python scripts/calibrate_vad.py recording.wav

    # also transcribe each segment, to see what the model receives
    .venv/bin/python scripts/calibrate_vad.py recording.wav --transcribe

Any format ffmpeg reads is fine; it is converted to mono 16kHz internally.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.asr.segmenter import (  # noqa: E402
    Event, SegmenterConfig, SpeechSegmenter, SAMPLE_RATE, WINDOW_SAMPLES,
)
from src.asr.vad import VAD  # noqa: E402

VAD_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "asr", "models", "silero_vad.onnx",
)

CANDIDATE_SILENCE_MS = [200, 300, 400, 500, 700, 1000, 1500]


def load_audio(path: str) -> np.ndarray:
    """Decode anything ffmpeg understands into mono float32 @16kHz."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
             "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", wav],
            check=True,
        )
        data, rate = sf.read(wav, dtype="float32")
    assert rate == SAMPLE_RATE
    return data


def probability_profile(audio: np.ndarray) -> np.ndarray:
    vad = VAD(VAD_MODEL)
    return np.array([
        float(vad.process_chunk(audio[i:i + WINDOW_SAMPLES]))
        for i in range(0, len(audio) - WINDOW_SAMPLES, WINDOW_SAMPLES)
    ])


def describe_pauses(probabilities: np.ndarray, threshold: float) -> None:
    """Where the natural boundaries actually are in this recording."""
    below = probabilities < threshold
    runs, current = [], 0
    for quiet in below:
        if quiet:
            current += 1
        elif current:
            runs.append(current * 100)
            current = 0
    if current:
        runs.append(current * 100)

    print(f"  speech probability: median {np.median(probabilities):.2f}  "
          f"above {threshold}: {(~below).mean() * 100:.0f}% of windows")
    if not runs:
        print("  no pauses at all — nothing to segment on")
        return

    runs.sort()
    print(f"  {len(runs)} pauses, longest {max(runs)}ms, median {int(np.median(runs))}ms")
    buckets = [(200, 300), (300, 500), (500, 800), (800, 1500), (1500, 10 ** 9)]
    for low, high in buckets:
        count = sum(1 for r in runs if low <= r < high)
        if count:
            label = f"{low}-{high}ms" if high < 10 ** 9 else f">{low}ms"
            print(f"    {label:>12}: {count}")


def segment_with(audio: np.ndarray, config: SegmenterConfig):
    vad = VAD(VAD_MODEL)
    segmenter = SpeechSegmenter(vad, config)
    segments = []
    step = int(0.6 * SAMPLE_RATE)      # the recorder's real chunk size
    for i in range(0, len(audio), step):
        for event, segment in segmenter.process(audio[i:i + step]):
            if event is Event.SPEECH_END:
                segments.append(segment)
    tail = segmenter.flush()
    if tail is not None:
        segments.append(tail)
    return segments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", help="a real recording of ordinary speech")
    parser.add_argument("--transcribe", action="store_true",
                        help="transcribe each segment (loads FunASR)")
    parser.add_argument("--speech-threshold", type=float, default=0.6)
    parser.add_argument("--silence-threshold", type=float, default=0.35)
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"[FATAL] no such file: {args.audio}")
        return 1

    audio = load_audio(args.audio)
    duration = len(audio) / SAMPLE_RATE
    print("=" * 70)
    print(f"{os.path.basename(args.audio)} — {duration:.1f}s")
    print("=" * 70)

    describe_pauses(probability_profile(audio), args.silence_threshold)
    print()

    print("=" * 70)
    print("SEGMENTATION vs min_silence_ms")
    print("=" * 70)
    print(f"  {'min_silence':>12} {'segments':>9} {'median':>8} {'longest':>8}   capped")
    results = {}
    for ms in CANDIDATE_SILENCE_MS:
        config = SegmenterConfig(
            speech_threshold=args.speech_threshold,
            silence_threshold=args.silence_threshold,
            min_silence_ms=ms,
        )
        segments = segment_with(audio, config)
        results[ms] = segments
        if not segments:
            print(f"  {ms:>10}ms {0:>9}")
            continue
        durations = [s.duration_s for s in segments]
        # Segments at the ceiling were cut by the cap, not by a real pause:
        # a high count means this threshold is not finding boundaries.
        capped = sum(1 for d in durations if d >= config.max_segment_s - 0.5)
        print(f"  {ms:>10}ms {len(segments):>9} {np.median(durations):>7.1f}s "
              f"{max(durations):>7.1f}s   {capped}")

    print()
    print("  A good setting gives segments of roughly one sentence (2-10s)")
    print("  with no capped ones. Many capped segments means the threshold is")
    print("  too long and no natural pause is ever reaching it.")

    if args.transcribe:
        print()
        print("=" * 70)
        print("TRANSCRIPTS")
        print("=" * 70)
        from src.TranscriberModels import get_model
        model = get_model(use_api=False)
        for ms in (300, 500, 700):
            segments = results.get(ms, [])
            print(f"\n--- min_silence={ms}ms, {len(segments)} segments ---")
            for index, segment in enumerate(segments[:8]):
                text = model.get_transcription_np(segment.audio).text
                print(f"  [{index}] {segment.duration_s:4.1f}s  {text!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


def _require_ffmpeg() -> bool:
    """
    Checked here rather than at app startup, where it used to be.

    Only this file decodes an audio file, so only this file needs ffmpeg. The
    app itself goes sounddevice -> numpy -> FunASR and never calls it -- but it
    refused to start without it, which cost somebody an install: no Homebrew
    meant no ffmpeg, and the app exited before drawing a window.
    """
    if shutil.which("ffmpeg"):
        return True
    print("This needs ffmpeg, which decodes the recording.")
    print("  brew install ffmpeg")
    print("  or https://ffmpeg.org/download.html")
    return False
