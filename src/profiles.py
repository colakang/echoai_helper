"""
src/profiles.py

Named bundles of the transcription settings that actually matter.

The two supported uses pull in opposite directions, and the difference is not
one knob but a consistent set of them:

  Meeting notes   accuracy and completeness; nobody is waiting on the text.
                  Transcribe once per pause, wait longer to be sure a pause
                  is real, drop sub-second backchannel.

  Live interview  the words have to appear while the other person is still
                  talking, or the prompt arrives after the moment has passed.
                  Transcribe continuously, cut sooner, keep short replies.

Exposing these as presets rather than as five separate spinboxes keeps the UI
honest: the individual values interact, and a user who lowers min_silence_ms
without also turning on live partials gets the worst of both.
"""

from dataclasses import dataclass
from typing import Dict, List

from .asr.segmenter import SegmenterConfig
from .config import AudioConfig


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    description: str

    # Re-transcribe the utterance in progress on every chunk. Roughly triples
    # the number of model calls, and is the only way text appears before the
    # speaker stops.
    live_partials: bool

    # How much silence ends an utterance. The dominant knob: shorter reacts
    # faster but splits sentences, denying the model the context it needs to
    # punctuate and to disambiguate.
    min_silence_ms: int

    # Utterances shorter than this are folded into the next one rather than
    # standing alone. Kept high for meetings because SenseVoice's language
    # detection is unreliable on very short audio -- sub-second Cantonese came
    # back as Japanese -- and because backchannel is noise in a transcript.
    min_speech_ms: int

    def segmenter_config(self) -> SegmenterConfig:
        return SegmenterConfig(
            min_silence_ms=self.min_silence_ms,
            min_speech_ms=self.min_speech_ms,
        )


MEETING = Profile(
    key="meeting",
    label="Meeting notes",
    description="Accurate transcript, one pass per pause. No live text.",
    live_partials=False,
    min_silence_ms=700,
    min_speech_ms=700,
)

INTERVIEW = Profile(
    key="interview",
    label="Live interview",
    description="Text as you speak, cut sooner. ~3x the model calls.",
    live_partials=True,
    min_silence_ms=450,
    min_speech_ms=300,
)

PROFILES: Dict[str, Profile] = {p.key: p for p in (MEETING, INTERVIEW)}
DEFAULT_PROFILE = MEETING


def labels() -> List[str]:
    return [p.label for p in PROFILES.values()]


def by_label(label: str) -> Profile:
    for profile in PROFILES.values():
        if profile.label == label:
            return profile
    return DEFAULT_PROFILE


def by_key(key: str) -> Profile:
    return PROFILES.get(key, DEFAULT_PROFILE)


def apply(profile: Profile, transcriber=None) -> None:
    """
    Make `profile` current.

    Settings the transcriber reads per call go through AudioConfig; the
    segmenters hold their config, so they are updated in place when a
    transcriber is supplied.
    """
    AudioConfig.set_live_partials(profile.live_partials)
    AudioConfig.set_profile(profile.key)
    if transcriber is not None:
        transcriber.apply_segmenter_config(profile.segmenter_config())
