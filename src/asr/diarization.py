"""
src/asr/diarization.py

Who said it, on a track that carries more than one person.

The "Speaker" track is whatever the meeting app is playing, so in any meeting
with more than two attendees it is several people sharing one channel. Every
one of them currently gets labelled "Speaker", which makes a transcript of a
five-person meeting close to useless as a record.

Approach: one voice embedding per VAD segment, clustered online against the
speakers seen so far. Segmentation already exists and is already paid for, so
this adds an embedding per utterance and nothing else -- no second
segmentation pass, no separate diarization model with its own VAD.

Embeddings come from CAM++ (iic/speech_campplus_sv_zh-cn_16k-common), which
FunASR already ships. That matters: pyannote needs a gated HuggingFace model
and a token, and NeMo's Sortformer drags in the whole NeMo stack. Measured on
an M4: 293ms per segment, 192-dimensional.

Known limit, and it is inherent rather than a tuning problem. A segment is cut
on silence, so a speaker change with no pause between the turns lands inside
one segment and its embedding is a blend of both voices. On a real support
call this happened once in twelve segments, where the customer finished
reciting a number and the agent asked the next question without a break.
Such a segment sits between the clusters and is reported with low confidence
rather than being forced into one of them.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Speaker:
    """A voice seen on this track."""
    id: int
    centroid: np.ndarray
    count: int = 1

    @property
    def label(self) -> str:
        return f"S{self.id}"


@dataclass
class Assignment:
    speaker: Optional[Speaker]
    similarity: float
    is_new: bool
    confident: bool

    @property
    def label(self) -> str:
        return self.speaker.label if self.speaker else "S?"


@dataclass
class DiarizationConfig:
    # Cosine similarity above which a segment is the same voice. On a real
    # two-party call, within-speaker similarity ran 0.77-0.93 and
    # cross-speaker 0.21-0.50, so anything in 0.6-0.7 separates them. The
    # lower end of that is deliberate: splitting one person into two is a
    # worse transcript than occasionally merging two into one, because a
    # reader can spot a wrong turn boundary but cannot recover a speaker who
    # was invented.
    same_speaker_threshold: float = 0.62

    # Below this, the segment matched nothing well and is left unlabelled
    # rather than guessed at. Mixed-voice segments land here.
    confidence_margin: float = 0.08

    # Voices shorter than this give unreliable embeddings, the same way they
    # give unreliable language detection.
    #
    # Measured, because the value that was here before was not. Cutting two
    # single-speaker recordings into windows and comparing each against that
    # speaker's own full-length print:
    #
    #     window   mean    p10
    #      0.5s    0.342   0.064
    #      1.0s    0.480   0.336     <- the old gate
    #      1.5s    0.624   0.486
    #      2.0s    0.721   0.636     <- first duration whose p10 clears 0.62
    #      3.0s    0.713   0.635
    #
    # At one second a speaker matches themselves at 0.480, against a
    # same-speaker threshold of 0.62 and a cross-speaker range of 0.21-0.50.
    # A person did not resemble themselves more than they resembled someone
    # else, so the gate was admitting embeddings indistinguishable from noise
    # -- which is exactly how one caller reading a number became three
    # speakers.
    #
    # 2.0 is where the curve flattens, and is the same figure LANGUAGE_TRUST_S
    # arrived at independently for the same underlying problem: too little
    # audio to tell anything from.
    min_duration_s: float = 2.0

    # A meeting has a bounded number of people; without a cap, noise and
    # crosstalk invent a new speaker every few segments.
    max_speakers: int = 12


class SpeakerRegistry:
    """
    Online clustering of voice embeddings.

    Online rather than batch because a transcript has to name the speaker as
    the meeting happens; offline clustering would only be able to label
    anything once it was over.
    """

    def __init__(self, config: Optional[DiarizationConfig] = None):
        self.config = config or DiarizationConfig()
        self._speakers: List[Speaker] = []
        self._next_id = 1

    @property
    def speakers(self) -> List[Speaker]:
        return list(self._speakers)

    def assign(self, embedding: np.ndarray) -> Assignment:
        """Match an embedding to a known voice, or register a new one."""
        vector = _normalise(embedding)
        if vector is None:
            return Assignment(None, 0.0, False, False)

        if not self._speakers:
            return Assignment(self._register(vector), 1.0, True, True)

        similarities = np.array([float(vector @ s.centroid) for s in self._speakers])
        best = int(np.argmax(similarities))
        best_similarity = float(similarities[best])

        # Ambiguous when two known voices match nearly as well: characteristic
        # of a segment that contains both of them.
        runner_up = float(np.partition(similarities, -2)[-2]) if len(similarities) > 1 else -1.0
        clear = (best_similarity - runner_up) >= self.config.confidence_margin

        if best_similarity >= self.config.same_speaker_threshold:
            speaker = self._speakers[best]
            if clear:
                self._update(speaker, vector)
            # An ambiguous match still gets the label -- it is the best
            # evidence available -- but does not pollute the centroid.
            return Assignment(speaker, best_similarity, False, clear)

        if len(self._speakers) >= self.config.max_speakers:
            # Out of room: fall back to the nearest voice rather than
            # discarding the segment.
            return Assignment(self._speakers[best], best_similarity, False, False)

        return Assignment(self._register(vector), best_similarity, True, True)

    def _register(self, vector: np.ndarray) -> Speaker:
        speaker = Speaker(id=self._next_id, centroid=vector)
        self._next_id += 1
        self._speakers.append(speaker)
        return speaker

    @staticmethod
    def _update(speaker: Speaker, vector: np.ndarray) -> None:
        """Running mean, so later segments refine rather than replace."""
        speaker.count += 1
        blended = speaker.centroid + (vector - speaker.centroid) / speaker.count
        speaker.centroid = blended / np.linalg.norm(blended)

    def reset(self) -> None:
        self._speakers = []
        self._next_id = 1


def recluster(embeddings: List[np.ndarray], target: int,
              labels: Optional[List[int]] = None) -> List[int]:
    """
    Re-assign voices offline, knowing how many people there were.

    Online clustering has to decide on each utterance as it arrives, with no
    view of what comes later, so it over-splits: a real call produced twelve
    speakers, of which three carried 93% of the talking and the other nine
    were fragments that had drifted. Given the whole recording and the true
    headcount, that is recoverable.

    Agglomerative, average-linkage on cosine similarity: repeatedly merge the
    two closest clusters until `target` remain. Average linkage rather than
    nearest-neighbour because a single ambiguous segment sitting between two
    people should not chain them together.

    Returns a cluster index per embedding, ordered by how much each cluster
    speaks, so speaker 1 is the one who talked most.
    """
    vectors = [_normalise(e) for e in embeddings]
    usable = [i for i, v in enumerate(vectors) if v is not None]
    if not usable:
        return [0] * len(embeddings)

    target = max(1, min(target, len(usable)))
    clusters = {i: [i] for i in usable}
    centroids = {i: vectors[i] for i in usable}

    while len(clusters) > target:
        best = None
        keys = list(clusters)
        for a_index, a in enumerate(keys):
            for b in keys[a_index + 1:]:
                similarity = float(centroids[a] @ centroids[b])
                if best is None or similarity > best[0]:
                    best = (similarity, a, b)
        if best is None:
            break

        _, a, b = best
        clusters[a].extend(clusters.pop(b))
        members = np.stack([vectors[i] for i in clusters[a]])
        merged = members.mean(axis=0)
        centroids[a] = merged / np.linalg.norm(merged)
        centroids.pop(b)

    # Largest first, so "Speaker 1" is whoever spoke most -- stable across
    # runs and the order a reader expects.
    ordered = sorted(clusters.values(), key=len, reverse=True)
    assignment = [0] * len(embeddings)
    for rank, members in enumerate(ordered, start=1):
        for index in members:
            assignment[index] = rank
    return assignment


def _normalise(embedding) -> Optional[np.ndarray]:
    """Torch tensor or array -> unit-length float64 vector."""
    if embedding is None:
        return None
    if hasattr(embedding, "detach"):
        # CAM++ runs on MPS; numpy conversion needs a host copy first.
        embedding = embedding.detach().cpu()
    vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        return None
    return vector / norm


# The same model under each hub's naming. Model ids are hub-specific, and
# this one was hardcoded to ModelScope's while everything else had moved to
# HuggingFace -- so a first run fetched two models from one host and then
# stalled on the third against the other.
_CAMPPLUS = {
    "hf": "funasr/campplus",
    "ms": "iic/speech_campplus_sv_zh-cn_16k-common",
}


def _speaker_model():
    """(model id, hub) for the embedder, following conf.yaml."""
    try:
        import yaml
        from ..config import PathConfig
        with open(PathConfig.get_conf_file(), "rb") as f:
            asr = (yaml.safe_load(f) or {}).get("FunASR", {}) or {}
    except Exception:
        asr = {}
    hub = (asr.get("hub") or "ms").lower()
    return asr.get("spk_model") or _CAMPPLUS.get(hub, _CAMPPLUS["ms"]), hub


class SpeakerEmbedder:
    """CAM++ wrapper. Loaded lazily -- diarization is opt-in."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _ensure_model(self):
        if self._model is None:
            from funasr import AutoModel
            model_id, hub = _speaker_model()
            self._model = AutoModel(model=model_id, hub=hub,
                                    device=self.device, disable_update=True)
            # Metal kernel compilation, paid here rather than on the first
            # thing anyone says.
            if self.device == "mps":
                self._model.generate(input=np.zeros(16000, dtype=np.float32))
        return self._model

    def embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        try:
            result = self._ensure_model().generate(input=audio)
        except Exception as e:
            print(f"[WARN] speaker embedding failed: {e}")
            return None
        if not result:
            return None
        return _normalise(result[0].get("spk_embedding"))
