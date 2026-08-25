#src/AudioTranscriber.py

#import whisper
import math
import os
import threading
from datetime import datetime, timedelta

import numpy as np
from scipy import signal

from .config import AudioConfig, SystemConfig
from .asr.hypothesis import HypothesisTracker
from .asr.diarization import SpeakerEmbedder, SpeakerRegistry
from .asr.segmenter import Event, SegmenterConfig, SpeechSegmenter
from .asr.vad import VAD


def _asr_device() -> str:
    """Reuse whatever device the ASR is configured for."""
    try:
        import yaml
        from .config import PathConfig
        with open(f"{PathConfig.get_project_root()}/conf.yaml", "rb") as f:
            config = yaml.safe_load(f)
        name = config.get("ASR_MODEL", "FunASR")
        return config.get(name, {}).get("device", "cpu")
    except Exception:
        return "cpu"


def _vad_model_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "asr", "models", "silero_vad.onnx")



# Segmentation is now driven by SegmenterConfig; these survive only because
# main.py's UI still exposes a "Phrase Timeout" control.
# Below this, SenseVoice's language detection is not reliable enough to
# trust: measured misdetections at 0.7-1.7s, none above ~2s.
LANGUAGE_TRUST_S = 2.0

PHRASE_TIMEOUT = 5.2
MAX_PHRASE_TIMEOUT = 30.2
MAX_PHRASES = 9999

class AudioTranscriber:
    def __init__(self, mic_source, speaker_source, model, response_manager, streaming_mode=False):
        # 添加response_manager
        self.response_manager = response_manager
        self.transcript_data = {"You": [], "Speaker": []}
        self.structured_transcript = {
            "you": [],      # [(text, timestamp, response_id), ...]
            "speaker": [],  # [(text, timestamp, response_id), ...]
            "combined": []  # [(text, timestamp, response_id, speaker_type), ...]
        }
        self.len_speaker = 0
        self.transcript_changed_event = threading.Event()
        self.audio_model = model

        # Phase 2: Thread safety for shared model
        self.model_lock = threading.Lock()  # Ensure serial model inference

        # Future: Streaming mode support (reserved interface)
        self.streaming_mode = streaming_mode

        # Either source may be absent: a Mac mini has no built-in microphone,
        # and macOS without a virtual audio device has no loopback. Build only
        # the tracks that actually exist so one missing device degrades the app
        # instead of crashing it.
        self.audio_sources = {}
        for name, source in (("You", mic_source), ("Speaker", speaker_source)):
            if source is None:
                print(f"[WARN] No audio source for '{name}' — that track is disabled.")
                continue
            self.audio_sources[name] = {
                "sample_rate": source.SAMPLE_RATE,
                "sample_width": source.SAMPLE_WIDTH,
                "channels": source.channels,
                "last_spoken": None,
                "first_spoken": None,
                "new_phrase": True
            }

        if not self.audio_sources:
            raise RuntimeError(
                "No audio sources available — neither a microphone nor a "
                "loopback device could be opened. See docs/macos-audio-setup.md."
            )

        # Phase 2: per-source data locks for thread-safe buffer operations.
        # Keyed off the tracks that exist, so a disabled track has no lock.
        self.source_locks = {name: threading.Lock() for name in self.audio_sources}

        # Two-pass transcription, one segmenter and one tracker per track.
        #
        #   fast lane  - re-transcribe the utterance so far on every chunk,
        #                stabilised by LocalAgreement, for live display;
        #   slow lane  - at the pause, transcribe the whole utterance once.
        #                Its result is authoritative and replaces the live text.
        #
        # The slow lane is not merely a bigger batch: cutting at a pause
        # rather than at a fixed timeout is most of why it is better. A fixed
        # cut lands mid-sentence and hands the model half a clause with no way
        # to tell what it was; a pause-aligned one gives it the whole
        # utterance. On real recordings the difference shows up as garbled
        # fragments becoming complete, correctly punctuated sentences.
        self.segmenters = {
            name: SpeechSegmenter(VAD(_vad_model_path()), SegmenterConfig())
            for name in self.audio_sources
        }
        self.trackers = {name: HypothesisTracker() for name in self.audio_sources}

        # Language stickiness. SenseVoice detects the language per utterance,
        # but needs enough audio to do it: on a real Cantonese call, segments
        # under ~2s came back as Japanese ('ですてま。', 'え。'). Raising
        # min_speech_ms helped but did not close it -- a 1.7s segment in an
        # English recording still came back as Chinese.
        #
        # So: trust detection only on segments long enough to be reliable,
        # remember what it said, and pin shorter ones to that instead of
        # letting them guess from too little signal.
        self._session_language = {name: None for name in self.audio_sources}

        # Voice embeddings, kept so the export can re-cluster them offline.
        # Online clustering must decide on each utterance as it arrives, with
        # no view of what follows, so it over-splits -- a real call produced
        # twelve speakers where three carried 93% of the talking. Given the
        # whole recording and the true headcount that is recoverable, but only
        # if the embeddings still exist. They did not, and the labels on that
        # meeting cannot be repaired.
        self.speaker_embeddings = {}     # response_id -> embedding

        # Written to as the meeting happens, so a crash costs the last
        # sentence rather than the whole thing. Optional: a transcript that
        # cannot be saved is still worth having on screen.
        self.session = None

        # Speaker labelling. The registry is per track: "Speaker" carries
        # everyone in the meeting and needs splitting, while "You" is one
        # person by construction. The embedder is shared and loaded lazily,
        # so nothing is paid unless diarization is switched on.
        self._embedder = SpeakerEmbedder(device=_asr_device())
        self._registries = {name: SpeakerRegistry() for name in self.audio_sources}


    def transcribe_audio_queue(self, audio_queue):
        """
        One thread per track. Audio is segmented on speech pauses rather than
        on a fixed timeout, and each utterance is transcribed twice: once
        incrementally while it is being spoken, and once in full when it ends.
        """
        while True:
            who_spoke, data, time_spoken = audio_queue.get()
            try:
                self._process_chunk(who_spoke, data, time_spoken)
            except Exception as e:
                print(f"Error in transcription: {e}")
                import traceback
                traceback.print_exc()

    def _process_chunk(self, who_spoke, data, time_spoken):
        audio_np = self.convert_bytes_to_numpy(data, who_spoke)
        if audio_np.size == 0:
            return

        with self.source_locks[who_spoke]:
            events = self.segmenters[who_spoke].process(audio_np)

        # Finalise any utterance that ended in this chunk before emitting a
        # partial, or the partial would describe the segment that just closed.
        for event, segment in events:
            if event is Event.SPEECH_END and segment is not None:
                self._finalize_segment(who_spoke, segment, time_spoken)

        if AudioConfig.get_live_partials():
            self._emit_partial(who_spoke, time_spoken)

    def _emit_partial(self, who_spoke, time_spoken):
        """
        Fast lane: re-transcribe the utterance so far.

        The result churns -- the same audio yields "good morning, everyone."
        then "good morning. Everyone." -- so it goes through LocalAgreement,
        which settles the agreeing prefix and leaves the rest provisional.
        """
        with self.source_locks[who_spoke]:
            segmenter = self.segmenters[who_spoke]
            if not segmenter.in_speech:
                return
            audio = segmenter.active_audio

        if audio.size == 0:
            return

        with self.model_lock:
            text = self._transcribe_audio(audio, who_spoke)
        if not self._is_usable(text):
            return

        with self.source_locks[who_spoke]:
            tracker = self.trackers[who_spoke]
            tracker.update(text)
            live = (tracker.committed + " " + tracker.pending).strip()
            if live:
                self.update_transcript(who_spoke, live, time_spoken)

    def _finalize_segment(self, who_spoke, segment, time_spoken):
        """
        Slow lane: transcribe the finished utterance in one pass.

        This is the authoritative text. It supersedes whatever the fast lane
        displayed, and it is the point at which the sentence is known to be
        complete -- which is what the responder waits for.
        """
        with self.model_lock:
            text = self._transcribe_audio(segment.audio, who_spoke)

        with self.source_locks[who_spoke]:
            self.trackers[who_spoke].reset()
            if not self._is_usable(text):
                # Nothing intelligible; leave the phrase open so the next
                # utterance does not inherit a blank record.
                return

            self._last_embedding = None
            label = self._identify_speaker(who_spoke, segment)
            display = f"{label}: {text}" if label else text
            print(f"Segment [{who_spoke} {segment.duration_s:.1f}s]"
                  f"{' ' + label if label else ''} {text}")
            self.update_transcript(who_spoke, display, time_spoken)

            if self.session is not None:
                self.session.append(
                    track=who_spoke, text=text, timestamp=time_spoken,
                    speaker=label,
                    response_id=self._last_response_id,
                    embedding=self._last_embedding)
            self.audio_sources[who_spoke]["new_phrase"] = True

            # Trigger the responder here, not when the phrase started. The
            # old code called create_response() with the first fragment of an
            # utterance, so the LLM was answering a truncated question.
            if (who_spoke.lower() == "speaker"
                    and not SystemConfig.get_record_only_mode()):
                self.transcript_changed_event.set()

    def preload_speaker_model(self) -> bool:
        """Load the embedding model off the transcription path."""
        try:
            self._embedder._ensure_model()
            print("[INFO] Speaker embedding model ready")
            return True
        except Exception as e:
            print(f"[WARN] Speaker embedding unavailable: {e}")
            AudioConfig.set_diarization(False)
            return False

    def _identify_speaker(self, who_spoke, segment):
        """
        Which of the voices on this track just spoke, or None.

        Only the far-end track is worth splitting: "You" is a single person
        by construction, and running this on it would spend an embedding per
        segment to rediscover that.
        """
        if not AudioConfig.get_diarization():
            return None
        if who_spoke.lower() != "speaker":
            return None

        registry = self._registries[who_spoke]
        # An expected headcount, when the user supplies one, caps the
        # registry: past it, every utterance goes to the nearest voice
        # already known rather than inventing another.
        expected = AudioConfig.get_speaker_count()
        if expected and registry.config.max_speakers != expected:
            registry.config.max_speakers = expected

        config = registry.config
        if segment.duration_s < config.min_duration_s:
            # Too short for a reliable embedding, the same way it is too short
            # for reliable language detection.
            return None

        if not self._embedder.ready:
            # Never block the transcription thread on a model load. The
            # first attempt used to trigger a lazy download and Metal warmup
            # inside the handling of an utterance, stalling that track for
            # 20s or more; preloading happens off this path instead.
            return None

        embedding = self._embedder.embed(segment.audio)
        if embedding is None:
            return None

        result = self._registries[who_spoke].assign(embedding)
        if result.speaker is None:
            return None
        self._last_embedding = embedding
        # A blended segment -- two voices with no pause between them -- is
        # marked rather than presented as certain.
        return result.label if result.confident else f"{result.label}?"

    @staticmethod
    def _is_usable(text) -> bool:
        return bool(text) and text.strip() != ""

    def apply_segmenter_config(self, config) -> None:
        """
        Swap segmentation settings on a running transcriber.

        Applied to the live objects rather than by rebuilding them: a
        segmenter holds the audio of the utterance in progress, and replacing
        it mid-sentence would drop whatever has been spoken so far.
        """
        for name, segmenter in self.segmenters.items():
            with self.source_locks[name]:
                segmenter.config = config

    def _transcribe_audio(self, audio_np, who_spoke=None):
        """
        Transcribe, applying language stickiness.

        Short audio is pinned to the language established by this track's
        longer utterances; long audio is trusted and updates that memory.
        """
        if self.streaming_mode:
            raise NotImplementedError("Streaming mode not yet implemented")

        duration = len(audio_np) / 16000.0
        remembered = self._session_language.get(who_spoke)
        trustworthy = duration >= LANGUAGE_TRUST_S

        override = None if trustworthy else remembered
        result = self.audio_model.get_transcription_np(audio_np, language=override)

        detected = getattr(result, "language", None)
        if trustworthy and detected and detected != "nospeech":
            if who_spoke in self._session_language and detected != remembered:
                print(f"[INFO] {who_spoke}: language now {detected!r}"
                      + (f" (was {remembered!r})" if remembered else ""))
                self._session_language[who_spoke] = detected

        return getattr(result, "text", result)

    def convert_bytes_to_numpy(self, audio_bytes, who_spoke):
        """
        Convert raw int16 PCM bytes to the mono float32 @16kHz array FunASR wants.

        Driven entirely by the source's declared format rather than by which
        track it is: both backends hand us interleaved int16 PCM, and a mic at
        48kHz stereo needs exactly the same treatment as a loopback at 48kHz
        stereo.
        """
        source_info = self.audio_sources[who_spoke]
        target_sample_rate = 16000  # FunASR expects 16kHz

        # Trim to a whole number of int16 samples: np.frombuffer raises on a
        # ragged buffer, which would kill the transcription thread outright.
        usable_bytes = len(audio_bytes) - (len(audio_bytes) % 2)
        if usable_bytes == 0:
            return np.zeros(0, dtype=np.float32)
        audio_np = np.frombuffer(audio_bytes[:usable_bytes], dtype=np.int16)

        # --- down-mix to mono ---
        channels = int(source_info["channels"])
        if channels > 1:
            usable = (audio_np.size // channels) * channels
            if usable == 0:
                return np.zeros(0, dtype=np.float32)
            # Average in float to avoid int16 overflow on summation.
            audio_np = audio_np[:usable].reshape(-1, channels).mean(axis=1)
        else:
            audio_np = audio_np.astype(np.float64)

        # --- normalise to [-1, 1] before resampling ---
        # Resampling in float avoids the double int16 round-trip the previous
        # version did, which quantised twice for no reason.
        audio_np = (audio_np / 32768.0).astype(np.float32)

        # --- resample to 16kHz ---
        source_rate = int(source_info["sample_rate"])
        if source_rate != target_sample_rate:
            # resample_poly (polyphase FIR) instead of signal.resample (FFT):
            # 48k -> 16k is an exact 1/3 decimation, so this is both far
            # cheaper and free of the circular-convolution edge artifacts the
            # FFT method introduces at chunk boundaries.
            gcd = math.gcd(source_rate, target_sample_rate)
            up = target_sample_rate // gcd
            down = source_rate // gcd
            audio_np = signal.resample_poly(audio_np, up, down).astype(np.float32)

        return audio_np

    def update_transcript(self, who_spoke, text, time_spoken):
        source_info = self.audio_sources[who_spoke]
        speaker_type = who_spoke.lower()
        
        # 为用户输入创建response记录
        response_id = None
        if speaker_type == 'speaker' and source_info["new_phrase"]:
            #print(f"\nDebug AudioTranscriber - New Speaker input:")
            #print(f"Text: {text}")
            
            response_id = self.response_manager.create_response(
                question_time=time_spoken,
                question_text=text
            )
            #print(f"Created new response_id: {response_id}")
        
        # 创建统一的记录结构
        record = {
            'transcript': (f"{who_spoke}: [{text}]\n\n", time_spoken),
            'structured': (text, time_spoken, response_id),
            'combined': (text, time_spoken, response_id, speaker_type)
        }

        embedding = getattr(self, "_last_embedding", None)
        if embedding is not None and response_id:
            self.speaker_embeddings[response_id] = embedding
        self._last_response_id = response_id
        #print (f"New record: {record}")
        # 更新数据结构
        update_method = 'insert' if source_info["new_phrase"] or not self.transcript_data[who_spoke] else 'update'
        self._update_all_transcripts(speaker_type, record, update_method)

        # The responder is no longer woken here. This ran when a phrase
        # *started*, so create_response() above was handed the first fragment
        # of an utterance and the LLM answered a truncated question.
        # _finalize_segment triggers it instead, once the sentence is complete.
        if source_info["new_phrase"]:
            source_info["new_phrase"] = False
            source_info["first_spoken"] = time_spoken
        source_info["last_spoken"] = time_spoken

    def _update_all_transcripts(self, speaker_type, record, method='insert'):
        """更新所有转录数据结构"""
        #print(f"\nDebug _update_all_transcripts:")
        #print(f"Speaker type: {speaker_type}")
        #print(f"Method: {method}")
        #print(f"Record response_id: {record['structured'][2]}")
        
        index = 0 if method == 'insert' else 0
        
        # 如果是更新操作，需要保留原有的 response_id
        if method == 'update' and self.structured_transcript[speaker_type]:
            original_response_id = self.structured_transcript[speaker_type][0][2]
            # 使用原有的 response_id 创建新的记录元组
            record = {
                'transcript': record['transcript'],
                'structured': (record['structured'][0], record['structured'][1], original_response_id),
                'combined': (record['combined'][0], record['combined'][1], original_response_id, record['combined'][3])
            }
            #print(f"Preserved response_id in update: {original_response_id}")
        
        # 更新原始transcript
        if method == 'insert':
            self.transcript_data[speaker_type.title()].insert(index, record['transcript'])
        else:
            if self.transcript_data[speaker_type.title()]:
                self.transcript_data[speaker_type.title()][index] = record['transcript']
            else:
                self.transcript_data[speaker_type.title()].insert(index, record['transcript'])
        
        # 更新结构化数据
        if method == 'insert':
            self.structured_transcript[speaker_type].insert(index, record['structured'])
        else:
            if self.structured_transcript[speaker_type]:
                self.structured_transcript[speaker_type][index] = record['structured']
            else:
                self.structured_transcript[speaker_type].insert(index, record['structured'])
        
        # 更新组合视图
        if method == 'insert':
            self.structured_transcript['combined'].insert(index, record['combined'])
        else:
            if self.structured_transcript['combined']:
                # 查找并更新对应speaker_type的最新消息
                for i, msg in enumerate(self.structured_transcript['combined']):
                    if msg[3] == speaker_type:  # 检查speaker_type
                        self.structured_transcript['combined'][i] = record['combined']
                        break
            else:
                self.structured_transcript['combined'].insert(index, record['combined'])
        
        #print(f"After update:")
        #print(f"Combined messages count: {len(self.structured_transcript['combined'])}")
        #if self.structured_transcript['combined']:
        #    print(f"Last combined message response_id: {self.structured_transcript['combined'][0][2]}")
        #    print(f"Last combined message question: {self.structured_transcript['combined'][0][0]}")


    def get_transcript(self):
        # 返回结构化的transcript数据
        return {
            'all': "".join([f"{t[3].title()}: [{t[0]}]\n\n" for t in self.structured_transcript["combined"]]),
            'speaker': [{'text': t[0], 'timestamp': t[1], 'response_id': t[2]} 
                       for t in self.structured_transcript["speaker"]],
            'you': [{'text': t[0], 'timestamp': t[1], 'response_id': t[2]} 
                    for t in self.structured_transcript["you"]]
        }

    def get_lastContent(self):
        """获取Speaker最后一条记录的内容"""
        try:
            # 从structured_transcript中获取speaker最新的记录
            if self.structured_transcript["speaker"]:
                # structured_transcript中的格式是 (text, timestamp, response_id)
                return self.structured_transcript["speaker"][0][0]
            return ''
        except Exception as e:
            print(f"Error in get_lastContent: {e}")
            return ''

    def clear_transcript_data(self):
        self.transcript_data["You"].clear()
        self.transcript_data["Speaker"].clear()
        self.structured_transcript["you"].clear()
        self.structured_transcript["speaker"].clear()
        self.structured_transcript["combined"].clear()

        for source_name, source_info in self.audio_sources.items():
            source_info["new_phrase"] = True
            source_info["last_spoken"] = None
            source_info["first_spoken"] = None
            self.segmenters[source_name].reset()
            self.trackers[source_name].reset()
            self._registries[source_name].reset()
            self._session_language[source_name] = None