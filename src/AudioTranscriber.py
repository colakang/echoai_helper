#src/AudioTranscriber.py

#import whisper
import math
import os
import threading
from datetime import datetime, timedelta

import numpy as np
from scipy import signal

from .config import AudioConfig, SystemConfig



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
                "last_sample": bytes(),
                "saved_sample": bytes(),
                "chunks_buffer": [],  # 使用列表存储chunks
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

    def transcribe_audio_queue(self, audio_queue):
        """
        Phase 2: Dual-queue support with thread-safe model access.
        This method can be called by multiple threads, each processing its own queue.
        """
        while True:
            who_spoke, data, time_spoken = audio_queue.get()

            # Phase 2: Thread-safe buffer update
            with self.source_locks[who_spoke]:
                self.update_last_sample_and_phrase_status(who_spoke, data, time_spoken)
                source_info = self.audio_sources[who_spoke]
                # Copy saved_sample for processing (avoid holding lock during inference)
                audio_data = source_info["saved_sample"]

            text = ''
            try:
                # Phase 1: Convert audio bytes to numpy (preprocessing, no lock needed)
                audio_np = self.convert_bytes_to_numpy(audio_data, who_spoke)

                # Phase 2: Thread-safe model inference (shared model, serial execution)
                with self.model_lock:
                    text = self._transcribe_audio(audio_np)

            except Exception as e:
                print(f"Error in transcription: {e}")
                import traceback
                traceback.print_exc()

            # Phase 2: Thread-safe result processing
            with self.source_locks[who_spoke]:
                source_info = self.audio_sources[who_spoke]
                if text != '' and text.lower() != 'you':
                    print("Catching: "+ text+"\n")
                    ## if text is end of 指定符号，则设定为new phrase
                    if (source_info["first_spoken"] and time_spoken - source_info["first_spoken"] > timedelta(seconds=AudioConfig.get_phrase_timeout())) :
                        print ("new phrase......\n")
                        source_info["new_phrase"] = True
                    self.update_transcript(who_spoke, text, time_spoken)
                else:
                    print("\r "+who_spoke+" text: Null, New_Phrase:"+str(source_info["new_phrase"])+"\r\n")

                self._enforce_buffer_cap(who_spoke, source_info, time_spoken)

    def _sample_seconds(self, who_spoke, sample: bytes) -> float:
        """How many seconds of audio a raw PCM buffer holds."""
        info = self.audio_sources[who_spoke]
        bytes_per_second = info["sample_rate"] * info["channels"] * info["sample_width"]
        return len(sample) / bytes_per_second if bytes_per_second else 0.0

    def _enforce_buffer_cap(self, who_spoke, source_info, time_spoken) -> None:
        """
        Hard ceiling on how much audio one phrase may accumulate.

        last_sample only ever shrinks in _reset_source_info, which runs from
        update_transcript -- and update_transcript is only reached when the
        model returns usable text. Audio that clears the capture energy gate
        but yields no transcript (music, keyboard noise, a fan, non-speech
        chatter) therefore accumulates forever: the buffer grows without
        bound, and every subsequent chunk re-transcribes all of it.

        Unnoticeable in a five-minute demo, fatal in the 1-2 hour meetings
        this is now aimed at. MAX_PHRASE_TIMEOUT has been declared since the
        first commit but never enforced; enforce it.
        """
        held = self._sample_seconds(who_spoke, source_info["last_sample"])
        if held <= MAX_PHRASE_TIMEOUT:
            return

        print(f"[WARN] {who_spoke}: buffer hit the {MAX_PHRASE_TIMEOUT}s cap "
              f"({held:.1f}s held) without a phrase break — forcing one.")
        source_info["new_phrase"] = True
        self._reset_source_info(source_info, time_spoken)

    def _transcribe_audio(self, audio_np):
        """
        Modularized transcription method for future streaming compatibility.
        In streaming mode, this will handle chunk-based processing.
        """
        if self.streaming_mode:
            # Future: Streaming implementation
            # return self._transcribe_streaming(audio_np)
            raise NotImplementedError("Streaming mode not yet implemented")
        else:
            # Current: Accumulate mode
            return self.audio_model.get_transcription_np(audio_np)

    def update_last_sample_and_phrase_status(self, who_spoke, data, time_spoken):
        source_info = self.audio_sources[who_spoke]
        #print("#1 "+who_spoke+" Now:"+str(time_spoken)+" First:"+str(source_info["first_spoken"])+" Last:"+str(source_info["last_spoken"])+"\r\n")
        # 更新chunks buffer
        max_chunks = AudioConfig.get_buffer_chunks()
        if max_chunks > 0:  # 只在需要buffer时处理
            source_info["chunks_buffer"].append(data)
            # 保持buffer大小不超过限制
            if len(source_info["chunks_buffer"]) > max_chunks:
                source_info["chunks_buffer"].pop(0)  # 移除最老的chunk
        if source_info["first_spoken"] == None:
                source_info["first_spoken"] = time_spoken
        source_info["last_sample"] += data
        source_info["last_spoken"] = time_spoken
        source_info["saved_sample"] = source_info["last_sample"] 

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
        #print (f"New record: {record}")
        # 更新数据结构
        update_method = 'insert' if source_info["new_phrase"] or not self.transcript_data[who_spoke] else 'update'
        self._update_all_transcripts(speaker_type, record, update_method)
        
        # 处理新短语的状态更新
        if source_info["new_phrase"]:
            # 如果是用户输入且创建了新的response_id，在所有数据更新完成后触发事件
            #if speaker_type == 'speaker' and response_id:
            if speaker_type == 'speaker' and response_id and not SystemConfig.get_record_only_mode():
    
                print("Setting transcript_changed_event after data update")
                self.transcript_changed_event.set()
                
            self._reset_source_info(source_info, time_spoken)

    def _reset_source_info(self, source_info, time_spoken):
        """重置source_info的状态"""
        buffered_data = b''.join(source_info["chunks_buffer"]) if source_info["chunks_buffer"] else bytes()
        source_info.update({
            'first_spoken': time_spoken,
            'last_sample': buffered_data,  # 使用合并后的buffer数据
            'new_phrase': False,
            'chunks_buffer': []  # 重置chunks buffer
        })
        print('Reset data with buffer.....\n')

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
            source_info["last_sample"] = bytes()
            source_info["saved_sample"] = bytes()
            source_info["chunks_buffer"].clear()  # 清除chunks buffer
            source_info["new_phrase"] = True
            source_info["last_spoken"] = None
            source_info["first_spoken"] = None