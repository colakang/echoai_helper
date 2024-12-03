#src/AudioTranscriber.py

#import whisper
import uuid
import torch
import wave
import os
import threading
import tempfile
import src.custom_speech_recognition as sr
import io
from datetime import timedelta
import pyaudiowpatch as pyaudio
from heapq import merge
from datetime import datetime
import time
from .config import AudioConfig, SystemConfig



PHRASE_TIMEOUT = 5.2
MAX_PHRASE_TIMEOUT = 30.2
MAX_PHRASES = 9999

class AudioTranscriber:
    def __init__(self, mic_source, speaker_source, model, response_manager):
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
        self.audio_sources = {
            "You": {
                "sample_rate": mic_source.SAMPLE_RATE,
                "sample_width": mic_source.SAMPLE_WIDTH,
                "channels": mic_source.channels,
                "last_sample": bytes(),
                "saved_sample": bytes(),
                "chunks_buffer": [],  # 使用列表存储chunks
                "last_spoken": None,
                "first_spoken": None,
                "new_phrase": True,
                "process_data_func": self.process_mic_data
            },
            "Speaker": {
                "sample_rate": speaker_source.SAMPLE_RATE,
                "sample_width": speaker_source.SAMPLE_WIDTH,
                "channels": speaker_source.channels,
                "last_sample": bytes(),
                "saved_sample": bytes(),
                "chunks_buffer": [],  # 使用列表存储chunks
                "last_spoken": None,
                "first_spoken": None,
                "new_phrase": True,
                "process_data_func": self.process_speaker_data
            }
        }

    def transcribe_audio_queue(self, audio_queue):
        while True:
            #print("Debug: "+ "-----" +"\n")
            who_spoke, data, time_spoken = audio_queue.get()
            self.update_last_sample_and_phrase_status(who_spoke, data, time_spoken)
            source_info = self.audio_sources[who_spoke]
            text = ''
            try:
                fd, path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                source_info["process_data_func"](source_info["saved_sample"], path)
                text = self.audio_model.get_transcription(path)
            except Exception as e:
                print(e)
            finally:
                os.unlink(path)
            if text != '' and text.lower() != 'you':
                print("Catching: "+ text+"\n")
                ## if text is end of 指定符号，则设定为new phrase
                if (source_info["first_spoken"] and time_spoken - source_info["first_spoken"] > timedelta(seconds=AudioConfig.get_phrase_timeout())) :
                    print ("new phrase......\n")
                    source_info["new_phrase"] = True
                    #if who_spoke.lower() == 'speaker':
                        #self.transcript_changed_event.set()
                self.update_transcript(who_spoke, text, time_spoken)
            else:
                print("\r "+who_spoke+" text: Null, New_Phrase:"+str(source_info["new_phrase"])+"\r\n")
                #self.transcript_changed_event.wait(1.5)

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

    def process_mic_data(self, data, temp_file_name):
        audio_data = sr.AudioData(data, self.audio_sources["You"]["sample_rate"], self.audio_sources["You"]["sample_width"])
        wav_data = io.BytesIO(audio_data.get_wav_data())
        with open(temp_file_name, 'w+b') as f:
            f.write(wav_data.read())

    def process_speaker_data(self, data, temp_file_name):
        with wave.open(temp_file_name, 'wb') as wf:
            wf.setnchannels(self.audio_sources["Speaker"]["channels"])
            p = pyaudio.PyAudio()
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.audio_sources["Speaker"]["sample_rate"])
            wf.writeframes(data)

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