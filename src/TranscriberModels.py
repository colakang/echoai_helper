#src/TranscriberModels.py

import openai
import yaml
#import whisper
#from faster_whisper import WhisperModel
import os
import torch
from src.asr.asr_factory import ASRFactory
from src.asr.asr_interface import ASRInterface
from .config import PathConfig


def get_model(use_api):
    if use_api:
        return APIWhisperTranscriber()
    else:
        return FunASRTranscriber()
        #return WhisperTranscriber()

class FunASRTranscriber:
    def __init__(self):
        #self.audio_model = whisper.load_model(os.path.join(os.getcwd(), 'small.pt'))
        with open(f"{PathConfig.get_project_root()}/conf.yaml", "rb") as f:
            self.config = yaml.safe_load(f)

        asr_model = "FunASR"
        asr_config = self.config.get(asr_model, {})

        self.audio_model = ASRFactory.get_asr_system(asr_model, **asr_config)

        print(f"[INFO] FunASR using GPU: " + str(torch.cuda.is_available()))

    def init_asr(self) -> ASRInterface:
        asr_model = self.config.get("ASR_MODEL")
        asr_config = self.config.get(asr_model, {})

        asr = ASRFactory.get_asr_system(asr_model, **asr_config)
        return asr

    def get_transcription(self, wav_file_path):
        try:
            #with open(wav_file_path, "rb") as audio_file:
            #    self.received_data_buffer = np.array([])
            result = self.audio_model.transcribe_wav(wav_file_path)
            #result = self.audio_model.transcribe(wav_file_path, fp16=torch.cuda.is_available())
        except Exception as e:
            print(e)
            return ''
        return result


class WhisperTranscriber:


    # Run on GPU with FP16
    #model = WhisperModel(model_size, device="cuda", compute_type="float16")

    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8

    def __init__(self):
        #self.audio_model = whisper.load_model(os.path.join(os.getcwd(), 'small.pt'))

        #model_size = "large-v3-turbo"
        #model_size = "distil-small.en"
        model_size = "small.en"
        self.audio_model = WhisperModel(model_size, device="cpu",cpu_threads=8, compute_type="int8")

        print(f"[INFO] Whisper using GPU: " + str(torch.cuda.is_available()))

    def get_transcription(self, wav_file_path):
        try:
            #result = self.audio_model.transcribe(wav_file_path, fp16=torch.cuda.is_available())
            segments, _ = self.audio_model.transcribe(wav_file_path, vad_filter=True,language="en",beam_size=5)
            result = list(segments)
        except Exception as e:
            print(e)
            return ''
        #return result['text'].strip()
        full_text = ""
        for segment in result:
        #print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            full_text += segment.text + " "  # 添加空格分隔不同段落
        return full_text.strip()

    
class APIWhisperTranscriber:
    def get_transcription(self, wav_file_path):
        try:
            with open(wav_file_path, "rb") as audio_file:
                result = openai.Audio.transcribe("whisper-1", audio_file)
        except Exception as e:
            print(e)
            return ''
        return result['text'].strip()