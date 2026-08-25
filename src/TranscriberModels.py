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


def resolve_device(requested="auto") -> str:
    """
    Pick the accelerator to run on.

    "auto" is the sensible default and the only portable one: a config file
    naming "mps" is wrong on Windows and a config naming "cuda" is wrong on a
    Mac, and this is not a preference the user should have to encode per
    machine.

    It matters more than a tuning knob. Measured on an M4: dual-track
    real-time factor is 2.04 on cpu -- falling behind twice over -- against
    0.35 on mps. Silently running on cpu because a config file was written on
    another machine means transcription that cannot keep up.
    """
    if requested and requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            print("[WARN] device 'mps' requested but unavailable; using cpu")
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            print("[WARN] device 'cuda' requested but unavailable; using cpu")
            return "cpu"
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

        # Honour ASR_MODEL from conf.yaml instead of hardcoding "FunASR",
        # which silently ignored the setting and made switching backends a
        # no-op.
        asr_model = self.config.get("ASR_MODEL", "FunASR")
        asr_config = dict(self.config.get(asr_model, {}))
        asr_config["device"] = resolve_device(asr_config.get("device", "auto"))

        self.audio_model = ASRFactory.get_asr_system(asr_model, **asr_config)

        device = resolve_device(asr_config.get("device", "auto"))
        asr_config["device"] = device
        print(f"[INFO] {asr_model} on device={device!r}")

        # First inference on Metal pays kernel-compilation cost (~1.3s) that
        # would otherwise land on the user's first spoken phrase. Burn it here.
        if device == "mps":
            try:
                import numpy as _np
                self.audio_model.transcribe_np(_np.zeros(16000, dtype=_np.float32))
                print("[INFO] MPS warmup complete")
            except Exception as e:
                print(f"[WARN] MPS warmup failed (non-fatal): {e}")

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

    def get_transcription_np(self, audio_data, language=None):
        """
        Transcribe a numpy array directly, no temp file.

        Returns an AsrResult (text, language). `language` pins this one call,
        letting the caller override automatic detection for audio too short
        for it to be reliable.
        """
        from src.asr.fun_asr import AsrResult
        try:
            return self.audio_model.transcribe_np(audio_data, language=language)
        except Exception as e:
            print(e)
            return AsrResult("", None)


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