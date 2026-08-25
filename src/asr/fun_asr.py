import numpy as np
from funasr import AutoModel
from .asr_interface import ASRInterface
from .asr_with_vad import VoiceRecognitionVAD

import re
import soundfile as sf
import io
import torch
from typing import NamedTuple, Optional


class AsrResult(NamedTuple):
    """Transcript plus the language SenseVoice thought it heard."""
    text: str
    language: Optional[str]


# SenseVoiceSmall prefixes its output with tags:
#   '<|zh|><|NEUTRAL|><|Speech|><|woitn|>欢迎大家来体验...'
# The first is the detected language. It used to be stripped and thrown away,
# which made language stickiness impossible: short segments get misdetected
# (sub-second Cantonese comes back as Japanese) and there was no way to tell
# that had happened, let alone to correct it from context.
_TAG = re.compile(r'<\s*\|\s*(.*?)\s*\|\s*>')
_LANGUAGES = {"zh", "en", "yue", "ja", "ko", "nospeech"}


def _split_tags(raw: str) -> AsrResult:
    tags = [t.replace(" ", "").lower() for t in _TAG.findall(raw)]
    language = next((t for t in tags if t in _LANGUAGES), None)
    return AsrResult(_TAG.sub("", raw).strip(), language)

# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need




class VoiceRecognition(ASRInterface):

    def __init__(
        self,
        model_name: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        language: str = "auto",
        #vad_model: str = "fsmn-vad",
        vad_model = None,
        punc_model=None,
        ncpu: int = None,
        hub: str = None,
        trust_remote_code: bool = False,
        device: str = "cpu",
        sample_rate: int = 16000,
        #use_itn: bool = False,
        itn: bool = True,
    ) -> None:

        self.model = AutoModel(
            model=model_name,
            vad_model=vad_model,
            ncpu=ncpu,
            hub=hub,
            device=device,
            punc_model=punc_model,
            disable_update=True,
            trust_remote_code=trust_remote_code,
            #spk_model="cam++",
        )
        self.sample_rate = sample_rate
        #self.use_itn = use_itn
        self.itn = itn
        self.language = language

        self.asr_with_vad = None

    def transcribe_with_local_vad(self) -> str:
        if self.asr_with_vad is None:
            self.asr_with_vad = VoiceRecognitionVAD(self.transcribe_np)
        return self.asr_with_vad.start_listening()
    
    def transcribe_wav(self, audio) -> str:
        
        #audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        res = self.model.generate(
            input=audio,
            batch_size_s=300,
            #use_itn=self.use_itn,
            itn=self.itn,
            language=self.language,
        )
        
        return _split_tags(res[0]["text"])

    def transcribe_np(self, audio: np.ndarray, language: Optional[str] = None) -> AsrResult:
        """
        `language` overrides the configured setting for this call only, which
        is what lets a caller pin a short segment to the language established
        by its neighbours instead of letting auto-detect guess from too
        little audio.
        """
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        res = self.model.generate(
            input=audio_tensor,
            batch_size_s=300,
            itn=self.itn,
            language=language or self.language,
        )
        
        return _split_tags(res[0]["text"])

    def _numpy_to_wav_in_memory(self, numpy_array: np.ndarray, sample_rate):

        memory_file = io.BytesIO()
        sf.write(memory_file, numpy_array, sample_rate, format='WAV')
        memory_file.seek(0)
        
        return memory_file
