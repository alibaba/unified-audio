import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from pathlib import Path
from typing import Any, Dict, Tuple
import math


import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from transformers import AutoModel

from .vq import Codec


class HCodecTokenizer(nn.Module):

    def __init__(self, pt_path: Path, **kwargs):
        super().__init__()

        self.model = Codec(None, None, None)
        state_dict = torch.load(pt_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.feature_extractor = AutoModel.from_pretrained("bosonai/hubert_base").eval()
        self.feature_extractor.requires_grad_(False)

        self.hop_length = 640  # 25hz
    

    @torch.no_grad()
    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""
        # wavs: (b,t)
        wavs = F.pad(wavs, (160, 160))
        
        feats = self.feature_extractor(wavs, output_hidden_states=True)
        feats_mix = torch.stack(feats.hidden_states, dim=1).mean(1)

        # 进行幅度压缩！！！
        symbol = (feats_mix > 0).float() * 2 - 1
        magnitude = feats_mix.abs() ** 0.3
        feats_mix = symbol * magnitude

        return feats_mix
    
    def pad_wav(self, wav):
        pad_length = math.ceil(wav.size(-1) / self.hop_length) * self.hop_length - wav.size(-1)
        wav = torch.nn.functional.pad(wav, (0, pad_length))
        return wav

    @torch.no_grad()
    def tokenize(self, wav: torch.Tensor):
        # wav: (b,t)
        wav = self.pad_wav(wav)
        feats = self.extract_wav2vec2_features(wav).transpose(-2, -1)  # (b,d,t)
        acoustic_codes, semantic_codes = self.model.encode(wav.unsqueeze(1), feats) # (b,nq,t)
        return acoustic_codes, semantic_codes

    @torch.no_grad()
    def detokenize(self, acoustic_codes: torch.Tensor, semantic_codes: torch.Tensor):
        wav_rec = self.model.decode(acoustic_codes, semantic_codes)  # b,t
        return wav_rec


# test
if __name__ == "__main__":
    import soundfile as sf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = HCodecTokenizer(pt_path='./weights.pt')
    tokenizer = tokenizer.to(device)

    wav_path = "/mnt/nas1/002_0313018212@0.wav"
    wav, _ = sf.read(wav_path, dtype='float32')
    wav = torch.from_numpy(wav).to(device)
    wav = wav.unsqueeze(0)

    acoustic_codes, semantic_codes = tokenizer.tokenize(wav)
    wav_rec = tokenizer.detokenize(acoustic_codes, semantic_codes)
    sf.write("./wav_rec.wav", wav_rec.squeeze(0).cpu().numpy(), 16000)


