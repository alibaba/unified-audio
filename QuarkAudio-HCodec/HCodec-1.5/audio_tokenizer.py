import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
from typing import Any, Dict, Tuple
import math

import librosa
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from transformers import AutoModel

from vq import Codec

def load_sub_weights(checkpoint_path, prefix= 'generator.'):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    
    if prefix is None:
        return checkpoint

    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            filtered_state_dict[new_key] = value
    
    return filtered_state_dict

class HCodecTokenizer(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.config = kwargs.get('config', None)
        self.model = Codec(config['encoder_config'], config['decoder_config'], config['quantizer_config'], config['adaptive_config'])
        state_dict = load_sub_weights(self.config['ckpt_path'], prefix=None)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.feature_extractor = AutoModel.from_pretrained("facebook/wav2vec2-large-xlsr-53").eval()
        self.feature_extractor.requires_grad_(False)
        self.hop_length = 640  # 25hz
    
    @torch.no_grad()
    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""
        # wavs: (b,t)
        wavs = F.pad(wavs, (160, 160))
        
        feat = self.feature_extractor(wavs, output_hidden_states=True)
        feats_mix = (
            feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
        ) / 3

        symbol = (feats_mix > 0).float() * 2 - 1
        magnitude = feats_mix.abs() ** 0.3
        feats_mix = symbol * magnitude

        return feats_mix
    
    def pad_wav(self, wav):
        hop_length = math.prod(self.config['encoder_config']['ratios']) * 2
        pad_length = math.ceil(wav.size(-1) / hop_length) * hop_length - wav.size(-1)
        wav = torch.nn.functional.pad(wav, (0, pad_length))
        return wav

    @torch.no_grad()
    def tokenize(self, wav: torch.Tensor):
        # wav: (b,t)
        wav = self.pad_wav(wav)
        feats = self.extract_wav2vec2_features(wav).transpose(-2, -1)  # (b,d,t)
        ret_dict = self.model.encode(wav.unsqueeze(1), feats) # (b,nq,t)
        return ret_dict

    @torch.no_grad()
    def detokenize(self, acoustic_codes: torch.Tensor, semantic_codes: torch.Tensor, token_lengths=None):
        wav_rec = self.model.decode(acoustic_codes, semantic_codes, token_lengths)  # b,t
        return wav_rec


def load_wav(info: str, fs=None):
    wav, fs_ = librosa.load(info, dtype=np.float32, sr=fs, mono=False)
    if wav.ndim == 1:
        wav = wav[None]  # (1, T)
    else:
        wav = wav[:1, :]  # 取第0通道
    return wav, fs_

# test
if __name__ == "__main__":
    import soundfile as sf
    config_path = './conf/config_adaptive_v3.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    threshold = config['adaptive_config']['manual_threshold']
    print(f'[+] Using threshold: {threshold:.2f} for dynamic frame rate')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = HCodecTokenizer(config=config)
    tokenizer = tokenizer.to(device)

    wav_path = "./sample.flac"
    wav, _ = load_wav(wav_path,fs=16000)
    
    wav = torch.from_numpy(wav).to(device).float()

    codes = tokenizer.tokenize(wav)
    wav_rec = tokenizer.detokenize(**codes)
    sf.write("./wav_rec.wav", wav_rec.squeeze(0).cpu().numpy(), 16000)


