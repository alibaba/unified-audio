import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


import math
import yaml
import librosa
from torchaudio.transforms import Resample

import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from transformers import AutoModel

from vq import Codec


class HCodecTokenizer(nn.Module):

    def __init__(self, pt_path: str, config_path: str, device='cpu'):
        super().__init__()

        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.model = Codec(
            config['encoder_config'],
            config['decoder_config'],
            config['quantizer_config'],
            config['semantic_encoder_config'],
            config['semantic_decoder_config'],
        ).to(device)
        state_dict = torch.load(pt_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.feature_extractor = AutoModel.from_pretrained("bosonai/hubert_base").eval()
        self.feature_extractor.requires_grad_(False).to(device)

        self.resample = Resample(config['sampling_rate'], 16000)

        self.hop_length = int(config['sampling_rate'] / config['encoder_config']['target_frame_rate'])
    

    @torch.no_grad()
    def extract_ssl_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract ssl features"""
        # wavs: (b,t)
        wavs = self.resample(wavs)
        wavs = F.pad(wavs, (160, 160))
        
        feats = self.feature_extractor(wavs, output_hidden_states=True)
        feats_mix = torch.stack(feats.hidden_states, dim=1).mean(1)

        # 幅度压缩
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
        feats = self.extract_ssl_features(wav).transpose(-2, -1)  # (b,d,t)
        acoustic_codes, semantic_codes = self.model.encode(wav, feats) # (b,nq,t)
        return acoustic_codes, semantic_codes

    @torch.no_grad()
    def detokenize(self, acoustic_codes: torch.Tensor, semantic_codes: torch.Tensor):
        wav_rec = self.model.decode(acoustic_codes, semantic_codes)  # b,t
        return wav_rec


# test
if __name__ == "__main__":
    import soundfile as sf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = HCodecTokenizer(pt_path='./ckpt/large_12.5hz_weights.pt', config_path='./conf/large_12.5hz_config.yaml', device=device)
    # tokenizer = tokenizer.to(device)

    wav_path = "./test.wav"
    wav, sr = librosa.load(wav_path, sr=48000, mono=False, dtype=np.float32)
    if wav.ndim == 1:
        wav = wav[None, :]
    else:
        wav = wav[:1, :]
    wav = torch.from_numpy(wav).to(device)

    acoustic_codes, semantic_codes = tokenizer.tokenize(wav)
    wav_rec = tokenizer.detokenize(acoustic_codes, semantic_codes)
    sf.write("./wav_rec.wav", wav_rec.squeeze(0).cpu().numpy(), 48000)


