from transformers import SeamlessM4TFeatureExtractor
import torch
import torchaudio
import librosa
import numpy as np


class FBankGen:
    def __init__(self, sr):
        import funasr
        from pathlib import Path
        assert sr == 16000
        cmvn_file = f'{str(Path(__file__).parent)}/am.mvn'
        self.frontend = funasr.frontends.wav_frontend.WavFrontend(
                cmvn_file=cmvn_file,
                n_mels=80,
                frame_length=25,
                frame_shift=10,
                lfr_m=7,
                lfr_n=6,
            )
    def extract_fbank(self, data, data_len=None, data_type: str = "sound", **kwargs):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            if len(data.shape) < 2:
                data = data[None, :]  # data: [batch, N]
            data_len = [data.shape[1]] if data_len is None else data_len
        elif isinstance(data, torch.Tensor):
            if len(data.shape) < 2:
                data = data[None, :]  # data: [batch, N]
            data_len = [data.shape[1]] if data_len is None else data_len
        elif isinstance(data, (list, tuple)):
            data_list, data_len = [], []
            for data_i in data:
                if isinstance(data_i, np.ndarray):
                    data_i = torch.from_numpy(data_i)
                data_list.append(data_i)
                data_len.append(data_i.shape[0])

        data, data_len = self.frontend(data, data_len, **kwargs) # [1, t, h]

        if isinstance(data_len, (list, tuple)):
            data_len = torch.tensor([data_len])
        return data.to(torch.float32), data_len.to(torch.int32)
    
    def extract_features(self, speech, fs=None):
        mel_feature, _ = self.extract_fbank(speech)
        return mel_feature.squeeze(0)  # (T, D)


class SeamlessM4TGen:
    def __init__(self, src_sr, frontend_path):
        self.frontend = SeamlessM4TFeatureExtractor.from_pretrained(frontend_path)
        self.src_sr = src_sr

    @torch.no_grad()
    def extract_features(self, speech, fs=None):
        speech_np = speech.squeeze(0).cpu().numpy()
        if self.src_sr != 16000:
            speech_np = librosa.resample(speech_np, orig_sr=self.src_sr, target_sr=16000)
        feats = self.frontend(speech_np, sampling_rate=16000, return_tensors="pt")
        speech_feat = feats["input_features"][0]
        # speech_feat_mask = feats["attention_mask"][0]
        return speech_feat 