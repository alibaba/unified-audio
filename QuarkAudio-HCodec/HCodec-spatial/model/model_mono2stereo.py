import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import soundfile as sf
from pathlib import Path
from transformers import AutoModel

from .vq import Codec_mono2stereo


class Model_mono2stereo(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.generator = Codec_mono2stereo(config['encoder_config'], config['decoder_config'], config['quantizer_config'], config['adaptive_config'])

        wav2vec2_path = config.get('wav2vec2_path', 'facebook/wav2vec2-large-xlsr-53')
        self.feature_extractor = AutoModel.from_pretrained(wav2vec2_path).eval()
        self.feature_extractor.requires_grad_(False)

    def pad_wav(self, wav):
        hop_length = math.prod(self.config['encoder_config']['ratios']) * 2
        pad_length = math.ceil(wav.size(-1) / hop_length) * hop_length - wav.size(-1)
        wav = torch.nn.functional.pad(wav, (0, pad_length))
        return wav

    @torch.no_grad()
    def extract_wav2vec2_features(self, wavs):
        wavs = F.pad(wavs, (160, 160))
        feat = self.feature_extractor(wavs, output_hidden_states=True)
        feats_mix = (feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]) / 3
        symbol = (feats_mix > 0).float() * 2 - 1
        magnitude = feats_mix.abs() ** 0.3
        feats_mix = symbol * magnitude
        return feats_mix

    def test_step(self, batch, batch_idx):
        wav, spatial_info, fs, lengths, names = batch
        if wav.ndim == 3 and wav.shape[1] == 2:
            wav = (wav[:,0] + wav[:,1]) * 0.5
        elif wav.ndim == 3 and wav.shape[1] == 1:
            wav = wav.squeeze(1)
        wav_pad = self.pad_wav(wav)
        feat = self.extract_wav2vec2_features(wav_pad).transpose(-2, -1)
        recon = self.generator.spatialize(wav_pad, feat, spatial_info)
        est = recon[..., :wav.size(-1)].squeeze().cpu().numpy()
        if 'save_enhanced' in self.config and self.config['save_enhanced'] is not None:
            sf.write(Path(self.config['save_enhanced']) / f"{names[0]}", est.transpose(), samplerate=int(fs[0]))
