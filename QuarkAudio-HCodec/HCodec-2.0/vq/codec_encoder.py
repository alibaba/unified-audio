import torch
from torch import nn
from typing import List, Tuple, Optional
import numpy as np
import torchaudio

from .conv import ConvNeXtBlock, ResnetBlock, AttnBlock, Normalize, Conv1d, Linear, init_weights, Transpose, ConvTranspose1d
from .heads import ISTFTHead
from .encoder_modules import Transformer


class CodecEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dimension: int,
        n_fft: int = 1920,
        hop_length: int = 960,
        convnext_layers: int = 12,
        transformer_layers: int = 2,
        target_frame_rate: float = 6.25,
        causal: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_freqs = n_fft // 2 + 1

        self.stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, center=False, power=None)
        # self.mel_spec = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, center=False, power=1,
        # )

        self.embed = Conv1d(self.n_freqs * 2, dim, kernel_size=3, causal=causal)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.prior_net = nn.Sequential(
            *[ConvNeXtBlock(dim=dim, intermediate_dim=intermediate_dim, causal=causal, layer_scale_init_value=1 / convnext_layers) for _ in range(convnext_layers)]
        )
        self.post_net = nn.Sequential(
            # Conv1d(dim, dimension, kernel_size=7, causal=causal),
            Transpose(-2, -1),
            Transformer(
                hidden_size=dim,
                intermediate_size=min(dim * 4, 4096),
                num_attention_heads=dim // 64,
                num_hidden_layers=transformer_layers,
                use_moe=False,
                causal=causal,
            ),
            Transpose(-2, -1),
        )

        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        assert target_frame_rate < 50
        stride = int(50 / target_frame_rate)
        self.out = Conv1d(dim, dimension, kernel_size=stride * 2 + 1, stride=stride, causal=causal)
        self.apply(init_weights)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        # 需要已经保证x的长度是 hop_length 的整数倍，在model.pad_wav里保证
        pad = (self.n_fft - self.hop_length) // 2
        x = torch.nn.functional.pad(x, (pad, pad))
        spec = self.stft(x)  # b,f,t, complex
        mag, phase = spec.abs(), spec.angle()
        mag = torch.log(torch.clip(mag, min=1e-5))
        phase = phase / torch.pi
        x = torch.cat([mag, phase], dim=1)

        x = self.embed(x)
        x = self.norm(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.prior_net(x)
        x = self.post_net(x)
        x = self.final_layer_norm(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.out(x)  # [B, C, T]
        return x


if __name__ == '__main__':
    m = CodecEncoder(dim=768, intermediate_dim=1536, dimension=512, convnext_layers=1)
    x = torch.randn(1, 48000)
    print(m(x).shape)


