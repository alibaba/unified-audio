import torch
from torch import nn
from typing import List, Tuple, Optional
import numpy as np


# from .conv import SEANetResnetBlock, ConvNeXtBlock, Conv1d, ResnetBlock
from .conv import ConvNeXtBlock, ResnetBlock, AttnBlock, Normalize, Conv1d, Linear, init_weights, Transpose, ConvTranspose1d
from .heads import ISTFTHead
from .encoder_modules import Transformer



class CodecDecoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        convnext_layers: int = 12,
        n_fft: int = 1280,
        hop_length: int = 320,
        causal: bool = False,
    ):
        super().__init__()
        # self.embed = Conv1d(input_channels, dim, kernel_size=7, causal=causal)
        self.embed = ConvTranspose1d(input_channels, dim, kernel_size=5, stride=2, causal=causal)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.post_net = nn.Sequential(
            *[ConvNeXtBlock(dim=dim, intermediate_dim=intermediate_dim, causal=causal, layer_scale_init_value=1 / convnext_layers) for _ in range(convnext_layers)]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(init_weights)

        self.prior_net = nn.Sequential(
            ResnetBlock(in_channels=dim, out_channels=dim, dropout=0.1, causal=causal),
            ResnetBlock(in_channels=dim, out_channels=dim, dropout=0.1, causal=causal),
            # AttnBlock(dim, causal=causal),
            Transpose(-2, -1),
            Transformer(
                hidden_size=dim,
                intermediate_size=dim * 4,
                num_attention_heads=8,
                num_hidden_layers=2,
                use_moe=False,
                causal=causal,
            ),
            Transpose(-2, -1),
            ResnetBlock(in_channels=dim, out_channels=dim, dropout=0.1, causal=causal),
            ResnetBlock(in_channels=dim, out_channels=dim, dropout=0.1, causal=causal),
            Normalize(dim),
        )

        self.head = ISTFTHead(dim, n_fft, hop_length, padding='same')
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x = self.embed(x)
        x = self.prior_net(x)
        x = self.norm(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.post_net(x)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.final_layer_norm(x)
        x = self.head(x)  # [B, T]
        return x


if __name__ == '__main__':
    m = CodecDecoder(
        input_channels=512,
        dim=768,
        intermediate_dim=768*3,
        convnext_layers=12,
        n_fft=1280,
        hop_length=320,
        causal=False,
    ).cuda()

    print(sum([p.numel() for p in m.parameters()]))
    # x = torch.randn(1, 512, 55).cuda()
    # print(m(x).shape)


