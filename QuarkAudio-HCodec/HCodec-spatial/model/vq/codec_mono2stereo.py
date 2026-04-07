import torch
from torch import nn
import torch.nn.functional as F

from .encoder_modules import SEANetEncoder as CodecEncoder
from .codec_decoder import CodecDecoder
from .core_vq import ResidualVectorQuantization
from vector_quantize_pytorch import ResidualVQ
from .semantic_module import Encoder as SemanticEncoder, Decoder as SemanticDecoder
from .conv import Conv1d


class FixedLowpass(nn.Module):
    def __init__(self, channels, k=9):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, k, padding=k//2, groups=channels, bias=False)
        w = torch.ones(channels, 1, k) / k
        self.dw.weight = nn.Parameter(w, requires_grad=False)
    def forward(self, x): return self.dw(x)


class CondMLP(nn.Module):
    def __init__(self, cond_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, c): return self.net(c)


class DWGatedResBlock(nn.Module):
    def __init__(self, channels, k=7, dilation=1, dropout=0.0, gn_groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=min(gn_groups, channels), num_channels=channels)
        self.dw = nn.Conv1d(channels, channels, k, padding=(k//2)*dilation, dilation=dilation, groups=channels)
        self.pw = nn.Conv1d(channels, channels*2, 1)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, x):
        h = self.gn(x)
        h = F.silu(h)
        h = self.dw(h)
        h = self.pw(h)
        a, b = h.chunk(2, dim=1)
        h = a * torch.sigmoid(b)
        h = self.drop(h)
        return x + h


class SpatialMapNet(nn.Module):
    def __init__(self, channels=512, cond_dim=256, cond_hidden=1024, width=768,
                 num_blocks=10, dropout=0.05, use_lowpass=True, lowpass_k=9):
        super().__init__()
        self.az_emb  = nn.Embedding(360, cond_dim)
        self.el_emb  = nn.Embedding(180, cond_dim)
        self.dis_emb = nn.Embedding(50,  cond_dim)

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim*3, cond_hidden), nn.SiLU(),
            nn.Linear(cond_hidden, cond_hidden), nn.SiLU(),
        )

        self.to_scale = CondMLP(cond_hidden, cond_hidden, width)
        self.to_shift = CondMLP(cond_hidden, cond_hidden, width)
        self.to_gate  = CondMLP(cond_hidden, cond_hidden, width)

        self.mid_in = nn.Conv1d(channels, width, 1)
        self.sem_in = nn.Conv1d(channels, width, 1)

        self.side_base = nn.Conv1d(width, width, 1)
        nn.init.zeros_(self.side_base.weight)
        nn.init.zeros_(self.side_base.bias)

        self.cross_gate = nn.Conv1d(width, width, 1)
        nn.init.zeros_(self.cross_gate.weight)
        nn.init.zeros_(self.cross_gate.bias)

        dilations = [1, 1, 2, 2, 4, 4, 8, 8, 16, 16][:num_blocks]
        self.blocks = nn.Sequential(*[
            DWGatedResBlock(width, k=7, dilation=d, dropout=dropout) for d in dilations
        ])

        self.out_gn = nn.GroupNorm(num_groups=min(32, width), num_channels=width)
        self.out = nn.Conv1d(width, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.out_scale = nn.Parameter(torch.tensor(0.15))
        self.lowpass = FixedLowpass(channels, k=lowpass_k) if use_lowpass else nn.Identity()

    def forward(self, mid, sem, spatial_cond):
        sc = spatial_cond.long()
        c = torch.cat([self.az_emb(sc[:,0]), self.el_emb(sc[:,1]), self.dis_emb(sc[:,2])], dim=-1)
        c = self.cond_proj(c)

        B, D, T = mid.shape
        m = self.mid_in(mid)
        s = self.sem_in(sem)

        scale = self.to_scale(c).unsqueeze(-1)
        shift = self.to_shift(c).unsqueeze(-1)
        gatec = torch.tanh(self.to_gate(c)).unsqueeze(-1)

        g_sem = torch.tanh(self.cross_gate(s))
        base = self.side_base(m) * (0.5 * g_sem + 0.5 * gatec)

        x = m + 0.5 * s + base
        x = x * (1.0 + scale) + shift
        x = self.blocks(x)
        x = self.out(self.out_gn(F.silu(x)))
        x = self.lowpass(x)
        side = x * self.out_scale
        return side


class Codec_mono2stereo(nn.Module):
    def __init__(self, encoder_kwargs, decoder_kwargs, quantizer_kwargs, adaptive_kwargs):
        super().__init__()
        self.encoder = CodecEncoder(**encoder_kwargs['encoder'])
        self.decoder = CodecDecoder(**decoder_kwargs['decoder'])
        self.quantizer = ResidualVQ(**quantizer_kwargs['quantizer'])
        self.semantic_quantizer = ResidualVQ(**quantizer_kwargs['semantic_quantizer'])
        self.semantic_encoder = SemanticEncoder(**encoder_kwargs['semantic_encoder'])
        self.semantic_decoder = SemanticDecoder(**decoder_kwargs['semantic_decoder'])
        self.spatial_encoder = CodecEncoder(**encoder_kwargs['spatial_encoder'])
        self.spatial_quantizer = ResidualVQ(**quantizer_kwargs['spatial_quantizer'])
        self.spatial_mapping = SpatialMapNet()
        for key, value in adaptive_kwargs.items():
            setattr(self, key, value)

    @torch.no_grad()
    def spatialize(self, x, feat, spatial_cond=None):
        acoustic_emb = self.encoder(x.unsqueeze(1))
        semantic_emb = self.semantic_encoder(feat)
        quantized, codes, _ = self.quantizer(acoustic_emb.transpose(-2, -1))
        quantized = quantized.transpose(-2, -1)
        quantized_semantic, codes_semantic, _ = self.semantic_quantizer(semantic_emb.transpose(-2, -1))
        quantized_semantic = quantized_semantic.transpose(-2, -1)

        pre_spatial = self.spatial_mapping(quantized, quantized_semantic, spatial_cond.long())
        recon_L = self.decoder(torch.cat([quantized+pre_spatial, quantized_semantic], dim=1))
        recon_R = self.decoder(torch.cat([quantized-pre_spatial, quantized_semantic], dim=1))
        recon = torch.stack((recon_L, recon_R), dim=1)
        return recon
