import torch
from torch import nn
import torch.nn.functional as F

from .encoder_modules import SEANetEncoder as CodecEncoder
from .codec_decoder import CodecDecoder
from .core_vq import ResidualVectorQuantization
from vector_quantize_pytorch import ResidualVQ
from .semantic_module import Encoder as SemanticEncoder, Decoder as SemanticDecoder
from .conv import Conv1d


class Codec(nn.Module):
    def __init__(self, encoder_kwargs, decoder_kwargs, quantizer_kwargs, adaptive_kwargs):
        super().__init__()
        self.encoder = CodecEncoder(**encoder_kwargs['encoder'])
        self.decoder = CodecDecoder(**decoder_kwargs['decoder'])
        self.quantizer = ResidualVQ(**quantizer_kwargs['quantizer'])
        self.semantic_quantizer = ResidualVQ(**quantizer_kwargs['semantic_quantizer'])
        self.semantic_encoder = SemanticEncoder(**encoder_kwargs['semantic_encoder'])
        self.semantic_decoder = SemanticDecoder(**decoder_kwargs['semantic_decoder'])
        for key, value in adaptive_kwargs.items():
            setattr(self, key, value)

    def forward(self, x, feat):
        # x: (b, 1, t)
        emb = self.encoder(x)
        semantic_emb = self.semantic_encoder(feat)

        quantized, codes, commit_loss = self.quantizer(emb.transpose(-2, -1))
        quantized = quantized.transpose(-2, -1)
        commit_loss = commit_loss.mean()
        quantized_semantic, _, commit_loss_semantic = self.semantic_quantizer(semantic_emb.transpose(-2, -1))
        quantized_semantic = quantized_semantic.transpose(-2, -1)
        commit_loss_semantic = commit_loss_semantic.mean()

        recon = self.decoder(torch.cat([quantized, quantized_semantic], dim=1))
        pred_feat = self.semantic_decoder(quantized_semantic)
        return recon, pred_feat, (commit_loss + commit_loss_semantic).mean()
