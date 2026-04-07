import torch
from torch import nn
import torch.nn.functional as F

from .encoder_modules import SEANetEncoder as CodecEncoder
from .codec_decoder import CodecDecoder
from .core_vq import ResidualVectorQuantization
from vector_quantize_pytorch import ResidualVQ
from .semantic_module import Encoder as SemanticEncoder, Decoder as SemanticDecoder
from .conv import Conv1d


class Codec_stereo(nn.Module):
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
        for key, value in adaptive_kwargs.items():
            setattr(self, key, value)

    def forward(self, x, feat):
        # x: (b, 2, t)
        wav_content = (x[:,0] + x[:,1]) * 0.5
        wav_spatial = (x[:,0] - x[:,1]) * 0.5
        acoustic_emb = self.encoder(wav_content.unsqueeze(1))
        semantic_emb = self.semantic_encoder(feat)
        spatial_emb = self.spatial_encoder(wav_spatial.unsqueeze(1))

        quantized, codes, commit_loss = self.quantizer(acoustic_emb.transpose(-2, -1))
        quantized = quantized.transpose(-2, -1)
        commit_loss = commit_loss.mean()
        quantized_semantic, _, commit_loss_semantic = self.semantic_quantizer(semantic_emb.transpose(-2, -1))
        quantized_semantic = quantized_semantic.transpose(-2, -1)
        commit_loss_semantic = commit_loss_semantic.mean()
        quantized_spatial, _, commit_loss_spatial = self.spatial_quantizer(spatial_emb.transpose(-2, -1))
        quantized_spatial = quantized_spatial.transpose(-2, -1)
        commit_loss_spatial = commit_loss_spatial.mean()

        recon_L = self.decoder(torch.cat([quantized+quantized_spatial, quantized_semantic], dim=1))
        recon_R = self.decoder(torch.cat([quantized-quantized_spatial, quantized_semantic], dim=1))
        recon = torch.stack((recon_L, recon_R), dim=1)
        pred_feat = self.semantic_decoder(quantized_semantic)
        return recon, pred_feat, (commit_loss + commit_loss_semantic + commit_loss_spatial).mean()
