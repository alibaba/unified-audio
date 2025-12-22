import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


from .codec_encoder import CodecEncoder
from .codec_decoder import CodecDecoder


from vector_quantize_pytorch import ResidualVQ
from .semantic_module import Encoder as SemanticEncoder, Decoder as SemanticDecoder
from .conv import Conv1d



class Codec(nn.Module):
    def __init__(
        self, 
        encoder_kwargs: dict,
        decoder_kwargs: dict,
        quantizer_kwargs: dict,
        semantic_encoder_kwargs: dict,
        semantic_decoder_kwargs: dict,
    ):
        super().__init__()

        self.encoder = CodecEncoder(
            **encoder_kwargs
        )
        
        self.decoder = CodecDecoder(
            **decoder_kwargs
        )


        self.quantizer = ResidualVQ(
            **quantizer_kwargs
        )

        self.semantic_quantizer = ResidualVQ(
            **quantizer_kwargs
        )


        self.semantic_encoder = SemanticEncoder(
            **semantic_encoder_kwargs
        )

        self.semantic_decoder = SemanticDecoder(
            **semantic_decoder_kwargs
        )
    
    def forward(self, x, feat):
        emb = self.encoder(x)
        semantic_emb = self.semantic_encoder(feat)
        
        # quantized: b,t,d
        # codes: b,t,layer
        # commit_loss: layer
        quantized, codes, commit_loss = self.quantizer(emb.transpose(-2, -1))
        quantized = quantized.transpose(-2, -1)
        commit_loss = commit_loss.mean()

        quantized_semantic, codes_semantic, commit_loss_semantic = self.semantic_quantizer(semantic_emb.transpose(-2, -1))
        quantized_semantic = quantized_semantic.transpose(-2, -1)
        commit_loss_semantic = commit_loss_semantic.mean()

        recon = self.decoder(torch.cat([quantized, quantized_semantic], dim=1))

        pred_feat = self.semantic_decoder(quantized_semantic)
        return recon, pred_feat, (commit_loss + commit_loss_semantic).mean()


    @torch.no_grad()
    def encode(self, x, feat):
        # [b,t], [b,d,t]
        emb = self.encoder(x)
        semantic_emb = self.semantic_encoder(feat)

        _, acoustic_codes, _ = self.quantizer(emb.transpose(-2, -1))  # b,t,nq
        _, semantic_codes, _ = self.semantic_quantizer(semantic_emb.transpose(-2, -1))  # b,t,nq

        acoustic_codes = acoustic_codes.transpose(-2, -1)
        semantic_codes = semantic_codes.transpose(-2, -1)  # b,nq,t
        
        return acoustic_codes, semantic_codes

    @torch.no_grad()
    def decode(self, acoustic_codes, semantic_codes):
        acoustic_codes = acoustic_codes.transpose(-2, -1)  # b,t,nq
        semantic_codes = semantic_codes.transpose(-2, -1)

        acoustic_emb = self.quantizer.get_output_from_indices(acoustic_codes).transpose(-2, -1)
        semantic_emb = self.semantic_quantizer.get_output_from_indices(semantic_codes).transpose(-2, -1)
        # import ipdb; ipdb.set_trace()
        recon = self.decoder(torch.cat([acoustic_emb, semantic_emb], dim=1))

        return recon



