import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


# from .codec_encoder import CodecEncoder
from .encoder_modules import SEANetEncoder as CodecEncoder
from .codec_decoder import CodecDecoder
# from .vocos import VocosBackbone as CodecDecoder
# from .simvq import SimVQ1D
# from .residual_vq import ResidualVQ
from .core_vq import ResidualVectorQuantization
from vector_quantize_pytorch import ResidualSimVQ, ResidualFSQ, ResidualVQ
# from .heads import ISTFTHead
from .semantic_module import Encoder as SemanticEncoder, Decoder as SemanticDecoder
from .conv import Conv1d



class Codec(nn.Module):
    def __init__(
        self, 
        encoder_kwargs: dict,
        decoder_kwargs: dict,
        quantizer_kwargs: dict,
    ):
        super().__init__()

        self.encoder = CodecEncoder(
            causal=False, n_residual_layers=1, norm='weight_norm', pad_mode='reflect', lstm=2,
            dimension=512, channels=1, n_filters=32, ratios=[8, 5, 4, 2], activation='ELU',
            kernel_size=7, residual_kernel_size=3, last_kernel_size=7, dilation_base=2,
            true_skip=False, compress=2, use_transformer=True,
        )
        # self.encoder = CodecEncoder(
        #     causal=False, n_residual_layers=1, lstm=2,
        #     dimension=512, channels=1, n_filters=32, ratios=[2, 4, 5, 8], activation='ELU',
        #     kernel_size=7, residual_kernel_size=3, last_kernel_size=7, dilation_base=2,
        #     true_skip=False, compress=2
        # )
        
        self.decoder = CodecDecoder(
            input_channels=512 * 2,
            dim=768,
            intermediate_dim=2304,
            # num_layers=12,
        )
        # self.quantizer = SimVQ1D(**quantizer_kwargs)
        # self.quantizer = ResidualVQ(
        #     **quantizer_kwargs
        # )
        # self.quantizer = ResidualVectorQuantization(
        #     dim = 512,
        #     codebook_size = 1024,
        #     num_quantizers = 2,
        #     decay = 0.99,
        #     kmeans_init = True,
        #     kmeans_iters = 200,
        #     threshold_ema_dead_code = 2,
        # )

        # self.semantic_quantizer = ResidualVectorQuantization(
        #     dim = 512,
        #     codebook_size = 1024,
        #     num_quantizers = 2,
        #     decay = 0.99,
        #     kmeans_init = True,
        #     kmeans_iters = 200,
        #     threshold_ema_dead_code = 2,
        # )

        # self.quantizer = ResidualSimVQ(
        #     dim = 512,
        #     codebook_size = 1024,
        #     num_quantizers = 4,
        #     rotation_trick = True,
        # )

        # self.semantic_quantizer = ResidualSimVQ(
        #     dim = 512,
        #     codebook_size = 1024,
        #     num_quantizers = 4,
        #     rotation_trick = True,
        # )


        # self.quantizer = ResidualFSQ(
        #     dim = 512,
        #     levels = [4, 4, 4, 4, 4],
        #     num_quantizers = 4,
        # )

        # self.semantic_quantizer = ResidualFSQ(
        #     dim = 512,
        #     levels = [4, 4, 4, 4, 4],
        #     num_quantizers = 4,
        # )


        self.quantizer = ResidualVQ(
            dim = 512,
            codebook_size = 1024,
            num_quantizers = 4,
            decay = 0.99,
            kmeans_init = True,
            kmeans_iters = 50,
            quantize_dropout = True,
        )

        self.semantic_quantizer = ResidualVQ(
            dim = 512,
            codebook_size = 1024,
            num_quantizers = 4,
            decay = 0.99,
            kmeans_init = True,
            kmeans_iters = 50,
            quantize_dropout = True,
        )


        self.semantic_encoder = SemanticEncoder(
            input_channels=768,
            encode_channels=768,
            out_channels=512,
            channel_ratios=(1, 1),
            strides=(2, 1),
        )

        self.semantic_decoder = SemanticDecoder(
            code_dim=512,
            output_channels=768,
            decode_channels=768,
            channel_ratios=(1, 1),
            strides=(2, 1),
        )
    
    def forward(self, x, feat, use_mask=False, domain_split=None):
        # [b,1,t]
        # cnn_feat, mask_indices, emb = self.encoder(x, use_mask=use_mask)
        # # quantized, codes, commit_loss = self.quantizer(emb, domain_split=domain_split)
        # quantized, codes, commit_loss = self.quantizer(emb)
        # recon = self.decoder(quantized)  # [b,t]
        # return recon, commit_loss.mean(), cnn_feat, mask_indices, quantized
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
        # recon = self.head(x)
        return recon, pred_feat, (commit_loss + commit_loss_semantic).mean()


    @torch.no_grad()
    def encode(self, x, feat):
        # [b,1,t]
        emb = self.encoder(x)
        semantic_emb = self.semantic_encoder(feat)
        _, acoustic_codes, _ = self.quantizer(emb.transpose(-2, -1))  # b,t,nq
        _, semantic_codes, _ = self.semantic_quantizer(semantic_emb.transpose(-2, -1))
        acoustic_codes = acoustic_codes.transpose(-2, -1)
        semantic_codes = semantic_codes.transpose(-2, -1)  # b,nq,t
        return acoustic_codes, semantic_codes

    
    @torch.no_grad()
    def decode(self, acoustic_codes, semantic_codes):
        acoustic_codes = acoustic_codes.transpose(-2, -1)  # b,t,nq
        semantic_codes = semantic_codes.transpose(-2, -1)

        acoustic_emb = self.quantizer.get_output_from_indices(acoustic_codes).transpose(-2, -1)
        semantic_emb = self.semantic_quantizer.get_output_from_indices(semantic_codes).transpose(-2, -1)
        recon = self.decoder(torch.cat([acoustic_emb, semantic_emb], dim=1))

        return recon
    
    
    @torch.no_grad()
    def get_quantized_emb(self, x, feat):
        # [b,1,t]
        # emb = self.encoder(x)
        # semantic_emb = self.semantic_encoder(feat)
        # quantized, codes, commit_loss = self.quantizer(emb)
        # quantized_semantic, codes_semantic, commit_loss_semantic = self.semantic_quantizer(semantic_emb)
        # return quantized, quantized_semantic  # [b,d,t]
        pass



