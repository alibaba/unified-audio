'''
API about dynmaic frame rate is implemented based on
[FlexiCodec](https://github.com/amphionspace/FlexiCodec) and [VARSTok](https://github.com/FunAudioLLM/FunResearch/tree/main/VARSTok)
'''


import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import random 

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

from adaptive.model_blocks.mimi.transformer import ProjectedTransformer, QueryTokenAggregator
from adaptive.modeling_flexicodec_new import FlexiCodec  

class Codec(nn.Module):
    def __init__(
        self, 
        encoder_kwargs: dict,
        decoder_kwargs: dict,
        quantizer_kwargs: dict,
        adaptive_kwargs: dict
    ):
        super().__init__()

        self.encoder = CodecEncoder(**encoder_kwargs['encoder'])          
        self.decoder = CodecDecoder(**decoder_kwargs['decoder'])    
        self.quantizer = ResidualVQ(**quantizer_kwargs['quantizer'])   

        self.semantic_quantizer = ResidualVQ(**quantizer_kwargs['semantic_quantizer'])
        self.semantic_encoder = SemanticEncoder(**encoder_kwargs['semantic_encoder']) 
        self.semantic_decoder = SemanticDecoder(**decoder_kwargs['semantic_decoder'])   
        ###################################### Adaptive Frame Rate ######################################
        for key, value in adaptive_kwargs.items():
            setattr(self, key, value)

        if self.use_bottleneck_transformer:
            transformer_kwargs = self.transformer_kwargs
            print('[#] using use_bottleneck_transformer ...')
            self.bottleneck_transformer = ProjectedTransformer(**transformer_kwargs)
        else:
            self.bottleneck_transformer = nn.Identity()

        if self.use_similarity_alignment:
            if not self.use_dynamic_similarity_threshold:
                assert self.similarity_threshold is not None, "similarity_threshold must be set when use_similarity_alignment=True and use_dynamic_similarity_threshold=False"
            else:
                assert self.similarity_threshold_lower < self.similarity_threshold_upper, "similarity_threshold_lower must be less than similarity_threshold_upper"
 
        if self.use_query_token_aggregator:
            self.semantic_aggregator = QueryTokenAggregator(**self.aggregators['semantic_aggregator'])
            self.acoustic_aggregator = QueryTokenAggregator(**self.aggregators['acoustic_aggregator'])
        
        self.codebook_size = quantizer_kwargs['quantizer']['codebook_size']
    
    def _inject_length_to_codes_index(self, codes, token_lengths):
        # codes.shape [B,num_qunant,T]; token_lengths.shape [B,T]
        # inject length information to codes via codes_new = (token_lengths - 1) * self.codebook_size + code_old
        tl = token_lengths.unsqueeze(1).to(dtype=codes.dtype)
        codes_new = (tl - 1) * self.codebook_size + codes
        return codes_new

    def _extract_length_from_codes_index(self, codes):
        # length_id = floor(codes / codebook_size) + 1
        length_id = torch.div(codes, self.codebook_size, rounding_mode="floor") + 1  # [B, num_quant, T]
        codes_plain = codes % self.codebook_size  # [B, num_quant, T]
        token_lengths = length_id[:, 0, :]  # [B, T]
        return codes_plain, token_lengths

    def _get_current_similarity_threshold(self) -> float:
        """
        Get the current similarity threshold for alignment.
        If using dynamic threshold, returns a random value between lower and upper bounds.
        Otherwise, returns the fixed threshold.
        
        Returns:
            float: Current similarity threshold value
        """
        if self.manual_threshold is not None:
            return float(self.manual_threshold)
        elif (self.use_dynamic_similarity_threshold and self.training) or self.infer_using_dynamic_threshold:
            # Sample a random threshold between lower and upper bounds
            threshold = random.uniform(self.similarity_threshold_lower, self.similarity_threshold_upper)
            return threshold
        else:
            return self.similarity_threshold      

    def forward(self, x, feat, use_mask=False, domain_split=None):
        # [b,1,t]
        
        # print('\n',x.shape,end='\n\n')  
        emb = self.encoder(x) # emb.shape [B, D, T]
        semantic_emb = self.semantic_encoder(feat) # [B, D, T]
        
        B, D, T = semantic_emb.shape  
        x_lens = torch.ones(B, dtype=torch.long)*T  
        x_lens = x_lens.to(semantic_emb.device) 
        current_threshold = self._get_current_similarity_threshold() 
        # _perform_similarity_alignment_vectorized input shape [B, T, D]
        alignment_matrices, sim, num_segments_per_item = FlexiCodec._perform_similarity_alignment_vectorized(semantic_emb.transpose(-2, -1), x_lens=x_lens, current_threshold=current_threshold, max_tokens_per_group=self.max_tokens_per_group)
               
        # print(f'AlignMatrix.shape:{alignment_matrices.shape}\tThreshold:{current_threshold}\n[Before] AcousticEmb.shape:{emb.shape}\tSemanticEmb.shape:{semantic_emb.shape}')      
        semantic_emb = self.semantic_aggregator(semantic_emb, alignment_matrices, num_segments_per_item)
        emb = self.acoustic_aggregator(emb, alignment_matrices, num_segments_per_item)        
        # print(f'[After] AcousticEmb.shape:{emb.shape}\tSemanticEmb.shape:{semantic_emb.shape}') 
 
        # quantized: b,t,d
        # codes: b,t,layer
        # commit_loss: layer
        quantized, codes, commit_loss = self.quantizer(emb.transpose(-2, -1))
        quantized = quantized.transpose(-2, -1)
        commit_loss = commit_loss.mean()
        quantized_semantic, codes_semantic, commit_loss_semantic = self.semantic_quantizer(semantic_emb.transpose(-2, -1))
        quantized_semantic = quantized_semantic.transpose(-2, -1)
        commit_loss_semantic = commit_loss_semantic.mean()

        # print(quantized.shape, quantized_semantic.shape)
        # assert quantized.shape[-1] == quantized_semantic.shape[-1], (quantized.shape, quantized_semantic.shape)   
        quantized = FlexiCodec.deaggregate_features(quantized, alignment_matrices, is_channel_last=False)    
        quantized_semantic = FlexiCodec.deaggregate_features(quantized_semantic, alignment_matrices, is_channel_last=False) 
        # [B,D,T]   
        # assert quantized.shape[-1] == quantized_semantic.shape[-1], (quantized.shape, quantized_semantic.shape)  
        cat_code = torch.cat([quantized, quantized_semantic], dim=1)
        cat_code_final = self.bottleneck_transformer(cat_code)
        recon = self.decoder(cat_code_final)  

        pred_feat = self.semantic_decoder(quantized_semantic)
        # recon = self.head(x)
        ret_dict={
            'recon': recon,
            'pred_feat': pred_feat,
            'commit_loss': (commit_loss + commit_loss_semantic).mean(),
            'token_lengths': alignment_matrices.sum(dim=2).long()
        }
        return ret_dict 

    @torch.no_grad()
    def encode(self, x, feat, use_mask=False, domain_split=None, threshold=0.0):
        assert 0 <= threshold <= 1.0
        # [b,1,t]
        emb = self.encoder(x)
        semantic_emb = self.semantic_encoder(feat)

        B, D, T = semantic_emb.shape  
        x_lens = torch.ones(B, dtype=torch.long)*T  
        x_lens = x_lens.to(semantic_emb.device) 
        current_threshold = self._get_current_similarity_threshold() if threshold <= 0.0 else threshold

        alignment_matrices, sim, num_segments_per_item = FlexiCodec._perform_similarity_alignment_vectorized(semantic_emb.transpose(-2, -1), x_lens=x_lens, current_threshold=current_threshold, max_tokens_per_group=self.max_tokens_per_group)     
        semantic_emb = self.semantic_aggregator(semantic_emb, alignment_matrices, num_segments_per_item)
        emb = self.acoustic_aggregator(emb, alignment_matrices, num_segments_per_item)      

        qa, acoustic_codes, _ = self.quantizer(emb.transpose(-2, -1))  # b,t,nq
        qs, semantic_codes, _ = self.semantic_quantizer(semantic_emb.transpose(-2, -1))
        
        acoustic_codes = acoustic_codes.transpose(-2, -1)
        semantic_codes = semantic_codes.transpose(-2, -1)  # b,nq,t
        token_lengths = alignment_matrices.sum(dim=2).long()
        acoustic_codes = self._inject_length_to_codes_index(acoustic_codes,token_lengths)
        semantic_codes = self._inject_length_to_codes_index(semantic_codes,token_lengths)
        ret_dict ={
            'acoustic_codes': acoustic_codes,
            'semantic_codes': semantic_codes,
            # 'token_lengths': token_lengths
        }
        return ret_dict
    
    @torch.no_grad()
    def decode(self, acoustic_codes, semantic_codes, token_lengths=None):
        
        if token_lengths is None:
            acoustic_codes, token_lengths = self._extract_length_from_codes_index(acoustic_codes)
            semantic_codes, token_lengths = self._extract_length_from_codes_index(semantic_codes)

        acoustic_codes = FlexiCodec._deaggregate_features_from_token_lengths(acoustic_codes, token_lengths)    # acoustic_emb.shape:[B,D,T]; token_lengths.shape:[B,T]
        semantic_codes = FlexiCodec._deaggregate_features_from_token_lengths(semantic_codes, token_lengths) 
        # [B,D,T]   
        
        quantized = self.quantizer.get_output_from_indices(acoustic_codes.transpose(-2, -1)).transpose(-2, -1)
        quantized_semantic = self.semantic_quantizer.get_output_from_indices(semantic_codes.transpose(-2, -1)).transpose(-2, -1)

        cat_code = torch.cat([quantized, quantized_semantic], dim=1)
        cat_code_final = self.bottleneck_transformer(cat_code)

        recon = self.decoder(cat_code_final)

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