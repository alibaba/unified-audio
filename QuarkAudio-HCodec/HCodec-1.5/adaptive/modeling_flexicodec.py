from .cnn import ConvNeXtBlock
import torch.nn as nn
import math
from typing import List
from typing import Union
import torch
from torch import nn
from transformers import Wav2Vec2BertModel
from dac.nn.layers import WNConv1d
from .dac_quantize import ResidualVectorQuantize
from .fsq_wrapper import FSQWrapper
from easydict import EasyDict as edict
import torch.nn.functional as F
import random
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
# Import FunASR for direct model usage
from funasr import AutoModel
from .model_blocks.mimi.transformer import ProjectedTransformer, QueryTokenAggregator

@torch.no_grad()
def _extract_semantic_code(
    semantic_model,
    input_features,
    attention_mask,
    mean,
    std,
    skip_normalize= False,
    sensevoice_prepend_inputs= True,
    sim_layer_idx=None,
    semantic_layer_idx=None,
):
    """Return `(semantic_repr, sim_repr)` in (B, T, C) format.

    * For Wav2Vec2-BERT both outputs are the same.
    * For SenseVoice we select hidden layers according to the supplied indices.
    """
    
    # Check if using FunASR model (SenseVoice)
    if isinstance(semantic_model, Wav2Vec2BertModel):
        # Wav2Vec2BertModel should run in full precision (ASR not in half)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            vq_emb = semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            output_idx = 17
            feat = vq_emb.hidden_states[output_idx]  # (B, T, C)
            if not skip_normalize:
                feat = (feat - mean) / std
        return feat, feat # layer selection not implemented
    
    elif hasattr(semantic_model, 'encoder'):
        # For FunASR model, we need to pass audio_features_lengths
        # Create dummy lengths based on attention_mask or input_features shape
        audio_features_lengths = torch.tensor([input_features.shape[1]] * input_features.shape[0], 
                                            device=input_features.device, dtype=torch.long)
        
        # Check if we need to prepend inputs (similar to SenseVoiceAudioEncoder.forward_encoder)
        # For SenseVoice, we typically want to prepend inputs
        if sensevoice_prepend_inputs:
            input_features, audio_features_lengths = semantic_model.prepend_inputs(
                input_features, audio_features_lengths
            ) # [b,t+4,c]
        
        # Call FunASR model encoder directly and explicitly disable autocast (ASR not in half)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            encoder_out, encoder_out_lengths, hidden_out, hiddens = semantic_model.encoder(
                input_features, audio_features_lengths, extract_hidden=True
            )
        
        if semantic_layer_idx is None:
            semantic_feat = hidden_out[:, 4:]
            return semantic_feat, semantic_feat
        else:
            # Support range/list for sim_layer_idx
            if isinstance(sim_layer_idx, (list, tuple, range)):
                sim_feat = torch.stack([hiddens[idx] for idx in sim_layer_idx], dim=0).mean(dim=0)
            elif sim_layer_idx is None:
                sim_feat = hidden_out
            else:
                sim_feat = hiddens[sim_layer_idx]
            if isinstance(semantic_layer_idx, (list, tuple, range)):
                if semantic_layer_idx[1] == -1:
                    semantic_layer_idx[1] = len(hiddens)
                semantic_feat = torch.stack([hiddens[idx] for idx in range(semantic_layer_idx[0], semantic_layer_idx[1])], dim=0).mean(dim=0)
            else:
                semantic_feat = hiddens[semantic_layer_idx]
            sim_feat = sim_feat[:, 4:]
            semantic_feat = semantic_feat[:, 4:]
            return semantic_feat, sim_feat

    else:
        raise ValueError(f"Unsupported semantic model type: {type(semantic_model)}")

class FlexiCodec(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 16384,
        codebook_dim: Union[int, list] = 8,
        semantic_codebook_dim=8,
        fsq_config: dict = None,
        quantizer_dropout: bool = True,
        sample_rate: int = 24000,
        distill_projection_out_dim=1024,
        convnext_dim=768,
        convnext_layers=4,
        decode_semantic_for_codec=True,
        is_causal=True,
        semantic_downsample_factor=2,
        semantic_encoder=None,
        semantic_decoder=None,
        ssl_dim=1024,
        semantic_model_path="./SenseVoiceSmall", 
        lambda_distill_loss=15.0,
        # New parameters for similarity-based alignment
        use_similarity_alignment: bool = False, # if false, is original dualcodec
        similarity_threshold=None,
        # Dynamic similarity threshold parameters
        use_dynamic_similarity_threshold: bool = False,
        similarity_threshold_lower: float = 0.8,
        similarity_threshold_upper: float = 1.0,
        skip_normalize=True,
        max_tokens_per_group: int | None = 8,
        semantic_model_type: str = "sensevoice", 
        sensevoice_prepend_inputs: bool = True,  # Whether to prepend inputs before encoding
        # Bottleneck transformer parameters
        use_bottleneck_transformer: bool = False, 
        bottleneck_transformer_config: dict = None,
        transformer_num_layers: int = 6,
        transformer_dim: int = 512,
        transformer_dim_feedforward: int = 2048,
        transformer_num_heads: int = 8,
        transformer_causal: bool = False,
        transformer_context_frames: int = 24,
        # Aggregator transformer parameters
        use_query_token_aggregator: bool = False,
        agg_transformer_num_layers: int = 6,
        agg_transformer_dim: int = 512,
        agg_transformer_num_heads: int = 8,
        agg_transformer_dim_feedforward: int = 2048,
        agg_transformer_causal: bool = False,
        agg_use_mean_pooling_init: bool = True,
        agg_transformer_context_frames: int = None,
        sensevoice_sim_layer_idx=None,
        sensevoice_semantic_layer_idx=None,
    ):
        super().__init__()
        self.semantic_downsample_factor = semantic_downsample_factor
        self.concat_downsample_factor = 1
        self.lambda_distill_loss = lambda_distill_loss
        self.semantic_encoder = semantic_encoder
        self.semantic_decoder = semantic_decoder
        self.use_fsq_for_semantic_vq = True
        if agg_transformer_context_frames is None:
            agg_transformer_context_frames = transformer_context_frames
        
        # Similarity alignment parameters
        self.use_similarity_alignment = use_similarity_alignment
        self.similarity_threshold = similarity_threshold
        self.use_dynamic_similarity_threshold = use_dynamic_similarity_threshold
        self.similarity_threshold_lower = similarity_threshold_lower
        self.similarity_threshold_upper = similarity_threshold_upper
        
        if self.use_similarity_alignment:
            if not self.use_dynamic_similarity_threshold:
                assert self.similarity_threshold is not None, "similarity_threshold must be set when use_similarity_alignment=True and use_dynamic_similarity_threshold=False"
            else:
                assert self.similarity_threshold_lower < self.similarity_threshold_upper, "similarity_threshold_lower must be less than similarity_threshold_upper"
        self.skip_normalize = skip_normalize
        self.max_tokens_per_group = max_tokens_per_group if (max_tokens_per_group is None or max_tokens_per_group > 0) else None
        self.use_query_token_aggregator = use_query_token_aggregator
        
        # Bottleneck transformer parameters
        self.use_bottleneck_transformer = use_bottleneck_transformer
        if self.use_bottleneck_transformer:
            transformer_kwargs = {
                'd_model': transformer_dim,
                'num_heads': transformer_num_heads,
                'num_layers': transformer_num_layers,
                'causal': transformer_causal,
                'layer_scale': 0.01,
                'context': transformer_context_frames,  # Use calculated context window
                'conv_layout': True,
                'max_period': 10000,
                'gating': 'none',
                'norm': 'layer_norm',
                'positional_embedding': 'rope',
                'dim_feedforward': transformer_dim_feedforward,
                'input_dimension': latent_dim,
                'output_dimensions': [latent_dim],
            }
            if transformer_num_layers == 0:
                self.bottleneck_transformer = nn.Identity()
            else:
                self.bottleneck_transformer = ProjectedTransformer(**transformer_kwargs)

        else:
            self.bottleneck_transformer = nn.Identity()
        
        # Initialize semantic model based on type
        self.semantic_model_type = semantic_model_type
        self.sensevoice_prepend_inputs = sensevoice_prepend_inputs
        
        if semantic_model_type == "sensevoice":
            # reset semantic downsample factor
            ssl_dim=512
            # Store SenseVoice specific parameters
            
            from pathlib import Path
            sensevoice_model_code_dir = f'{str(Path(__file__).parent)}/customized_sensevoice/model.py'
            # Initialize FunASR model directly
            funasr_model = AutoModel(
                model=semantic_model_path,
                trust_remote_code=True,
                remote_code=sensevoice_model_code_dir,
                device="cpu",
                disable_update=True
            )
            # Set semantic_model to the model directly, similar to audio_encoder.py
            self.semantic_model = funasr_model.model
            # For FunASR model, we don't need mean/var stats as normalization is handled internally
            self.register_buffer("semantic_mean", torch.zeros(1))
            self.register_buffer("semantic_std", torch.ones(1))
            self.semantic_model_path = semantic_model_path
        else:
            from pathlib import Path
            # Default Wav2Vec2BertModel initialization
            mean_var_path = f'{str(Path(__file__).parent)}/w2vbert2_mean_var_stats_emilia.pt'
            stat_mean_var = torch.load(mean_var_path)
            self.register_buffer("semantic_mean", stat_mean_var["mean"])
            self.register_buffer("semantic_std", stat_mean_var["var"])
            self.semantic_model_path = semantic_model_path
            self.semantic_model = Wav2Vec2BertModel.from_pretrained(self.semantic_model_path).eval()
        
        self.freeze_semantic_model()
        self.trainer_callbacks = None
        self.sample_rate = sample_rate
        self.infer_using_dynamic_threshold = False
        from .dac_model import DAC
        self.dac = DAC(
            encoder_dim,
            encoder_rates,
            latent_dim,
            decoder_dim,
            decoder_rates,
            n_codebooks,
            codebook_size,
            codebook_dim,
            quantizer_dropout,
            sample_rate,
            distill_projection_out_dim,
            distill=False,
        )

        self.decode_semantic_for_codec = decode_semantic_for_codec
        self.encoder_rates = encoder_rates
        self.ssl_dim = ssl_dim
        self.dac_bn_dim = self.dac.latent_dim
        self.manual_threshold = None

        self.convnext_encoder = nn.Sequential(
            WNConv1d(
                self.ssl_dim, convnext_dim, kernel_size=1,
            ),
            *[ConvNeXtBlock(
                dim=convnext_dim,
                intermediate_dim=2048,
                is_causal=is_causal
            ) for _ in range(convnext_layers)],  # Unpack the list directly into nn.Sequential
        )
        if semantic_encoder is not None:
            self.convnext_encoder = semantic_encoder

        if self.use_fsq_for_semantic_vq:
            fsq_params = (fsq_config or {}).copy()
            self.semantic_vq = FSQWrapper(
                input_dim=convnext_dim,
                **fsq_params
            )
        else:
            self.semantic_vq = ResidualVectorQuantize(
                convnext_dim, n_codebooks=1, codebook_size=semantic_codebook_size,
                codebook_dim=semantic_codebook_dim,
            )

        self.convnext_decoder = nn.Sequential(
            *[ConvNeXtBlock(
                dim=convnext_dim,
                intermediate_dim=2048,
                is_causal=is_causal,
            ) for _ in range(convnext_layers)],  # Unpack the list directly into nn.Sequential
            WNConv1d(
                convnext_dim, self.dac_bn_dim, kernel_size=1,
            ),
        )
        if semantic_decoder is not None:
            self.convnext_decoder = semantic_decoder
            if self.use_fsq_for_semantic_vq:
                fsq_params = (fsq_config or {}).copy()
                self.semantic_vq = FSQWrapper(
                    input_dim=1024,
                    **fsq_params
                )
            else:
                self.semantic_vq = ResidualVectorQuantize(
                    1024, n_codebooks=1, codebook_size=semantic_codebook_size,
                    codebook_dim=semantic_codebook_dim,
                )
        # if not self.decode_semantic_for_codec:
        #     assert convnext_dim == 1024
        self.sensevoice_sim_layer_idx = sensevoice_sim_layer_idx
        self.sensevoice_semantic_layer_idx = sensevoice_semantic_layer_idx

        if self.use_query_token_aggregator:
            self.semantic_aggregator = QueryTokenAggregator(
                dim=agg_transformer_dim,
                in_out_dim=ssl_dim,
                num_heads=agg_transformer_num_heads,
                num_layers=agg_transformer_num_layers,
                dim_feedforward=agg_transformer_dim_feedforward,
                causal=agg_transformer_causal,
                use_mean_pooling_init=agg_use_mean_pooling_init,
                context_frames=transformer_context_frames,
            )
            self.acoustic_aggregator = QueryTokenAggregator(
                dim=self.dac_bn_dim, # latent_dim
                in_out_dim=self.dac_bn_dim,
                num_heads=agg_transformer_num_heads,
                num_layers=agg_transformer_num_layers,
                dim_feedforward=agg_transformer_dim_feedforward,
                causal=agg_transformer_causal,
                use_mean_pooling_init=agg_use_mean_pooling_init,
                context_frames=agg_transformer_context_frames,
            )
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"FlexiCodec - Total Parameters: {total_params / 1e6:.2f}M")
        print(f"FlexiCodec - Trainable Parameters: {trainable_params / 1e6:.2f}M")
        
        # Print detailed submodule analysis
        # self.print_submodule_params()
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

    def _get_current_aggregator_downsample_ratio(self) -> int:
        """
        Get the current aggregator downsample ratio.
        If aggregator_downsample_ratio_options is provided, randomly selects from the options during training.
        Otherwise, returns the fixed aggregator_downsample_ratio.
        
        Returns:
            int: Current aggregator downsample ratio value
        """
        if self.aggregator_downsample_ratio_options is not None and self.training:
            # Select randomly from aggregator_downsample_ratio_options during training
            ratio = random.choice(self.aggregator_downsample_ratio_options)
            return int(ratio)
        else:
            return int(self.aggregator_downsample_ratio)

    def _downsample_semantic_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Downsample semantic features using either avg_pool1d (for integer factors) 
        or interpolate (for fractional factors).
        
        Args:
            features: torch.Tensor, shape (B, C, T) - semantic features to downsample
            
        Returns:
            torch.Tensor: downsampled features
        """
        if self.semantic_downsample_factor == 1:
            return features
        
        # Check if downsample factor is an integer
        if self.semantic_downsample_factor == int(self.semantic_downsample_factor):
            # Use avg_pool1d for integer factors
            return torch.nn.functional.avg_pool1d(
                features,
                self.semantic_downsample_factor,
                self.semantic_downsample_factor,
            )
        else:
            # Use interpolate for fractional factors
            target_length = int(features.shape[-1] / self.semantic_downsample_factor)
            return torch.nn.functional.interpolate(
                features,
                size=target_length,
                mode='linear',
                align_corners=False
            )

    def _downsample_x_lens(self, x_lens: torch.Tensor) -> torch.Tensor:
        """
        Downsample x_lens by the same factor as semantic features.
        
        Args:
            x_lens: torch.Tensor, shape (B,) - original feature lengths
            
        Returns:
            torch.Tensor: downsampled lengths
        """
        if self.semantic_downsample_factor == 1 or x_lens is None:
            return x_lens
        
        # Check if downsample factor is an integer
        if self.semantic_downsample_factor == int(self.semantic_downsample_factor):
            # For integer factors, divide and round up to ensure we don't lose valid frames
            downsampled_lens = torch.div(x_lens, self.semantic_downsample_factor, rounding_mode='floor')
        else:
            # For fractional factors, apply the same scaling
            downsampled_lens = (x_lens.float() / self.semantic_downsample_factor).long()
        
        # Ensure we have at least 1 frame if original length > 0
        downsampled_lens = torch.where(x_lens > 0, torch.clamp(downsampled_lens, min=1), downsampled_lens)
        
        return downsampled_lens

    def freeze_semantic_model(self):
        """Freeze all parameters in the semantic model."""
        # Freeze FunASR model parameters
        for param in self.semantic_model.parameters():
            param.requires_grad = False
    def semantic_quantize(self, semantic_repr):
        semantic = self.convnext_encoder(semantic_repr)
            
        semantic, codes, latents, commitment_loss, codebook_loss, first_layer_quantized = self.semantic_vq(semantic)
        codes = rearrange(codes, 'b 1 t -> b t')
        return codes

    def get_semantic_feature(self, semantic_codes):
        """
        Get the semantic feature from semantic codes.
        Args:
            semantic_codes: torch.Tensor, shape (B, n_q_s, G)
        Returns:
            torch.Tensor: semantic feature, shape (B, C, T)
        """
        semantic = self.semantic_vq.from_codes(semantic_codes)[0]
        if self.decode_semantic_for_codec:
            semantic_decoded = self.convnext_decoder(semantic)
        else:
            semantic_decoded = semantic
        return semantic_decoded

    def decode_from_codes(self, semantic_codes, acoustic_codes, token_lengths=None):
        """
        Decodes from semantic and acoustic codes. If token_lengths are provided,
        it assumes features are aggregated and will de-aggregate them.
        
        Args:
            semantic_codes (torch.Tensor): semantic codes of shape [B, n_q_s, G]
            acoustic_codes (torch.Tensor): acoustic codes of shape [B, n_q_a, G] or None
            token_lengths: Optional[torch.Tensor], shape (B, G)
                If provided, will de-aggregate the features after VQ decoding.
                Each value is the number of frames in the corresponding group.
        """
        semantic = self.semantic_vq.from_codes(semantic_codes)[0]
        if self.decode_semantic_for_codec:
            semantic_decoded = self.convnext_decoder(semantic)
        else:
            semantic_decoded = semantic

        # Handle alignment-based decoding (de-aggregation)
        is_aggregated = self.use_similarity_alignment
        if is_aggregated and token_lengths is not None:
            # De-aggregate semantic features to match acoustic dimensions
            semantic_expanded = self._deaggregate_features_from_token_lengths(semantic_decoded, token_lengths)
            
            # Decode acoustic codes normally first
            if acoustic_codes is not None:
                acoustic_vq = self.dac.quantizer.from_codes(acoustic_codes)[0]
                
                # De-aggregate acoustic features
                acoustic_expanded = self._deaggregate_features_from_token_lengths(acoustic_vq, token_lengths)
            else:
                acoustic_expanded = 0.0
            
            # Add semantic contribution to acoustic features
            acoustic_final = acoustic_expanded + semantic_expanded
            
            # Apply bottleneck transformers
            acoustic_final = self.bottleneck_transformer(acoustic_final)
            
            # Use original DAC decoder
            audio = self.dac.decoder(acoustic_final)
        else:
            # Original decoding without alignment/aggregation
            audio = self.dac.decode_from_codes(acoustic_codes, semantic_latent=semantic_decoded)
        
        return audio
    
    def decode_from_latent(self, latent, token_lengths):
        acoustic_final = self._deaggregate_features_from_token_lengths(latent, token_lengths)
        acoustic_final = self.bottleneck_transformer(acoustic_final)
        audio_output = self.dac.decoder(acoustic_final)
        return audio_output
    
    def forward(self, dl_output, encode_only=False, infer_using_dynamic_threshold=False):
        audio_data = dl_output.get("audio", dl_output).float()
        if len(audio_data.shape) == 2:
            audio_data = audio_data.unsqueeze(1) # [B, 1, T]
        audio_features = dl_output.get("x", dl_output).float()
        x_lens = dl_output.get("x_lens", None)  # Optional input lengths [batch_size]
        mel_mask = dl_output.get("mel_mask", None)  # Optional mask for flow matching
        manual_threshold = dl_output.get("manual_threshold", None)  # Optional mask for flow matching
        audio_features_masks = torch.ones_like(audio_features[:,:,0])
        semantic_repr, sim_repr = _extract_semantic_code(
            self.semantic_model,
            audio_features,
            audio_features_masks,
            self.semantic_mean,
            self.semantic_std,
            skip_normalize=self.skip_normalize,
            sensevoice_prepend_inputs=self.sensevoice_prepend_inputs,
            sim_layer_idx=self.sensevoice_sim_layer_idx,
            semantic_layer_idx=self.sensevoice_semantic_layer_idx,
        )
        semantic_repr = semantic_repr.transpose(1,2)
        sim_repr = sim_repr.transpose(1,2)
        out_dict = self.forward_features(
            audio_data,
            self.sample_rate,
            semantic_repr=semantic_repr,
            alignment_hidden=sim_repr,
            n_quantizers=dl_output.get("num_quantizers", None),
            possibly_no_quantizer=False,
            mel=dl_output.get("mel", None),
            encode_only=encode_only,
            x_lens=x_lens,
            mel_mask=mel_mask,
            infer_using_dynamic_threshold=infer_using_dynamic_threshold,
            manual_threshold=manual_threshold,
        )
        return out_dict
    def forward_features(self, 
            audio_data: torch.Tensor,
            sample_rate: int = 24000,
            n_quantizers: int = None,
            semantic_repr=None,
            alignment_hidden=None,
            bypass_quantize_rate=0.125,
            possibly_no_quantizer=False,
            mel=None,
            encode_only: bool = False,
            x_lens=None,
            mel_mask=None,
            infer_using_dynamic_threshold: bool = False,
            manual_threshold=None,
        ):
        """
        semantic_repr: [B, C, T] at same frame rate as acoustic codes
        alignment_hidden: Optional representation [B, C, T] to compute similarity alignment; if None, use semantic_repr
        """
        if manual_threshold is not None:
            self.manual_threshold = manual_threshold
            assert not infer_using_dynamic_threshold
            self.training = False
        if infer_using_dynamic_threshold:
            self.training = False
            n_quantizers = None
            self.infer_using_dynamic_threshold = True
        else:
            if encode_only:
                self.training = False
        if mel is not None:
            mel = mel.transpose(1,2)
        semantic_repr_before_downsample = semantic_repr.clone().detach()
        semantic_repr = self._downsample_semantic_features(semantic_repr)

        if alignment_hidden is not None:
            alignment_hidden = self._downsample_semantic_features(alignment_hidden)
        else:
            alignment_hidden = semantic_repr
        semantic_repr_ret = semantic_repr.clone().detach()
        
        # Downsample x_lens by the same factor as semantic features
        x_lens_downsampled = self._downsample_x_lens(x_lens)
        
        # Regular audio processing
        audio_data_preprocessed = self.dac.preprocess(audio_data, sample_rate)
        acoustic_features = self.dac.encoder(audio_data_preprocessed)
        
        # Generate alignment matrix if using similarity-based alignment
        alignment_matrices = None
        num_segments_per_item = None
        is_aggregated = self.use_similarity_alignment
        sim = None

        # Ensure time dimensions match
        if acoustic_features.shape[-1] != semantic_repr.shape[-1]:
            # assert the shape difference is at most 2
            
            min_len = min(acoustic_features.shape[-1], semantic_repr.shape[-1])
            acoustic_features = acoustic_features[..., :min_len]
            semantic_repr = semantic_repr[..., :min_len]
            semantic_repr_ret = semantic_repr_ret[..., :min_len]
        if self.use_similarity_alignment:
            # Vectorized alignment computation for the whole batch, based on semantic_repr
            h_frames_batch = semantic_repr.transpose(1, 2)  # (B, T, D)
            alignment_matrices, sim, num_segments_per_item = self._perform_similarity_alignment_vectorized(h_frames_batch, x_lens=x_lens_downsampled)

        if is_aggregated:

            if self.use_query_token_aggregator:
                semantic_repr = self.semantic_aggregator(semantic_repr, alignment_matrices, num_segments_per_item)

                acoustic_aggregated = self.acoustic_aggregator(acoustic_features, alignment_matrices, num_segments_per_item)
                semantic_repr_gt_agg = self.aggregate_features(semantic_repr_ret, alignment_matrices)
            else:
                # Aggregate `semantic_repr` BEFORE convnext using simple mean-pooling
                semantic_repr = self.aggregate_features(semantic_repr, alignment_matrices)
                # Aggregate acoustic features
                acoustic_aggregated = self.aggregate_features(acoustic_features, alignment_matrices)
                # Aggregate ground truth representation for distillation loss
                semantic_repr_gt_agg = self.aggregate_features(semantic_repr_ret, alignment_matrices)
            
            # Process aggregated semantic stream
            semantic_aggregated = self.convnext_encoder(semantic_repr)

        else:
            # No alignment - process semantic features directly
            semantic_aggregated = self.convnext_encoder(semantic_repr)
            acoustic_aggregated = acoustic_features
            semantic_repr_gt_agg = semantic_repr_ret  # already match shape
        
        # Quantize semantic stream
        semantic_vq, semantic_codes, latents, commitment_loss, codebook_loss, first_layer_quantized = self.semantic_vq(semantic_aggregated)
        
        if self.decode_semantic_for_codec:
            semantic_decoded = self.convnext_decoder(semantic_vq)
        else:
            semantic_decoded = semantic_vq

        # Prepare for acoustic quantization
        bypass_quantize = random.random() < bypass_quantize_rate
        if not self.training:
            bypass_quantize = False
        if n_quantizers == 1:
            bypass_quantize = True
        if n_quantizers is not None:
            n_quantizers = n_quantizers - 1
        
        # Use aggregated semantic latent for subtraction
        subtracted_latent_agg = semantic_decoded  # already aggregated if alignment enabled, else passthrough
        
        # Quantize acoustic stream (with aggregated features if using alignment)
        if is_aggregated:
            # For aggregated acoustic features, we need to modify DAC's quantization
            acoustic_vq_input = acoustic_aggregated - subtracted_latent_agg
            
            if bypass_quantize:
                acoustic_codes, acoustic_latents, acoustic_commitment_loss, acoustic_codebook_loss = \
                    None, None, 0.0, 0.0
                acoustic_vq = 0.0
            else:
                acoustic_vq, acoustic_codes, acoustic_latents, acoustic_commitment_loss, acoustic_codebook_loss, _ = \
                    self.dac.quantizer(acoustic_vq_input, n_quantizers, possibly_no_quantizer=possibly_no_quantizer)

            
            if not bypass_quantize:
                # De-aggregate acoustic and semantic features separately, then sum them.
                # This keeps the computation graph cleaner for DDP to trace.
                acoustic_expanded = self.deaggregate_features(acoustic_vq, alignment_matrices)
                semantic_expanded = self.deaggregate_features(semantic_decoded, alignment_matrices)
                acoustic_final = acoustic_expanded + semantic_expanded
            else:
                # Bypassed quantization, directly expand semantic only
                acoustic_final = self.deaggregate_features(semantic_decoded, alignment_matrices)
                semantic_expanded = acoustic_final
            
            if encode_only:
                assert not self.training
                token_lengths = alignment_matrices.sum(dim=2).long()
                
                # Deaggregate semantic codes to match original frame rate
                semantic_codes_deaggregated = self._deaggregate_features_from_token_lengths(
                    semantic_codes.float(), token_lengths
                ).long()
                
                return_dict = edict({
                    "semantic_codes": semantic_codes,  # Aggregated codes [B, 1, G]
                    "semantic_codes_deaggregated": semantic_codes_deaggregated,  # Deaggregated codes [B, 1, T]
                    "acoustic_codes": acoustic_codes,
                    "token_lengths": token_lengths,
                    "alignment_matrix": alignment_matrices,
                    # "semantic_features": semantic_expanded.squeeze(0).cpu().detach().transpose(0,1) if not self.training else None,
                    "semantic_features": semantic_expanded.cpu().detach() if not self.training else None,
                    "speech_token_len": num_segments_per_item,  # Valid speech token lengths after aggregation
                    "semantic_repr_ret": semantic_repr_ret,
                    "decoder_latent": acoustic_vq+semantic_decoded,
                    # "decoder_latent": acoustic_expanded+semantic_expanded,
                    "decoder_latent_before_agg": acoustic_final,
                    "semantic_repr_before_downsample": semantic_repr_before_downsample,
                    "sim": None if sim is None else sim
                })
                return return_dict
            # Decode to audio
            acoustic_final = self.bottleneck_transformer(acoustic_final) # TODO match the shape of acoustic_final
            audio_output = self.dac.decoder(acoustic_final)
            
            acoustic_edict = edict({
                "x": audio_output,
                "z": acoustic_final,
                "codes": acoustic_codes,
                "latents": acoustic_latents,
                "penalty": acoustic_commitment_loss,
                "vq/codebook_loss": acoustic_codebook_loss,
                "metrics": {},
            })
        else:
            # Original DAC processing (non-dynamic frame rate codec)
            if encode_only:
                # For non-aggregated case, we need to get acoustic codes without full decoding
                acoustic_vq_input = acoustic_features - subtracted_latent_agg
                
                if bypass_quantize:
                    acoustic_codes = None
                    acoustic_latents = None
                    acoustic_commitment_loss = 0.0
                    acoustic_codebook_loss = 0.0
                else:
                    _, acoustic_codes, acoustic_latents, acoustic_commitment_loss, acoustic_codebook_loss, _ = \
                        self.dac.quantizer(acoustic_vq_input, n_quantizers, possibly_no_quantizer=possibly_no_quantizer)
                
                # Calculate speech_token_len for non-aggregated case
                if x_lens_downsampled is not None:
                    speech_token_len = x_lens_downsampled
                else:
                    # Fallback to semantic_codes length if no x_lens provided
                    speech_token_len = torch.tensor([semantic_codes.shape[-1]] * semantic_codes.shape[0], 
                                                  device=semantic_codes.device, dtype=torch.long)
                return_dict = edict({
                    "semantic_codes": semantic_codes,
                    "semantic_codes_deaggregated": semantic_codes,
                    "acoustic_codes": acoustic_codes,
                    "token_lengths": None,  # No aggregation for non-dynamic frame rate
                    "alignment_matrix": torch.zeros(1,1,1),  # No alignment for non-dynamic frame rate
                    "semantic_features": semantic_decoded.cpu().detach() if not self.training else None,
                    "speech_token_len": speech_token_len,
                    "semantic_repr_ret": semantic_repr_ret,
                    # "decoder_latent": semantic_decoded+self.dac.quantizer.from_codes(acoustic_codes)[0],
                })
                if acoustic_codes is not None:
                    return_dict['decoder_latent_before_agg'] = semantic_decoded+self.dac.quantizer.from_codes(acoustic_codes)[0]
                return return_dict
            
            # Continue with full DAC processing for non-encode_only case
            acoustic_edict = self.dac(
                encoded_feature=acoustic_features, 
                sample_rate=sample_rate, 
                n_quantizers=n_quantizers, 
                subtracted_latent=subtracted_latent_agg, 
                bypass_quantize=bypass_quantize, 
                possibly_no_quantizer=possibly_no_quantizer,
                cut_from_front=False
            )
            semantic_expanded = semantic_decoded
        
        if not self.decode_semantic_for_codec:
            semantic_decoded = self.convnext_decoder(semantic_vq)
            semantic_repr_ret = semantic_repr_ret[..., :semantic_decoded.shape[-1]]

        # Distillation loss uses aggregated ground truth and quantized semantic
        distill_loss = F.mse_loss(semantic_repr_gt_agg.detach(), semantic_decoded) * self.lambda_distill_loss

        length = audio_data.shape[-1]
        if acoustic_edict['x'] is not None:
            if acoustic_edict['x'].shape[-1] < length:
                acoustic_edict['x'] = nn.functional.pad(acoustic_edict['x'], (0, length - acoustic_edict['x'].shape[-1]))
            else:
                acoustic_edict['x'] = acoustic_edict['x'][..., :length]

        merged_edict = edict({
            "audio": acoustic_edict['x'],
            # "x": acoustic_edict['x'],
            "acoustic_codes": acoustic_edict['codes'],
            "semantic_codes": semantic_codes,
            "latents": acoustic_edict['latents'],
            # "penalty": acoustic_edict['penalty'] + commitment_loss,
            "vq/commitment_loss": acoustic_edict['penalty'] + commitment_loss,
            "vq/codebook_loss": acoustic_edict['vq/codebook_loss'] + codebook_loss,
            "metrics": acoustic_edict['metrics'],
            "semantic_repr": semantic_repr_ret,
            "distill_loss": distill_loss,
            "bypassed_quantize": bypass_quantize,
            "semantic_features": semantic_expanded.squeeze(0).cpu().detach().transpose(0,1) if not self.training else None,
            # "semantic_features": semantic_vq.squeeze(0).cpu().detach().transpose(0,1) if not self.training else None,
            "token_lengths": None,
        })
        # Add compression ratio and speech token lengths if alignment is used
        if self.use_similarity_alignment:
            original_frames_lengths = alignment_matrices.shape[-1]
            num_segments = num_segments_per_item.float().mean().item()
            compression_ratio = (num_segments / original_frames_lengths)
            merged_edict["token_ratio"] = compression_ratio
            merged_edict["speech_token_len"] = num_segments_per_item  # Valid speech token lengths after aggregation
        else:
            # If not using alignment, speech_token_len is None (indicating no aggregation)
            merged_edict["speech_token_len"] = None
        return merged_edict

    def _perform_similarity_alignment_vectorized(self, h_frames_v: torch.Tensor, x_lens=None):
        """
        Perform similarity-based alignment for an entire batch in a fully vectorized manner.
        
        Args:
            h_frames_v: torch.Tensor, shape (B, T, D)
                Hidden features for similarity calculation where B is the batch size,
                T is the number of frames and D is the hidden dimension.
            x_lens: torch.Tensor, shape (B,), optional
                Valid lengths for each item in the batch. If provided, only these frames
                will be considered for alignment computation.
                
        Returns:
            torch.Tensor, shape (B, max_groups, T)
                Padded binary alignment matrices for the batch where 1 indicates
                that the frame (column) belongs to the group (row). `max_groups`
                is the maximum number of groups among all items in the batch.
        """
        B, T, D = h_frames_v.shape

        if T <= 1:
            # All sequences are length 1 – return identity matrices with a single group
            return torch.ones(B, 1, T, device=h_frames_v.device, dtype=h_frames_v.dtype), \
                   torch.ones(B, T-1, device=h_frames_v.device, dtype=h_frames_v.dtype), \
                   torch.ones(B, device=h_frames_v.device, dtype=torch.long)

        # Create valid frame mask if x_lens is provided
        if x_lens is not None:
            # Create mask for valid frames
            valid_frame_mask = torch.arange(T, device=h_frames_v.device).unsqueeze(0) < x_lens.unsqueeze(1)  # (B, T)
        else:
            valid_frame_mask = torch.ones(B, T, device=h_frames_v.device, dtype=torch.bool)

        # 1. Cosine similarity between consecutive frames -> (B, T-1)
        sim = F.cosine_similarity(h_frames_v[:, :-1, :], h_frames_v[:, 1:, :], dim=2)

        # Mask out similarities for invalid transitions (where either frame is padding)
        if x_lens is not None:
            valid_transition_mask = valid_frame_mask[:, :-1] & valid_frame_mask[:, 1:]  # (B, T-1)
            # Set similarity to 0.0 (low similarity) for invalid transitions to force new segments
            # This prevents padding frames from being grouped with valid content
            sim = torch.where(valid_transition_mask, sim, torch.zeros_like(sim))

        # 2. Determine new segment boundaries (B, T-1)
        current_threshold = self._get_current_similarity_threshold()
        is_new_group_boundary = sim <= current_threshold

        # Pad a leading True to mark the start of the first segment -> (B, T)
        is_new_group_padded = torch.cat(
            [torch.ones(B, 1, dtype=torch.bool, device=h_frames_v.device), is_new_group_boundary], dim=1
        )

        # If max_tokens_per_group is set, we must also split segments that
        # are too long. This can be done efficiently in a vectorized way.
        if self.max_tokens_per_group is not None:
            # Find the index of each frame within its original segment
            arange_t = torch.arange(T, device=h_frames_v.device, dtype=torch.long).unsqueeze(0)
            segment_start_markers = arange_t * is_new_group_padded.long()
            last_segment_start_indices = torch.cummax(segment_start_markers, dim=1).values
            frame_indices_in_segment = arange_t - last_segment_start_indices
            
            # A new boundary is formed either by the original similarity score
            # or by reaching the maximum token limit.
            is_split_boundary = (frame_indices_in_segment % self.max_tokens_per_group) == 0
            
            # The final segment map is the cumulative sum of all boundaries.
            frame_to_segment_map = torch.cumsum(is_split_boundary.long(), dim=1) - 1
        else:
            # Original logic: only split based on similarity.
            frame_to_segment_map = torch.cumsum(is_new_group_padded.long(), dim=1) - 1
        
        # 4. Determine number of segments per item and global max
        # Only count segments that contain valid frames
        if x_lens is not None:
            # For each batch item, find the maximum segment index for valid frames
            num_segments_per_item = torch.zeros(B, device=h_frames_v.device, dtype=torch.long)
            for b in range(B):
                valid_length = x_lens[b]
                num_segments_per_item[b] = frame_to_segment_map[b, valid_length - 1] + 1
        else:
            num_segments_per_item = frame_to_segment_map.max(dim=1).values + 1  # (B,)
        # max_segments = int(num_segments_per_item.max().item())
        max_segments = int((frame_to_segment_map.max(dim=1).values + 1).max().item())

        # 5. Build alignment matrices using advanced indexing / scatter
        alignment_matrix = torch.zeros(B, max_segments, T, device=h_frames_v.device, dtype=torch.bool)

        # Prepare indices for scatter
        batch_indices = torch.arange(B, device=h_frames_v.device).unsqueeze(1).expand(B, T)  # (B, T)
        frame_indices = torch.arange(T, device=h_frames_v.device).unsqueeze(0).expand(B, T)  # (B, T)

        alignment_matrix[batch_indices, frame_to_segment_map, frame_indices] = True

        return alignment_matrix.float(), sim, num_segments_per_item

    def aggregate_features(
        self,
        features: torch.Tensor,
        alignment_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate features using alignment matrix.
        
        Args:
            features: torch.Tensor, shape (batch_size, feat_len, feature_dim) or (batch_size, feature_dim, feat_len)
            alignment_matrix: torch.Tensor, shape (batch_size, num_groups, feat_len)
                
        Returns:
            torch.Tensor, shape (batch_size, num_groups, feature_dim) or (batch_size, feature_dim, num_groups)
        """
        # Handle both (B, T, D) and (B, D, T) formats
        is_channel_last = features.dim() == 3 and features.shape[-1] != alignment_matrix.shape[-1]
        
        if not is_channel_last:
            # Features are (B, D, T), need to transpose for aggregation
            features = features.transpose(1, 2)  # (B, T, D)
        
        # Ensure alignment matrix is float and on the correct device
        alignment_float = alignment_matrix.to(features.device, dtype=features.dtype)
        
        # Calculate the sum of features for each group via vectorized operation
        summed_features = torch.einsum('bgt,btd->bgd', alignment_float, features)
        
        # Calculate the number of frames assigned to each group
        group_frame_counts = alignment_float.sum(dim=2)  # (batch_size, num_groups)
        
        # To avoid division by zero, clamp counts to a minimum of 1
        group_frame_counts = group_frame_counts.clamp(min=1)
        
        # Reshape counts for broadcasting
        counts_reshaped = group_frame_counts.unsqueeze(-1)  # (batch_size, num_groups, 1)
        
        # Compute the average
        aggregated_features = summed_features / counts_reshaped
        
        if not is_channel_last:
            # Transpose back to (B, D, T) format
            aggregated_features = aggregated_features.transpose(1, 2)
        
        return aggregated_features

    def deaggregate_features(
        self,
        grouped_features: torch.Tensor,
        alignment_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        De-aggregate features back to original frame structure.
        
        Args:
            grouped_features: torch.Tensor, shape (batch_size, num_groups, feature_dim) or (batch_size, feature_dim, num_groups)
            alignment_matrix: torch.Tensor, shape (batch_size, num_groups, feat_len)
                
        Returns:
            torch.Tensor, shape (batch_size, feat_len, feature_dim) or (batch_size, feature_dim, feat_len)
        """
        # Handle both (B, G, D) and (B, D, G) formats
        is_channel_last = grouped_features.dim() == 3 and grouped_features.shape[1] == alignment_matrix.shape[1]
        
        if not is_channel_last:
            # Features are (B, D, G), need to transpose for de-aggregation
            grouped_features = grouped_features.transpose(1, 2)  # (B, G, D)
        
        # Ensure alignment matrix is float and on the correct device
        alignment_float = alignment_matrix.to(grouped_features.device, dtype=grouped_features.dtype)
        
        # De-aggregate: expand group features to frame features
        expanded_features = torch.einsum('bgd,bgt->btd', grouped_features, alignment_float)
        
        if not is_channel_last:
            # Transpose back to (B, D, T) format
            expanded_features = expanded_features.transpose(1, 2)
        
        return expanded_features

    def _deaggregate_features_from_token_lengths(
        self,
        grouped_features: torch.Tensor,
        token_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        De-aggregate features back to original frame structure using token_lengths.
        This is a replacement for `deaggregate_features` when only token lengths are available.
        
        Args:
            grouped_features: torch.Tensor, shape (batch_size, feature_dim, num_groups)
            token_lengths: torch.Tensor, shape (batch_size, num_groups)
                
        Returns:
            torch.Tensor, shape (batch_size, feature_dim, feat_len)
        """
        B, D, G = grouped_features.shape
        assert G == token_lengths.shape[1], "Number of groups in features and token_lengths must match."
        
        # Permute features to be (B, G, D) for repeat_interleave
        grouped_features_permuted = grouped_features.permute(0, 2, 1)
        
        expanded_features_list = []
        for i in range(B):
            # For each item in the batch, repeat its features according to token_lengths
            # token_lengths contains the number of repetitions for each feature vector in the group
            expanded_item = torch.repeat_interleave(grouped_features_permuted[i], token_lengths[i], dim=0)
            expanded_features_list.append(expanded_item)
        
        # Pad the list of tensors to the same length and stack them
        expanded_features = pad_sequence(expanded_features_list, batch_first=True, padding_value=0.0)
        
        # Transpose back to (B, D, T) format
        expanded_features = expanded_features.transpose(1, 2)
        
        return expanded_features
