# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch import nn
import numpy as np

from pathlib import Path
from typing import Any, Dict, Tuple
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from .utils.file import load_config
from .utils.audio import load_audio
from .bicodec import BiCodec


class BiCodecTokenizer(nn.Module):
    """BiCodec tokenizer for handling audio input and tokenization."""

    def __init__(self, model_dir: Path, **kwargs):
        super().__init__()
        """
        Args:
            model_dir: Path to the model directory.
        """
        self.model_dir = model_dir
        self.config = load_config(f"{model_dir}/config.yaml")
        self._initialize_model()

    def _initialize_model(self):
        """Load and initialize the BiCodec model and Wav2Vec2 feature extractor."""
        self.model = BiCodec.load_from_checkpoint(f"{self.model_dir}/BiCodec")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            f"{self.model_dir}/wav2vec2-large-xlsr-53"
        )
        self.feature_extractor = Wav2Vec2Model.from_pretrained(
            f"{self.model_dir}/wav2vec2-large-xlsr-53"
        )
        self.feature_extractor.config.output_hidden_states = True

    def get_ref_clip(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Get reference audio clip for speaker embedding.
        Args:
            wav: audio waveform. shape: (batch_size, seq_len)
        """
        ref_segment_length = (
            int(self.config["sample_rate"] * self.config["ref_segment_duration"])
            // self.config["latent_hop_length"]
            * self.config["latent_hop_length"]
        )
        wav_length = wav.size(-1)

        if ref_segment_length > wav_length:
            # Repeat and truncate to handle insufficient length
            # wav = np.tile(wav, ref_segment_length // wav_length + 1)
            wav = torch.tile(wav, (1, ref_segment_length // wav_length + 1))

        return wav[:, :ref_segment_length]

    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""
        device = wavs.device
        wavs = np.asarray(wavs.cpu())  # (batch_size, seq_len)
        inputs = self.processor(
            wavs,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            output_hidden_states=True,
        ).input_values
        feat = self.feature_extractor(inputs.to(device))
        feats_mix = (
            feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
        ) / 3

        return feats_mix

    @torch.no_grad()
    def tokenize(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """tokenize the audio"""
        ref_wav = self.get_ref_clip(wav)  # shape: (batch_size, seq_len)

        feat = self.extract_wav2vec2_features(wav)
        batch = {
            "wav": wav,
            "ref_wav": ref_wav,
            "feat": feat,
        }
        semantic_tokens, global_tokens = self.model.tokenize(batch)

        return global_tokens, semantic_tokens

    @torch.no_grad()
    def detokenize(
        self, global_tokens: torch.Tensor, semantic_tokens: torch.Tensor
    ) -> np.array:
        """detokenize the tokens to waveform

        Args:
            global_tokens: global tokens. shape: (batch_size, 1, global_dim)
            semantic_tokens: semantic tokens. shape: (batch_size, latent_dim)

        Returns:
            wav_rec: waveform. shape: (batch_size, seq_len) for batch or (seq_len,) for single
        """
        wav_rec = self.model.detokenize(semantic_tokens, global_tokens)
        return wav_rec


# test
if __name__ == "__main__":
    # import soundfile as sf

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tokenizer = BiCodecTokenizer(
    #     model_dir="pretrained_models/Spark-TTS-0.5B",
    #     device=device,
    # )
    # wav_path = "example/prompt_audio.wav"


    # import ipdb; ipdb.set_trace()
    # global_tokens, semantic_tokens = tokenizer.tokenize(wav_path)  # (1, b, 32), (b, T)
    # wav_rec = tokenizer.detokenize(global_tokens.squeeze(0), semantic_tokens)

    # sf.write("example/prompt_recon.wav", wav_rec, 16000)
    
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "/mnt/nas1/project/unified_speech_generation/bicodec_test/model/pretrained_ckpt/wav2vec2-large-xlsr-53"
    )

    # feature_extractor = Wav2Vec2Model.from_pretrained(
    #     "/mnt/nas1/project/unified_speech_generation/bicodec_test/model/pretrained_ckpt/wav2vec2-large-xlsr-53"
    # )
    # feature_extractor.config.output_hidden_states = True

    # wavs = np.random.randn(16000,)
    wavs = torch.randn((4, 16000))
    inputs = processor(
        wavs,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        output_hidden_states=True,
    ).input_values
    print(inputs.shape)

