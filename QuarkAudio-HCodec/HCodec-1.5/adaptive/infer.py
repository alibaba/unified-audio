
import torch
import time
import torchaudio
from .feature_extractors import FBankGen
from huggingface_hub import snapshot_download, hf_hub_download

get_params = lambda model: sum(p.numel() for p in model.parameters()) / 1e6

def prepare_model(sensevoice_small_path=None, device='auto', ckpt_path=None, config_path=None):
    """Loads the FlexiCodec model and its corresponding feature extractor."""
    if sensevoice_small_path is None:
        sensevoice_small_path = snapshot_download('FunAudioLLM/SenseVoiceSmall')
    if config_path is None:
        config_path = hf_hub_download('jiaqili3/flexicodec', '12hz_v1_half_config.yaml')
    if ckpt_path is None:
        ckpt_path = hf_hub_download('jiaqili3/flexicodec', '12hz_v1_half.safetensors')
    
    
    import yaml
    
    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    model_config['model']['semantic_model_path'] = sensevoice_small_path
    model_config['model']['semantic_model_type'] = 'sensevoice'

    def build_codec_model(config):
        from pathlib import Path
        import copy
        from .modeling_flexicodec import FlexiCodec
        codec_model_config = copy.deepcopy(config)
        codec_model = FlexiCodec(
            **codec_model_config
        )
        return codec_model

    model = build_codec_model(model_config['model'])
    if ckpt_path.endswith('.safetensors'):
        import safetensors.torch
        model.load_state_dict(safetensors.torch.load_file(ckpt_path), strict=False)
    else:
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    print('weight loaded.')
    model.eval()
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    feature_extractor = FBankGen(sr=16000)
    return {"model": model, "feature_extractor": feature_extractor, "type": "sensevoice"}

@torch.no_grad()
def encode_flexicodec(audio: torch.Tensor, model: dict, sample_rate: int=16000, num_quantizers: int = 8, merging_threshold: float = 0.91, audio_lens=None):
    """
    Encodes the audio using the FlexiCodec model.
    Audio: [B,T]
    audio_lens: [B]
    """
    assert len(audio.shape) == 2, "audio should be [B, T]"
    batch_size = audio.shape[0]
    codec_model = model['model']
    feature_extractor = model['feature_extractor']
    device = next(codec_model.parameters()).device
    
    # Ensure audio is mono and on the correct device
    audio = audio.to(device)
    
    # 1. Resample audio: 16kHz for semantic features
    resampler_16k = torchaudio.transforms.Resample(sample_rate, 16000).to(device)
    audio_16k = resampler_16k(audio)
    duration = audio_16k.shape[-1] / 16000
    sim = None
    # Process each batch item individually and concatenate features
    features_list = []
    for i in range(audio_16k.shape[0]):
        if audio_lens is not None:
            features_i, _ = feature_extractor.extract_fbank(audio_16k[i:i+1,:audio_lens[i]].cpu())
        else:
            features_i, _ = feature_extractor.extract_fbank(audio_16k[i:i+1].cpu())
        features_list.append(features_i.squeeze(0))
    features_lens = torch.tensor([features_i.shape[0] for features_i in features_list])
    features_lens = features_lens.to(device)
    features = torch.nn.utils.rnn.pad_sequence(features_list, batch_first=True)
        # Now features.shape = [B, max_T, D]
    audio_features = features.to(device)


    dl_output = {
        "audio": audio_16k,
        "x": audio_features,
        'x_lens': features_lens,
        "num_quantizers": num_quantizers,
        "manual_threshold": merging_threshold,
    }
    # Encode the audio to get semantic and acoustic codes
    encoded_output = codec_model(
        dl_output,
        encode_only=True,
    )
    # Extract the codes and token lengths
    semantic_codes = encoded_output['semantic_codes']
    acoustic_codes = encoded_output['acoustic_codes']
    token_lengths = encoded_output['token_lengths']
    alignmnet_matrix = encoded_output['alignment_matrix']
    sim = encoded_output.get('sim', None)
    return {
        'semantic_codes': semantic_codes,
        'acoustic_codes': acoustic_codes,
        'token_lengths': token_lengths,   # [B, T] - duration for each speech token
        'alignment_matrix': alignmnet_matrix,
        'sim_matrix': sim,
        'total_frames': encoded_output['speech_token_len'] # [B] - speech token length
    }
    # Decode from codes to reconstruct the audio
    # reconstructed_audio = codec_model.decode_from_codes(
    #     semantic_codes=semantic_codes,
    #     acoustic_codes=acoustic_codes,
    #     token_lengths=token_lengths,
    # )


@torch.no_grad()
def infer(audio: torch.Tensor, model: dict, sample_rate: int=16000, num_quantizers: int = 8, manual_threshold: float = 0.87):
    """
    Performs end-to-end inference with the DualCodec model.
    Uses the correct feature extraction pipeline based on model['type'].
    """
    audio = audio.reshape(1,-1)
    codec_model = model['model']
    feature_extractor = model['feature_extractor']
    device = next(codec_model.parameters()).device
    
    # Ensure audio is mono and on the correct device
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.to(device)

    # 1. Resample audio: 16kHz for semantic features
    resampler_16k = torchaudio.transforms.Resample(sample_rate, 16000).to(device)
    audio_16k = resampler_16k(audio)
    duration = audio_16k.shape[-1] / 16000
    sim = None

    if model.get('type') == 'sensevoice':
        # Use SenseVoice's FBankGen (on CPU)
        features, _ = feature_extractor.extract_fbank(audio_16k.cpu())
        audio_features = features.to(device)

    # 3. Time the encoding and decoding steps separately
    start_time = time.time()
    dl_output = {
        "audio": audio_16k,
        "x": audio_features,
        "num_quantizers": num_quantizers,
        "manual_threshold": manual_threshold,
    }
    # Encode the audio to get semantic and acoustic codes
    encoded_output = codec_model(
        dl_output,
        encode_only=True,
    )
    # Extract the codes and token lengths
    semantic_codes = encoded_output['semantic_codes']
    acoustic_codes = encoded_output['acoustic_codes']
    token_lengths = encoded_output['token_lengths']
    alignmnet_matrix = encoded_output['alignment_matrix']
    sim = encoded_output.get('sim', None)
    
    # Decode from codes to reconstruct the audio
    reconstructed_audio = codec_model.decode_from_codes(
        semantic_codes=semantic_codes,
        acoustic_codes=acoustic_codes,
        token_lengths=token_lengths,
    )
    
    semantic_features = encoded_output.get("semantic_features", None)

    print(token_lengths)
    return {
        "out": reconstructed_audio.cpu().to(torch.float32),
        "compressed": semantic_codes,
        "semantic_features": semantic_features,
        "token_lengths": token_lengths,
        "alignment_matrix": alignmnet_matrix,
        "sim": sim, # shape: [1, T]
    }


# Example usage
if __name__ == '__main__':
    model_dict = prepare_model()
    
    # Load a real audio file
    audio_path = '/Users/jiaqi/Downloads/00000.wav'
    audio, sample_rate = torchaudio.load(audio_path)
    # audio = torch.cat([audio, audio], dim=0)
    with torch.no_grad():
        encoded_output = encode_flexicodec(audio, model_dict, sample_rate, num_quantizers=8, merging_threshold=0.91)
        reconstructed_audio = model_dict['model'].decode_from_codes(
            semantic_codes=encoded_output['semantic_codes'],
            acoustic_codes=encoded_output['acoustic_codes'],
            token_lengths=encoded_output['token_lengths'],
        )

    duration = audio.shape[-1] / 16000
    # # Save the reconstructed audio (output is at 16kHz)
    output_path = 'decoded_audio.wav'
    torchaudio.save(output_path, reconstructed_audio.cpu().squeeze(1), 16000)
    
    print(f"Saved decoded audio to {output_path}")
    print(f"This sample avg frame rate: {encoded_output['token_lengths'].shape[-1] / duration:.4f} frames/sec")