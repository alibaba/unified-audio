# HCodec-Spatial: Spatial Audio Codec for Stereo Encoding and Mono-to-Stereo Spatialization

<p align="center">
  <a href="https://arxiv.org/pdf/2512.20151">
    <img src="https://img.shields.io/badge/Paper-ArXiv-red.svg" alt="Paper">
  </a>
  <a href="https://huggingface.co/QuarkAudio/QuarkAudio-HCodec/">
    <img src="https://img.shields.io/badge/Model-Hugging%20Face-yellow.svg" alt="Hugging Face">
  </a>
  <a href="https://www.modelscope.cn/models/QuarkAudio/QuarkAudio-HCodec/">
    <img src="https://img.shields.io/badge/Model-%20%E9%AD%94%E6%90%AD-orange.svg" alt="ModelScope">
  </a>
</p>

> **HCodec-Spatial** extends the H-Codec framework to support **spatial audio**, enabling both stereo encoding/decoding and mono-to-stereo spatialization conditioned on spatial metadata (azimuth, elevation, distance).

## Key Features

- **Stereo Codec**: Encode and decode stereo audio with high fidelity using three-stream (acoustic + semantic + spatial) quantization.
- **Mono-to-Stereo Spatialization**: Convert mono audio to spatial stereo given target direction metadata.
- **Three Codebook Groups**: Independent acoustic, semantic, and spatial quantizers for disentangled representation.
- **16kHz / Fixed Frame Rate**: Consistent with the H-Codec family design.

## Architecture

| Component | Description |
|-----------|-------------|
| Acoustic Encoder | Conv1D-based encoder with LSTM and Transformer layers |
| Semantic Encoder | Processes wav2vec2 features into semantic codebook space |
| Spatial Encoder | Encodes spatial channel differences |
| Quantizer | 3 groups: acoustic (4 codebooks), semantic (4 codebooks), spatial (2-4 codebooks) |
| Decoder | Bottleneck Transformer decoder reconstructing stereo waveform |

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/alibaba/unified-audio.git
cd QuarkAudio-HCodec/HCodec-spatial
```

### 2. Install Dependencies

```bash
conda create -n hcodec_spatial python=3.9
conda activate hcodec_spatial
pip install -r requirements.txt
```

### 3. Download Checkpoints

Download pretrained checkpoints from [Hugging Face](https://huggingface.co/QuarkAudio/QuarkAudio-HCodec/) or [ModelScope](https://www.modelscope.cn/models/QuarkAudio/QuarkAudio-HCodec/) and place them in `./checkpoints/`:

```
checkpoints/
  hcodec_spatial_stereo.ckpt
  hcodec_spatial_mono2stereo.ckpt
```

The wav2vec2 feature extractor will be automatically downloaded from Hugging Face, or you can specify a local path via `wav2vec2_path` in the config.

### 4. Stereo Encoding/Decoding

```bash
python test_stereo.py \
    --config ./conf/config_base_stereo.yaml \
    --input_dir ./test_wavs_stereo \
    --output_dir ./output_stereo
```

Input: stereo `.wav` or `.flac` files (16kHz, 2-channel).

### 5. Mono-to-Stereo Spatialization

```bash
python test_mono2stereo.py \
    --config ./conf/config_base_mono2stereo.yaml \
    --input_dir ./test_wavs_mono \
    --output_dir ./output_mono2stereo
```

Input: mono `.wav` or `.flac` files (16kHz, 1-channel).

#### Spatial Information

Spatial metadata is parsed from the input filename. Files should follow this naming convention:

```
<name>_azi_<azimuth>_ele_<elevation>_dist_<distance>.wav
```

| Parameter | Range | Description |
|-----------|-------|-------------|
| azimuth | 0-359 | Horizontal angle in degrees |
| elevation | -90 to 89 | Vertical angle (internally mapped to 0-179) |
| distance | 0.0-4.9 | Distance in meters (internally mapped to 0-49 via *10) |

Example filename: `speech_azi_45_ele_0_dist_1.5.wav`

If the filename does not match the expected pattern, default spatial parameters `(azimuth=0, elevation=90, distance=10)` will be used.

## Configuration

Config files are located in `./conf/`:

- `config_base_stereo.yaml` - Stereo codec inference config
- `config_base_mono2stereo.yaml` - Mono-to-stereo inference config

Key config parameters:

| Parameter | Description |
|-----------|-------------|
| `ckpt_path` | Path to model checkpoint |
| `wav2vec2_path` | Path or HF model ID for wav2vec2-xlsr-large |
| `accelerator` | `gpu` or `cpu` |
| `devices` | GPU device indices |

## Project Structure

```
HCodec-spatial/
├── conf/                   # Configuration files
│   ├── config_base_stereo.yaml
│   └── config_base_mono2stereo.yaml
├── model/                  # Model definitions
│   ├── model_stereo.py     # Stereo codec model
│   ├── model_mono2stereo.py # Mono-to-stereo model
│   └── vq/                 # Vector quantization modules
├── dataloader/             # Data loading utilities
│   └── data_module.py
├── test_stereo.py          # Stereo inference script
├── test_mono2stereo.py     # Mono-to-stereo inference script
├── requirements.txt
└── README.md
```

## Citation

```bibtex
@article{liu2024hcodec,
  title={UniTok-Audio: A Unified Tokenizer for High-Fidelity, Multi-task Audio Generation},
  author={Liu, Yaoxun and others},
  journal={arXiv preprint arXiv:2512.20151},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](../LICENSE) file for details.
