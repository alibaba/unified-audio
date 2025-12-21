# UniSE: A Unified Framework for Decoder-only Autoregressive LM-based Speech Enhancement

<p align="center">
  <a href="https://arxiv.org/abs/2510.20441">
    <img src="https://img.shields.io/badge/Paper-ArXiv-red.svg" alt="Paper">
  </a>
  <a href="https://github.com/hyyan2k/UniSE/">
    <img src="https://img.shields.io/badge/Demo-Page-blue.svg" alt="Demo">
  </a>
  <a href="https://huggingface.co/spaces/QuarkAudio/">
    <img src="https://img.shields.io/badge/Model-Hugging%20Face-yellow.svg" alt="Hugging Face">
  </a>
</p>

![UniSE](UniSE.png)


## Introduction

QuarkAudio-UniSE is a unified speech process model capable of handling multiple tasks without extra task prompts, including:

- **SR**: Speech Restoration (⛳ supported)
- **SE**: Speech Enhancement (⛳ supported)
- **TSE**: Target Speaker Extraction (⛳ supported)
- **SS**: Speech Separation (⛳ supported)
- **AEC**: Acoustic Echo Cancellation (⛳ developing)

**QuarkAudio-UniSE**, a unified decoder-only LM-based framework to handle different SE tasks including speech restoration, target speaker extraction and speech separation. It takes input speech features as conditions and generates discrete tokens of the target speech using AR modeling, which facilitates a compatibility between distinct learning patterns of multiple tasks. Comprising **WavLM** with adapter to extract continuous speech feature, a discrete speech codec **BiCodec** to produce discrete tokens and reconstruct waveforms and a decoder-only LM backbone
to model conditional probability

For more details, refer to our paper: [UniSE Paper](https://arxiv.org/abs/2510.20441)

## Demo

You can listen to the enhancement results on our [Demo Page](https://github.com/hyyan2k/UniSE/).

## Installation

Checkpoints are at [huggingface](https://huggingface.co/ASLP-lab/LLaSE-G1).

### 1. Clone the repository

```bash
git https://github.com/alibaba/unified-audio.git
cd QuarkAudio-UniSE
```

### 2. Create a Conda environment and install dependencies

```bash
conda create -n llase python=3.10
conda activate QuarkAudio
pip install -r requirements.txt
```

### 3. Download Pretrained Models

LLaSE-G1 requires three additional **WavLM** and **BiCodec** pre-trained models and checkpoint of the middle LM on Huggingface to function properly. You can download three of them using the provided shell script:

```bash
cd checkpoints
bash download.sh
```
Additionally, download WavLM-Large.pt from this [URL](https://huggingface.co/microsoft/wavlm-base-plus) and put it at `./ckpt/WavLM-Large.pt` .

Alternatively, you can download them manually and place them in the `./model/bicodec/` directory.

After Downloading, the tree should be like this:

## Train
+ Quick start

```bash
#!/bin/bash
python ./train.py --config conf/config.yaml
```
| Parameter        | Description                                                                                                                                                            |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `resume` | if want to resume, specify ckpt path                                                                                                  |
| `simulation_config` | data simulate config                                                                                                                        |
| `speech_scp_path`        | SCP of clean audio files                                                       |
| `noise_scp_path`        | SCP of noise audio files                                                                    | `rir_scp_path`        | SCP of rir audio files                                                                       |
| `mode`           | Task type: `se` (Noise Suppression,Speech Restoration,Packet Loss Concealment), `se` (Target Speaker Extraction), `SS` (Speech Separation). |


## Inference
+ Quick start
The main inference script is **`test.py`**. The inference process consists of two stages:

1. Extract the 6th-layer features from WavLM.
2. Use the language model (LM) to predict speech tokens, and then decode them into audio using **BiCodec**.

### Running Inference
+ Quick start
To run test.py, configure the parameters in `./conf/config.yaml`:

| Parameter        | Description                                                                                                                                                            |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ckpt_path` | pretrained weight                                                                                                             |
| `enroll_duration` | Number of inference iterations.                                                                                                                                        |
| `data_src_dir`        | Directory of processed audio files directory.                                                        |
| `data_tgt_dir`        | Directory of processed audio files directory.                                                                                                                                    |
| `mode`           | Task type: `se` (Noise Suppression,Speech Restoration,Packet Loss Concealment), `se` (Target Speaker Extraction), `SS` (Speech Separation). |

Command to run inference:

```python
python test.py
```

## Results

Samples processed by LLaSE-G1 can be found on our [Demo Page](https://github.com/hyyan2k/UniSE/).

## Model Checkpoints

Our pretrained model is available on [Hugging Face](https://huggingface.co/spaces/QuarkAudio/).

## Hints

Our approach focuses on leveraging the LLM's comprehension capabilities to enable autonomous determination of task types, though this may exhibit instability in certain scenarios. A more stable and robust iteration will be released in the upcoming version.

## Citation

```
@misc{yan2025uniseunifiedframeworkdecoderonly,
      title={UniSE: A Unified Framework for Decoder-only Autoregressive LM-based Speech Enhancement}, 
      author={Haoyin Yan and Chengwei Liu and Shaofei Xue and Xiaotao Liang and Zheng Xue},
      year={2025},
      eprint={2510.20441},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2510.20441}, 
}
```


## Contact
For any questions, please contact: `yanhaoyin.yhy@alibaba-inc.com`
 
