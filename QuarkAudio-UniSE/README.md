# UniSE: A Unified Framework for Decoder-only Autoregressive LM-based Speech Enhancement

<p align="center">
  <a href="https://arxiv.org/abs/2510.20441">
    <img src="https://img.shields.io/badge/Paper-ArXiv-red.svg" alt="Paper">
  </a>
  <a href="https://huggingface.co/QuarkAudio/QuarkAudio-UniSE/">
    <img src="https://img.shields.io/badge/Model-Hugging%20Face-yellow.svg" alt="Hugging Face">
  </a>
  <a href="https://www.modelscope.cn/models/QuarkAudio/QuarkAudio-UniSE/">
    <img src="https://img.shields.io/badge/Model-%20%E9%AD%94%E6%90%AD-orange.svg" alt="ModelScope">
  </a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.20441"><img src="QuarkAudio-UniSE.png" width="70%" /></a>
</p>

🚀 **Key Highlights**:
- ✅ **Unified & Prompt-Free**: Handles multiple tasks without explicit instruction.
- ⚙️ **Decoder-only AR-LM Backbone**: Leverages LLM-style autoregressive generation for speech token prediction.
- 🔄 **End-to-End Compatible**: Integrates WavLM (feature extractor), BiCodec (discrete codec), and LM into one pipeline.
- 🌍 **Multitask Support**: SR, TSE, SS, and more — all in a single model.

📄 **Paper**: [arXiv:2510.20441](https://arxiv.org/abs/2510.20441)  | 🤗 **Model**: [Hugging Face Spaces](https://huggingface.co/QuarkAudio/QuarkAudio-UniSE/)

---

## 📋 Supported Tasks

| Task | Full Name | Status | Description |
|------|-----------|--------|-------------|
| **SR** | Speech Restoration | ✅ Stable | General-purpose denoising and clarity improvemen (e.g., noise, reverb, packet loss) |
| **TSE** | Target Speaker Extraction | ✅ Stable | Extract target speaker using reference enrollment audio |
| **SS** | Speech Separation | ✅ Stable | Separate mixed speakers or sound sources |
| **AEC** | Acoustic Echo Cancellation | ⏳ Developing | Coming soon in next release |

---

## 🎯 Quick Start: Run Inference in 3 Minutes

### 1. Clone Repository

```bash
git clone https://github.com/alibaba/unified-audio.git
cd QuarkAudio-UniSE
```

### 2. Create a Conda environment and install dependencies

```bash
conda create -n unise python=3.10
conda activate unise
pip install -r requirements.txt
```

### 3. Download Checkpoints


UniSE needs the checkpoints of BiCodec, please download the files in https://huggingface.co/SparkAudio/Spark-TTS-0.5B and put them into `./checkpoints`
After Downloading, the tree should be like this:

```
./checkpoints
|-- BiCodec
|   |-- config.yaml
|   `-- model.safetensors
|-- config.yaml
`-- wav2vec2-large-xlsr-53
    |-- README.md
    |-- config.json
    |-- preprocessor_config.json
    `-- pytorch_model.bin
```

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
| `noise_scp_path`        | SCP of noise audio files                                                                   
 | `rir_scp_path`        | SCP of rir audio files                                                                       |


## Inference
+ Quick start
The main inference script is **`test.py`**. The inference process consists of two stages:

1. Extract hidden states from all WavLM layers and obtain a single representation by averaging them across layers.
2. Use the language model (LM) to predict speech tokens autoregressively, and then decode them into audio using **BiCodec**.

### Running Inference
+ Quick start
To run test.py, configure the parameters in `./conf/config.yaml`:

| Parameter        | Description                                                                                                                                                            |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ckpt_path` | pretrained weight                                                                                                             |
| `enroll_duration` | Number of inference iterations.                                                                                                                                        |
| `data_src_dir`        | Directory of processed audio files directory.                                                        |
| `data_tgt_dir`        | Directory of processed audio files directory.                                                                                                                                    |
| `mode`           | Task type: `se` (Speech Restoration), `tse` (Target Speaker Extraction), `ss` (Speech Separation). |

Command to run inference:

```python
python test.py
```


## Model Checkpoints

Our pretrained model is available on [Hugging Face](https://huggingface.co/QuarkAudio/QuarkAudio-UniSE/).


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
 
