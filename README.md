# QuarkAudio: An Open-Source Project to Unify Audio Processing and Generation
<p align="center">
  <img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arXiv" alt="arXiv">
  <img src="https://img.shields.io/badge/Demo-Page-blue?logo=github" alt="Demo">
  <img src="https://img.shields.io/badge/Model-Hugging%20Face-yellow?logo=huggingface" alt="Hugging Face">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.20441"><img src="QuarkAudio.jpg" width="70%" /></a>
</p>

This project contains a series of works developed for audio (including speech, music, and general audio events) processing and generation, which helps reproducible research in the field of audio. 
🚀 **Key Highlights**:
- ✅ **Unified & Prompt-Free**: Handles multiple tasks without explicit instruction.
- ⚙️ **Decoder-only AR-LM Backbone**: Leverages LLM-style autoregressive generation for speech token prediction.
- 🔄 **End-to-End Compatible**: Integrates WavLM (feature extractor), BiCodec (discrete codec), and LM into one pipeline.
- 🌍 **Multitask Support**: SE, SR, TSE, SS, and more — all in a single model.

📄 **Paper**: [arXiv:2510.20441](https://arxiv.org/abs/2510.20441) | 🎤 **Listen**: [Demo Page](https://hyyan2k.github.io/UniSE/) | 🤗 **Model**: [Hugging Face Spaces](https://huggingface.co/spaces/QuarkAudio/)

The target of **QuarkAudio** is to explore a unified framework to handle **different audio processing and generation tasks**, including:
## 📋 Supported Tasks
- **SR**: Speech Restoration (⛳ supported)
- **TSE**: Target Speaker Extraction (⛳ supported)
- **SS**: Speech Separation (⛳ supported)
- **VC**: Voice Conversion (⛳ supported)
- **LASS**: Language-Queried Audio Source Separation (⛳ supported)
- **CODEC**: Audio Tokenization (⛳ supported)
- **AE**: Audio Editing (⛳ supported)
- **TTA**: Text to Audio (⛳ developing)
- more...

In addition to the frameworks for specific audio tasks, **QuarkAudio** also provides works involving **neural audio codec (NAC)**, which is the fundamental module to combine audio modality with language models.


## 🚀 News
- **2025/09/22**: We release [***UniSE***](https://github.com/hyyan2k/UniSE), a foundation model for unified speech generation. The system supports target speaker extraction, universal speech enhancement.[***demo***](https://hyyan2k.github.io/UniSE/), [![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2510.20441),Code will comming soon. 
- **2025/10/26**: We release [***UniTok-Audio***](https://github.com/alibaba/unified-audio), The system supports target speaker extraction, universal speech enhancement, Speech Restoration, Voice Conversion, Language-Queried Audio Source Separation, Audio Tokenization,[***demo***](https://alibaba.github.io/unified-audio/), [![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2510.26372)
## key Works
### UniSE
[UniSE](https://github.com/alibaba/unified-audio/tree/main/QuarkAudio-UniSE): A Unified Framework for Decoder-Only Autoregressive LM-Based Speech Enhancement[![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2510.20441) 
supported tasks: **SR**, **TSE**, **SS**
    