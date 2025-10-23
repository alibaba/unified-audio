# Unified-Audio: An Open-Source Project to Unify Audio Processing and Generation

This project contains a series of works developed for audio (including speech, music, and general audio events) processing and generation, which helps reproducible research in the field of audio. The target of **Unified-Audio** is to explore a unified framework to handle **different audio processing and generation tasks**, including:

- **SR**: Speech Restoration (⛳ supported)
- **TSE**: Target Speaker Extraction (⛳ supported)
- **SS**: Speech Separation (⛳ supported)
- **VC**: Voice Conversion (⛳ supported)
- **LASS**: Language-Queried Audio Source Separation (⛳ supported)
- **AE**: Audio Editing (⛳ developing)
- **TTA**: Text to Audio (⛳ developing)
- more...

In addition to the frameworks for specific audio tasks, **Unified-Audio** also provides works involving **neural audio codec (NAC)**, which is the fundamental module to combine audio modality with language models.


## 🚀 News
- **2025/09/22**: We release [***UniSE***](https://github.com/hyyan2k/UniSE), a foundation model for unified speech generation. The system supports target speaker extraction, universal speech enhancement.[***demo***](https://hyyan2k.github.io/UniSE/), Code will comming soon.

## key Works
### UniSE
[UniSE](https://github.com/alibaba/unified-audio/tree/main/UniSE): A Unified Framework for Decoder-Only Autoregressive LM-Based Speech Enhancement
supported tasks: **SR**, **TSE**, **SS**
