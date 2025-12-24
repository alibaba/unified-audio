# HCodec-1.5 with adaptive frame rate
## Installation
1. Install dependencies from requirement.txt via pypi
2. Download pretrained weights from Huggingface &#x1F917;: [QuarkAudio/HCodec-1.5-adaptive](https://huggingface.co/QuarkAudio/HCodec-1.5-adaptive) and save them to ./checkpoints/ 

## Quick start
+ Generate tokens from audio
+ Reconstruct audio from tokens

```bash
#!/bin/bash
python audio_tokenizer.py
```

## Optional configuration
+ Customize your testing options about adaptive frame rate

```yaml
  # hyperparameter configuration in conf/config_adaptive_v3.yaml

  training: false # keep false when testing
  use_similarity_alignment: true
  use_dynamic_similarity_threshold: false
  infer_using_dynamic_threshold: true # work when manual_threshold is null
  similarity_threshold: 0.7
  similarity_threshold_lower: 0.7
  similarity_threshold_upper: 1.0 # valid interval of dynamic threshold when 'infer_using_dynamic_threshold' turns on
  max_tokens_per_group: 8
  manual_threshold: 0.6 # set to a fixed value when evaluate specific threshold
```

## 😘 Acknowlegement
We would like to thank the great work of following projects:

- The adaptive mechanism implementation is based on the work from [FlexiCodec](https://github.com/amphionspace/FlexiCodec) and [VARSTok](https://github.com/FunAudioLLM/FunResearch/tree/main/VARSTok).
- Transformer implementation is based on the work from [Mimi Codec](https://github.com/kyutai-labs/moshi)