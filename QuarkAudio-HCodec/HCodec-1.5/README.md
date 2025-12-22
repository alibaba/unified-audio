# Hcodec with adpative frame rate
## Installation
1. Install dependencies from requirement.txt via pypi or environment.yml via anaconda
2. Feel free to download our pretrained model from [gdrive]() to <project_root>/checkpoints/hcodec_adaptive.ckpt

## Tokenizer
+ Quick start

```bash
#!/bin/bash
python audio_tokenizer.py
```

## Train
+ Quick start

```bash
#!/bin/bash
python ./train.py --config conf/config_adaptive_v3.yaml
```

+ Customize your training options about adaptive frame rate

```yaml
###  part of file conf/config_adaptive_v3.yaml
adaptive_config: 
  training: true
  use_similarity_alignment: true
  use_dynamic_similarity_threshold: true
  infer_using_dynamic_threshold: false
  similarity_threshold: 0.7
  similarity_threshold_lower: 0.7
  similarity_threshold_upper: 1.0 # range of threshold for training [lower,upper]
  max_tokens_per_group: 8
  manual_threshold: null # only for infering stage with fixed threshold, keep null when training
  use_query_token_aggregator: true
  aggregators:
    semantic_aggregator:
      dim: 512
      in_out_dim: 512
      num_heads: 8
      num_layers: 32
      dim_feedforward: 2048
      causal: false
      use_mean_pooling_init: true
      context_frames: 16
    acoustic_aggregator:
      dim: 512
      in_out_dim: 512   # 512*4
      num_heads: 8
      num_layers: 32
      dim_feedforward: 2048
      causal: false
      use_mean_pooling_init: true
      context_frames: 16
  use_bottleneck_transformer: true  # apply bottleneck when combining disaggregated acoustic and semantic features
  transformer_kwargs:
    d_model: 1024                # transformer_dim
    num_heads: 8                 # transformer_num_heads
    num_layers: 32               # transformer_num_layers
    causal: false                # transformer_causal
    layer_scale: 0.01
    context: 16                  # transformer_context_frames, calculated context window
    conv_layout: true
    max_period: 10000
    gating: "none"
    norm: "layer_norm"
    positional_embedding: "rope"
    dim_feedforward: 2048        # transformer_dim_feedforward
    input_dimension: 1024        # latent_dim
    output_dimensions:           
      - 1024
```

## Test
+ Quick start

```bash
###  part of file scripts/eval_libri.sh

# raw audio directory
speech_dir=/mnt/nas1/datasets/raw/LibriSpeech/test_clean
# test config file
config_path=./conf/config_adaptive_v3.yaml
# pretrained weight
ckpt_path=./checkpoints/epoch=86-step=111999-pesq=3.02-utmos=4.03.ckpt
# save directory for reconstructed audio
save_enhanced=/mnt/nas1/datasets/raw/enhanced/libri_adaptive_v3_step111999_1201_thr7-10

# log directory for recording std/err output from scripts
log_dir=./log/test/librispeech/
```

```bash
bash ./scripts/eval_libri.sh
```

+ Customize your testing options about adaptive frame rate

```yaml
  training: false # keep false when testing
  use_similarity_alignment: true
  use_dynamic_similarity_threshold: false
  infer_using_dynamic_threshold: true # work when manual_threshold is null
  similarity_threshold: 0.7
  similarity_threshold_lower: 0.7
  similarity_threshold_upper: 1.0 # valid interval of dynamic threshold
  max_tokens_per_group: 8
  manual_threshold: null # set to a fixed value when evaluate specific threshold
```

## 😘 Acknowlegement
We would like to thank the great work of following projects:

- The adaptive mechanism implementation is based on the work from [FlexiCodec](https://github.com/amphionspace/FlexiCodec)
- Transformer implementation is based on the work from [Mimi Codec](https://github.com/kyutai-labs/moshi)