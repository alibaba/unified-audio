python download_script.py \
    --source hf \
    --repo_id facebook/w2v-bert-2.0 \
    --filename model.safetensors \
    --save_path \
    ./codec_ckpt/hub/models--facebook--w2v-bert-2.0/model.safetensors

python download_script.py \
     --source hf \
     --repo_id HKUSTAudio/xcodec2 \
     --filename ckpt/epoch=4-step=1400000.ckpt \
     --save_path ./codec_ckpt/epoch=4-step=1400000.ckpt

python download_script.py \
     --source hf \
     --repo_id ASLP-lab/LLaSE-G1 \
     --filename ckpt/model.pt.tar \
     --save_path ./model.pt.tar
