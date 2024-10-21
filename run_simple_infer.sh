#!/bin/bash

ckpt_path=/ailab/user/huanghaifeng/work/groundingLMM/output/outputs/test_1020/ckpt_model_best/hf_model
image_path=/ailab/user/huanghaifeng/work/groundingLMM/data/real_images/20241018-003114.jpeg
prompt="pick the pink object"

# real_images/20241018-003020.jpeg pick the water bottle
# real_images/20241018-003114.jpeg pick the pink object
# real_images/20241018-003138.jpeg pick the yellow object

# droid_images/1.jpg pick the round object
# droid_images/2.jpg pick the mouse
# droid_images/4.jpg pick the cup

python simple_infer.py \
    --hf_model_path "$ckpt_path" \
    --image_path "$image_path" \
    --prompt "$prompt"

# ckpt_path=/ailab/user/huanghaifeng/work/groundingLMM/GLaMM-FullScope

# python simple_infer.py \
#     --hf_model_path "$ckpt_path" \
#     --image_path "$image_path" \
#     --prompt "$prompt" \
#     --use_spacy