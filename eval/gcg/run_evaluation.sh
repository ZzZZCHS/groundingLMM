#!/bin/sh

## USAGE

## bash eval/gcg/run_evaluation.sh <path to the HF checkpoints path> <path to the directory to save the evaluation results>

## USAGE


export PYTHONPATH="./:$PYTHONPATH"
MASTER_PORT=24999
NUM_GPUS=8  # Adjust it as per the available #GPU

# Positional arguments for the bash scripts
CKPT_PATH=$1
RESULT_PATH=$2
MASK_PATH=$3
POLICY_CONFIG=$4

# -------------------------- original glamm -----------------
# Path to the GranD-f evaluation dataset images directory
# IMAGE_DIR=./data/GranDf/GranDf_HA_images/val_test

# Run Inference
# srun --partition=mozi-S1 --gres=gpu:"$NUM_GPUS" --ntasks-per-node=1 --kill-on-bad-exit --quotatype=reserved \
# torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" eval/gcg/infer.py --hf_model_path "$CKPT_PATH" --img_dir "$IMAGE_DIR" --output_dir "$RESULT_PATH"

# Path to the GranD-f evaluation dataset ground-truths directory
# GT_DIR=./data/GranDf/annotations/val_test

# Evaluate
# srun --partition=mozi-S1 --gres=gpu:0 --ntasks-per-node=1 --kill-on-bad-exit --quotatype=reserved \
#     python eval/gcg/evaluate.py --prediction_dir_path "$RESULT_PATH" --gt_dir_path "$GT_DIR" --split "val"
# srun --partition=mozi-S1 --gres=gpu:0 --ntasks-per-node=1 --kill-on-bad-exit --quotatype=reserved \
#     python eval/gcg/evaluate.py --prediction_dir_path "$RESULT_PATH" --gt_dir_path "$GT_DIR" --split "test"


# --------------------------- evaluate robocasa ----------------
IMAGE_DIR="data/robocasa_datasets/v0.1/generated_1024/images"
ANNO_PATH="data/GranDf/annotations/train/robocasa_GCG_val.json"

# Run Inference
# srun --partition=mozi-S1 --gres=gpu:"$NUM_GPUS" --ntasks-per-node=1 --kill-on-bad-exit --quotatype=reserved \
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" eval/gcg/infer_robocasa.py --version "$CKPT_PATH" --img_dir "$IMAGE_DIR" --output_dir "$RESULT_PATH" --anno_path "$ANNO_PATH" 
# --policy_config "$POLICY_CONFIG"

# Evaluate
# srun --partition=mozi-S1 --gres=gpu:0 --ntasks-per-node=1 --kill-on-bad-exit --quotatype=reserved \
python eval/gcg/evaluate_robocasa.py --prediction_dir_path "$RESULT_PATH" --anno_path "$ANNO_PATH" --mask_results_path "$MASK_PATH"