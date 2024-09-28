export MASTER_PORT=$((54000 + $RANDOM % 10000))
export MASTER_ADDR=localhost

gpu_num=1
PRETRAINED_HF_PATH="GLaMM-GranD-Pretrained"
GROUNDING_ENC_CKPT_PATH="checkpoints/sam_vit_h_4b8939.pth"
OUTPUT_DIR_PATH="outputs/test_0927"

srun --partition=mozi-S1 --gres=gpu:${gpu_num} --ntasks-per-node=1 --kill-on-bad-exit --quotatype=reserved \
deepspeed --master_port $MASTER_PORT train.py --version $PRETRAINED_HF_PATH --dataset_dir ./data/ --vision_pretrained $GROUNDING_ENC_CKPT_PATH --exp_name $OUTPUT_DIR_PATH --lora_r 8 --lr 3e-4 --pretrained --use_segm_data --seg_dataset "Semantic_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG" --segm_sample_rates "1,3,3,3,1" --val_dataset "FlickrGCGVal|RefCocoGCGVal|PsgGCGVal" --epochs 10 --steps_per_epoch 500 --mask_validation