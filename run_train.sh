export MASTER_PORT=$((54000 + $RANDOM % 10000))
export MASTER_ADDR=localhost

PRETRAINED_HF_PATH="GLaMM-GranD-Pretrained"  # GLaMM-GranD-Pretrained GLaMM-GCG
GROUNDING_ENC_CKPT_PATH="checkpoints/sam_vit_h_4b8939.pth"

gpu_num=8
batch_size=8
accum_steps=1
total_batch_size=$(($batch_size * $accum_steps))
lr=1e-4
epochs=5
steps_per_epoch=50000
warmup_steps=10000
weight_decay=0.01
exp_name="train_joint_gcg"

# OUTPUT_DIR_PATH=outputs/"$(date +"%Y%m%d_%H%M%S")"_"$exp_name"_bs"$total_batch_size"_lr"$lr"_steps"$epochs"-"$steps_per_epoch"-"$warmup_steps"_decay"$weight_decay"
OUTPUT_DIR_PATH=outputs/"$(date +"%Y%m%d_%H%M%S")"_"$exp_name"

export PYTHONPATH="./:$PYTHONPATH"

# Grounded Conversation Generation (GCG)
# srun --partition=mozi-S1 --gres=gpu:${gpu_num} --ntasks-per-node=1 --kill-on-bad-exit --quotatype=reserved \
deepspeed --master_port $MASTER_PORT train.py --version $PRETRAINED_HF_PATH --dataset_dir ./data/ --vision_pretrained $GROUNDING_ENC_CKPT_PATH --exp_name $OUTPUT_DIR_PATH --lora_r 8 --lr 3e-4 --pretrained --use_segm_data --seg_dataset "Semantic_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG||Robocasa_GCG" --segm_sample_rates "1,3,3,3,1,3" --val_dataset "FlickrGCGVal|RefCocoGCGVal|PsgGCGVal|Robocasa_GCG" --epochs 10 --steps_per_epoch 500 --mask_validation

# only robocasa
# srun --partition=mozi-S1 --gres=gpu:${gpu_num} --ntasks-per-node=1 --kill-on-bad-exit --quotatype=reserved \
# deepspeed --master_port $MASTER_PORT train.py --version $PRETRAINED_HF_PATH --dataset_dir ./data/ --vision_pretrained $GROUNDING_ENC_CKPT_PATH --exp_name $OUTPUT_DIR_PATH --lora_r 8 --lr 3e-4 --pretrained --use_segm_data --seg_dataset "Robocasa_GCG" --segm_sample_rates "1" --val_dataset "Robocasa_GCG" --epochs 10 --steps_per_epoch 100 --mask_validation

# gcg + policy
# deepspeed --master_port $MASTER_PORT train.py --version $PRETRAINED_HF_PATH --dataset_dir ./data/ --vision_pretrained $GROUNDING_ENC_CKPT_PATH --exp_name $OUTPUT_DIR_PATH --lora_r 8 --lr 3e-4 --pretrained --use_segm_data --seg_dataset "Robocasa_GCG" --segm_sample_rates "1" --use_gvla_data --val_dataset "Robocasa_GCG" --epochs 10 --steps_per_epoch 500 --mask_validation --policy_config "config/bc.json" --raw_data_dir "data/robocasa_datasets/v0.1/generated_1013"

# add raw data
# deepspeed --master_port $MASTER_PORT train.py --version $PRETRAINED_HF_PATH --dataset_dir ./data/ --vision_pretrained $GROUNDING_ENC_CKPT_PATH --exp_name $OUTPUT_DIR_PATH --lora_r 8 --lr 3e-4 --pretrained --use_segm_data --seg_dataset "Semantic_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG||Robocasa_GCG" --segm_sample_rates "1,3,3,3,1,5" --use_gvla_data --val_dataset "Robocasa_GCG" --epochs 10 --steps_per_epoch 1000 --mask_validation --policy_config "config/bc.json" --raw_data_dir "data/robocasa_datasets/v0.1/generated_1013"

# only policy
# deepspeed --master_port $MASTER_PORT train.py \
#     --version $PRETRAINED_HF_PATH --dataset_dir ./data/ --vision_pretrained $GROUNDING_ENC_CKPT_PATH --exp_name $OUTPUT_DIR_PATH --lora_r 8  --pretrained --use_gvla_data --mask_validation  \
#     --lr ${lr} --epochs ${epochs} --steps_per_epoch ${steps_per_epoch} --warmup_steps ${warmup_steps} --batch_size ${batch_size} --grad_accumulation_steps ${accum_steps} --weight_decay ${weight_decay} \
#     --policy_config "config/bc.json" --raw_data_dir "data/robocasa_datasets/v0.1/generated_1024" --only_policy --use_gt_mask

# Combined tasks (stuck)
# srun --partition=mozi-S1 --gres=gpu:${gpu_num} --ntasks-per-node=1 --kill-on-bad-exit --quotatype=reserved \
# deepspeed --master_port $MASTER_PORT train.py --version $PRETRAINED_HF_PATH --dataset_dir ./data/ --vision_pretrained $GROUNDING_ENC_CKPT_PATH --exp_name $OUTPUT_DIR_PATH --lora_r 8 --lr 3e-4 --pretrained --use_cap_data --use_reg_data --use_segm_data --cap_dataset "CocoCap||LLaVaInstruct" --cap_sample_rate "1,2" --reg_dataset "RefCoco_Reg||RefCocoG_Reg||RefCocoP_Reg||VisGen_Reg" --reg_sample_rates "1,1,1,1" --seg_dataset "Semantic_Segm||Refer_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG" --segm_sample_rates "4,3,2,2,2,1" --val_dataset "FlickrGCGVal|RefCocoGCGVal|PsgGCGVal" --epochs 10 --steps_per_epoch 500
