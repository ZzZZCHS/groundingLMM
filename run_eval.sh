#!/bin/bash
exp_name="20241101_125511_train_joint_gcg"
save_dir="/ailab/user/huanghaifeng/work/groundingLMM/output/outputs/$exp_name/ckpt_model_best" # ckpt_model_best
ckpt_path="$save_dir/pytorch_model.bin"
hf_dir="$save_dir/hf_model"
eval_results_dir="$save_dir/eval_results"
mask_results_dir="$save_dir/mask_results"
policy_config="config/bc.json"
raw_data_dir="/ailab/user/huanghaifeng/work/groundingLMM/data/robocasa_datasets/v0.1/generated_1024"
only_policy=False

export PYTHONPATH="./:$PYTHONPATH"


# --------------------  get pytorch_model.bin
cd $save_dir
python zero_to_fp32.py "$save_dir" "$ckpt_path"
cd -

# --------------------  merge pretrained lora weights

python scripts/merge_lora_weights.py --version 'GLaMM-GranD-Pretrained' --weight $ckpt_path --save_path $hf_dir

# add policy
# python scripts/merge_lora_weights.py --version 'GLaMM-GranD-Pretrained' --weight $ckpt_path --save_path $hf_dir --policy_config $policy_config --only_policy $only_policy


# --------------------- infer & evaluate

# Evaluate on GCG
bash eval/gcg/run_evaluation.sh ${hf_dir} ${eval_results_dir} ${mask_results_dir} ${policy_config}


# Evaluate policy
# python eval/policy/infer.py \
#     --version ${hf_dir} \
#     --exp_dir ${save_dir} \
#     --policy_config ${policy_config} \
#     --raw_data_dir ${raw_data_dir} \
#     --only_policy ${only_policy} \
#     --use_gt_mask


