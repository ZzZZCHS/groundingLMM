save_dir="/ailab/user/huanghaifeng/work/groundingLMM/output/outputs/test_1020/ckpt_model_best" # ckpt_model_best
ckpt_path="$save_dir/pytorch_model.bin"
hf_dir="$save_dir/hf_model"
eval_results_dir="$save_dir/eval_results"
mask_results_dir="$save_dir/mask_results"

# cd $save_dir
# # srun --partition=mozi-S1 --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit --quotatype=reserved \
# python zero_to_fp32.py "$save_dir" "$ckpt_path"
# cd -


# export PYTHONPATH="./:$PYTHONPATH"
# # srun --partition=mozi-S1 --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit --quotatype=reserved \
# python scripts/merge_lora_weights.py --version 'GLaMM-GranD-Pretrained' --weight $ckpt_path --save_path $hf_dir


# hf_dir="/mnt/petrelfs/huanghaifeng/share_hw/groundingLMM/GLaMM-FullScope"
# eval_results_dir="/mnt/petrelfs/huanghaifeng/share_hw/groundingLMM/output/outputs/test_origin_full/eval_results"
# mask_results_dir="/mnt/petrelfs/huanghaifeng/share_hw/groundingLMM/output/outputs/test_origin_full/mask_results"
# Evaluate on GCG
bash eval/gcg/run_evaluation.sh ${hf_dir} ${eval_results_dir} ${mask_results_dir}