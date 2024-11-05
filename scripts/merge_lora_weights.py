import os
import torch
import argparse
from peft import get_peft_model
from train import setup_tokenizer_and_special_tokens, initialize_model, prepare_model_for_training, setup_lora_config
import json
from config import config_factory


def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM: Merge lora weights and save model in hf format")

    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained", help='Path to the base model.')
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--weight", required=True, type=str, help="Path to the .bin model "
                                                                  "(generated using the script zero_to_fp32.py)")
    parser.add_argument("--save_path", required=True, type=str, help="Path to save the hf model.")
    parser.add_argument("--policy_config", default=None, type=str)
    parser.add_argument("--only_policy", default=False, type=bool)

    # Model-specific settings
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true", default=False)
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true", default=False)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--with_region", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--pretrained", action="store_true", default=True)
    # Training settings
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")

    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.policy_config is not None:
        ext_cfg = json.load(open(args.policy_config, 'r'))
        policy_config = config_factory(ext_cfg["algo_name"])
        with policy_config.unlocked():
            policy_config.update(ext_cfg)
    else:
        policy_config = None

    # Create output directory if not exists already
    os.makedirs(args.save_path, exist_ok=True)

    # Initialize the tokenizer and model
    tokenizer = setup_tokenizer_and_special_tokens(args)
    
    model, shape_meta = initialize_model(args, tokenizer, policy_config=policy_config)
    
    if not args.only_policy:
        if policy_config:
            glamm_model = model.glamm_model
        else:
            glamm_model = model
        
        glamm_model.get_model().initialize_vision_modules(glamm_model.get_model().config)
        vision_tower = glamm_model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16)
        glamm_model.get_model().initialize_glamm_model(glamm_model.get_model().config)
        lora_r = args.lora_r
        if lora_r > 0:
            if policy_config:
                lora_config = setup_lora_config(glamm_model, args)
                model.glamm_model = get_peft_model(glamm_model, lora_config)
                model.glamm_model.print_trainable_parameters()
            else:
                lora_config = setup_lora_config(model, args)
                model = get_peft_model(model, lora_config)
                
        glamm_model.resize_token_embeddings(len(tokenizer))

    # Load the state-dict from --weights
    state_dict = torch.load(args.weight, map_location="cpu")
    if policy_config:
        model.load_state_dict(state_dict, strict=True)
    else:
        updated_state_dict = {}
        for key in state_dict.keys():
            updated_key = f"base_model.model.{key}"
            updated_state_dict[updated_key] = state_dict[key]
        model.load_state_dict(updated_state_dict, strict=True)

    # Merge and save
    if not args.only_policy:
        if policy_config:
            model.glamm_model = model.glamm_model.merge_and_unload()
        else:
            model = model.merge_and_unload()
    state_dict = {}
    for k, v in model.state_dict().items():
        if "vision_tower" not in k:
            state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()