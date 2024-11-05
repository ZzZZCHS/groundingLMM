"""
train.py - GLaMM Model Training on Mixed Datasets

Trains the GLaMM model using Caption, Region, and Segmentation datasets with a random sampling approach. This method
is crucial for developing a versatile model capable of handling diverse applications effectively.
"""
import os
import sys
import time
import tqdm
import random
import torch
import argparse
import deepspeed
import numpy as np
import transformers
import glob
import json
import imageio
from functools import partial
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.GLaMM import GLaMMForCausalLM, GLaMMWithPolicy
from model.llava import conversation as conversation_lib

from dataset.dataset import custom_collate_fn, gvla_custom_collate_fn, HybridSegDataset, HybridRegDataset, HybridCapDataset
from tools.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, ProgressMeter, dict_to_cuda, dict_to_bfloat16, Summary, intersectionAndUnionGPU)

from dataset.segm_datasets.RefCOCO_Segm_ds import ReferSegmDataset
from dataset.region_datasets.RefCOCO_VG_Region_ds import RefCocoGRegDataset, VisualGenomeRegDataset
from dataset.caption_datasets.COCO_Caption_ds import CocoCapDataset
from dataset.gcg_datasets.GranDf_gcg_ds import OpenPsgGCGDataset, Flickr30kGCGDataset, RefCOCOgGCGDataset, RobocasaGCGDataset
from dataset.gvla_datasets.grounded_vla_ds import GroundedVLADataset, GroundedVLAMetaDataset
from config import config_factory
import tools.file_utils as FileUtils
import tools.obs_utils as ObsUtils
import tools.log_utils as LogUtils
import tools.tensor_utils as TensorUtils
import tools.env_utils as EnvUtils
import tools.lang_utils as LangUtils
from tools.script_utils import deep_update
from tools.train_utils import VAL_ENV_INFOS
from transformers import LlamaConfig
import traceback
import robomimic.utils.obs_utils as RobomimicObsUtils
RobomimicObsUtils.RESIZE_TO_128 = False
from copy import deepcopy
import cv2


eval_env_meta_list, eval_env_name_list, eval_env_horizon_list = [], [], []

def parse_args(args):
    parser = argparse.ArgumentParser(description="GLaMM Model Training")

    # Model-specific settings
    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained")
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true")
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--with_region", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--precision", default='bf16', type=str)
    parser.add_argument("--policy_config", default=None, type=str)
    parser.add_argument("--only_policy", action="store_true")
    parser.add_argument("--use_gt_mask", action="store_true")

    # Dataset settings
    parser.add_argument("--use_cap_data", action="store_true", help="Use caption data")
    parser.add_argument("--use_reg_data", action="store_true", help="Use region data")
    parser.add_argument("--use_segm_data", action="store_true", help="Use segmentation data")
    parser.add_argument("--use_gvla_data", action="store_true", help="Use groundedvla data")
    parser.add_argument("--weight_cap", default=0.15, type=float, help="Sampling weight for caption data")
    parser.add_argument("--weight_reg", default=0.40, type=float, help="Sampling weight for region data")
    parser.add_argument("--weight_segm", default=0.45, type=float, help="Sampling weight for segmentation data")
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument("--seg_dataset", default="Semantic_Segm||Refer_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG",
                        type=str, help="Choose from: Semantic_Segm, Refer_Segm, RefCoco_GCG, GranDf_GCG, PSG_GCG, Flickr_GCG, GrandRefer_Segm")
    parser.add_argument("--segm_sample_rates", default="5,4,3,3,3,1", type=str)
    parser.add_argument("--reg_dataset", default="RefCoco_Reg||RefCocoG_Reg||RefCocoP_Reg||VisGen_Reg",
                        type=str, help="Choose from: RefCoco_Reg, RefCocoG_Reg, RefCocoP_Reg, VisGen_Reg, Flickr_Reg, GrandRefer_Reg")
    parser.add_argument("--reg_sample_rates", default="1,1,1,1", type=str)
    parser.add_argument("--cap_dataset", default="CocoCap||LLaVaInstruct", type=str,
                        help="Choose from: CocoCap, LLaVaInstruct, GrandCaptionDataset")
    parser.add_argument("--cap_sample_rates", default="1,1", type=str)
    # parser.add_argument("--gvla_dataset", default="Robocasa_GVLA", type=str)
    # parser.add_argument("--gvla_sample_rates", default="1", type=str)
    parser.add_argument("--semantic_segm_data", default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary", type=str)
    parser.add_argument("--refer_segm_data", default="refcoco||refcoco+||refcocog||refclef", type=str)
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--raw_data_dir", default=None, type=str, help="raw data dir of the hdf5 files")

    # Training settings
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--weight_decay", default=0., type=float)
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=2, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--print_freq", default=10, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")

    # Evaluation settings
    parser.add_argument("--val_dataset", default="CocoCapVal|RefCOCOgRegVal|RefCOCOgSegmVal", type=str,
                        help="Choose from: CocoCapVal, RefCOCOgRegVal, VisGenomeRegVal, RefCOCOgSegmVal, PsgGCGVal, "
                             "RefCocoGCGVal, FlickrGCGVal")
    parser.add_argument("--mask_validation", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    # Experiment settings
    parser.add_argument("--log_base_dir", default="./output", type=str)
    parser.add_argument("--exp_name", default="GlamFinetuneOS", type=str)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args(args)


def prepare_observation(ob, ep_lang_emb, model):
    if len(ob["robot0_eef_pos"].shape) == 1:
        ob["lang_emb"] = ep_lang_emb
    else:
        ob["lang_emb"] = np.repeat(ep_lang_emb[np.newaxis], len(ob["robot0_eef_pos"]), axis=0)
    ob = TensorUtils.to_tensor(ob)
    ob = TensorUtils.to_batch(ob)
    ob = TensorUtils.to_device(ob, model.policy_net.device)
    ob = TensorUtils.to_float(ob)
    return ob


def postprocess_action(ac):
    ac = ac[0]
    ac = TensorUtils.to_numpy(ac)
    ori_ac = ac
    # if action_normalization_stats is not None:
    #     action_keys = model.policy_net.global_config.train.action_keys
    #     action_shapes = {k: action_normalization_stats[k]["offset"].shape[1:] for k in action_normalization_stats}
    #     ac_dict = AcUtils.vector_to_action_dict(ac, action_shapes=action_shapes, action_keys=action_keys)
    #     ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=action_normalization_stats)
    #     action_config = model.policy_net.global_config.train.action_config
    #     for key, value in ac_dict.items():
    #         this_format = action_config[key].get("format", None)
    #         if this_format == "rot_6d":
    #             rot_6d = torch.from_numpy(value).unsqueeze(0)
    #             conversion_format = action_config[key].get("convert_at_runtime", "rot_axis_angle")
    #             if conversion_format == "rot_axis_angle":
    #                 rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).squeeze().numpy()
    #             elif conversion_format == "rot_euler":
    #                 rot = TorchUtils.rot_6d_to_euler_angles(rot_6d=rot_6d, convention="XYZ").squeeze().numpy()
    #             else:
    #                 raise ValueError
    #             ac_dict[key] = rot
    #     ac = AcUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)
    return ac


def initialize_environment(args):
    """ Set up logging and model directories. """
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        return SummaryWriter(args.log_dir)
    return None


def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token

    if not args.pretrained:
        if args.use_mm_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        # modifications specific for regions
        reg_tokens = ['<bbox>', '<point>']
        # Adding special tokens for pixel grounding
        segmentation_tokens = ['[SEG]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        special_tokens = reg_tokens + segmentation_tokens + phrase_tokens
        tokenizer.add_tokens(special_tokens, special_tokens=True)

    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]

    return tokenizer


def initialize_model(args, tokenizer, policy_config):
    """ Initialize the GLaMM model. """
    model_args = {k: getattr(args, k) for k in
                  ["train_mask_decoder", "out_dim", "ce_loss_weight", "dice_loss_weight", "bce_loss_weight",
                   "seg_token_idx", "vision_pretrained", "vision_tower", "use_mm_start_end", "mm_vision_select_layer",
                   "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end",
                   "with_region", "bbox_token_idx", "eop_token_idx", "bop_token_idx"]}
    model_args["num_level_reg_features"] = 4
    shape_meta = None
    if policy_config is None:
        model = GLaMMForCausalLM.from_pretrained(
            args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args
        )
        
        # Configure model tokens
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        ObsUtils.initialize_obs_utils_with_config(policy_config)
        RobomimicObsUtils.initialize_obs_utils_with_config(policy_config)
        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=policy_config.train.data[0]["path"],
            action_keys=policy_config.train.action_keys,
            all_obs_keys=policy_config.all_obs_keys,
            ds_format=policy_config.train.data_format,
            verbose=False
        )
        model = GLaMMWithPolicy(
            config=LlamaConfig(),
            glamm_version=args.version, 
            glamm_model_args=model_args, 
            obs_key_shapes=shape_meta["all_shapes"],
            ac_dim=shape_meta["ac_dim"],
            policy_config=policy_config,
            only_policy=args.only_policy
        )
        # Configure model tokens
        if not args.only_policy:
            model.glamm_model.config.eos_token_id = tokenizer.eos_token_id
            model.glamm_model.config.bos_token_id = tokenizer.bos_token_id
            model.glamm_model.config.pad_token_id = tokenizer.pad_token_id
    return model, shape_meta


def prepare_model_for_training(model, tokenizer, args, policy_config):
    
    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]
    
    if args.only_policy:
        set_trainable_modules(model)
        return
    
    # Enable input gradients
    if policy_config:
        glamm_model = model.glamm_model
    else:
        glamm_model = model
    
    glamm_model.enable_input_require_grads()
    glamm_model.gradient_checkpointing_enable()

    # Initialize vision tower
    print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.vision_tower
        ) + '\033[0m'
    )
    glamm_model.get_model().initialize_vision_modules(glamm_model.get_model().config)
    vision_tower = glamm_model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=args.local_rank)

    # Initialize GLaMM model and adjust requires_grad
    if not args.pretrained:
        glamm_model.get_model().initialize_glamm_model(glamm_model.get_model().config)
    else:
        for param in glamm_model.get_model().grounding_encoder.parameters():
            param.requires_grad = False
        if glamm_model.get_model().config.train_mask_decoder:
            glamm_model.get_model().grounding_encoder.mask_decoder.train()
            for param in glamm_model.get_model().grounding_encoder.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        glamm_model.get_model().text_hidden_fcs.train()
        for param in glamm_model.get_model().text_hidden_fcs.parameters():
            param.requires_grad = True

    # Set requires_grad for vision tower and mm projector
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in glamm_model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Set requires_grad based on LoRA training
    lora_r = args.lora_r
    if lora_r == 0:
        for p in glamm_model.get_model().layers.parameters():
            p.requires_grad = True
        for p in glamm_model.get_model().mm_projector.parameters():
            p.requires_grad = True

    # Configure LoRA if applicable
    if lora_r > 0:
        if policy_config:
            lora_config = setup_lora_config(glamm_model, args)
            model.glamm_model = get_peft_model(glamm_model, lora_config)
            glamm_model = model.glamm_model
        else:
            lora_config = setup_lora_config(model, args)
            model = get_peft_model(model, lora_config)

    # Resize token embeddings
    glamm_model.resize_token_embeddings(len(tokenizer))

    # Make certain modules trainable
    set_trainable_modules(model)


def setup_lora_config(model, args):
    """ Configure LoRA settings for the model. """

    def find_proj_layers(model, target_modules):
        """ Identify projection layers in the model for LoRA adaptation. """
        linear_cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (isinstance(module, linear_cls) and all(
                    x not in name for x in ["grounding_encoder", "vision_tower", "mm_projector", "text_hidden_fcs"]
            ) and any(x in name for x in target_modules)):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    # Extracting LoRA target modules
    lora_target_modules = args.lora_target_modules.split(",")
    lora_module_names = find_proj_layers(model, lora_target_modules)

    # Configuring LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    return lora_config


def set_trainable_modules(model):
    """ Make specified modules in the model trainable. """
    trainable_modules = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "region_encoder", "policy_net"]
    for name, param in model.named_parameters():
        if any(module in name for module in trainable_modules):
            print(f"Making trainable: {name}, Shape: {param.shape}")
            param.requires_grad = True

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('\033[92m' + "---- Total parameters: ----{}".format(total_params) + '\033[0m')
        print('\033[92m' + "---- Trainable parameters: ----{}".format(trainable_params) + '\033[0m')

    count_parameters(model)


def initialize_datasets_and_loaders(args, tokenizer, policy_config, shape_meta=None):
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # Common dataset arguments
    common_ds_args = {"dataset_dir": args.dataset_dir, "tokenizer": tokenizer,
                      "global_image_encoder": args.vision_tower,
                      "epoch_samples": args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,
                      "precision": args.precision, "image_size": args.image_size,
                      "num_classes_per_sample": args.num_classes_per_sample}
    
    # Training datasets
    cap_train_dataset = HybridCapDataset(
        **common_ds_args, dataset=args.cap_dataset, sample_rate=[float(x) for x in args.cap_sample_rates.split(",")],
        batch_size=args.batch_size, ) if args.use_cap_data else None
    reg_train_dataset = HybridRegDataset(
        **common_ds_args, dataset=args.reg_dataset, sample_rate=[float(x) for x in args.reg_sample_rates.split(",")],
        batch_size=args.batch_size, ) if args.use_reg_data else None
    seg_train_dataset = HybridSegDataset(
        **common_ds_args, dataset=args.seg_dataset, sample_rate=[float(x) for x in args.segm_sample_rates.split(",")],
        semantic_segm_data=args.semantic_segm_data, refer_segm_data=args.refer_segm_data,
        batch_size=args.batch_size, ) if args.use_segm_data else None
    
    if args.use_gvla_data:
        gvla_datasets = []
        for hdf5_path in glob.glob(os.path.join(args.raw_data_dir, "*.hdf5")):
            if "PnPCounterToCab" not in hdf5_path:
                continue
            gvla_dataset = GroundedVLADataset(**common_ds_args, hdf5_path=hdf5_path, raw_data_dir=args.raw_data_dir, obs_keys=shape_meta["all_obs_keys"], action_keys=policy_config.train.action_keys, dataset_keys=policy_config.train.dataset_keys, action_config=policy_config.train.action_config)
            gvla_datasets.append(gvla_dataset)
            if args.debug:
                break
        gvla_train_dataset = GroundedVLAMetaDataset(gvla_datasets)
    else:
        gvla_train_dataset = None
    

    # Validation datasets
    val_datasets = []
    if not args.no_eval:
        val_dataset_classes = {
            'CocoCapVal': CocoCapDataset,
            'RefCOCOgRegVal': RefCocoGRegDataset,
            'VisGenomeRegVal': VisualGenomeRegDataset,
            'RefCOCOgSegmVal': ReferSegmDataset,
            'PsgGCGVal': OpenPsgGCGDataset,
            'RefCocoGCGVal': RefCOCOgGCGDataset,
            'FlickrGCGVal': Flickr30kGCGDataset,
            'Robocasa_GCG': RobocasaGCGDataset
        }
        for val_dataset_name in args.val_dataset.split('|'):
            val_dataset_class = val_dataset_classes.get(val_dataset_name)
            if val_dataset_class:
                if val_dataset_class == ReferSegmDataset:
                    # Modify this if other datasets in refer_segm_data need to be included in val
                    refer_segm_data = 'refcocog'
                    all_datasets = refer_segm_data.split("||")
                    for d in all_datasets:
                        val_dataset_class = val_dataset_class(
                            **common_ds_args, validation=True, refer_segm_data=d, split='val'
                        )
                        val_dataset_class._set_len(len(val_dataset_class.refer_segm_data[d]['images']))
                        val_datasets.append(val_dataset_class)
                else:
                    val_datasets.append(val_dataset_class(**common_ds_args, validation=True))

    return cap_train_dataset, reg_train_dataset, seg_train_dataset, gvla_train_dataset, val_datasets


def setup_data_loaders(args, cap_train_dataset, reg_train_dataset, seg_train_dataset, gvla_train_dataset, val_datasets, tokenizer):
    sampler_args = {"shuffle": False, "drop_last": False}
    train_loader_args = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.workers,
                         "pin_memory": False}
    val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.workers,
                       "pin_memory": False}
    collate_fn_args_train = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=False
    )
    gvla_collate_fn_args_train = partial(
        gvla_custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=False
    )
    inference_mode = args.mask_validation
    collate_fn_args_val = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=inference_mode
    )

    # Training loaders
    cap_train_loader = torch.utils.data.DataLoader(
        cap_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            cap_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if cap_train_dataset is not None else None
    reg_train_loader = torch.utils.data.DataLoader(
        reg_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            reg_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if reg_train_dataset is not None else None
    seg_train_loader = torch.utils.data.DataLoader(
        seg_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            seg_train_dataset, **sampler_args
        ), collate_fn=collate_fn_args_train, **train_loader_args
    ) if seg_train_dataset is not None else None
    gvla_train_loader = torch.utils.data.DataLoader(
        gvla_train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            gvla_train_dataset, **sampler_args
        ), collate_fn=gvla_collate_fn_args_train, **train_loader_args
    ) if gvla_train_dataset is not None else None

    # Validation loader
    val_loader = None
    if val_datasets:
        combined_val_datasets = ConcatDataset(val_datasets)
        val_loader = torch.utils.data.DataLoader(
            combined_val_datasets, **val_loader_args, collate_fn=collate_fn_args_val,
            sampler=torch.utils.data.distributed.DistributedSampler(combined_val_datasets, **sampler_args), )

    return cap_train_loader, reg_train_loader, seg_train_loader, gvla_train_loader, val_loader


def initialize_deepspeed(model, tokenizer, args):
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW", 
            "params": {
                "lr": args.lr, 
                "weight_decay": args.weight_decay,
                "betas": (args.beta1, args.beta2)
            }
        },
        "scheduler": {
            # "type": "WarmupLR",
            # "params": {
            #     "warmup_num_steps": args.warmup_steps, 
            #     "warmup_type": "log",
            #     "warmup_max_lr": args.lr
            # }
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup_steps,
                "warmup_type": "linear"
            }
        },
        "fp16": {
            "enabled": args.precision == "fp16"
        }, 
        "bf16": {
            "enabled": args.precision == "bf16"
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2, 
            "contiguous_gradients": True, 
            "overlap_comm": True,
            "reduce_scatter": True, 
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8
        }, 
    }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), collate_fn=partial(
            custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank
        ), config=ds_config
    )

    return model_engine, optimizer, scheduler


def resume_training_from_checkpoint(model_engine, args):
    if args.auto_resume and not args.resume:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"Resume training from {args.resume}, start from epoch {args.start_epoch}")
        

def run_rollout(
        env, 
        horizon,
        initial_state=None,
        model=None,
        lang_encoder=None,
        render=False,
        video_writer=None,
        video_skip=5,
        ep_i=0,
        args=None
    ):
    
    env.env.env.add_object_num = 0
    ob_dict = env.reset_to(initial_state)
    assert env.env.env.unique_attr == json.loads(initial_state["ep_meta"])["unique_attr"]
    
    # policy start episode
    ep_lang_emb = TensorUtils.to_numpy(lang_encoder.get_lang_emb(env._ep_lang_str))
    model.policy_net.set_eval()
    model.policy_net.reset()
    

    results = {}
    video_count = 0  # video frame counter

    rews = []
    success = None #{ k: False for k in env.is_success() } # success metrics

    end_step = None

    video_frames = []
    
    camera_names = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
        
    masked_dict = {}
    if args.use_gt_mask:
        target_obj_str = env.env.env.target_obj_str
        if target_obj_str == "obj":
            target_obj_str += "_main"
        target_place_str = env.env.env.target_place_str
        masked_dict = {}
        geom2body_id_mapping = {geom_id: body_id for geom_id, body_id in enumerate(env.env.env.sim.model.geom_bodyid)}
        name2id = env.env.env.sim.model._body_name2id
        for cam_name in camera_names:
            seg = env.env.env.sim.render(
                camera_name=cam_name,
                width=256,
                height=256,
                depth=False,
                segmentation=True
            )
            seg = seg[::-1, :, 1]
            tmp_seg = (
                np.fromiter(
                    map(
                        lambda x: geom2body_id_mapping.get(x, -1),
                        seg.flatten()
                    ),
                    dtype=np.int32
                ).reshape(256, 256)
            )
            tmp_mask = np.zeros(tmp_seg.shape, dtype=np.uint8)
            for tmp_target_obj_str in target_obj_str.split('/'):
                tmp_mask[tmp_seg == name2id[tmp_target_obj_str]] = 1
            if target_place_str:
                tmp_mask[tmp_seg == name2id[target_place_str]] = 2
                if (tmp_seg == name2id[target_place_str]).sum() == 0 and target_place_str == "container_main" and name2id[target_place_str] == name2id[None] - 1:
                    tmp_mask[tmp_seg == name2id[None]] = 2
            tmp_mask = tmp_mask.astype(np.float32) / 2.
            tmp_mask = np.expand_dims(tmp_mask, axis=0)
            tmp_mask = np.expand_dims(tmp_mask, axis=0).repeat(ob_dict[f"{cam_name}_image"].shape[0], axis=0)
            masked_dict[f"{cam_name}_mask"] = tmp_mask
    # else:

    obs_keys = ["robot0_agentview_left_image"]
    # , "robot0_agentview_right_image", "robot0_eye_in_hand_image"]
    pred_embeddings = None
    if not args.only_policy:
        for obs_key in obs_keys:
            tmp_img = ob_dict[obs_key][0]
            tmp_img = np.uint8(tmp_img*255).transpose(1, 2, 0)
            
            cleaned_str, pred_masks, phrases, pred_embeddings = inference(env._ep_lang_str, tmp_img)
            print(cleaned_str, phrases)
            
            pred_embeddings = pred_embeddings[0]
        
            if pred_embeddings.shape[0] == 1:
                pred_embeddings = pred_embeddings.repeat(2, 1)
            tmp_norm = pred_embeddings.unsqueeze(0).repeat(2, 1, 1).norm()
            pred_embeddings = pred_embeddings / tmp_norm
            pred_embeddings = pred_embeddings.flatten(0, 1)
            seq_len = ob_dict[obs_key].shape[0]
            pred_embeddings = pred_embeddings.unsqueeze(0).repeat(seq_len, 1)
            pred_embeddings = pred_embeddings.unsqueeze(0)
    for step_i in range(horizon): #LogUtils.tqdm(range(horizon)):
        # for cam_name in camera_names:
        #     depth_name = f"{cam_name}_depth"
        #     _, depth = env.env.env.sim.render(
        #         camera_name=cam_name,
        #         width=256,
        #         height=256,
        #         depth=True
        #     )
        #     depth = np.expand_dims(depth[::-1], axis=0)
        #     if depth_name not in env.obs_history:
        #         env.obs_history[depth_name] = deque(
        #             [depth[None]] * env.num_frames,
        #             maxlen=env.num_frames
        #         )
        #     else:
        #         env.obs_history[depth_name].append(depth[None])
        #     ob_dict = env._get_stacked_obs_from_history()
        
        ob_dict.update(masked_dict)
        
        if ObsUtils.MASK_CHANNEL == 1:
            for cam_name in camera_names:
                image_key = f"{cam_name}_image"
                mask_key = f"{cam_name}_mask"
                # ob_dict[image_key] = np.concatenate([ob_dict[image_key], ob_dict[mask_key]], axis=1)
                ob_dict[image_key][:, 3:4, ...] = ob_dict[mask_key]
                del ob_dict[mask_key]
        if ObsUtils.DEPTH_CHANNEL == 1:
            for cam_name in camera_names:
                image_key = f"{cam_name}_image"
                depth_key = f"{cam_name}_depth"
                # ob_dict[image_key] = np.concatenate([ob_dict[image_key], ob_dict[depth_key]], axis=1)
                ob_dict[image_key][:, -1:, ...] = ob_dict[depth_key]
                del ob_dict[depth_key]
            
        ob_input = prepare_observation(ob_dict, ep_lang_emb, model)
        ac = model.policy_net.get_action(obs_dict=ob_input, mask_embeds=pred_embeddings)
        
        ac = postprocess_action(ac)

        # play action
        ob_dict, r, done, info = env.step(ac)

        # compute reward
        rews.append(r)

        cur_success_metrics = info["is_success"]

        if success is None:
            success = deepcopy(cur_success_metrics)
        else:
            for k in success:
                success[k] = success[k] | cur_success_metrics[k]

        # visualization
        if video_writer is not None:
            if video_count % video_skip == 0:
                frame = env.render(mode="rgb_array", height=512, width=512)
                frame = frame.copy()
                text1 = env._ep_lang_str
                position1 = (10, 50)
                color = (255, 0, 0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 1
                font_scale = 0.5
                cv2.putText(frame, text1, position1, font, font_scale, color, thickness)
                text2 = f"demo idx: {ep_i}"
                position2 = (10, 100)
                cv2.putText(frame, text2, position2, font, font_scale, color, thickness)
                video_frames.append(frame)

            video_count += 1

        if done or success["task"]:
            end_step = step_i
            break


    if video_writer is not None:
        for frame in video_frames:
            video_writer.append_data(frame)

    end_step = end_step or step_i
    total_reward = np.sum(rews[:end_step + 1])
    
    results["Return"] = total_reward
    results["Horizon"] = end_step + 1
    results["Success_Rate"] = float(success["task"])

    # log additional success metrics
    for k in success:
        if k != "task":
            if batched:
                results["{}_Success_Rate".format(k)] = success[k].astype(float)
            else:
                results["{}_Success_Rate".format(k)] = float(success[k])

    return results

def eval_policy(args, epoch_i, policy_config, model, lang_encoder):
    num_episodes = 20
    envs = env_iterator(policy_config)
    video_dir = os.path.join(args.log_dir, f"epoch{epoch_i}_videos")
    os.makedirs(video_dir, exist_ok=True)
    
    success_rate_list = []
    
    for env, horizon in zip(envs, eval_env_horizon_list):
        env_name = env.name
        video_path = os.path.join(video_dir, f"{env_name}.mp4")
        video_writer = imageio.get_writer(video_path, fps=20)
        print("rollout: env={}, horizon={}, num_episodes={}".format(
            env_name, horizon, num_episodes,
        ))
        num_success = 0
        for ep_i in LogUtils.custom_tqdm(range(num_episodes), total=num_episodes):
            initial_state = VAL_ENV_INFOS[env_name][ep_i]
            try:
                rollout_info = run_rollout(
                    env=env,
                    horizon=horizon,
                    initial_state=initial_state,
                    model=model,
                    lang_encoder=lang_encoder,
                    video_writer=video_writer,
                    ep_i=ep_i,
                    args=args
                )
            except Exception as e:
                print(traceback.format_exc())
                print(env_name, "Rollout exception at episode number {}!".format(ep_i))
                # break
                continue
            print(f"{num_success} / {ep_i+1}")
        video_writer.close()
        del env
        success_rate_list.append(num_success / num_episodes)
    
    return sum(success_rate_list) / len(success_rate_list) if len(success_rate_list) > 0 else 0.


def main(args):
    if args.policy_config is not None:
        ext_cfg = json.load(open(args.policy_config, 'r'))
        policy_config = config_factory(ext_cfg["algo_name"])
        with policy_config.unlocked():
            policy_config.update(ext_cfg)
    else:
        policy_config = None
    tokenizer = setup_tokenizer_and_special_tokens(args)
    
    if policy_config:
        load_eval_env_infos(args, policy_config)
    model, shape_meta = initialize_model(args, tokenizer, policy_config)
    prepare_model_for_training(model, tokenizer, args, policy_config)
    
    if policy_config:
        device="cuda:0"
        lang_encoder = LangUtils.LangEncoder(
            device=device,
        )
        model.policy_net.device = device

    model_engine, optimizer, scheduler = initialize_deepspeed(model, tokenizer, args)
    resume_training_from_checkpoint(model_engine, args)

    cap_train_dataset, reg_train_dataset, seg_train_dataset, gvla_train_dataset, val_datasets = (
        initialize_datasets_and_loaders(args, tokenizer, policy_config, shape_meta))
    cap_train_loader, reg_train_loader, seg_train_loader, gvla_train_loader, val_loader = (
        setup_data_loaders(args, cap_train_dataset, reg_train_dataset, seg_train_dataset, gvla_train_dataset, val_datasets, tokenizer))

    # Determine active datasets and their weights
    active_dataloaders = []
    weights = []

    if args.use_cap_data:
        active_dataloaders.append(('cap', cap_train_loader))
        weights.append(args.weight_cap)
    if args.use_reg_data:
        active_dataloaders.append(('reg', reg_train_loader))
        weights.append(args.weight_reg)
    if args.use_segm_data:
        active_dataloaders.append(('seg', seg_train_loader))
        weights.append(args.weight_segm)

    # Assert that at least one dataset is active
    # assert active_dataloaders, "Error: At least one dataset (segm, reg, or cap) must be active."

    dataset_iters = {'cap': iter(cap_train_loader) if args.use_cap_data else None,
                     'reg': iter(reg_train_loader) if args.use_reg_data else None,
                     'seg': iter(seg_train_loader) if args.use_segm_data else None, }
    gvla_dataset_iter = iter(gvla_train_loader) if args.use_gvla_data else None

    writer = initialize_environment(args)

    if args.eval_only:
        cur_val_loss = validate_model_performance(val_loader, model_engine, 0, writer, args)[0]
        exit()

    epoch_seeds = [random.randint(0, 100000) for _ in range(args.epochs)]
    dataset_choices = [idx for idx, _ in enumerate(active_dataloaders)]

    best_giou, best_ciou, best_val_loss = 0.0, 0.0, np.inf
    for epoch in range(args.start_epoch, args.epochs):
        random.seed(epoch_seeds[epoch])

        if args.only_policy:
            step_choices = None
        else:
            step_choices = random.choices(dataset_choices, weights=weights, k=args.steps_per_epoch)

        dataset_iters, gvla_dataset_iter, min_loss = train(
            active_dataloaders, model_engine, epoch, scheduler, writer, dataset_iters, gvla_dataset_iter, args, step_choices, gvla_train_loader
        )

        if args.no_eval:
            save_checkpoint(model_engine, args, epoch, 'loss', f"{min_loss:.4f}", True)
        elif args.policy_config:
            success_rate = -1.
            if args.local_rank == 0:
                success_rate = eval_policy(args, epoch, policy_config, model=model, lang_encoder=lang_encoder)
                
            torch.distributed.barrier()
            # breakpoint()
            save_checkpoint(model_engine, args, epoch, 'success', f"{min_loss:.4f}_{success_rate:.4f}", True)
        else:
            if args.mask_validation:
                giou, ciou = validate_model_performance(val_loader, model_engine, epoch, writer, args)
                is_best = giou > best_giou
                best_giou = max(giou, best_giou)
                best_ciou = ciou if is_best else best_ciou
                if args.local_rank == 0:  # Log the progress
                    print(f"Epoch: {epoch}, giou: {giou}, ciou: {ciou}, best_giou: {best_giou}, best_ciou: {best_ciou}")
                save_checkpoint(model_engine, args, epoch, 'giou-ciou', f"{giou:.4f}-{ciou:.4f}", is_best)
            else:
                cur_val_loss = validate_model_performance(val_loader, model_engine, epoch, writer, args)
                is_best = cur_val_loss < best_val_loss
                best_val_loss = min(cur_val_loss, best_val_loss)
                if args.local_rank == 0:  # Log the progress
                    print(f"Epoch: {epoch}, Current Validation Loss: {cur_val_loss:.4f}, Best Validation Loss: {best_val_loss:}")
                save_checkpoint(model_engine, args, epoch, 'loss', f"{cur_val_loss:.4f}", is_best)


def save_checkpoint(model_engine, args, epoch, metric_name, metric_value, is_best):
    """ Saves the model checkpoint. """
    # If the checkpoint is the best, save it in ckpt_model_best, else in ckpt_model_last_epoch
    if not is_best:
        return
    save_dir_name = "ckpt_model_best" if is_best else "ckpt_model_last_epoch"
    save_dir = os.path.join(args.log_dir, save_dir_name)
    # Ensure the directory exists
    if args.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_filename = f"epoch_{epoch}_val_{metric_name}_{metric_value}.pth"
        torch.save({"epoch": epoch, f"val_{metric_name}": metric_value}, os.path.join(save_dir, ckpt_filename))
    torch.distributed.barrier()
    model_engine.save_checkpoint(save_dir)


def train(active_datasets, model, epoch, scheduler, writer, dataset_iters, gvla_dataset_iter, args, step_choices, gvla_train_loader):
    """Main training loop."""

    def get_next_input(iterator, data_loader):
        """Retrieve next input from the iterator, or reinitialize if necessary."""
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iterator = iter(data_loader)
            return next(new_iterator), new_iterator

    def log_progress():
        """Log training progress."""
        global_global_step = epoch * args.steps_per_epoch + global_step
        if global_step % args.print_freq == 0:
            if args.distributed:
                for tracker in trackers.values():
                    tracker.all_reduce()

            if args.local_rank == 0:
                progress.display(global_global_step + 1)
                for key, tracker in trackers.items():
                    writer.add_scalar(f"train/{key}", tracker.avg, global_global_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_global_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_global_step)

            for tracker in trackers.values():
                tracker.reset()

    batch_time = AverageMeter("Time", ":.4f")
    data_time = AverageMeter("Data", ":.4f")
    trackers = {"loss": AverageMeter("Loss", ":.4f"),
                "ce_loss": AverageMeter("CeLoss", ":.4f"),
                "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
                "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                "mask_loss": AverageMeter("MaskLoss", ":.4f"),
                "policy_loss": AverageMeter("PolicyLoss", ":.4f")}
    progress = ProgressMeter(args.steps_per_epoch, list(trackers.values()), prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()
    min_loss = 100
    for global_step in range(args.steps_per_epoch):
        for _ in range(args.grad_accumulation_steps):
            # Select data loader based on step choice
            if args.only_policy or args.use_gvla_data and _ % 2 == 1:
                data_batch, new_iter = get_next_input(gvla_dataset_iter, gvla_train_loader)
                gvla_dataset_iter = new_iter
            else:
                dataset_type, data_loader = active_datasets[step_choices[global_step]]
                data_batch, new_iter = get_next_input(dataset_iters[dataset_type], data_loader)
                dataset_iters[dataset_type] = new_iter

            data_time.update(time.time() - end)
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                if data_batch[key] is not None:
                    data_batch[key] = data_batch[key].bfloat16()
            # if "batch_meta" in data_batch:
            #     data_batch["batch_meta"] = dict_to_bfloat16(data_batch["batch_meta"])

            output_dict = model(only_policy=args.only_policy, **data_batch)

            # Update training metrics
            for key, tracker in trackers.items():
                if key in output_dict:
                    tracker.update(output_dict[key].item(), data_batch["global_enc_images"].size(0))
            min_loss = min(min_loss, output_dict["loss"].item())
            model.backward(output_dict["loss"])
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()
        log_progress()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return dataset_iters, gvla_dataset_iter, min_loss


def validate_model_performance(validation_loader, training_model, current_epoch, tensorboard_writer, args):
    if args.mask_validation:
        # For use with only segmentation/GCG type datasets
        trackers = {"intersection": AverageMeter("Intersec", ":.4f", Summary.SUM),
                    "union": AverageMeter("Union", ":.4f", Summary.SUM),
                    "gIoU": AverageMeter("gIoU", ":.4f", Summary.SUM)}

        training_model.eval()
        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                results = training_model(**data_batch)

            predictions = results["pred_masks"]
            gt_masks = results["gt_masks"][0].int()
            # Note: An error at this line may suggest that the dataset used for validation does not support
            # segmentation tasks. Ensure that the dataset is appropriate for segmentation analysis.
            predicted_masks = (predictions[0] > 0).int()
            assert len(predictions) == 1

            intersection, union, accuracy_iou = 0.0, 0.0, 0.0
            for target, prediction in zip(gt_masks, predicted_masks):
                intersect, union_, _ = intersectionAndUnionGPU(
                    prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                )
                intersection += intersect
                union += union_
                accuracy_iou += intersect / (union_ + 1e-5)
                # handles no-object targets
                accuracy_iou[union_ == 0] += 1.0

            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            accuracy_iou = accuracy_iou.cpu().numpy() / gt_masks.shape[0]
            trackers["intersection"].update(intersection)
            trackers["union"].update(union)
            trackers["gIoU"].update(accuracy_iou, n=gt_masks.shape[0])

        for meter in trackers.values():
            meter.all_reduce()

        iou_per_class = trackers["intersection"].sum / (trackers["union"].sum + 1e-10)
        class_iou = iou_per_class[1]
        global_iou = trackers["gIoU"].avg[1]

        if args.local_rank == 0:
            tensorboard_writer.add_scalar("val/giou", global_iou, current_epoch)
            tensorboard_writer.add_scalar("val/ciou", class_iou, current_epoch)
            print("giou: {:.4f}, ciou: {:.4f}".format(global_iou, class_iou))

        return global_iou, class_iou
    else:
        # Initializing performance trackers
        trackers = {"loss": AverageMeter("Loss", ":.4f"), "ce_loss": AverageMeter("CeLoss", ":.4f"),
                    "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
                    "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                    "mask_loss": AverageMeter("MaskLoss", ":.4f")}

        # Prepare model for validation phase
        # Hack to get the loss
        training_model.train()

        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                if data_batch[key] is not None:
                    data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                predictions = training_model(**data_batch)
            # Update performance metrics)
            for key, tracker in trackers.items():
                tracker.update(predictions[key].item(), data_batch["global_enc_images"].size(0))

        # Synchronize metrics across processes
        for tracker in trackers.values():
            tracker.all_reduce()
        # Calculate average validation loss
        avg_val_loss = trackers["ce_loss"].avg
        # Tensorboard logging for primary process
        if args.local_rank == 0:
            tensorboard_writer.add_scalar("val/loss", avg_val_loss, current_epoch)

        return avg_val_loss

def load_eval_env_infos(args, config):
    global eval_env_meta_list, eval_env_name_list, eval_env_horizon_list
    # extract the metadata across all datasets
    # eval_env_meta_list = []
    # eval_env_name_list = []
    # eval_env_horizon_list = []
    for dataset_cfg in config.train.data:
        dataset_path = os.path.expanduser(dataset_cfg["path"])
        ds_format = config.train.data_format
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

        # load basic metadata from training file
        print("\n============= Loaded Environment Metadata =============")
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path, ds_format=ds_format)
        # populate language instruction for env in env_meta
        env_meta["env_lang"] = dataset_cfg.get("lang", None)

        # update env meta if applicable
        deep_update(env_meta, dataset_cfg.get("env_meta_update_dict", {}))
        deep_update(env_meta, config.experiment.env_meta_update_dict)
        # env_meta_list.append(env_meta)
        
        if env_meta["env_name"] not in VAL_ENV_INFOS or dataset_cfg.get("do_eval", True) == False:
            continue
        eval_env_meta_list.append(env_meta)
        eval_env_name_list.append(env_meta["env_name"])
        horizon = dataset_cfg.get("horizon", config.experiment.rollout.horizon)
        eval_env_horizon_list.append(horizon)


def env_iterator(config):
    for (env_meta, env_name) in zip(eval_env_meta_list, eval_env_name_list):
        if env_name != "PnPCounterToCab":  # for one task
            continue
        def create_env_helper(env_i=0):
            env_kwargs = dict(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=True,
                seed=config.train.seed * 1000 + env_i,
            )
            env = EnvUtils.create_env_from_metadata(**env_kwargs)
            # handle environment wrappers
            env = EnvUtils.wrap_env_from_config(env, config=config)  # apply environment warpper, if applicable

            return env

        env = create_env_helper()
        # print(env)
        yield env


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
