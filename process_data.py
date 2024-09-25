import h5py
import cv2
import os
import numpy as np
from PIL import Image
import json
from spacy_utils import extract_direct_object_phrases
import argparse
import sys
from transformers import AutoTokenizer, CLIPImageProcessor
from model.GLaMM import GLaMMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
import torch
import torch.nn.functional as F
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
import random
from tools.markdown_utils import colors
from tqdm import tqdm
import shutil
import time
from fastprogress.fastprogress import progress_bar


def parse_args(args):
    parser = argparse.ArgumentParser(description="GLaMM Model Demo")
    parser.add_argument("--version", default="/ailab/user/huanghaifeng/work/robocasa_exps/groundingLMM/GLaMM-FullScope")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--precision", default='bf16', type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])

    return parser.parse_args(args)


def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token
    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]

    return tokenizer


def initialize_model(args, tokenizer):
    """ Initialize the GLaMM model. """
    model_args = {k: getattr(args, k) for k in
                  ["seg_token_idx", "bbox_token_idx", "eop_token_idx", "bop_token_idx"]}

    model = GLaMMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args)
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def prepare_model_for_inference(model, args):
    # Initialize vision tower
    print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.vision_tower
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=args.local_rank)
    model = model.bfloat16().cuda()
    return model


def grounding_enc_processor(x: torch.Tensor) -> torch.Tensor:
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    x = (x - IMG_MEAN) / IMG_STD
    h, w = x.shape[-2:]
    x = F.pad(x, (0, IMG_SIZE - w, 0, IMG_SIZE - h))
    return x


def prepare_mask(image_np, pred_masks, text_output):
    save_img = None
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue
        pred_mask = pred_mask.detach().cpu().numpy()
        mask_list = [pred_mask[i] for i in range(pred_mask.shape[0])]
        if len(mask_list) > 0:
            save_img = image_np.copy()
            # colors_temp = colors
            seg_count = text_output.count("[SEG]")
            mask_list = mask_list[-seg_count:]
            for curr_mask in mask_list:
                # color = random.choice(colors_temp)
                color = [255, 0, 0]
                # if len(colors_temp) > 0:
                #     colors_temp.remove(color)
                # else:
                #     colors_temp = colors
                # color_history.append(color)
                curr_mask = curr_mask > 0
                save_img[curr_mask] = (image_np * 0.5 + curr_mask[:, :, None].astype(np.uint8) * np.array(color) * 0.5)[curr_mask]
                # breakpoint()
                # save_img[curr_mask, 0] = 255
                # breakpoint()
    seg_mask = np.zeros((curr_mask.shape[0], curr_mask.shape[1], 3), dtype=np.uint8)
    seg_mask[curr_mask] = [255, 255, 255]  # white for True values
    seg_mask[~curr_mask] = [0, 0, 0]  # black for False values
    # seg_mask = Image.fromarray(seg_mask)
    # mask_path = input_image.replace('image', 'mask')
    # seg_mask.save(mask_path)

    return save_img, seg_mask


def inference(input_str, input_image, draw_image=None):

    # print("input_str: ", input_str, "input_image: ", image_np)
    
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    conv_history = {'user': [], 'model': []}
    conv_history["user"].append(input_str)

    input_str = input_str.replace('&lt;', '<').replace('&gt;', '>')
    prompt = input_str
    prompt = f"The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture." + "\n" + prompt
    if args.use_mm_start_end:
        replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    image_np = input_image
    orig_h, orig_w = image_np.shape[:2]
    original_size_list = [image_np.shape[:2]]

    # Prepare input for Global Image Encoder
    global_enc_image = global_enc_processor.preprocess(
        image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda()
    global_enc_image = global_enc_image.bfloat16()

    # Prepare input for Grounding Image Encoder
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    grounding_enc_image = (grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).
                                                   contiguous()).unsqueeze(0).cuda())
    grounding_enc_image = grounding_enc_image.bfloat16()

    # Prepare input for Region Image Encoder
    post_h, post_w = global_enc_image.shape[1:3]
    bboxes = None

    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    # Pass prepared inputs to model
    output_ids, pred_masks = model.evaluate(
        global_enc_image, grounding_enc_image, input_ids, resize_list, original_size_list, max_tokens_new=512,
        bboxes=bboxes)
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]
    # print("text_output: ", text_output)

    save_img = None
    if draw_image is None:
        draw_image = image_np
        # color_history = []
    if "[SEG]" in text_output:
        save_img, seg_mask = prepare_mask(draw_image, pred_masks, text_output)

    return save_img


if __name__ == "__main__":
    data_root = '/ailab/user/huanghaifeng/work/robocasa_exps/robocasa/datasets'
    ori_data_dir = os.path.join(data_root, 'v0.1/single_stage')

    obs_keys = ["robot0_agentview_left_image", "robot0_agentview_right_image", "robot0_eye_in_hand_image"]

    prompt_template = "Can you segment {}?"

    # src_name = 'demo_gentex_im128_randcams.hdf5'
    # tgt_name = 'demo_gentex_im128_randcams_addmask.hdf5'
    src_name = 'demo_addobj_gentex_im128_randcams.hdf5'
    tgt_name = 'demo_addobj_gentex_im128_randcams_addmask_new.hdf5'

    args = parse_args(sys.argv[1:])
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    model = prepare_model_for_inference(model, args)
    global_enc_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()

    if not os.path.exists(ori_data_dir):
        raise FileNotFoundError

    for root, dirs, files in tqdm(os.walk(ori_data_dir)):
        if '/mg/' not in root or 'PnPCounterToCab' not in root:
            continue
        for file_name in files:
            if file_name != src_name:
                continue
            ori_file_path = os.path.join(root, src_name)
            tgt_file_path = os.path.join(root, tgt_name)
            print(ori_file_path)
            # if os.path.exists(tgt_file_path):
            #     continue
            if not os.path.exists(tgt_file_path):
                shutil.copy(ori_file_path, tgt_file_path)
            # print(tgt_file_path)
            f = h5py.File(tgt_file_path, 'a')
            # f = h5py.File(ori_file_path, 'a')
            data = f['data']
            for demo_id in progress_bar(data.keys()):
                tmp_demo = data[demo_id]
                tmp_obs = tmp_demo['obs']
                ep_meta = tmp_demo.attrs.get('ep_meta')
                ep_meta = json.loads(ep_meta)
                lang = ep_meta.get('lang')
                noun_phrases = extract_direct_object_phrases(lang)
                
                for k in tmp_obs.keys():
                    if k not in obs_keys:
                        continue
                    if f"masked_{k}" in tmp_obs.keys():
                        continue
                    all_masked_imgs = []
                    # for img_idx in range(tmp_obs[k][()].shape[0]):
                    for img_idx in range(1):
                        tmp_img = tmp_obs[k][()][img_idx]
                        # breakpoint()
                        # print(tmp_img.shape)
                        # im = Image.fromarray(tmp_img)
                        # im.save(f"{k}.jpg")
                        # st_time = time.time()
                        tmp_masked_img = None
                        for i, tmp_phrase in enumerate(noun_phrases):
                            tmp_prompt = prompt_template.format(tmp_phrase)
                            # print(tmp_prompt)
                            tmp_masked_img = inference(tmp_prompt, tmp_img, tmp_masked_img)
                            # im = Image.fromarray(tmp_masked_img)
                            # im.save(f"masked_img{i}.jpg")
                        if tmp_masked_img is None:
                            tmp_masked_img = tmp_img.copy()
                        tmp_masked_img = np.expand_dims(tmp_masked_img, axis=0)
                        all_masked_imgs.append(tmp_masked_img)
                        # ed_time = time.time()
                        # print("time:", ed_time - st_time)
                    # breakpoint()
                    all_masked_imgs = np.concatenate(all_masked_imgs, axis=0)
                    dset = tmp_obs.create_dataset(f'masked_{k}', data=all_masked_imgs)
    