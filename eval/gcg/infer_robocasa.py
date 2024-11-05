import re
import cv2
import json
import bleach
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, CLIPImageProcessor

from eval.utils import *
from eval.ddp import *
from model.GLaMM import GLaMMForCausalLM, GLaMMWithPolicy
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from tools.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from spacy_utils import extract_direct_object_phrases
from config import config_factory
from train import setup_tokenizer_and_special_tokens, initialize_model
import tools.file_utils as FileUtils
import tools.obs_utils as ObsUtils


def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Inference - GCG")

    parser.add_argument("--version", required=True, help="The model path in huggingface format.")
    parser.add_argument("--img_dir", required=False, default="./data/GranDf/GranDf_HA_images/val_test",
                        help="The directory containing images to run inference.")
    parser.add_argument("--output_dir", required=True, help="The directory to store the response in json format.")
    parser.add_argument("--anno_path", required=True, help="The val annotation file path")
    parser.add_argument("--policy_config", default=None, type=str)

    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def inference(instructions, image_path):
    # Filter out special chars
    instructions = bleach.clean(instructions)
    instructions = instructions.replace('&lt;', '<').replace('&gt;', '>')

    # Prepare prompt for model Inference
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
    prompt = begin_str + instructions
    if args.use_mm_start_end:
        replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    # Read and preprocess the image (Global image encoder - CLIP)
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]
    image_clip = (clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda())
    image_clip = image_clip.bfloat16()  # Precision is bf16 by default

    # Preprocess the image (Grounding image encoder)
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = (
        grounding_image_ecoder_preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda())
    image = image.bfloat16()  # Precision is bf16 by default

    # Prepare inputs for inference
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()
    bboxes = None  # No box/region is input in GCG task

    # Generate output
    output_ids, pred_masks = model.evaluate(
        global_enc_images=image_clip, 
        grounding_enc_images=image, 
        input_ids=input_ids, 
        resize_list=resize_list, 
        orig_sizes=original_size_list, 
        max_tokens_new=512, 
        bboxes=bboxes, 
        device=input_ids.device
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    # Post-processing
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r'<.*?>', '', text_output)

    pattern = re.compile(r'<p>(.*?)<\/p>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    # Remove the [SEG] token
    cleaned_str = cleaned_str.replace('[SEG]', '')

    # Strip unnecessary spaces
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    return cleaned_str, pred_masks, phrases


def custom_collate_fn(batch):
    image_id = [item[0] for item in batch]
    image_path = [item[1] for item in batch]
    prompt = [item[2] for item in batch]
    return image_id, image_path, prompt


if __name__ == "__main__":
    args = parse_args()
    if args.policy_config is not None:
        ext_cfg = json.load(open(args.policy_config, 'r'))
        policy_config = config_factory(ext_cfg["algo_name"])
        with policy_config.unlocked():
            policy_config.update(ext_cfg)
    else:
        policy_config = None
    
    init_distributed_mode(args)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.version, cache_dir=None,
                                              model_max_length=args.model_max_length, padding_side="right",
                                              use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    torch_dtype = torch.bfloat16  # By default, using bf16
    kwargs = {
        # "torch_dtype": torch_dtype,
        "seg_token_idx": seg_token_idx
    }
    
    # tokenizer = setup_tokenizer_and_special_tokens(args)
    
    # model = initialize_model
    
    if policy_config is None:
        model = GLaMMForCausalLM.from_pretrained(args.version, low_cpu_mem_usage=True, **kwargs)
        # Update model config
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        # breakpoint()
        ObsUtils.initialize_obs_utils_with_config(policy_config)
        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=policy_config.train.data[0]["path"],
            action_keys=policy_config.train.action_keys,
            all_obs_keys=policy_config.all_obs_keys,
            ds_format=policy_config.train.data_format,
            verbose=False
        )
        
        model = GLaMMWithPolicy.from_pretrained(
            args.version, 
            glamm_version=None,
            glamm_model_args=kwargs,
            obs_key_shapes=shape_meta["all_shapes"],
            ac_dim=shape_meta["ac_dim"],
            policy_config=policy_config,
            low_cpu_mem_usage=False,
            _fast_init=False
        )
        model.glamm_model.seg_token_idx = seg_token_idx
        model.glamm_model.config.eos_token_id = tokenizer.eos_token_id
        model.glamm_model.config.bos_token_id = tokenizer.bos_token_id
        model.glamm_model.config.pad_token_id = tokenizer.pad_token_id

    if policy_config:
        glamm_model = model.glamm_model
    else:
        glamm_model = model
    
    # Initialize Global Image Encoder (CLIP)
    glamm_model.get_model().initialize_vision_modules(glamm_model.get_model().config)
    vision_tower = glamm_model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    # Transfer the model to GPU
    model = model.bfloat16().cuda()  # Replace with model = model.float().cuda() for 32 bit inference
    vision_tower = glamm_model.get_model().get_vision_tower()
    vision_tower.to(device="cuda")

    # Initialize Image Processor for GLobal Image Encoder (CLIP)
    clip_image_processor = CLIPImageProcessor.from_pretrained(glamm_model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()  # Model should be in evaluation mode for inference

    # Prompt model to return grounded conversations
    instruction = "Please respond with interleaved \
    segmentation masks for the corresponding parts of the answer."

    # Create output directory if not exists already
    os.makedirs(args.output_dir, exist_ok=True)

    # Create DDP Dataset
    dataset = RobocasaGCGEvalDDP(args.img_dir, args.anno_path)
    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, num_workers=2,
                            sampler=distributed_sampler, collate_fn=custom_collate_fn)

    # Iterate over all the images, run inference and save results
    for (image_id, image_path, prompt) in tqdm(dataloader):
        image_id, image_path, prompt = image_id[0], image_path[0], prompt[0]

        output_path = f"{args.output_dir}/{image_id[:-4]}.json"
        tmp_instruction = prompt + ' ' + instruction
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # ori_instruction = prompt.split('instruction: ')[-1]
        # phrases = extract_direct_object_phrases(ori_instruction)
        # phrase = phrases[0] if len(phrases) > 0 else "the object"
        # tmp_instruction = f"Can you segment {phrase}?"
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        result_caption, pred_masks, phrases = inference(tmp_instruction, image_path)  # GLaMM Inference

        # Convert the predicted masks into RLE format
        pred_masks_tensor = pred_masks[0].cpu()
        binary_pred_masks = pred_masks_tensor > 0
        uncompressed_mask_rles = mask_to_rle_pytorch(binary_pred_masks)
        rle_masks = []
        for m in uncompressed_mask_rles:
            rle_masks.append(coco_encode_rle(m))

        # Create results dictionary
        result_dict = {
            "image_id": image_id[:-4],
            "caption": result_caption,
            "phrases": phrases,
            "pred_masks": rle_masks
        }

        # Save the inference results
        with open(output_path, 'w') as f:
            json.dump(result_dict, f)
