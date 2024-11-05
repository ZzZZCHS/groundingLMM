import re
import cv2
import json
import bleach
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, CLIPImageProcessor
import os
import sys
import glob
import imageio
from collections import OrderedDict
import time
import traceback
from copy import deepcopy

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

from dataset.gvla_datasets.grounded_vla_ds import GroundedVLADataset, GroundedVLAMetaDataset

import tools.file_utils as FileUtils
import tools.obs_utils as ObsUtils
import tools.env_utils as EnvUtils
import tools.train_utils as TrainUtils
import tools.log_utils as LogUtils
import tools.tensor_utils as TensorUtils
import tools.action_utils as AcUtils
import tools.torch_utils as TorchUtils
import tools.lang_utils as LangUtils
from tools.log_utils import PrintLogger, DataLogger, flush_warnings
from tools.script_utils import deep_update
from tools.train_utils import VAL_ENV_INFOS

import robomimic.utils.obs_utils as RobomimicObsUtils
RobomimicObsUtils.RESIZE_TO_128 = False

from PIL import Image

action_normalization_stats = None

def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Inference - GCG")

    parser.add_argument("--version", required=True, help="The model path in huggingface format.")
    parser.add_argument("--img_dir", required=False, default="./data/GranDf/GranDf_HA_images/val_test",
                        help="The directory containing images to run inference.")
    parser.add_argument("--exp_dir", required=True, help="The directory to store exp results.")
    parser.add_argument("--policy_config", default=None, type=str)
    parser.add_argument("--raw_data_dir", required=True, default=None, type=str)
    parser.add_argument("--use_gt_mask", action="store_true")
    parser.add_argument("--only_policy", default=False, type=bool)

    parser.add_argument("--precision", default='bf16', type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )
    
    parser.add_argument(
        "--generate_val_data_path",
        type=str,
        default=None,
        help="path to the generated new val data"
    )

    return parser.parse_args()


def inference(lang, image_np):
    prompt = f"Given a robot manipulation instruction: {lang}, identify the target object for manipulation and, if applicable, the target placement area."
    # Prompt model to return grounded conversations
    instruction = "Please respond with interleaved \
    segmentation masks for the corresponding parts of the answer."
    
    instructions = prompt + " " + instruction
    
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
    # image_np = cv2.imread(image_path)
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]
    image_clip = (clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).to(device))
    image_clip = image_clip.bfloat16()  # Precision is bf16 by default

    # Preprocess the image (Grounding image encoder)
    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = (
        grounding_image_ecoder_preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).to(device))
    image = image.bfloat16()  # Precision is bf16 by default

    # Prepare inputs for inference
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(device)
    bboxes = None  # No box/region is input in GCG task

    # Generate output
    output_ids, pred_masks, pred_embeddings = model.evaluate(
        global_enc_images=image_clip, 
        grounding_enc_images=image, 
        input_ids=input_ids, 
        resize_list=resize_list, 
        orig_sizes=original_size_list, 
        max_tokens_new=512, 
        bboxes=bboxes, 
        device=input_ids.device,
        return_hidden_embedding=True
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

    return cleaned_str, pred_masks, phrases, pred_embeddings


def custom_collate_fn(batch):
    image_id = [item[0] for item in batch]
    image_path = [item[1] for item in batch]
    prompt = [item[2] for item in batch]
    return image_id, image_path, prompt


def init_model(args):
    shape_meta = None
    if args.only_policy:
        tokenizer = None
    else:
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
    
    if policy_config is None:
        model = GLaMMForCausalLM.from_pretrained(args.version, low_cpu_mem_usage=True, **kwargs)
        # Update model config
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
        
        if args.only_policy:
            model = GLaMMWithPolicy.from_pretrained(
                args.version, 
                glamm_version=None,
                glamm_model_args=None,
                obs_key_shapes=shape_meta["all_shapes"],
                ac_dim=shape_meta["ac_dim"],
                policy_config=policy_config,
                low_cpu_mem_usage=False,
                _fast_init=False,
                only_policy=True
            )
        else:
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

    if args.only_policy:
        clip_image_processor = None
        transform = None
        model = model.bfloat16().to(device)
    else:
        if policy_config:
            glamm_model = model.glamm_model
        else:
            glamm_model = model
        
        # Initialize Global Image Encoder (CLIP)
        glamm_model.get_model().initialize_vision_modules(glamm_model.get_model().config)
        vision_tower = glamm_model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)

        # Transfer the model to GPU
        model = model.bfloat16().to(device)  # Replace with model = model.float().to(device) for 32 bit inference
        vision_tower = glamm_model.get_model().get_vision_tower()
        vision_tower.to(device=device)

        # Initialize Image Processor for GLobal Image Encoder (CLIP)
        clip_image_processor = CLIPImageProcessor.from_pretrained(glamm_model.config.vision_tower)
        transform = ResizeLongestSide(args.image_size)

    model.eval()  # Model should be in evaluation mode for inference
    
    return model, tokenizer, clip_image_processor, transform, shape_meta


def init_envs(args, config, shape_meta=None):
    log_dir = os.path.join(args.exp_dir, 'logs')
    video_dir = os.path.join(args.exp_dir, 'videos')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    
    if config.experiment.logging.terminal_output_to_txt:
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger
    
    ObsUtils.initialize_obs_utils_with_config(config)
    
    # extract the metadata across all datasets
    eval_env_meta_list = []
    eval_env_name_list = []
    eval_env_horizon_list = []
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

    # create environments
    def env_iterator():
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
        
    # load all dataset for action_normalization_stats
    # common_ds_args = {"precision": args.precision, "image_size": args.image_size}
    # gvla_datasets = []
    # for hdf5_path in glob.glob(os.path.join(args.raw_data_dir, "*.hdf5")):
    #     gvla_dataset = GroundedVLADataset(**common_ds_args, hdf5_path=hdf5_path, raw_data_dir=args.raw_data_dir, obs_keys=shape_meta["all_obs_keys"])
    #     gvla_datasets.append(gvla_dataset)
    # gvla_train_dataset = GroundedVLAMetaDataset(gvla_datasets)
    
    # action_normalization_stats = gvla_train_dataset.get_action_normalization_stats()
    action_normalization_stats = None
    
    return env_iterator(), eval_env_horizon_list, action_normalization_stats, log_dir, video_dir
    
    
def prepare_observation(ob, ep_lang_emb):
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
    if action_normalization_stats is not None:
        action_keys = model.policy_net.global_config.train.action_keys
        action_shapes = {k: action_normalization_stats[k]["offset"].shape[1:] for k in action_normalization_stats}
        ac_dict = AcUtils.vector_to_action_dict(ac, action_shapes=action_shapes, action_keys=action_keys)
        ac_dict = ObsUtils.unnormalize_dict(ac_dict, normalization_stats=action_normalization_stats)
        action_config = model.policy_net.global_config.train.action_config
        for key, value in ac_dict.items():
            this_format = action_config[key].get("format", None)
            if this_format == "rot_6d":
                rot_6d = torch.from_numpy(value).unsqueeze(0)
                conversion_format = action_config[key].get("convert_at_runtime", "rot_axis_angle")
                if conversion_format == "rot_axis_angle":
                    rot = TorchUtils.rot_6d_to_axis_angle(rot_6d=rot_6d).squeeze().numpy()
                elif conversion_format == "rot_euler":
                    rot = TorchUtils.rot_6d_to_euler_angles(rot_6d=rot_6d, convention="XYZ").squeeze().numpy()
                else:
                    raise ValueError
                ac_dict[key] = rot
        ac = AcUtils.action_dict_to_vector(ac_dict, action_keys=action_keys)
    return ac


def run_rollout(
        env, 
        horizon,
        initial_state=None,
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
        
        # if grounding_model is not None or args.use_gt_mask:
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
            
        ob_input = prepare_observation(ob_dict, ep_lang_emb)
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


if __name__ == "__main__":
    args = parse_args()
    if args.policy_config is not None:
        ext_cfg = json.load(open(args.policy_config, 'r'))
        policy_config = config_factory(ext_cfg["algo_name"])
        with policy_config.unlocked():
            policy_config.update(ext_cfg)
    else:
        policy_config = None
    
    # init_distributed_mode(args)
    device = "cuda:0"

    model, tokenizer, clip_image_processor, transform, shape_meta = init_model(args)
    model.policy_net.device = device
    
    lang_encoder = LangUtils.LangEncoder(
        device=device,
    )
    
    envs, horizon_list, action_normalization_stats, log_dir, video_dir = init_envs(args, policy_config, shape_meta)

    num_episodes = 50
    all_rollout_logs = OrderedDict()
    
    for env, horizon in zip(envs, horizon_list):
        env_name = env.name
        
        if video_dir is not None:
            video_path = os.path.join(video_dir, f"{env_name}.mp4")
            video_writer = imageio.get_writer(video_path, fps=20)
        
        print("rollout: env={}, horizon={}, num_episodes={}".format(
            env_name, horizon, num_episodes,
        ))
        
        rollout_logs = []
        num_success = 0
        for ep_i in LogUtils.custom_tqdm(range(num_episodes), total=num_episodes):
            initial_state = VAL_ENV_INFOS[env_name][ep_i]
            rollout_timestamp = time.time()
            try:
                rollout_info = run_rollout(
                    env=env,
                    horizon=horizon,
                    initial_state=initial_state,
                    video_writer=video_writer,
                    ep_i=ep_i,
                    args=args
                )
            except Exception as e:
                print(traceback.format_exc())
                print(env_name, "Rollout exception at episode number {}!".format(ep_i))
                # break
                continue
            
            rollout_info["time"] = time.time() - rollout_timestamp
            rollout_logs.append(rollout_info)
            num_success += rollout_info["Success_Rate"]
            
            print(f"{num_success} / {ep_i+1}")
            # print("Episode {}, horizon={}, num_success={}".format(ep_i + 1, horizon, num_success))
            # print(json.dumps(rollout_info, sort_keys=True, indent=4))
        
        if video_dir is not None:
            # close this env's video writer (next env has it's own)
            video_writer.close()
        
        # delete the environment after use
        del env
        
        # average metric across all episodes
        if len(rollout_logs) > 0:
            rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
            rollout_logs_mean = dict((k, np.mean(v)) for k, v in rollout_logs.items())
            rollout_logs_mean["Time_Episode"] = np.sum(rollout_logs["time"]) / 60. # total time taken for rollouts in minutes
            all_rollout_logs[env_name] = rollout_logs_mean
        else:
            all_rollout_logs[env_name] = {"Time_Episode": -1, "Return": -1, "Success_Rate": -1, "time": -1}
        
        break

    if log_dir:
        with open(os.path.join(log_dir, "all_rollout_logs.json"), 'w') as f:
            json.dump(all_rollout_logs, f, indent=4)
