import h5py
from pycocotools import mask
import json
import numpy as np
from eval.utils import coco_encode_rle, mask_to_rle_numpy
import torch
from PIL import Image
import os
from tqdm import tqdm
from tools.prompt_templates import GVLA_PROMPT_TEMPLATE


# annos = json.load(open('/mnt/petrelfs/huanghaifeng/share_hw/groundingLMM/data/GranDf/annotations/train/GranDf_HA_GCG_train.json', 'r'))
# img_dir = "/mnt/petrelfs/huanghaifeng/share_hw/groundingLMM/data/GranDf/GranDf_HA_images/train"

# for anno in annos:
#     height = anno['height']
#     width = anno['width']
#     print(height, width)
#     print(anno['caption'])
#     img = np.array(Image.open(os.path.join(img_dir, anno['file_name'])))
#     for word, grounding in anno['groundings'].items():
#         print(word, grounding['token_positives'])
#         binary_mask = np.zeros((height, width), dtype=np.uint8)
#         for rle in grounding["rle_masks"]:
#             m = mask.decode(rle).astype(np.uint8)
#             binary_mask += m.squeeze()
#         new_img = img.copy()
#         new_img[binary_mask == 1] //= 2
#         new_img[binary_mask == 1, 0] += 127
#         save_img = Image.fromarray(new_img)
#         save_img.save('tmp.jpg')
#         breakpoint()

# ori = np.random.rand(10, 10)
# ori_mask = (ori > 0.5).astype(np.uint8)
# uncompressed_mask_rles = [mask_to_rle_numpy(ori_mask)]
# rle_masks = [coco_encode_rle(m) for m in uncompressed_mask_rles]
# breakpoint()
train_annos = []
cam_names = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
generated_data_dir = 'data/robocasa_datasets/v0.1/generated_1013'
img_dir = os.path.join(generated_data_dir, "images")
if not os.path.exists(img_dir):
    os.mkdir(img_dir)
# img_dir = '/mnt/petrelfs/huanghaifeng/share_hw/groundingLMM/data/GranDf/robocasa_images'
train_annos = []
val_annos = []

for hdf5_file_name in tqdm(os.listdir(generated_data_dir)):
    if not hdf5_file_name.endswith(".hdf5"):
        continue
    for split in ["train", "val"]:
        data_path = os.path.join(generated_data_dir, hdf5_file_name)
        dataset_name = os.path.basename(hdf5_file_name).split('.hdf5')[0]
        f = h5py.File(data_path, 'r')
        demos = list(f['data'].keys())
        inds = np.argsort([int(elem[5:]) for elem in demos])
        inds = inds[:-len(inds)//10] if split == "train" else inds[-len(inds)//10:]
        demos = [demos[i] for i in inds]
        for demo_id in tqdm(demos):
            ep_data_grp = f[f"data/{demo_id}"]
            obs_data_grp = ep_data_grp["obs"]
            ep_meta = ep_data_grp.attrs["ep_meta"]
            ep_meta = json.loads(ep_meta)
            lang = ep_meta["lang"]
            target_obj_phrase = "the " + ep_meta["target_obj_phrase"]
            target_place_phrase = ep_meta["target_place_phrase"]
            if target_place_phrase is not None:
                target_place_phrase = "the " + target_place_phrase
            prompt = GVLA_PROMPT_TEMPLATE.format(lang)
            caption = f"The target object is {target_obj_phrase}."
            if target_place_phrase is not None:
                caption += f" The target placement area is {target_place_phrase}."
            else:
                caption += f" No target placement area."
            for i, cam_name in enumerate(cam_names):
                image_key = f"{cam_name}_image"
                mask_key = f"{cam_name}_mask"
                image_id = f"{dataset_name}_{demo_id}_{image_key}"
                file_name = f"{image_id}.jpg"
                save_path = os.path.join(img_dir, file_name)
                cur_image, cur_mask = obs_data_grp[image_key][0], obs_data_grp[mask_key][0]
                tmp_image = Image.fromarray(cur_image)
                tmp_image.save(save_path)
                groundings = {}
                
                for target_phrase, mask_idx in [(target_obj_phrase, 1), (target_place_phrase, 2)]:
                    if target_phrase is None:
                        continue
                    tmp_mask = (cur_mask == mask_idx).astype(np.uint8)
                    uncompressed_mask_rles = [mask_to_rle_numpy(tmp_mask)]
                    rle_masks = [coco_encode_rle(m) for m in uncompressed_mask_rles]
                    st_idx = caption.find(target_phrase)
                    ed_idx = st_idx + len(target_phrase)
                    groundings[target_phrase] = {
                        "token_positives": [st_idx, ed_idx],
                        "rle_masks": rle_masks
                    }
                
                tmp_anno = {
                    "file_name": file_name,
                    "height": 256,
                    "width": 256,
                    "image_id": image_id,
                    "caption": caption,
                    "prompt": prompt,
                    "groundings": groundings
                }
                if split == "train":
                    train_annos.append(tmp_anno)
                else:
                    val_annos.append(tmp_anno)

out_dir = "data/GranDf/annotations/train"
with open(os.path.join(out_dir, 'robocasa_GCG_train.json'), 'w') as f:
    json.dump(train_annos, f, indent=4)

with open(os.path.join(out_dir, 'robocasa_GCG_val.json'), 'w') as f:
    json.dump(val_annos, f, indent=4)


# train_annos = []
# val_annos = []

# data = torch.load('/mnt/petrelfs/huanghaifeng/share_hw/groundingLMM/gpt/data/demo_gentex_im128_randcams_cp_addobj_use_actions.pt', map_location='cpu')

# prompt_template = "Identify the target object in the instruction: {}."
# answer_template = "The target object is {}."

# for demo_id, demo_data in tqdm(data.items()):
#     images = demo_data['images']
#     masks = demo_data['masks']
#     prompt = prompt_template.format(demo_data['lang'])
#     target_phrase = "the " + demo_data['target_phrase']
#     answer = answer_template.format(target_phrase)
#     for i, cam_name in [(0, 'left'), (1, 'right'), (2, 'hand')]:
#         file_name = f"{demo_id}_{cam_name}.jpg"
#         save_path = os.path.join(img_dir, file_name)
#         tmp_image, tmp_mask = images[i], masks[i]
#         tmp_image = Image.fromarray(tmp_image)
#         tmp_image.save(save_path)
#         tmp_mask = (tmp_mask[:, :, 0] > 0).astype(np.uint8)
#         uncompressed_mask_rles = [mask_to_rle_numpy(tmp_mask)]
#         rle_masks = [coco_encode_rle(m) for m in uncompressed_mask_rles]
#         st_idx = answer.find(target_phrase)
#         ed_idx = st_idx + len(target_phrase)
#         tmp_anno = {
#             "file_name": file_name,
#             "height": 256,
#             "width": 256,
#             "image_id": f"{demo_id}_{cam_name}",
#             "caption": answer,
#             "prompt": prompt,
#             "groundings": {
#                 target_phrase: {
#                     "token_positives": [st_idx, ed_idx],
#                     "rle_masks": rle_masks
#                 }
#             }
#         }
#         if int(demo_id.split('_')[-1]) < 2000:
#             train_annos.append(tmp_anno)
#         else:
#             val_annos.append(tmp_anno)
    
# out_dir = "/mnt/petrelfs/huanghaifeng/share_hw/groundingLMM/data/GranDf/annotations/train"
# with open(os.path.join(out_dir, 'robocasa_GCG_train.json'), 'w') as f:
#     json.dump(train_annos, f, indent=4)

# with open(os.path.join(out_dir, 'robocasa_GCG_val.json'), 'w') as f:
#     json.dump(val_annos, f, indent=4)
