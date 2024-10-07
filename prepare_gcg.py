import h5py
from pycocotools import mask
import json
import numpy as np
from eval.utils import coco_encode_rle, mask_to_rle_numpy
import torch
from PIL import Image
import os
from tqdm import tqdm


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

img_dir = '/mnt/petrelfs/huanghaifeng/share_hw/groundingLMM/data/GranDf/robocasa_images'
train_annos = []
val_annos = []

data = torch.load('/mnt/petrelfs/huanghaifeng/share_hw/groundingLMM/gpt/data/demo_gentex_im128_randcams_cp_addobj_use_actions.pt', map_location='cpu')

prompt_template = "Identify the target object in the instruction: {}."
answer_template = "The target object is {}."

for demo_id, demo_data in tqdm(data.items()):
    images = demo_data['images']
    masks = demo_data['masks']
    prompt = prompt_template.format(demo_data['lang'])
    target_phrase = "the " + demo_data['target_phrase']
    answer = answer_template.format(target_phrase)
    for i, cam_name in [(0, 'left'), (1, 'right'), (2, 'hand')]:
        file_name = f"{demo_id}_{cam_name}.jpg"
        save_path = os.path.join(img_dir, file_name)
        tmp_image, tmp_mask = images[i], masks[i]
        tmp_image = Image.fromarray(tmp_image)
        tmp_image.save(save_path)
        tmp_mask = (tmp_mask[:, :, 0] > 0).astype(np.uint8)
        uncompressed_mask_rles = [mask_to_rle_numpy(tmp_mask)]
        rle_masks = [coco_encode_rle(m) for m in uncompressed_mask_rles]
        st_idx = answer.find(target_phrase)
        ed_idx = st_idx + len(target_phrase)
        tmp_anno = {
            "file_name": file_name,
            "height": 512,
            "width": 512,
            "image_id": f"{demo_id}_{cam_name}",
            "caption": answer,
            "prompt": prompt,
            "groundings": {
                target_phrase: {
                    "token_positives": [st_idx, ed_idx],
                    "rle_masks": rle_masks
                }
            }
        }
        if int(demo_id.split('_')[-1]) < 2000:
            train_annos.append(tmp_anno)
        else:
            val_annos.append(tmp_anno)
    
out_dir = "/mnt/petrelfs/huanghaifeng/share_hw/groundingLMM/data/GranDf/annotations/train"
with open(os.path.join(out_dir, 'robocasa_GCG_train.json'), 'w') as f:
    json.dump(train_annos, f, indent=4)

with open(os.path.join(out_dir, 'robocasa_GCG_val.json'), 'w') as f:
    json.dump(val_annos, f, indent=4)
