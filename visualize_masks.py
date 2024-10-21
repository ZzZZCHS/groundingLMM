import json
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import random

annos = json.load(open('data/GranDf/annotations/train/robocasa_GCG_val.json', 'r'))
image_ids = [x['image_id'] for x in annos]
exp_name = 'test_1020'
mask_dir = f'output/outputs/{exp_name}/ckpt_model_best/mask_results'
# ori_mask_dir = "output/outputs/test_origin_full/mask_results"
image_dir = 'data/robocasa_datasets/v0.1/generated_1013/images'
save_dir = f'output/outputs/{exp_name}/ckpt_model_best/concat_masks'
os.makedirs(save_dir, exist_ok=True)

image_ids = random.sample(image_ids, 100)

for image_id in tqdm(image_ids):
    gt_mask = Image.open(os.path.join(mask_dir, f"{image_id}_gt.jpg"))
    pred_mask = Image.open(os.path.join(mask_dir, f"{image_id}_pred.jpg"))
    # ori_pred_mask = Image.open(os.path.join(ori_mask_dir, f"{image_id}_pred.jpg"))
    ori_img = Image.open(os.path.join(image_dir, f"{image_id}.jpg"))
    
    gt_mask = np.array(gt_mask)
    pred_mask = np.array(pred_mask)
    # ori_pred_mask = np.array(ori_pred_mask)
    ori_img = np.array(ori_img)
    
    gt_mask = np.repeat(np.expand_dims(gt_mask, axis=-1), 3, axis=-1)
    pred_mask = np.repeat(np.expand_dims(pred_mask, axis=-1), 3, axis=-1)
    # ori_pred_mask = np.repeat(np.expand_dims(ori_pred_mask, axis=-1), 3, axis=-1)
    
    gt_mask[:, :, 1:] = 0
    pred_mask[:, :, 1:] = 0
    # ori_pred_mask[:, :, 1:] = 0
    
    gt_img = ori_img // 3 + gt_mask // 3 * 2
    pred_img = ori_img // 3 + pred_mask // 3 * 2
    # ori_pred_img = ori_img // 3 + ori_pred_mask // 3 * 2
    
    save_img = np.concatenate([ori_img, gt_img, pred_img], axis=1)
    save_img = Image.fromarray(save_img)
    save_img.save(os.path.join(save_dir, f"{image_id}.jpg"))
