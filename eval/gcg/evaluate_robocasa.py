import os
import json
import argparse
from tqdm import tqdm
from pycocotools import mask as maskUtils
from PIL import Image

from eval.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--prediction_dir_path", required=True, help="The path where the inference results are stored.")
    parser.add_argument("--anno_path", required=True, help="path to the val annotation file")
    parser.add_argument("--mask_results_path", required=True, help="path to save the gt masks and pred masks")

    args = parser.parse_args()

    return args


def compute_miou(pred_masks, gt_masks):
    # Computing mIoU between predicted masks and ground truth masks
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

    # One-to-one pairing and mean IoU calculation
    paired_iou = []
    while iou_matrix.size > 0 and np.max(iou_matrix) > 0:
        max_iou_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
        paired_iou.append(iou_matrix[max_iou_idx])
        iou_matrix = np.delete(iou_matrix, max_iou_idx[0], axis=0)
        iou_matrix = np.delete(iou_matrix, max_iou_idx[1], axis=1)

    return np.mean(paired_iou) if paired_iou else 0.0


def evaluate_mask_miou(image_ids, pred_masks, gt_masks, save_mask_dir):
    mious = []
    for image_id in tqdm(image_ids):
        # Getting ground truth masks

        tmp_gt_masks = [maskUtils.decode(x) for x in gt_masks[image_id]]
        tmp_pred_masks = [maskUtils.decode(x) for x in pred_masks[image_id]]
        
        save_gt_mask = tmp_gt_masks[0] * 255
        save_pred_mask = tmp_pred_masks[0] * 255
        save_gt_mask = Image.fromarray(save_gt_mask)
        save_pred_mask = Image.fromarray(save_pred_mask)
        save_gt_mask.save(os.path.join(save_mask_dir, f"{image_id}_gt.jpg"))
        save_pred_mask.save(os.path.join(save_mask_dir, f"{image_id}_pred.jpg"))

        # Compute and save the mIoU for the current image
        mious.append(compute_miou(tmp_pred_masks, tmp_gt_masks))

    # Report mean IoU across all images
    mean_miou = np.mean(mious) if mious else 0.0  # If list is empty, return 0.0

    print(f"Mean IoU (mIoU) across all images: {mean_miou:.3f}")



def main():
    args = parse_args()

    # Get the image names of the split
    annos = json.load(open(args.anno_path, 'r'))
    all_image_ids = [x['image_id'] for x in annos]
    gt_masks = {x['image_id']: list(x['groundings'].values())[0]['rle_masks'] for x in annos}
    
    pred_masks = {}
    for image_id in all_image_ids:
        pred_path = os.path.join(args.prediction_dir_path, f"{image_id}.json")
        tmp_pred = json.load(open(pred_path))
        pred_masks[image_id] = tmp_pred['pred_masks']
    
    save_mask_dir = args.mask_results_path
    os.makedirs(save_mask_dir, exist_ok=True)

    evaluate_mask_miou(all_image_ids, pred_masks, gt_masks, save_mask_dir)


if __name__ == "__main__":
    main()
