import argparse
import math
import random
import os
import shutil

import cv2
import numpy as np
from tqdm.auto import tqdm
from pycocotools.coco import COCO

from utils.path import get_subfolders_with_files, is_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Input path')
    parser.add_argument('output_path', type=str, help='Output folder path')
    parser.add_argument('--min_crop_size', type=int, default=15, help='Minimum size of crop to save [Default 15]')
    parser.add_argument('--debug', action='store_true', help='Debug mode [Default False]')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    img_num = 0
    os.makedirs(args.output_path, exist_ok=True)
    # Process annotations for each coco file in input path.
    coco_paths = get_subfolders_with_files(args.input_path, lambda s: s.endswith('.json'), yield_by_one=True)
    for coco_path in coco_paths:
        print('coco', coco_path)
        # If coco in annotations folder, data path is in the upper folder,
        # otherwise in the same folder
        if os.path.split(coco_path)[0].endswith('annotations'):
            data_path = os.path.normpath(coco_path).rsplit(os.path.sep, 2)[0]
        else:
            data_path = os.path.split(coco_path)[0]
        # Get relative paths to folders
        rel_data_path = os.path.relpath(data_path, start=args.input_path)
        rel_coco_path = os.path.relpath(coco_path, start=args.input_path)
        
        coco = COCO(coco_path)
        for i, (img_id, img_info) in tqdm(enumerate(coco.imgs.items()), total=len(coco.imgs), desc=rel_coco_path):
            # Get annotations and load image
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
            im_anns = coco.loadAnns(ann_ids)
            im_path = os.path.join(data_path, img_info['file_name'])
            image = cv2.imread(im_path)
            assert image is not None, f"image at {im_path} is None!"


            for obj in im_anns:
                bbox = np.array(obj['bbox'], dtype=np.int32)
                assert len(bbox) == 4, "BBOX should have 4 elements!"

                # Get bbox with random padding
                b = np.array([
                    max(bbox[0] - bbox[2] * np.random.uniform(0.05, 0.25), 0),
                    max(bbox[1] - bbox[3] * np.random.uniform(0.02, 0.20), 0),
                    min(bbox[0] + bbox[2] * np.random.uniform(1.05, 1.25), image.shape[1]),
                    min(bbox[1] + bbox[3] * np.random.uniform(1.02, 1.20), image.shape[0]),
                ], dtype=np.int32)
                # If side is less than min_crop_size, skip.
                if b[3] - b[1] < args.min_crop_size or b[2] - b[0] < args.min_crop_size:
                    continue
                
                # Crop image
                cropped_image = image[b[1]:b[3], b[0]:b[2]].copy()
                # Get actual bbox for cropped image
                bbox = [
                    [bbox[0], bbox[1]],
                    [bbox[0] + bbox[2], bbox[1]],
                    [bbox[0], bbox[1] + bbox[3]],
                    [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                ]
                bbox = (np.array(bbox, dtype=np.float32) - b[:2]) / cropped_image.shape[:2][::-1]
                
                # Save image
                img_num += 1
                im_out_name = f"{bbox[0][0]:.6}_{bbox[0][1]:.6}_{bbox[1][0]:.6}_{bbox[1][1]:.6}_{bbox[2][0]:.6}_{bbox[2][1]:.6}_{bbox[3][0]:.6}_{bbox[3][1]:.6}_{img_num}.png"
                im_out_path = os.path.join(args.output_path, im_out_name)
                cv2.imwrite(im_out_path, cropped_image)

                # Show image for debugging
                if args.debug:
                    y_s, x_s = cropped_image.shape[:2]
                    cv2.rectangle(
                        cropped_image,
                        (int(bbox[0][0] * x_s), int(bbox[0][1] * y_s)),
                        (int(bbox[3][0] * x_s), int(bbox[3][1] * y_s)),
                        (0,0,255), 2
                    )
                    cv2.imshow('cropped', cropped_image)
                    cv2.waitKey(0)