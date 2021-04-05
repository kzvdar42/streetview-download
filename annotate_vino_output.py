import os
import re
import argparse
from collections import defaultdict

import cv2
import numpy as np
from tqdm.auto import tqdm
from pycocotools.coco import COCO

from utils.writer import get_coco_writer
from utils.image import pad_img, draw_text
from utils.get_image_info import get_image_size
from utils.path import get_subfolders_with_files, is_image


def get_crop(image, x, y, w,h):
    height, width, _ = image.shape

    ymin = np.clip(y-10, 0, height)
    ymax = np.clip(y+h+10, 0, height)
    xmin = np.clip(x-10, 0, width)
    xmax = np.clip(x+w+10, 0 , width)    
    crop = image[ymin:ymax,xmin:xmax]
    crop = cv2.resize(crop, (-1,-1), fx=5, fy=5)
    return crop


def display_crop(image, bbox, target_window_sizes, cat_id):
    image = image.copy()
    crop_size, image_size = target_window_sizes

    crop = get_crop(image, *bbox)
    crop = cv2.resize(crop, crop_size)
    (label_width, label_height), baseline = cv2.getTextSize(str(cat_id), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    crop = cv2.copyMakeBorder(crop, label_height + baseline, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(crop, str(cat_id), (0, label_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.imshow(f'crop', crop)

    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x+w, y+h), (22, 48, 163), 2)
    cv2.imshow(f'full_image', cv2.resize(image, image_size))


def write_results(images, annotations, save_path):
    writer = get_coco_writer()
    for i, anns in annotations.items():
        img_h, img_w, img_path = images[i]
        if len(anns):
            image_id, _ = writer.add_frame(img_h, img_w, filename=img_path)
            for (bbox, category_id) in anns.values():
                writer.add_annotation(image_id, bbox, category_id)
    writer.write_result(save_path, verbose=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to the output folder of `super_download_script`')
    parser.add_argument('--min_size', type=int, default=-1, help='min_size_to_classify')
    parser.add_argument('--auto', action='store_true', help='skip manual work')
    args = parser.parse_args()
    return args

def is_changed_folder(root_path:str, crop_name: str, annotation_num: int, class_num: int) -> bool:
    """Check if folder is changed for image."""
    image_name = f'{crop_name}_{annotation_num}.jpg'
    image_name_2 = f'{crop_name}_{annotation_num}..jpg'
    changed_dir_path = os.path.join(root_path, f'misclass_{class_num}')
    return (os.path.isfile(os.path.join(changed_dir_path, image_name)) or 
            os.path.isfile(os.path.join(changed_dir_path, image_name_2)))


def in_which_folder(root_path:str, crop_name: str, annotation_num: int) -> int:
    """Return class number based on detection location"""
    image_name = f'{crop_name}_{annotation_num}.jpg'
    image_name_2 = f'{crop_name}_{annotation_num}..jpg'
    for cat_id, folder_name in zip([0, 0, 1, 1], ['classified_0', 'misclass_1', 'classified_1', 'misclass_0']):
        infer_dir_path = os.path.join(root_path, folder_name)
        if (os.path.isfile(os.path.join(infer_dir_path, image_name)) or 
                os.path.isfile(os.path.join(infer_dir_path, image_name_2))):
            return cat_id
    return -1


if __name__ == "__main__":

    args = get_args()
    print(args)
    
    target_window_sizes = [
        ((64, 64), (640, 360)), 
        ((128, 128), (1280, 720)), 
        ((256, 256), (1920, 1080)),
    ]
    target_size_marker = 2

    coco = COCO(os.path.join(args.data_path, 'hor_crops_annotations.json'))
    hor_crops_path = os.path.join(args.data_path, 'hor_crops')
    hor_crops_infer_path = os.path.join(args.data_path, 'hor_crops_infer')
    save_path = os.path.join(args.data_path, 'human_checked_hor_crop_annotations.json')
    anns = coco.anns
    imgs = coco.imgs

    keys = np.array(list(imgs.keys()))
    image_num = 0
    prev_image_num = -1
    annotation_num = 0
    
    images, annotations = dict(), defaultdict(dict)
    pbar = tqdm(total=len(keys))
    while image_num < len(keys):
        pbar.update(image_num - pbar.n)
        # Load image and it's annotations
        if image_num != prev_image_num:
            img_info = imgs.get(keys[image_num])
            crop_name = os.path.splitext(os.path.split(img_info['file_name'])[1])[0]
            ann_ids = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            # Skip if no annotations
            if len(anns) == 0:
                prev_image_num = image_num
                image_num += 1
                continue
            annotation_num = 0
            # Load image, if none, skip
            image = cv2.imread(img_info['file_name'])
            if image is None:
                print('Image not found at: ', img_info['file_name'])
                prev_image_num = image_num
                image_num += 1
                continue
            # Get image id
            images[image_num] = (*image.shape[:2], img_info['file_name'])
        
        # If no more object in frame, go to next frame
        if annotation_num >= len(anns):
            prev_image_num = image_num
            image_num += 1
            continue
        # If annotation_num < 0, go to previous image
        elif annotation_num < 0:
            prev_image_num = image_num
            image_num -= 1
            continue
        
        # Get bbox and annotation
        bbox = np.array(anns[annotation_num]['bbox'], dtype=int)
        # Check if already classified
        if annotations[image_num].get(annotation_num):
            cat_id = annotations[image_num][annotation_num][1]
        else:
            # Check if already mapped as different class
            cat_id = in_which_folder(hor_crops_infer_path, crop_name, annotation_num)
            # Don't store if deleted
            if not cat_id == -1:
                annotations[image_num][annotation_num] = (bbox, cat_id)

        # Skip displaying if in auto mode
        if args.auto:
            prev_image_num = image_num
            annotation_num += 1
            continue

        # Display bbox
        display_crop(image, bbox, target_window_sizes[target_size_marker], cat_id)
        
        isBadKey = True
        while isBadKey:
            key = cv2.waitKey()
            isBadKey = False
            print(key)
            # Increase window size
            if key == ord('+'):
                target_size_marker = min(target_size_marker+1, 2)
            # Decrease window size
            elif key == ord('-'):
                target_size_marker = max(target_size_marker-1, 0)
            # Go to previous object
            elif key == ord(','):
                annotation_num -= 1
            # Go to next object
            elif key == ord('.'):
                annotation_num += 1
            # TP
            elif key == ord('j'):
                annotations[image_num][annotation_num] = (bbox, 1)
                annotation_num += 1
            # FP
            elif key == ord('o'):
                annotations[image_num][annotation_num] = (bbox, 0)
                annotation_num += 1
            # Save result
            elif key == 13:
                write_results(images, annotations, save_path)
            # Exit on q
            elif key == ord('q'):
                break
            else:
                isBadKey = True
        
        prev_image_num = image_num
        # Exit on q
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
    write_results(images, annotations, save_path)
