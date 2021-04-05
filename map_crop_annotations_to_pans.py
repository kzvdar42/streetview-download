import os
import re
import argparse
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO

from horizontal_crop_pans import get_thetas
from core.mapper import get_mapping
from utils.writer import get_coco_writer
from utils.image import pad_img, draw_text
from utils.get_image_info import get_image_size
from utils.path import get_subfolders_with_files, is_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to the output folder of `super_download_script`')
    parser.add_argument('--phi', type=float, default=15, help='Phi angle (pitch) in range [-90, 90) degrees [default: 15]')
    parser.add_argument('--res_x', type=int, default=1920, help='Resolution of the output image width [default: 256]')
    parser.add_argument('--res_y', type=int, default=1080, help='Resolution of the output image height [default: 256]')
    parser.add_argument('--fov', type=float, default=60.0, help='Field of View for image height in range [0, 180] degrees [default: 60.0]')
    parser.add_argument('--n_cuts_per_image', type=int, default=5, help='Number of cuts [default: 5]')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()
    return args


def _map_crop_annot_to_pan(map_y, map_x, bboxes):
    new_bboxes = []
    for bbox in bboxes:
        # convert to xyxy
        b = np.array([
            bbox[0], bbox[1],
            bbox[0] + bbox[2],
            bbox[1] + bbox[3],
        ])
        # Map to panorama coordinates
        for i in range(2):
            b[i*2], b[i*2 + 1] = map_x[b[i*2+1], b[i*2]], map_y[b[i*2+1], b[i*2]]
        # Convert back
        b[2] -= b[0]
        b[3] -= b[1]
        # Save
        new_bboxes.append(b)
    return new_bboxes


def map_crop_annot_to_pan(pano_height, pano_width, bboxes, theta, phi, res_y, res_x, fov):
    map_y, map_x = get_mapping(
        pano_height, pano_width, theta=theta, phi=phi,
        res_y=res_y, res_x=res_x, fov=fov
    )
    return _map_crop_annot_to_pan(map_y, map_x, bboxes)

def pan_name_from_crop_name(crop_name):
    pan_name = '_'.join(crop_name.split('_')[:-1])
    pan_name = re.sub('-?\d\d.\d+_\-?\d\d.\d+_', '', pan_name)
    return pan_name


def get_pannames2paths(path):
    pan_name2path = dict()
    for pan_path in get_subfolders_with_files(path, is_image, yield_by_one=True):
        pan_name = os.path.splitext(os.path.split(pan_path)[1])[0]
        pan_name = re.sub('-?\d\d.\d+_\-?\d\d.\d+_', '', pan_name)
        pan_name2path[pan_name] = pan_path
    return pan_name2path


def get_annots_from_folders(paths):
    pass

def get_annots_from_coco(paths):
    pass


if __name__ == "__main__":

    args = get_args()
    thetas = get_thetas(fov=args.fov, n_cuts=args.n_cuts_per_image)
    print(args)

    classes = ['non-sign', 'sign']

    coco = COCO(os.path.join(args.data_path, 'hor_crops_annotations.json'))
    file_name2imgid = dict()
    for img_obj in coco.imgs.values():
        img_name = os.path.split(img_obj['file_name'])[1]
        file_name2imgid[img_name] = img_obj['id']

    # Check if path exists
    crops_path = os.path.join(args.data_path, 'hor_crops_human')
    if not os.path.isdir(crops_path):
        print('Firstly, create `hor_crops_human` folder with hand classication of crops')
        exit(0)
    
    # Get image paths
    class_0_crop_paths = (
        list(get_subfolders_with_files(os.path.join(crops_path, 'classified_0'), is_image, yield_by_one=True)) +
        list(get_subfolders_with_files(os.path.join(crops_path, 'misclass_1'), is_image, yield_by_one=True))
    )

    class_1_crop_paths = (
        list(get_subfolders_with_files(os.path.join(crops_path, 'classified_1'), is_image, yield_by_one=True)) +
        list(get_subfolders_with_files(os.path.join(crops_path, 'misclass_0'), is_image, yield_by_one=True))
    )

    # Create mapping from hor crops annotations to their classes
    hor_crop_id2annotcats = defaultdict(dict)
    hor_crop_id2crop_path = defaultdict(dict)
    for image_paths, cls_id in zip([class_0_crop_paths, class_1_crop_paths], [0, 1]):
        for image_path in image_paths:
            image_name = os.path.split(image_path)[1]
            hor_crop_name = image_name.split('_')
            hor_crop_name, hor_crop_id = '_'.join(hor_crop_name[:-1]) + '.jpg', int(hor_crop_name[-1].split('.')[0])
            img_id = file_name2imgid.get(hor_crop_name)
            if img_id is None:
                print(f'Not found pano for {image_path}')
                continue
            hor_crop_id2annotcats[img_id][hor_crop_id] = cls_id
            hor_crop_id2crop_path[img_id][hor_crop_id] = image_path
    
    # Create mapping from panoramas names to panoramas paths
    pan_name2path = get_pannames2paths(os.path.join(args.data_path, 'pans'))

    mappings = defaultdict(dict)
    
    writer = get_coco_writer()
    
    for img_id in tqdm(coco.imgs, desc="Mapping annotations to panoramas"):
        # Skip if no anotations
        if not hor_crop_id2annotcats[img_id]:
            continue
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        im_anns = coco.loadAnns(ann_ids)
        
        # Extract paths, n_cut for hor_crop
        hor_crop_path = coco.imgs[img_id]['file_name']
        n_hor_crop = int(hor_crop_path.split('_')[-1].split('.')[0])
        pan_name = pan_name_from_crop_name(os.path.split(hor_crop_path)[1])
        pan_path = pan_name2path[pan_name]
        
        if args.debug:
            pan_img = cv2.imread(pan_path)
        try:
            pan_height, pan_width = get_image_size(pan_path)
        except FileNotFoundError:
            print(f"No pan image at {pan_path}")
            continue

        image_id, _ = writer.add_frame(pan_height, pan_width, filename=pan_path)
        
        # If mapping doesn't exist, create it
        if not mappings[(pan_height, pan_width)].get(n_hor_crop):
            mappings[(pan_height, pan_width)][n_hor_crop] = get_mapping(
                pan_height, pan_width, thetas[n_hor_crop], args.phi, 
                args.res_y, args.res_x, args.fov
            )
        
        # Get mapping
        map_y, map_x = mappings[(pan_height, pan_width)].get(n_hor_crop)

        # Map to panorama and save
        for ann_id, im_ann in enumerate(im_anns):
            b = _map_crop_annot_to_pan(map_y, map_x, [im_ann['bbox']])[0]
            cat_name = classes[hor_crop_id2annotcats[img_id][ann_id]]
            writer.add_annotation(
                image_id, b, category_id=writer.get_cat_id(cat_name)
            )

            if args.debug:
                b[2] += b[0]
                b[3] += b[1]
                cv2.rectangle(pan_img, tuple(b[:2]), tuple(b[2:]), (0,0,255), 5)

                crop_img = cv2.imread(hor_crop_id2crop_path[img_id][ann_id])
                cv2.imshow(f'crop_img_{ann_id}', pad_img(crop_img, (256, 256)))

                draw_text(
                    pan_img, cat_name, (b[0], b[1]-10),
                )
        
        if args.debug:
            cv2.imshow('pan_img', cv2.resize(pan_img, (1920, 1080)))
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
    writer.write_result(os.path.join(args.data_path, 'pans_annotations.json'))
