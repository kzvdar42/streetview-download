import argparse
import os
import re

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO

from core.mapper import get_mapping, get_point_coords, get_bbox_coord
from utils.writer import get_coco_writer
from utils.path import get_subfolders_with_files, is_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_crops_path', type=str, help='Input crops path')
    parser.add_argument('input_pans_path', type=str, help='Input pans path')
    parser.add_argument('--n_cuts_per_image', type=int, default=5, help='Number of cuts [default: 5]')
    parser.add_argument('--phi', type=float, default=15, help='Phi angle (pitch) in range [-90, 90) degrees [default: 15]')
    parser.add_argument('--resolution_x', type=int, default=1080, help='Resolution of the output image width [default: 256]')
    parser.add_argument('--resolution_y', type=int, default=1920, help='Resolution of the output image height [default: 256]')
    parser.add_argument('--fov', type=float, default=60.0, help='Field of View for image height in range [0, 180] degrees [default: 60.0]')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    start_theta = 0
    end_theta = 360 - args.fov // 2
    thetas = np.linspace(start_theta, end_theta, args.n_cuts_per_image)

    # Get pans paths
    pan_name2path = dict()
    for pan_path in get_subfolders_with_files(args.input_pans_path, is_image, yield_by_one=True):
        pan_name = os.path.splitext(os.path.split(pan_path)[1])[0]
        pan_name = re.sub('-?\d\d.\d+_\-?\d\d.\d+_', '', pan_name)
        pan_name2path[pan_name] = pan_path

    print(args.input_crops_path)
    for coco_path in get_subfolders_with_files(args.input_crops_path, lambda s: s.endswith('.json'), yield_by_one=True):
        print('coco', coco_path)
        data_path = os.path.normpath(coco_path).rsplit(os.path.sep, 2)[0]
        rel_data_path = os.path.relpath(data_path, start=args.input_crops_path)
        out_folder_path = os.path.join(args.output_path, rel_data_path)
        rel_coco_path = os.path.relpath(coco_path, start=args.input_crops_path)
        annots_out_path = os.path.join(args.output_path, rel_coco_path)
        
        coco = COCO(coco_path)
        writer = get_coco_writer()
        for i, (img_id, img_info) in tqdm(enumerate(coco.imgs.items()), total=len(coco.imgs), desc=rel_coco_path):
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
            im_anns = coco.loadAnns(ann_ids)
            if len(im_anns) == 0:
                continue
            im_name = os.path.splitext(os.path.split(img_info['file_name'])[1])[0]
            pan_name = '_'.join(im_name.split('_')[:-1])
            pan_name = re.sub('-?\d\d.\d+_\-?\d\d.\d+_', '', pan_name)
            im_folder = os.path.split(img_info['file_name'])[0]
            im_path = os.path.join(data_path, img_info['file_name'])
            image = cv2.imread(im_path)
            if image is None:
                print('Empty image at', im_path)
            if pan_name not in pan_name2path:
                print(f"Didn't found panorama for crop at {im_path}")
                continue
            if args.debug:
                cv2.imshow('image', cv2.resize(image, (1080, 720)))
                # cv2.imshow('image', cv2.resize(image, (1920, 1080)))

            bboxes = [np.array(obj['bbox'], dtype=np.int32) for obj in im_anns]

            # Get mapping
            img_n_cut = int(im_name.split('_')[-1])
            pan_img = cv2.imread(pan_name2path[pan_name])
            map_x, map_y = get_mapping(
                pan_img, theta=thetas[img_n_cut], phi=args.phi, res_x=args.resolution_x,
                res_y=args.resolution_y, fov=args.fov
            )
            if args.debug:
                cropped_img = cv2.remap(pan_img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                cv2.imshow('cropped_img', cv2.resize(cropped_img, (1080, 720)))
                # cv2.imshow('cropped_img', cv2.resize(cropped_img, (1920, 1080)))
            
            for bbox in bboxes:
                # convert to xyxy
                b = np.array([
                    bbox[0],
                    bbox[1],
                    bbox[0] + bbox[2],
                    bbox[1] + bbox[3],
                ])
                # Map to panorama coordinates
                for i in range(2):
                    print(i, i*2+1, i*2)
                    print(i, b[i*2+1], b[i*2])
                    b[i*2], b[i*2 + 1] = map_y[b[i*2+1], b[i*2]], map_x[b[i*2+1], b[i*2]]
                
                
                if args.debug:
                    print('b', b)
                    cv2.rectangle(pan_img, tuple(b[:2]), tuple(b[2:]), (0,0,255), 10)

            if args.debug:
                cv2.imshow('pan', cv2.resize(pan_img, (1920, 1080)))
                # cv2.imshow('pan', cv2.resize(pan_img, (1080, 720)))
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break
        os.makedirs(os.path.split(annots_out_path)[0], exist_ok=True)
        writer.write_result(annots_out_path)

    cv2.destroyAllWindows()
