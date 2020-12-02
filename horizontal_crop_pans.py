import argparse
import math
import random
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO

from core.mapper import get_mapping
from utils.path import get_subfolders_with_files, is_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Input path')
    parser.add_argument('output_path', type=str, help='Output folder path')
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
    image_folders = list(get_subfolders_with_files(args.input_path, is_image))

    for image_paths in tqdm(image_folders, desc='Folders'):
        image_folder_path = os.path.split(image_paths[0])[0]
        rel_folder_path = os.path.relpath(image_folder_path, args.input_path)
        out_path = os.path.join(args.output_path, rel_folder_path)
        for image_path in tqdm(image_paths, leave=False, desc=rel_folder_path):
            image_name = os.path.split(image_path)[1]
            image_name, image_ext = os.path.splitext(image_name)
            img = cv2.imread(image_path)
            img_out_path = os.path.join(out_path, image_folder_path)
            os.makedirs(img_out_path, exist_ok=True)
            for n_cut in range(args.n_cuts_per_image):
                map_x, map_y = get_mapping(
                    img, theta=thetas[n_cut], phi=args.phi, res_x=args.resolution_x,
                    res_y=args.resolution_y, fov=args.fov
                )
                out_img = cv2.remap(img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                # print(os.path.join(img_out_path, image_name + f'_{n_cut}' + image_ext))
                cv2.imwrite(os.path.join(img_out_path, image_name + f'_{n_cut}' + image_ext), out_img)
