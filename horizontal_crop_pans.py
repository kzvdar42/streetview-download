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
from utils.pool_helper import PoolHelper, return_with_code


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Input path')
    parser.add_argument('output_path', type=str, help='Output folder path')
    parser.add_argument('--n_cuts_per_image', type=int, default=5, help='Number of cuts [default: 5]')
    parser.add_argument('--phi', type=float, default=15, help='Phi angle (pitch) in range [-90, 90) degrees [default: 15]')
    parser.add_argument('--resolution_x', type=int, default=1080, help='Resolution of the output image width [default: 256]')
    parser.add_argument('--resolution_y', type=int, default=1920, help='Resolution of the output image height [default: 256]')
    parser.add_argument('--fov', type=float, default=60.0, help='Field of View for image height in range [0, 180] degrees [default: 60.0]')
    parser.add_argument('--max_workers', type=int, default=20, help='Max number of parallel workers to crop panoramas [Defaul: 20]')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--skip_cropped', action='store_true', help='Skip cropped')
    return parser.parse_args()


def get_crop_and_save(img, theta, phi, res_x, res_y, fov, save_path, out_queue=None):
    map_x, map_y = get_mapping(
        img, theta=theta, phi=phi, res_x=res_x, res_y=res_y, fov=fov
    )
    out_img = cv2.remap(
        img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
    )
    folder_path, image_name = os.path.split(save_path)
    os.makedirs(folder_path, exist_ok=True)
    cv2.imwrite(save_path, out_img)
    if out_queue is not None:
        out_queue.put(image_name)


def crops_from_queue(executor, in_queue, rel_path, save_path, out_queue=None, n_cuts_per_image=5,
                     phi=15, res_x=1080, res_y=1920, fov=60, skip_cropped=True):
    pbar = tqdm(desc="Horizontal crops", total=in_queue.total_amount)
    in_queue.pbar = pbar
    executor = PoolHelper(pool=executor)
    start_theta = 0
    end_theta = 360 - fov // 2
    thetas = np.linspace(start_theta, end_theta, n_cuts_per_image)
    for image_paths in in_queue:
        for image_path in image_paths:
            image_name, image_ext = os.path.splitext(os.path.split(image_path)[1])
            img = None
            for n_cut in range(n_cuts_per_image):
                crop_out_path = os.path.join(save_path, image_name + f'_{n_cut}' + image_ext)
                if skip_cropped and os.path.isfile(crop_out_path):
                    if out_queue is not None:
                        out_queue.put(image_name + f'_{n_cut}' + image_ext)
                    if n_cut == n_cuts_per_image - 1:
                        pbar.update(1)
                    continue
                if img is None:
                    img = cv2.imread(os.path.join(rel_path, image_path))
                if n_cut == n_cuts_per_image - 1:
                    f_done = lambda f: pbar.update(1)
                else:
                    f_done = None
                executor.submit(
                    return_with_code(get_crop_and_save),
                    img, thetas[n_cut], phi,
                    res_x, res_y,
                    fov, crop_out_path,
                    f_done = f_done,
                    out_queue=out_queue
                )
    # Do not exit till all tasks are complete
    if out_queue is not None:
        out_queue.put('exit', increment=False)
    executor.wait_for_all()
    pbar.close()


if __name__ == '__main__':
    args = get_args()

    start_theta = 0
    end_theta = 360 - args.fov // 2
    thetas = np.linspace(start_theta, end_theta, args.n_cuts_per_image)
    image_folders = list(get_subfolders_with_files(args.input_path, is_image))

    executor = PoolHelper(args.max_workers)
    for image_paths in tqdm(image_folders, desc='Folders'):
        image_folder_path = os.path.split(image_paths[0])[0]
        rel_folder_path = os.path.relpath(image_folder_path, args.input_path)
        out_path = os.path.join(args.output_path, rel_folder_path)
        for image_path in tqdm(image_paths, leave=False, desc=rel_folder_path):
            image_name = os.path.split(image_path)[1]
            image_name, image_ext = os.path.splitext(image_name)
            img = None
            img_out_path = os.path.join(out_path, image_folder_path)
            os.makedirs(img_out_path, exist_ok=True)
            for n_cut in range(args.n_cuts_per_image):
                crop_out_path =os.path.join(img_out_path, image_name + f'_{n_cut}' + image_ext)
                if args.skip_cropped and os.path.isfile(crop_out_path):
                    continue
                if img is None:
                    img = cv2.imread(image_path)
                executor.submit(
                    return_with_code(get_crop_and_save),
                    img, thetas[n_cut], args.phi,
                    args.resolution_x, args.resolution_y,
                    args.fov, crop_out_path
                )
    print('Waiting for last crops to save...')
    executor.shutdown()
    print('Finished')
