import numpy as np
import argparse
import math
import os

import cv2

from core.mapper import convert_panorama_image
from utils.path import get_subfolders_with_files, is_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Input Panorama Image')
    parser.add_argument('output_path', type=str, help='Output Panorama Image')
    parser.add_argument('--theta', type=float, default=0.0, help='Theta angle (yaw) in range [-180, 180] degrees [default: 0.0]')
    parser.add_argument('--phi', type=float, default=0.0, help='Phi angle (pitch) in range [-90, 90) degrees [default: 0.0]')
    parser.add_argument('--resolution_x', type=int, default=800, help='Resolution of the output image width [default: 800]')
    parser.add_argument('--resolution_y', type=int, default=400, help='Resolution of the output image height [default: 400]')
    parser.add_argument('--move', type=float, default=0.5, help='Move forward a bit [default: 0.5]')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')
    args = parser.parse_args()

    for image_paths in get_subfolders_with_images(args.input_path):
        rel_folder_path = os.path.relpath(image_paths, args.input_path)
        image_folder_path = os.path.split(image_paths[0])[0]
        rel_folder_path = os.path.relpath(image_folder_path, args.input_path)
        out_path = os.path.join(args.output_path, rel_folder_path)
        os.makedirs(out_path, exists_ok=True)
        for image_path in image_paths:
                image_name = os.path.split(image_path)[1]
                img = cv2.imread(image_path)
                out_img = convert_panorama_image(img, theta=args.theta, phi=args.phi, res_x=args.resolution_x,
                        res_y=args.resolution_y, debug=args.debug)
                cv2.imwrite(os.path.join(out_path, image_name), out_img)
