import argparse
import math
import random
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO

from core.mapper import get_mapping, get_point_coords, get_bbox_coord
from utils.writer import COCO_writer, get_coco_writer
from utils.path import get_subfolders_with_files, is_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Input path')
    parser.add_argument('output_path', type=str, help='Output folder path')
    parser.add_argument('--n_cuts_per_image', type=int, default=12, help='Number of random crops to do on image [default: 12]')
    parser.add_argument('--resolution_x', type=int, default=1080, help='Resolution of the output image width [default: 256]')
    parser.add_argument('--resolution_y', type=int, default=1920, help='Resolution of the output image height [default: 256]')
    parser.add_argument('--mean_fov', type=float, default=50.0, help='Mean Field of View for image height [default: 50.0]')
    parser.add_argument('--max_fov_offset', type=float, default=40.0, help='Max Field of View offset [default: 40.0]')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--skip_annotated', action='store_true', help='Skip annotated')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Constants
    STANDARD_FOV = 60 # DO NOT CHANGE

    print(args.input_path)
    # Clean out path
    # shutil.rmtree(args.output_path, ignore_errors=True)

    for coco_path in get_subfolders_with_files(args.input_path, lambda s: s.endswith('.json'), yield_by_one=True):
        print('coco', coco_path)
        data_path = os.path.normpath(coco_path).rsplit(os.path.sep, 2)[0]
        rel_data_path = os.path.relpath(data_path, start=args.input_path)
        out_folder_path = os.path.join(args.output_path, rel_data_path)
        rel_coco_path = os.path.relpath(coco_path, start=args.input_path)
        annots_out_path = os.path.join(args.output_path, rel_coco_path)

        # If annotation exists, skip
        if os.path.isfile(annots_out_path) and args.skip_annotated:
            print('Already processed, skipping')
            continue
        # else:
        #     shutil.rmtree(out_folder_path, ignore_errors=True)
        
        coco = COCO(coco_path)
        writer = get_coco_writer()

        for i, (img_id, img_info) in tqdm(enumerate(coco.imgs.items()), total=len(coco.imgs), desc=rel_coco_path):
            ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
            im_anns = coco.loadAnns(ann_ids)
            if len(im_anns) == 0:
                continue
            im_name = os.path.splitext(os.path.split(img_info['file_name'])[1])[0]
            im_folder = os.path.split(img_info['file_name'])[0]
            im_path = os.path.join(data_path, img_info['file_name'])
            image = cv2.imread(im_path)
            if image is None:
                print('Empty image at', im_path)
            if args.debug:
                cv2.imshow('image', cv2.resize(image, (1920, 1080)))

            bboxes = [np.array(obj['bbox'], dtype=np.int32) for obj in im_anns]

            out_image_folder_path = os.path.join(out_folder_path, im_folder)
            os.makedirs(out_image_folder_path, exist_ok=True)
            for i in tqdm(range(args.n_cuts_per_image), total=args.n_cuts_per_image, leave=False):
                out_image_path = os.path.join(out_image_folder_path, im_name + f'_{i}.jpg')
                # Choose one bbox to center on
                bbox_to_center = random.choice(bboxes)

                for try_num in range(5):
                    # Random fov
                    fov = args.mean_fov + np.random.uniform(-args.max_fov_offset, args.max_fov_offset)
                    # Center on bbox
                    phi = (bbox_to_center[1] + bbox_to_center[3] / 2 - image.shape[0] / 2) / image.shape[0] * 180
                    theta = (bbox_to_center[0] + bbox_to_center[2] / 2) / image.shape[1] * 360

                    fov_ratio = STANDARD_FOV / fov
                    max_theta_offset = (args.resolution_y - bbox_to_center[2]) / image.shape[1] * 180 / fov_ratio
                    max_phi_offset = (args.resolution_x - bbox_to_center[3]) / image.shape[0] * 180 / fov_ratio

                    # Add random offset
                    theta += np.random.uniform(-max_theta_offset, max_theta_offset)
                    phi += np.random.uniform(-max_phi_offset, max_phi_offset)

                    # Get mapping
                    map_x, map_y = get_mapping(
                        image, theta=theta, phi=phi, res_x=args.resolution_x,
                        res_y=args.resolution_x, fov=fov,
                    )
                    coord_map = np.array([map_x.flatten(), map_y.flatten()]).T

                    # Check mapping to include bbox_to_center
                    bbox = get_bbox_coord(bbox_to_center, coord_map, map_x.shape)
                    if bbox is None:
                        continue
                    else:
                        break
                else:
                    continue
                
                # Make crop
                cropped_image = cv2.remap(image, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
                # Save crop
                os.makedirs(os.path.split(out_image_path)[0], exist_ok=True)
                cv2.imwrite(out_image_path, cropped_image)
                image_id, _ = writer.add_frame(
                    *cropped_image.shape[:2],
                    filename=os.path.join(im_folder, im_name + f'_{i}.jpg')
                )

                for obj in im_anns:
                    bbox = np.array(obj['bbox'], dtype=np.int32)
                    assert len(bbox) == 4, "BBOX should have 4 elements!"
                    bbox = get_bbox_coord(bbox, coord_map, map_x.shape)
                    if bbox is None:
                        continue
                    bbox = np.array(bbox, dtype=np.int32)
                    writer.add_annotation(
                        image_id, bbox,
                        track_id=-1,
                        category_id=writer.get_cat_id('sign')
                    )
                    if args.debug:
                        bbox[2] += bbox[0]
                        bbox[3] += bbox[1]
                        cv2.rectangle(cropped_image, tuple(bbox[:2]), tuple(bbox[2:]), (0,0,255), 2)
                        cat_name = coco.cats[obj['category_id']]['name']
                        cv2.putText(cropped_image, cat_name, (int(bbox[0]), int(bbox[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 255, 255), 2)

                if args.debug:
                    cv2.imshow('cropped', cropped_image)

                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        break
        os.makedirs(os.path.split(annots_out_path)[0], exist_ok=True)
        writer.write_result(annots_out_path)

    cv2.destroyAllWindows()
