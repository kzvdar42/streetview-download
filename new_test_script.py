from collections import defaultdict
import os
import argparse
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from threading import Thread
from glob import glob
import time

import multiprocessing as mp
from addict import Dict
from tqdm.auto import tqdm
import numpy as np
import cv2
import grequests
from turbojpeg import TurboJPEG

from core.pans_download import get_tiles_info
from core.model import VinoModel, load_model_config
from utils.writer import get_coco_writer
from utils.pool_helper import QueueIterator, PoolHelper, return_with_code
from utils.image import get_crop
from download_pans import read_pos_boxes_file, get_panos, create_grid
from horizontal_crop_pans import _get_crop, get_thetas, get_mapping


jpeg_reader = TurboJPEG()

def load_img(path):
    with open(path, 'rb') as in_file:
        return jpeg_reader.decode(in_file.read(), 1)


def save_img(path, bgr_array):
    with open(path, 'wb') as out_file:
        out_file.write(jpeg_reader.encode(bgr_array))


def stich_tiles(tiles_info, tiles, img_h, img_w, tile_h, tile_w):
    panorama = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    i = 0
    # jpeg_reader = TurboJPEG()
    for tile, (x, y, fname, url) in zip(tiles, tiles_info):
        if tile is None:
            continue
        try:
            tile = jpeg_reader.decode(tile, 1)
            panorama[y*tile_h:(y+1)*tile_h, x*tile_w:(x+1)*tile_w] = tile
        except:
            continue
    return panorama


def download_panorama_v5(panoid, zoom=5, max_retry=3):
    '''
    v3: save image information in a buffer. (v2: save image to dist then read)
    input:
        panoid: which is an id of image on google maps
        zoom: larger number -> higher resolution, from 1 to 5, better less than 3, some location will fail when zoom larger than 3
        disp: verbose of downloading progress, basically you don't need it
    output:
        panorama image (uncropped)
    '''
    tile_height, tile_width = 512, 512
    img_w, img_h = 416*(2**zoom), 416*( 2**(zoom-1) )
    tiles = get_tiles_info(panoid, zoom=zoom)
    
    # Try to download the image file
    tile_urls = np.array([tile[-1] for tile in tiles], dtype=str)
    valid_tiles = np.array([None] * len(tile_urls), dtype=object)
    is_empty = valid_tiles == None
    n_retry = 0
    while (is_empty).any() and n_retry < max_retry:
        if n_retry > 0:
            tqdm.write("Connection error. Trying again in 1 seconds.")
            time.sleep(1)
        idxs = (is_empty).nonzero()[0]
        rs = (grequests.get(t_url, stream=True) for t_url in tile_urls[idxs])
        res = grequests.map(rs)
        for i, res_i in zip(idxs, res):
            if res_i:
                valid_tiles[i] = res_i.content
        n_retry += 1
        is_empty = valid_tiles == None
    
    # If failed to download at least 30% of panorama, return None
    if np.sum(is_empty) > 0.3 * len(valid_tiles):
        return None

    return stich_tiles(
        tiles, valid_tiles, img_h, img_w, tile_height, tile_width
    )


def process_image(detector, classifier, image, img_path, writer, out_folder,
                  debug=False, out_queue=None, image_ext='.jpg'):
    image_name, image_ext = os.path.splitext(os.path.split(img_path)[1])
    class_names = ['sign']
    h, w = image.shape[:2]
    # Detect signs
    out = detector.predict(image)
    preds = out[0][0]

    image_id = None
    predicted = False
    for pred_n, pred in enumerate(preds):
        # If batch number is -1, it's the end of predictions
        if int(pred[0]) == -1:
            break
        if pred[2] > 0.2:
            # Save annotations
            if image_id is None:
                image_id, _ = writer.add_frame(h, w, img_path)
            pred[3:7] = np.clip(pred[3:7], 0, 1)
            bbox = np.array([
                pred[3]*w,
                pred[4]*h,
                pred[5]*w - pred[3]*w,
                pred[6]*h - pred[4]*h,
            ], dtype=np.int32)
            pred[1] = 0 if pred[1] > 0 else pred[1]
            category_id = writer.get_cat_id(class_names[int(pred[1])])
            # Make crops
            # b = bbox.copy()
            # save_b = np.array([
            #     max(b[0] - b[2] * 0.05, 0),
            #     max(b[1] - b[3] * 0.05, 0),
            #     min(b[0] + b[2] * 1.05, image.shape[1]),
            #     min(b[1] + b[3] * 1.05, image.shape[0]),
            # ], dtype=np.int32)
            # infer_b = np.array([
            #     max(b[0] - max(4, b[2] * 0.1), 0),
            #     max(b[1] - max(4, b[3] * 0.1), 0),
            #     min(b[0] + max(4, b[2] * 1.1), image.shape[1]),
            #     min(b[1] + max(4, b[3] * 1.1), image.shape[0]),
            # ], dtype=np.int32)
            # infer_crop = image[infer_b[1]:infer_b[3], infer_b[0]:infer_b[2]]
            # save_crop = image[save_b[1]:save_b[3], save_b[0]:save_b[2]]
            save_crop = get_crop(image, bbox, 0.05, min_padding=0)
            infer_crop = get_crop(image, bbox, 0.1, min_padding=4)
            is_sign = classifier.predict(infer_crop)
            is_sign = np.argmax(is_sign)
            predicted = True
            if is_sign:
                os.makedirs(f'{out_folder}/classified_1', exist_ok=True)
                save_img(f'{out_folder}/classified_1/' + f'{image_name}_{pred_n}{image_ext}', save_crop)
                # cv2.imwrite(f'{out_folder}/classified_1/' + f'{image_name}_{pred_n}{image_ext}', save_crop)
            else:
                category_id = writer.get_cat_id('non-sign')
                os.makedirs(f'{out_folder}/classified_0', exist_ok=True)
                save_img(f'{out_folder}/classified_0/' + f'{image_name}_{pred_n}{image_ext}', save_crop)
                # cv2.imwrite(f'{out_folder}/classified_0/' + f'{image_name}_{pred_n}{image_ext}', save_crop)
            writer.add_annotation(image_id, bbox, category_id=category_id)
    return predicted



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pos_file', type=str, help='Positions file path')
    parser.add_argument('output_path', type=str, help='Output folder path')
    parser.add_argument('--redo_processed', action='store_true',
        help='Provide this flag if want to re-process the files that already processed')
    # Download args
    parser.add_argument('--step', type=float, default=0.0001, help='Grid step size [Default: 0.0002]')
    parser.add_argument('--zoom', type=int, default=5, help='Panorama quality in range [0, 5] [Default: 5]')
    parser.add_argument('--min_zoom', type=int, default=3,
        help='Minimun panorama quality to download, if better quality is not available in range [0, 5] [Default: 5]'
    )
    parser.add_argument('--max_workers', type=int, default=2, help='Max number of parallel workers to index panoramas [Defaul: 2]')
    parser.add_argument('--save_ext', type=str, default='jpg', help='Format to save panoramas [Default: jpg]')
    parser.add_argument('--min_year', type=int, default=2013, help='Minimum year to download [Default: 2013]')
    parser.add_argument('--grid_radius', type=float, default=0.0001, help='Grid radius for single points [Default: 0.0001]')
    parser.add_argument('--download_all', action='store_true', help='Download all founded panoramas, not only the closest ones')
    # Horizontal crop args
    parser.add_argument('--n_cuts_per_image', type=int, default=5, help='Number of cuts [default: 5]')
    parser.add_argument('--phi', type=float, default=15, help='Phi angle (pitch) in range [-90, 90) degrees [default: 15]')
    parser.add_argument('--res_x', type=int, default=1920, help='Resolution of the output image width [default: 256]')
    parser.add_argument('--res_y', type=int, default=1080, help='Resolution of the output image height [default: 256]')
    parser.add_argument('--fov', type=float, default=60.0, help='Field of View for image height in range [0, 180] degrees [default: 60.0]')
    # Detector args
    parser.add_argument('detector_config', help='Path to the detector model config')
    parser.add_argument('classifier_config', default=None, help='Path to the classifier model config')
    parser.add_argument('--detector_n_threads', default='1')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    args.closest = not args.download_all
    args.skip_processed = not args.redo_processed
    return args


def main(args):
    pos_boxes = read_pos_boxes_file(args.pos_file, args.grid_radius)

    executor = PoolExecutor(max_workers=args.max_workers)
    m = mp.Manager()

    # Launch indexation
    pan_info_queue = QueueIterator(m.Queue(), batch_size=args.max_workers)
    get_panos_thread = Thread(
        target = get_panos, daemon=True,
        args = (pos_boxes, executor, pan_info_queue, args)
    )
    get_panos_thread.start()
    # Give time to start indexing
    time.sleep(3)

    thetas = get_thetas(fov=args.fov, n_cuts=args.n_cuts_per_image)
    hor_maps = defaultdict(dict)

    pans_out_path = os.path.join(args.output_path, 'pans')
    hor_crop_out_path = os.path.join(args.output_path, 'hor_crops')
    classifier_out_path = os.path.join(args.output_path, 'hor_crops_infer')
    annotations_out_path = os.path.join(args.output_path, 'hor_crops_annotations.json')

    # Load model configs
    det_config = load_model_config(args.detector_config)
    cls_config = load_model_config(args.classifier_config)

    detector = VinoModel(
        config=det_config,
        num_proc=args.detector_n_threads
    )

    classifier = VinoModel(
        config=cls_config,
        num_proc=args.detector_n_threads
    )

    writer = get_coco_writer()
    n_same, n_empty = 0, 0
    pbar = tqdm(total=0, desc='Downloading & Processing')
    for pan_infos in pan_info_queue:
        pbar.total = pan_info_queue.total_amount
        for pan_info in pan_infos:
            pbar.set_postfix(n_same=n_same, n_empty=n_empty)
            pan_id, lat, lon, year, month = (
                pan_info['panoid'], pan_info['lat'], pan_info['lon'],
                pan_info['year'], pan_info['month']
            )
            pano_name = f'{lat}_{lon}_{pan_id}_{month}_{year}'
            same_files = glob(os.path.join(pans_out_path, f'*{pan_id}_{month}_{year}*'))
            hor_crop_paths = glob(os.path.join(hor_crop_out_path, f'*{pan_id}_{month}_{year}*'))
            # If found panorama on disk, skip
            if same_files:
                n_same += 1
                pano_img = load_img(same_files[0])
            # If found horizontal crops on disk, skip downloading too
            elif hor_crop_paths and args.skip_processed:
                n_same += 1
                hor_crop_paths = {
                    int(os.path.splitext(hor_crop_path)[0].rsplit('_', 1)[1]):hor_crop_path for hor_crop_path in hor_crop_paths
                }
                pano_img = None
            else:
                pano_img = download_panorama_v5(pan_id)
                # If failed to download, skip
                if pano_img is None:
                    n_empty += 1
                    pbar.update(1)
                    continue
            for n_hor_crop, theta in enumerate(thetas):
                # If panorama is not downloaded, load the crop
                if pano_img is None:
                    hor_crop_path = hor_crop_paths.get(n_hor_crop)
                    if hor_crop_path is None:
                        continue
                    else:
                        hor_img = load_img(hor_crop_path)
                else:
                    # Get theta
                    if hor_maps[pano_img.shape].get(theta) is None:
                        hor_maps[pano_img.shape][theta] = get_mapping(
                            *pano_img.shape[:2], theta, args.phi, args.res_y, args.res_x, args.fov
                        )
                    hor_img = _get_crop(pano_img, *hor_maps[pano_img.shape][theta])
                hor_img_name = f'{pano_name}_{n_hor_crop}'
                hor_img_path = os.path.join(hor_crop_out_path, f'{hor_img_name}.jpg')
                
                is_predicted = process_image(detector, classifier, hor_img, hor_img_path, writer, classifier_out_path)
                if is_predicted:
                    os.makedirs(hor_crop_out_path, exist_ok=True)
                    save_img(hor_img_path, hor_img)
            pbar.update(1)
            # Save result every 500 pans
            if pbar.n % 500 == 0:
                writer.write_result(annotations_out_path)
    writer.write_result(annotations_out_path)


if __name__ == '__main__':

    args = get_args()
    print('Args:', args, sep='\n')
    main(args)