import os
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from threading import Thread
import multiprocessing as mp

import cv2
import numpy as np
from tqdm import tqdm

from core.model import VinoModel, load_model_config
from core.pans_download import get_panoids, download_panorama_v3
from download_pans import read_pos_boxes_file, get_panos, download_and_save_queue
from horizontal_crop_pans import crops_from_queue
from run_vino_detector import detect_from_queue, VinoModel
from utils.writer import get_coco_writer
from utils.pool_helper import QueueIterator, PoolHelper, return_with_code


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
    parser.add_argument('--max_workers', type=int, default=20, help='Max number of parallel workers to download panoramas [Defaul: 20]')
    parser.add_argument('--save_ext', type=str, default='jpg', help='Format to save panoramas [Default: jpg]')
    parser.add_argument('--min_year', type=int, default=2010, help='Minimum year to download [Default: 2010]')
    parser.add_argument('--grid_radius', type=float, default=0.0001, help='Grid radius for single points [Default: 0.0004]')
    parser.add_argument('--download_all', action='store_true', help='Download all founded panoramas, not only the closest ones')
    # Horizontal crop args
    parser.add_argument('--n_cuts_per_image', type=int, default=5, help='Number of cuts [default: 5]')
    parser.add_argument('--phi', type=float, default=15, help='Phi angle (pitch) in range [-90, 90) degrees [default: 15]')
    parser.add_argument('--res_x', type=int, default=1080, help='Resolution of the output image width [default: 256]')
    parser.add_argument('--res_y', type=int, default=1920, help='Resolution of the output image height [default: 256]')
    parser.add_argument('--fov', type=float, default=60.0, help='Field of View for image height in range [0, 180] degrees [default: 60.0]')
    # Detector args
    parser.add_argument('detector_config', help='Path to the detector model config')
    parser.add_argument('classifier_config', default=None, help='Path to the classifier model config')
    parser.add_argument('--detector_n_threads', default='4')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    args.closest = not args.download_all
    args.skip_processed = not args.redo_processed
    return args


if __name__ == "__main__":

    args = get_args()
    print(args)
    # Read position boxes
    pos_boxes = read_pos_boxes_file(args.pos_file, args.grid_radius)
    
    # Create executors
    executor = PoolExecutor(max_workers=args.max_workers)
    executor2 = PoolExecutor(max_workers=args.max_workers)
    executor3 = PoolExecutor(max_workers=args.max_workers)
    executor4 = PoolExecutor(max_workers=2)
    m = mp.Manager()

    # Launch indexation
    panos_queue = QueueIterator(m.Queue(), batch_size=args.max_workers)
    get_panos_thread = Thread(
        target = get_panos,
        daemon=True,
        args = (pos_boxes, executor, panos_queue, args)
    )

    # Launch downloading
    panos_path_queue = QueueIterator(m.Queue(), batch_size=args.max_workers)
    download_and_save_thread = Thread(
        target = download_and_save_queue,
        daemon=True,
        args = (
            executor2, panos_queue,
            panos_path_queue, args,
        ),
        kwargs=dict(skip_downloaded=args.skip_processed)
    )

    # Launch horizontal cropping
    hor_crops_path_queue = QueueIterator(m.Queue(), batch_size=args.max_workers)
    hor_crops_thread = Thread(
        target=crops_from_queue,
        daemon=True,
        kwargs=dict(
            executor=executor3,
            in_queue=panos_path_queue,
            rel_path=f'{args.output_path}/pans', 
            save_path=f'{args.output_path}/hor_crops',
            out_queue=hor_crops_path_queue,
            n_cuts_per_image=args.n_cuts_per_image,
            phi=args.phi, res_x=args.res_x, res_y=args.res_y,
            fov=args.fov, skip_cropped=args.skip_processed
        )
    )

    # Create annotations writer
    writer = get_coco_writer()

    # Load models
    det_config = load_model_config(args.detector_config)
    class_config = load_model_config(args.classifier_config)

    detector = VinoModel(
        config=det_config,
        num_proc=args.detector_n_threads
    )
    
    classifier = VinoModel(
        config=class_config,
        num_proc=args.detector_n_threads
    )

    detector_thread = Thread(
        target=detect_from_queue,
        daemon=True,
        kwargs=dict(
            executor=executor4,
            detector=detector,
            classifier=classifier,
            writer=writer,
            in_queue=hor_crops_path_queue,
            rel_path=f'{args.output_path}/hor_crops', 
            save_path=f'{args.output_path}/hor_crops_infer',
            out_queue=None, debug=False,
        )
    )

    threads = [
        get_panos_thread,
        download_and_save_thread,
        hor_crops_thread,
        detector_thread,
    ]
    
    # Start threads with small waiting between
    for thread in threads:
        thread.start()
        time.sleep(3)

    # Try to join threads
    while any(thread.is_alive() for thread in threads):
        try:
            for thread in threads:
                thread.join(timeout=1)
        except KeyboardInterrupt:
            exit(0)

