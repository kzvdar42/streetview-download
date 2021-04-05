"""Script to index and download panoramas from input file."""

import os
import time
import json
from glob import glob
import argparse
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
from threading import Thread
import multiprocessing as mp

import cv2
import numpy as np
from tqdm.auto import tqdm

from core.pans_download import get_panoids, get_grid_panoids, download_panorama_v3
from utils.writer import get_coco_writer
from utils.path import get_subfolders_with_files, is_image
from utils.pool_helper import QueueIterator, PoolHelper, return_with_code


def create_grid(x1, y1, x2, y2, step=0.0005):
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])
    
    num_steps_x = max(1, int(np.ceil((x_max - x_min) / step)))
    num_steps_y = max(1, int(np.ceil(np.abs(y_max - y_min) / step)))
    
    grid = []
    for i in range(num_steps_x):
        for j in range(num_steps_y):
            grid.append((x_min + i*step, y_max - j*step))
    
    return np.array(grid)


def download_and_save(panoid, save_path, zoom=5, min_zoom=3,
                      out_queue=None, skip_downloaded=True):
    folder_path, image_name = os.path.split(save_path)
    # If already exists, skip saving, but write to queue
    if not (os.path.isfile(save_path) and skip_downloaded):
        # if returned empty image, try lowering the resolution
        for z in range(zoom, min_zoom-1, -1):
            panorama = download_panorama_v3(panoid, zoom=z)
            if np.sum(panorama) > 0:
                break
        else:
            # If no image is found, skip
            return
        panorama = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
        os.makedirs(folder_path, exist_ok=True)
        cv2.imwrite(save_path, panorama)
    # Write to out_queue
    if out_queue is not None:
        out_queue.put(image_name)


def download_and_save_queue(executor, in_queue, out_queue, args, skip_downloaded=True):
    pbar = tqdm(desc="Downloading panoramas", total=in_queue.total_amount)
    in_queue.pbar = pbar
    n_skipped = 0
    executor = PoolHelper(pool=executor)
    for panos in in_queue:
        for pano in panos:
            year, month, panoid = pano['year'], pano['month'], pano['panoid']
            lat, lon = pano['lat'], pano['lon']
            # If pan with this panoid, year, month and zoom exists, skip
            same_files = glob(f'{args.output_path}/pans/*_{panoid}_{month}_{year}_{args.zoom}.{args.save_ext}')
            if same_files:
                n_skipped += 1
                out_queue.put(os.path.split(same_files[0])[1])
                pbar.update(1)
                pbar.set_postfix(n_skipped=n_skipped)
                continue
            save_path = f'{args.output_path}/pans/{lat}_{lon}_{panoid}_{month}_{year}_{args.zoom}.{args.save_ext}'
            executor.submit(
                return_with_code(download_and_save),
                f_done=lambda f: pbar.update(1),
                panoid=panoid, save_path=save_path, zoom=args.zoom,
                min_zoom=args.min_zoom, out_queue=out_queue,
                skip_downloaded=skip_downloaded
            )
    # Do not exit till all tasks are complete
    executor.wait_for_all()
    # Indicate that no new values would be passed
    out_queue.put('exit', increment=False)
    pbar.close()


def get_panos(pos_boxes, executor, out_queue, args, downloaded_panoids=None):
    downloaded_panoids = downloaded_panoids or set()
    indexed_panoids_set = downloaded_panoids.copy()
    n_all_pans, n_filtered_pans = 0, 0
    kwargs = []
    total = 0
    try:
        for pos_box in pos_boxes:
            grid = create_grid(*pos_box, step=args.step)
            total += len(grid)
            kwargs.extend([dict(lat=lat, lon=lon, closest=args.closest) for lat, lon in grid])

        pbar = tqdm(desc='Indexing panoids', total=total)
        for (lat, lon), pans in executor.map(lambda kwargs: get_panoids(**kwargs), kwargs):
            pans = [p for p in pans if p['panoid'] not in indexed_panoids_set]
            filtered_pans = [p for p in pans if p['year'] >= args.min_year]
            panoids = set(p['panoid'] for p in pans)
            filtered_panoids = set(p['panoid'] for p in filtered_pans)
            if len(pans) > 0:
                n_all_pans += len(panoids.difference(indexed_panoids_set))
                n_filtered_pans += len(filtered_panoids.difference(indexed_panoids_set))
                indexed_panoids_set.update(panoids)
                out_queue.put_iter(filtered_pans)
                pbar.set_postfix(
                    all_pans=n_all_pans,
                    filtered_pans=n_filtered_pans,
                )
            pbar.update(1)
    finally:
        pbar.close()
        # Indicate that no new values would be passed
        out_queue.put('exit', increment=False)
    return indexed_panoids_set.difference(downloaded_panoids)


def check_downloaded_panoids(panos_path):
    downloaded_panoids = set()
    for image_path in get_subfolders_with_files(panos_path, is_image, yield_by_one=True):
        pano_name = os.path.split(image_path)[1]
        panoid = '_'.join(pano_name.split('_')[2:-3])
        downloaded_panoids.add(panoid)
    return downloaded_panoids


def read_pos_boxes_file(path, grid_radius):
    pos_boxes = []
    with open(path) as in_file:
        for n_line, line in enumerate(in_file):
            line = line.strip()
            # Skip commented lines
            if line.startswith('#'):
                continue
            # split by comma
            pos_box = [float(coord.strip()) for coord in line.split(',')]
            # If just one point, make a grid
            if len(pos_box) == 2:
                pos_box = [
                    pos_box[0] - grid_radius,
                    pos_box[1] - grid_radius,
                    pos_box[0] + grid_radius,
                    pos_box[1] + grid_radius,
                ]
            if len(pos_box) != 4:
                raise ValueError(f'Bad line ({n_line}) in input!')
            pos_boxes.append(pos_box)
    return pos_boxes


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pos_file', type=str, help='Positions file path')
    parser.add_argument('output_path', type=str, help='Output folder path')
    parser.add_argument('--step', type=float, default=0.0001, help='Grid step size [Default: 0.0002]')
    parser.add_argument('--zoom', type=int, default=5, help='Panorama quality in range [0, 5] [Default: 5]')
    parser.add_argument('--min_zoom', type=int, default=3,
        help='Minimun panorama quality to download, if better quality is not available in range [0, 5] [Default: 5]'
    )
    parser.add_argument('--max_workers', type=int, default=20, help='Max number of parallel workers to download panoramas [Defaul: 20]')
    parser.add_argument('--save_ext', type=str, default='jpg', help='Format to save panoramas [Default: jpg]')
    parser.add_argument('--min_year', type=int, default=2010, help='Minimum year to download [Default: 2010]')
    parser.add_argument('--grid_radius', type=float, default=0.0005, help='Grid radius for single points [Default: 0.0004]')
    parser.add_argument('--download_all', action='store_true', help='Download all founded panoramas, not only the closest ones')
    
    args = parser.parse_args()
    args.closest = not args.download_all
    return args


if __name__ == "__main__":
    # Two closest points on the road
    # 41.1265508,-73.8616821
    # 41.1264585,-73.8616855

    args = get_args()
    print(args)

    # Check which pans are already downloaded
    downloaded_panoids = check_downloaded_panoids(os.path.join(args.output_path, 'pans'))

    # Read position boxes
    pos_boxes = read_pos_boxes_file(args.pos_file, args.grid_radius)
    
    executor = PoolExecutor(max_workers=args.max_workers)
    executor2 = PoolExecutor(max_workers=args.max_workers)
    m = mp.Manager()

    panos_queue = QueueIterator(m.Queue(), batch_size=args.max_workers)
    # Launch indexation
    get_panos_thread = Thread(
        target = get_panos,
        daemon=True,
        args = (pos_boxes, executor, panos_queue, args),
        kwargs= dict(downloaded_panoids=downloaded_panoids),
    )

    panos_path_queue = QueueIterator(m.Queue(), batch_size=args.max_workers)
    # Launch downloading
    download_and_save_thread = Thread(
        target = download_and_save_queue,
        daemon=True,
        args = (executor2, panos_queue,
                panos_path_queue, args)
    )

    threads = [
        get_panos_thread,
        download_and_save_thread,
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

