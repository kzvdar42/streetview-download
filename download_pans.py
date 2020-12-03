import os
import argparse
from concurrent.futures import ThreadPoolExecutor as PoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from core.pans_download import get_panoids, download_panorama_v3


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


def download_and_save(panoid, save_path, zoom=5, min_zoom=3, queue=None):
    if os.path.isfile(save_path):
        return
    # if returned empty image, try lowering the resolution
    for z in range(zoom, min_zoom-1, -1):
        panorama = download_panorama_v3(panoid, zoom=z, disp=False)
        if np.sum(panorama) > 0:
            break
    # If no image is found, do not save
    else:
        return
    panorama = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    cv2.imwrite(save_path, panorama)
    if queue is None:
        queue.append(save_path)


def get_panos_for_grid(grid, executor, closest=True, downloaded_panoids=None):
    downloaded_panoids = downloaded_panoids or set()
    indexed_panoids_set = downloaded_panoids.copy()
    indexed_pans_dict = dict()
    found_panoids = 0
    args = [dict(lat=lat, lon=lon, closest=closest) for lat, lon in grid]
    pbar = tqdm(desc='Indexing panoids', total=len(args))
    for (lat, lon), pans in executor.map(lambda kwargs: get_panoids(**kwargs), args):
        pbar.update(1)
        pans = [p for p in pans if p['panoid'] not in indexed_panoids_set]
        panoids = [pano['panoid'] for pano in pans]
        if len(pans) > 0:
            n_pans = len(indexed_panoids_set)
            indexed_panoids_set.update(panoids)
            found_panoids += len(indexed_panoids_set) - n_pans
            indexed_pans_dict[(lat, lon)] = pans
            pbar.set_postfix(
                found_pans=found_panoids
            )
    pbar.close()
    return indexed_pans_dict, indexed_panoids_set.difference(downloaded_panoids)


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
    return parser.parse_args()


if __name__ == "__main__":
    # Two closest points on the road
    # 41.1265508,-73.8616821
    # 41.1264585,-73.8616855

    args = get_args()
    print(args)
    # Read position boxes
    with open(args.pos_file) as in_file:
        pos_boxes = []
        for n_line, line in enumerate(in_file):
            line = line.strip()
            # Skip commented lines
            if line.startswith('#'):
                continue
            pos_box = [float(coord.strip()) for coord in line.split(',')]
            if len(pos_box) == 2:
                pos_box = [
                    pos_box[0] - args.grid_radius,
                    pos_box[1] - args.grid_radius,
                    pos_box[0] + args.grid_radius,
                    pos_box[1] + args.grid_radius,
                ]
            if len(pos_box) != 4:
                raise ValueError(f'Bad line ({n_line}) in input!')
            pos_boxes.append(pos_box)
    
    closest = not args.download_all
    downloaded_panoids = set()
    with PoolExecutor(max_workers=args.max_workers) as executor:
        for (lat1, lon1, lat2, lon2) in pos_boxes:
            print(f'Creating grid for {lat1, lon1, lat2, lon2}')
            grid = create_grid(lat1, lon1, lat2, lon2, step=args.step)
            print(f'Grid size {len(grid)}')
            indexed_pans_dict, indexed_panoids_set = get_panos_for_grid(
                grid, executor, closest, downloaded_panoids
            )
            downloaded_panoids.update(indexed_panoids_set)
            print(f'Found {len(indexed_panoids_set)} new panos')
            kwargs = []
            for (lat, lon), panoids in indexed_pans_dict.items():
                for pano in panoids:
                    year, month, panoid = pano['year'], pano['month'], pano['panoid']
                    if int(year) >= args.min_year:
                        save_path = f'{args.output_path}/{lat}_{lon}/{panoid}_{month}_{year}_{args.zoom}.{args.save_ext}'
                        # save_path = f'{args.output_path}/{lat}_{lon}_{panoid}_{month}_{year}_{args.zoom}.{args.save_ext}'
                        kwargs.append(dict(panoid=panoid, save_path=save_path, zoom=args.zoom, min_zoom=args.min_zoom))
            pbar = tqdm(total=len(kwargs))
            for _ in executor.map(lambda kwargs: download_and_save(**kwargs), kwargs):
                pbar.update(1)
            pbar.close()
