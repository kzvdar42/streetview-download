"""
Original code is from https://github.com/robolyst/streetview
Functions added in this file are
download_panorama_v1, download_panorama_v2, download_panorama_v3
Usage: 
    given latitude and longitude
    panoids = panoids( lat, lon )
    panoid = panoids[0]['panoid']
    panorama_img = download_panorama_v3(panoid, zoom=2)
"""

import re
from datetime import datetime
import grequests
import requests
import time
import shutil
import itertools
import os

import PIL
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
from skimage import io
from io import BytesIO

def get_panoids_url(lat, lon):
    """
    Builds the URL of the script on Google's servers that returns the closest
    panoramas (ids) to a give GPS coordinate.
    """
    url = "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
    return url.format(lat, lon)


def get_panoids_data(lat, lon, proxies=None):
    """
    Gets the response of the script on Google's servers that returns the
    closest panoramas (ids) to a give GPS coordinate.
    """
    url = get_panoids_url(lat, lon)
    try:
        return requests.get(url, proxies=proxies)
    except requests.exceptions.ConnectionError:
        time.sleep(2)
        return get_panoids_data(lat, lon, proxies=proxies)

def postprocess_panoids(resp, closest=False, disp=False, proxies=None):
    # Get all the panorama ids and coordinates
    # I think the latest panorama should be the first one. And the previous
    # successive ones ought to be in reverse order from bottom to top. The final
    # images don't seem to correspond to a particular year. So if there is one
    # image per year I expect them to be orded like:
    # 2015
    # XXXX
    # XXXX
    # 2012
    # 2013
    # 2014
    pans = re.findall('\[[0-9]+,"(.+?)"\].+?\[\[null,null,(-?[0-9]+.[0-9]+),(-?[0-9]+.[0-9]+)', resp)
    pans = [{
        "panoid": p[0],
        "lat": float(p[1]),
        "lon": float(p[2])} for p in pans]  # Convert to floats
    
    if len(pans) == 0:
        return []

    # Remove duplicate panoramas
    pans = [p for i, p in enumerate(pans) if p not in pans[:i]]

    if disp:
        for pan in pans:
            print(pan)

    # Get all the dates
    # The dates seem to be at the end of the file. They have a strange format but
    # are in the same order as the panoids except that the latest date is last
    # instead of first.
    dates = re.findall('([0-9]?[0-9]?[0-9])?,?\[(20[0-9][0-9]),([0-9]+)\]', resp)
    dates = [list(d)[1:] for d in dates]  # Convert to lists and drop the index

    if len(dates) > 0:
        # Convert all values to integers
        dates = [[int(v) for v in d] for d in dates]

        # Make sure the month value is between 1-12
        dates = [d for d in dates if d[1] <= 12 and d[1] >= 1]

        # The last date belongs to the first panorama
        year, month = dates.pop(-1)
        pans[0].update({'year': year, "month": month})

        # The dates then apply in reverse order to the bottom panoramas
        dates.reverse()
        for i, (year, month) in enumerate(dates):
            pans[-1-i].update({'year': year, "month": month})

    # # Make the first value of the dates the index
    # if len(dates) > 0 and dates[-1][0] == '':
    #     dates[-1][0] = '0'
    # dates = [[int(v) for v in d] for d in dates]  # Convert all values to integers
    #
    # # Merge the dates into the panorama dictionaries
    # for i, year, month in dates:
    #     pans[i].update({'year': year, "month": month})

    # Sort the pans array
    def func(x):
        if 'year'in x:
            return datetime(year=x['year'], month=x['month'], day=1)
        else:
            return datetime(year=3000, month=1, day=1)
    pans.sort(key=func)

    if closest:
        return [pans[i] for i in range(len(dates)+1)]
    else:
        return pans


def get_panoids(lat, lon, closest=False, disp=False, proxies=None):
    """
    Gets the closest panoramas (ids) to the GPS coordinates.
    If the 'closest' boolean parameter is set to true, only the closest panorama
    will be gotten (at all the available dates)
    """

    responce = get_panoids_data(lat, lon)
    return ((lat, lon), postprocess_panoids(responce.text, closest=closest, disp=disp, proxies=proxies))


def get_grid_panoids(grid, closest=False, proxies=None, max_retry=3):
    """Get the closest panoramas (ids) for the GPS coordinates list.
    
    If the 'closest' boolean parameter is set to true, only the closest panorama
    will be gotten (at all the available dates)
    """
    def _gen_result(result):
        for r in result:
            yield r

    pan_urls = []
    for lat, lon in grid:
        pan_urls.append(get_panoids_url(lat, lon))
    pan_urls = np.array(pan_urls)
    
    # Try to get the panorama's data
    responses = np.zeros(len(pan_urls), dtype=bool)
    result = []
    n_retry = 0
    while any(responses == False) and n_retry < max_retry:
        if n_retry > 0:
            tqdm.write("[get_grid_panoids] Connection error. Trying again in 2 seconds.")
            time.sleep(2)
        idxs = (responses == False).nonzero()[0]
        rs = (grequests.get(p_url, stream=True) for p_url in pan_urls[idxs])
        res = grequests.map(rs)
        for i, res_i in zip(idxs, res):
            if res_i is not None:
                pans = postprocess_panoids(res_i.text, closest=closest, proxies=proxies)
                result.extend(pans)
                # result.append((grid[i], pans))
                # yield (grid[i], pans)
                responses[i] = True
        n_retry += 1
    return result
    # return _gen_result(result)


def get_tiles_info(panoid, zoom=1):
    """
    Generate a list of a panorama's tiles and their position.

    The format is (x, y, filename, fileurl)
    """
#     image_url = 'http://maps.google.com/cbk?output=tile&panoid={}&zoom={}&x={}&y={}'
    image_url = "http://cbk0.google.com/cbk?output=tile&panoid={}&zoom={}&x={}&y={}"

    # The tiles positions
    coord = list(itertools.product(range(26), range(13)))

    tiles = [(x, y, "%s_%dx%d.jpg" % (panoid, x, y), image_url.format(panoid, zoom, x, y)) for x, y in coord]

    return tiles


def download_tiles(tiles, directory, disp=False):
    """
    Downloads all the tiles in a Google Stree View panorama into a directory.

    Params:
        tiles - the list of tiles. This is generated by get_tiles_info(panoid).
        directory - the directory to dump the tiles to.
    """

    for i, (x, y, fname, url) in enumerate(tiles):

        if disp and i % 20 == 0:
            print("Image %d / %d" % (i, len(tiles)))

        # Try to download the image file
        while True:
            try:
                response = requests.get(url, stream=True)
                break
            except requests.ConnectionError:
                tqdm.write("Connection error. Trying again in 2 seconds.")
                time.sleep(2)

        with open(directory + '/' + fname, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response


def stich_tiles(panoid, tiles, directory, final_directory):
    """
    Stiches all the tiles of a panorama together. The tiles are located in
    `directory'.
    """

    tile_width = 512
    tile_height = 512

    panorama = Image.new('RGB', (26*tile_width, 13*tile_height))

    for x, y, fname, url in tiles:

        fname = directory + "/" + fname
        tile = Image.open(fname)

        panorama.paste(im=tile, box=(x*tile_width, y*tile_height))

        del tile

    panorama.save(final_directory + ("/%s.jpg" % panoid))
    del panorama


def download_panorama_v3(panoid, zoom=5):
    '''
    v3: save image information in a buffer. (v2: save image to dist then read)
    input:
        panoid: which is an id of image on google maps
        zoom: larger number -> higher resolution, from 1 to 5, better less than 3, some location will fail when zoom larger than 3
    output:
        panorama image (uncropped)
    '''
    tile_width = 512
    tile_height = 512
    # img_w, img_h = int(np.ceil(416*(2**zoom)/tile_width)*tile_width), int(np.ceil(416*( 2**(zoom-1) )/tile_width)*tile_width)
    img_w, img_h = 416*(2**zoom), 416*( 2**(zoom-1) )
    tiles = get_tiles_info(panoid, zoom=zoom)
    valid_tiles = []
    # function of download_tiles
    for i, tile in enumerate(tiles):
        x, y, fname, url = tile
        if x*tile_width < img_w and y*tile_height < img_h: # tile is valid
            # Try to download the image file
            while True:
                try:
                    response = requests.get(url, stream=True)
                    break
                except requests.ConnectionError:
                    tqdm.write("Connection error. Trying again in 2 seconds.")
                    time.sleep(2)
            # If Not a valid tile, try once more, after that fill with black pixels
            try:
                valid_tiles.append( Image.open(BytesIO(response.content)) )
            except PIL.UnidentifiedImageError:
                try:
                    time.sleep(2)
                    response = requests.get(url, stream=True)
                    valid_tiles.append( Image.open(BytesIO(response.content)) )
                except (requests.ConnectionError, PIL.UnidentifiedImageError):
                    valid_tiles.append( Image.new('RGB', (tile_width, tile_height)) )
            del response
            
    # function to stich
    panorama = Image.new('RGB', (img_w, img_h))
    i = 0
    for x, y, fname, url in tiles:
        if x*tile_width < img_w and y*tile_height < img_h: # tile is valid
            tile = valid_tiles[i]
            i+=1
            panorama.paste(im=tile, box=(x*tile_width, y*tile_height))
    return np.array(panorama)
    

def download_panorama_v4(panoid, zoom=5, max_retry=3):
    '''
    v4: save image information in a buffer. (v2: save image to dist then read)
    input:
        panoid: which is an id of image on google maps
        zoom: larger number -> higher resolution, from 1 to 5, better less than 3, some location will fail when zoom larger than 3
    output:
        panorama image (uncropped)
    '''
    tile_height, tile_width = 512, 512
    # img_w, img_h = int(np.ceil(416*(2**zoom)/tile_width)*tile_width), int(np.ceil(416*( 2**(zoom-1) )/tile_width)*tile_width)
    img_w, img_h = 416*(2**zoom), 416*( 2**(zoom-1) )
    tiles = get_tiles_info(panoid, zoom=zoom)
    valid_tiles = []
    tile_urls = []
    # function of download_tiles
    for i, tile in enumerate(tiles):
        x, y, fname, url = tile
        if x*tile_width < img_w and y*tile_height < img_h: # tile is valid
            tile_urls.append(url)
    tile_urls = np.array(tile_urls, dtype=str)
    
    # Try to download the image file
    valid_tiles = np.array([None] * len(tile_urls), dtype=object)
    n_retry = 0
    while any(valid_tiles == None) and n_retry < max_retry:
        idxs = (valid_tiles == None).nonzero()[0]
        if n_retry > 0:
            tqdm.write(f"[download_panorama_v3] Connection error. Trying again in 2 seconds. {len(idxs)}/{len(valid_tiles)}")
            time.sleep(2)
        rs = (grequests.get(t_url, stream=True) for t_url in tile_urls[idxs])
        res = grequests.map(rs)
        for i, res_i in zip(idxs, res):
            if res_i is not None:
                try:
                    valid_tiles[i] = Image.open(BytesIO(res_i.content))
                except PIL.UnidentifiedImageError:
                    continue
        n_retry += 1
    # If Not a valid tile, fill with black pixels
    idxs = (valid_tiles == None).nonzero()[0]
    if idxs.size > 0:
        for idx in idxs:
            valid_tiles[idx] = Image.new('RGB', (tile_width, tile_height))

    # function to stich
    panorama = Image.new('RGB', (img_w, img_h))
    i = 0
    for x, y, fname, url in tiles:
        if x*tile_width < img_w and y*tile_height < img_h: # tile is valid
            tile = valid_tiles[i]
            i += 1
            panorama.paste(im=tile, box=(x*tile_width, y*tile_height))
    return np.array(panorama)


def api_download(panoid, heading, flat_dir, key, width=640, height=640,
                 fov=120, pitch=0, extension='jpg', year=2017, fname=None):
    """
    Download an image using the official API. These are not panoramas.

    Params:
        :panoid: the panorama id
        :heading: the heading of the photo. Each photo is taken with a 360
            camera. You need to specify a direction in degrees as the photo
            will only cover a partial region of the panorama. The recommended
            headings to use are 0, 90, 180, or 270.
        :flat_dir: the direction to save the image to.
        :key: your API key.
        :width: downloaded image width (max 640 for non-premium downloads).
        :height: downloaded image height (max 640 for non-premium downloads).
        :fov: image field-of-view.
        :image_format: desired image format.
        :fname: file name

    You can find instructions to obtain an API key here: https://developers.google.com/maps/documentation/streetview/
    """
    if not fname:
        fname = "%s_%s_%s" % (year, panoid, str(heading))
    image_format = extension if extension != 'jpg' else 'jpeg'

    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        # maximum permitted size for free calls
        "size": "%dx%d" % (width, height),
        "fov": fov,
        "pitch": pitch,
        "heading": heading,
        "pano": panoid,
        "key": key
    }

    response = requests.get(url, params=params, stream=True)
    try:
        img = Image.open(BytesIO(response.content))
        filename = '%s/%s.%s' % (flat_dir, fname, extension)
        img.save(filename, image_format)
    except:
        print("Image not found")
        filename = None
    del response
    return filename


def download_flats(panoid, flat_dir, key, width=400, height=300,
                   fov=120, pitch=0, extension='jpg', year=2017):
    for heading in [0, 90, 180, 270]:
        api_download(panoid, heading, flat_dir, key, width, height, fov, pitch, extension, year)