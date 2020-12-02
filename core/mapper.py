import math

import numpy as np
import cv2

def get_mapping(img, theta=0.0, phi=0.0, res_x=512, res_y=512, fov=60.0, debug=False):
    img_x = img.shape[0]
    img_y = img.shape[1]

    theta = theta / 180 * math.pi
    phi = phi / 180 * math.pi

    fov_x = fov
    aspect_ratio = res_y * 1.0 / res_x
    half_len_x = math.tan(fov_x / 180 * math.pi / 2)
    half_len_y = aspect_ratio * half_len_x

    pixel_len_x = 2 * half_len_x / res_x
    pixel_len_y = 2 * half_len_y / res_y

    axis_y = math.cos(theta)
    axis_z = math.sin(theta)
    axis_x = 0

    # theta rotation matrix
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    theta_rot_mat = np.array([
        [1, 0, 0],
        [0, cos_theta, -sin_theta],
        [0, sin_theta, cos_theta]
    ], dtype=np.float32)

    # phi rotation matrix
    cos_phi = math.cos(phi)
    sin_phi = -math.sin(phi)
    phi_rot_mat = np.array([
        [
        cos_phi + axis_x**2 * (1 - cos_phi),
        axis_x * axis_y * (1 - cos_phi) - axis_z * sin_phi,
        axis_x * axis_z * (1 - cos_phi) + axis_y * sin_phi
        ],
        [
            axis_y * axis_x * (1 - cos_phi) + axis_z * sin_phi,
            cos_phi + axis_y**2 * (1 - cos_phi),
            axis_y * axis_z * (1 - cos_phi) - axis_x * sin_phi
        ],
        [
            axis_z * axis_x * (1 - cos_phi) - axis_y * sin_phi,
            axis_z * axis_y * (1 - cos_phi) + axis_x * sin_phi,
            cos_phi + axis_z**2 * (1 - cos_phi)
        ]
    ], dtype=np.float32)

    map_x = np.tile(np.array(np.arange(res_x), dtype=np.float32), (res_y, 1)).T
    map_y = np.tile(np.array(np.arange(res_y), dtype=np.float32), (res_x, 1))

    map_x = map_x * pixel_len_x + pixel_len_x / 2 - half_len_x
    map_y = map_y * pixel_len_y + pixel_len_y / 2 - half_len_y
    map_z = np.ones((res_x, res_y)).astype(np.float32) * -1

    ind = np.reshape(np.concatenate((np.expand_dims(map_x, 2), np.expand_dims(map_y, 2), \
            np.expand_dims(map_z, 2)), axis=2), [-1, 3]).T

    ind = theta_rot_mat.dot(ind)
    ind = phi_rot_mat.dot(ind)

    vec_len = np.sqrt(np.sum(ind**2, axis=0))
    ind /= np.tile(vec_len, (3, 1))

    cur_phi = np.arcsin(ind[0, :])
    cur_theta = np.arctan2(ind[1, :], -ind[2, :])

    map_x = (cur_phi + math.pi/2) / math.pi * img_x
    map_y = cur_theta % (2 * math.pi) / (2 * math.pi) * img_y

    map_x = np.reshape(map_x, [res_x, res_y])
    map_y = np.reshape(map_y, [res_x, res_y])

    if debug:
        for x in range(res_x):
            for y in range(res_y):
                print(x, y, '-> (%.2f, %.2f)\t' % (map_x[x, y], map_y[x, y]),)
            print()

    return map_x, map_y


def convert_panorama_image(img, theta=0.0, phi=0.0, move=0.5, res_x=400, res_y=800, debug=False):
    map_x, map_y = get_mapping(img, theta, phi, res_x, res_y, fov, debug)
    return cv2.remap(img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def get_point_coords(point, coord_map, map_shape, max_diff):
    res = point[:2][::-1] - np.array([map_x.flatten(), map_y.flatten()]).T
    ress = np.sum(np.abs(res), axis=1)
    ind = ress.argmin()
    if ress[ind] > max_diff:
        return None
    y = ind // map_shape[1]
    x = ind % map_shape[1]
    return x, y


def get_bbox_coord(bbox, coord_map, map_shape):
    max_diff = bbox[2] + bbox[3]
    bbox = [
        [bbox[0], bbox[1]],
        [bbox[0] + bbox[2], bbox[1]],
        [bbox[0], bbox[1] + bbox[3]],
        [bbox[0] + bbox[2], bbox[1] + bbox[3]],
    ]
    res = []
    for point in bbox:
        new_point = get_point_coords(point, coord_map, map_shape, max_diff)
        if new_point == None:
            return None
        res.append(new_point)
    return cv2.boundingRect(np.array(res))