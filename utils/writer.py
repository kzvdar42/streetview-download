import json
import os

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Class to encode the numpy arrays in json."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class CocoWriter:

    def __init__(self, categories=None, synonyms=None):
        categories = [] if categories is None else categories
        self.categories = categories
        self.annotations = []
        self.images = []
        self.cat_to_id = dict()
        self.id_to_cat = dict()
        self.max_im_id = 0
        for cat in self.categories:
            cat_id = cat['id']
            cat_name = cat['name'].lower()
            self.cat_to_id[cat_name] = cat_id
            self.id_to_cat[cat_id] = cat_name
            # Add synonyms
            if synonyms is not None:
                for s in synonyms.get(cat['name'], []):
                    self.cat_to_id[s.lower()] = cat_id

    def get_cat_name(self, cat_id):
        cat_name = self.id_to_cat.get(cat_id, None)
        if cat_name is None:
            raise ValueError(f'Unknown category id ({cat_id})')
        else:
            return cat_name


    def get_cat_id(self, cat_name):
        cat_id = self.cat_to_id.get(cat_name.lower(), None)
        if cat_id is None:
            raise ValueError(f'Unknown category ({cat_name})')
        else:
            return cat_id

    def add_category(self, name, supercategory, cat_id=None):
        existing_ids = [x['id'] for x in self.categories]
        if cat_id is not None and any([x == cat_id for x in existing_ids]):
            raise ValueError(f"Category with id {cat_id} already exists")

        if cat_id is None:
            cat_id = max(existing_ids) + 1 if len(existing_ids) else 1

        self.categories.append({
            'name': name,
            'supercategory': supercategory,
            'id': cat_id,
        })

    def add_frame(self, height, width, filename=None, file_ext=None, image_id=None):
        if image_id:
            self.max_im_id = max(image_id, self.max_im_id)
        else:
            self.max_im_id += 1
            image_id = self.max_im_id
        if filename is None:
            file_ext = file_ext if file_ext is not None else 'jpg'
            filename = f'{image_id:0>7}.{file_ext}'
        filename = filename.replace('\\', '/')
        self.images.append({
            'height': height,
            'date_captured': None,
            'dataset': 'Roadar',
            'id': image_id,
            'file_name': filename,
            'image': filename,
            'flickr_url': None,
            'coco_url': None,
            'width': width,
            'license': None,
        })
        return image_id, filename

    def add_annotation(self, image_id, bbox, category_id, track_id=None, segmentation=None):
        assert image_id is not None

        area = int(bbox[2] * bbox[3])
        self.annotations.append({
            'image_id': image_id,
            'segmentation': segmentation,
            'iscrowd': 0,
            'bbox': bbox,
            'attributes': {},
            'area': area,
            'track_id': track_id,
            'is_occluded': False,
            'id': len(self.annotations) + 1,
            'category_id': category_id,
        })
    
    def _update_with_coco(self, coco_json):
        for image_dict in coco_json['images']:
            self.add_frame(
                height=image_dict['height'],
                width=image_dict['width'],
                filename=image_dict['file_name'],
                image_id=image_dict['id'],
            )
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_json['categories']}
        for annot_dict in coco_json['annotations']:
            self.add_annotation(
                image_id=annot_dict['image_id'],
                bbox=annot_dict['bbox'],
                track_id=annot_dict.get('track_id', None),
                category_id=self.get_cat_id(cat_id_to_name[annot_dict['category_id']]),
                segmentation=annot_dict.get('segmentation', None)
            )

    def update_with_coco(self, coco_path):
        with open(coco_path) as in_file:
            self._update_with_coco(json.load(in_file))
    
    def get_json(self):
        result = dict()
        result['annotations'] = self.annotations
        result['categories'] = self.categories
        result['images'] = self.images
        result['licenses'] = None
        result['info'] = None
        return result

    def write_result(self, save_path, verbose=False):
        if len(self.images) == 0 or len(self.annotations) == 0:
            print('Empty annotations, do not write to the file.')
            return
        if verbose:
            print(
                f'Saving coco annotation file to {save_path}' +
                f'\n with {len(self.annotations)} annotations in {len(self.images)} images'
            )
        result = self.get_json()
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        with open(save_path, 'w') as out_file:
            json.dump(
                result,
                out_file,
                indent=2,
                ensure_ascii=False,
                cls=NumpyEncoder,
            )


def get_coco_writer():
    return CocoWriter([
        {
            'id': 0,
            'name': 'non-sign',
        },
        {
            'id': 1,
            'name': 'sign',
        }
    ])

class TxtWriter:

    def __init__(self, categories=None, synonyms=None):
        categories = [] if categories is None else categories
        self.categories = categories
        self.annotations = []
        self.cat_to_id = dict()
        for cat in self.categories:
            cat_id = cat['id']
            self.cat_to_id[cat['name'].lower()] = cat_id
            # Add synonyms
            if synonyms is not None:
                for s in synonyms.get(cat['name'], []):
                    self.cat_to_id[s.lower()] = cat_id

    def get_cat_id(self, cat_name):
        cat_id = self.cat_to_id.get(cat_name.lower(), None)
        if cat_id is None:
            raise ValueError(f'Unknown category ({cat_name})')
        else:
            return cat_id

    def add_category(self, name, supercategory, cat_id=None):
        existing_ids = [x['id'] for x in self.categories]
        if cat_id is not None and any([x == cat_id for x in existing_ids]):
            raise ValueError(f"Category with id {cat_id} already exists")

        if cat_id is None:
            cat_id = max(existing_ids) + 1 if len(existing_ids) else 1

        self.categories.append({
            'name': name,
            'supercategory': supercategory,
            'id': cat_id,
        })

    def add_frame(self, height, width, filename=None, file_ext=None, image_id=None):
        return image_id, filename

    def add_annotation(self, img_path, bbox, track_id, category_id):
        x1, y1, x2, y2 = bbox
        #name, pred, x1, y1 ,x2, y2
        self.annotations.append([
            img_path,
            category_id,
            x1,y1,
            x2,y2,
        ])


    def get_txt(self):
        return '\n'.join(
            [' '.join(ann) for ann in self.annotations] 
        )

    def write_result(self, save_path):
        if len(self.annotations) == 0:
            print('Empty annotations, do not write to the file.')
            return
        
        result = self.get_txt()
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        with open(save_path, 'w') as out_file:
            out_file.write(result)
