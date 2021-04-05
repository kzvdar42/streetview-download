import os
import argparse

from tqdm import tqdm
import numpy as np
import cv2

from core.model import VinoModel, load_model_config
from utils.writer import get_coco_writer
from utils.path import is_image, get_subfolders_with_files
from utils.pool_helper import PoolHelper, return_with_code
from utils.image import get_crop


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('detector_config', help='Path to the detector model config')
    parser.add_argument('in_path')
    parser.add_argument('out_path')
    parser.add_argument('--classifier_config', default=None, help='Path to the classifier model config')
    parser.add_argument('--n_threads', default='4')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Load model configs
    det_config = load_model_config(args.detector_config)

    cls_config = None
    if args.classifier_config:
        cls_config = load_model_config(args.classifier_config)

    return args, det_config, cls_config


def process_image(detector, classifier, img_path, writer, out_folder,
                  debug=False, out_queue=None):
    class_names = ['sign']
    # Load image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"[detector] No image at {img_path}")
    image_name, image_ext = os.path.splitext(os.path.split(img_path)[1])
    h, w = image.shape[:2]
    # Detect signs
    out = detector.predict(image)
    preds = out[0][0]

    image_id = None
    for pred_n, pred in enumerate(preds):
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
            if classifier is not None:
                # Make crops
                save_crop = get_crop(image, bbox, 0.05, min_padding=0)
                infer_crop = get_crop(image, bbox, 0.1, min_padding=4)
                is_sign = classifier.predict(infer_crop)
                is_sign = np.argmax(is_sign)
                if is_sign:
                    os.makedirs(f'{out_folder}/classified_1', exist_ok=True)
                    cv2.imwrite(f'{out_folder}/classified_1/' + f'{image_name}_{pred_n}{image_ext}', save_crop)
                else:
                    category_id = writer.get_cat_id('non-sign')
                    os.makedirs(f'{out_folder}/classified_0', exist_ok=True)
                    cv2.imwrite(f'{out_folder}/classified_0/' + f'{image_name}_{pred_n}{image_ext}', save_crop)
            writer.add_annotation(image_id, bbox, category_id=category_id)

            # if debug, save bboxes in it's folder
            if debug:
                debug_image = image.copy()
                cv2.putText(debug_image, class_names[int(pred[1])] + str(pred[2]), (int(pred[3]*ow),int(pred[4]*oh)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.rectangle(debug_image, tuple(bbox[:2]), tuple(bbox[2:]), (22,48,163), 1)
                os.makedirs(f'{out_folder}/classified_crops_with_bbox/', exist_ok=True)
                cv2.imwrite(f'{out_folder}/classified_crops_with_bbox/'+f'{pred[2]:.3f}_{os.path.split(im)[1]}', debug_image)
    # if debug, save crop
    if debug and image_id is not None:
        os.makedirs(f'{out_folder}/classified_crops/', exist_ok=True)
        cv2.imwrite(f'{out_folder}/classified_crops/'+f'{os.path.split(img_path)[1]}', image)


def detect_from_queue(executor, detector, classifier, writer,
                      in_queue, rel_path, save_path, out_queue=None,
                      writer_save_path=None, debug=False):
    pbar = tqdm(desc="Detection", total=in_queue.total_amount)
    in_queue.pbar = pbar
    executor = PoolHelper(pool=executor)
    for image_paths in in_queue:
        for image_path in image_paths:
            image_path = os.path.join(rel_path, image_path)
            executor.submit(
                return_with_code(process_image),
                detector, classifier, image_path,
                writer, save_path, debug=debug,
                out_queue=out_queue,
                f_done=lambda f: pbar.update(1)
            )

    # Do not exit till all tasks are complete
    executor.wait_for_all()
    # Indicate that no new values would be passed
    if out_queue is not None:
        out_queue.put('exit', increment=False)
    pbar.close()
    if writer_save_path is not None:
        writer.write_result(writer_save_path)


if __name__ == "__main__":
    args, det_config, cls_config = get_args()

    detector = VinoModel(
        config=det_config,
        num_proc=args.n_threads
    )

    if cls_config is not None:
        classifier = VinoModel(
            config=cls_config,
            num_proc=args.n_threads
        )

    writer = get_coco_writer()

    images = list(get_subfolders_with_files(args.in_path, is_image, yield_by_one=True))
    classifier_out_path = os.path.join(args.out_path, 'hor_crops_infer')
    for counter, img_path in enumerate(tqdm(images)):
        process_image(detector, classifier, img_path, writer, classifier_out_path)
        if counter % 500 == 0 and counter:
            writer.write_result(os.path.join(f'{args.out_path}', 'hor_crops_annotations.json'))
    
    writer.write_result(os.path.join(f'{args.out_path}', 'hor_crops_annotations.json'))
