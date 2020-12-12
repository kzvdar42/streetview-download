import os
import argparse

from tqdm import tqdm
import numpy as np
import cv2

from core.model import VinoModel, load_model_config
from utils.writer import get_coco_writer
from utils.pool_helper import PoolHelper, return_with_code


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
            category_id = writer.get_cat_id(class_names[int(pred[1])])
            writer.add_annotation(image_id, bbox, track_id=-1, category_id=category_id)
            # Make crops
            if classifier is not None:
                b = bbox.copy()
                b = np.array([
                    max(b[0] - b[2] * np.random.uniform(0.05, 0.25), 0),
                    max(b[1] - b[3] * np.random.uniform(0.02, 0.20), 0),
                    min(b[0] + b[2] * np.random.uniform(1.05, 1.25), image.shape[1]),
                    min(b[1] + b[3] * np.random.uniform(1.02, 1.20), image.shape[0]),
                ], dtype=np.int32)
                crop = image[b[1]:b[3], b[0]:b[2]].copy()
                is_sign = classifier.predict(crop)
                is_sign = np.argmax(is_sign)
                if is_sign:
                    os.makedirs(f'{out_folder}/classified_1', exist_ok=True)
                    cv2.imwrite(f'{out_folder}/classified_1/' + f'{image_name}_{pred_n}.{image_ext}', crop)
                else:
                    os.makedirs(f'{out_folder}/classified_0', exist_ok=True)
                    cv2.imwrite(f'{out_folder}/classified_0/' + f'{image_name}_{pred_n}.{image_ext}', crop)


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
                      in_queue, rel_path, save_path, out_queue=None, debug=False):
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
    
    # Indicate that no new values would be passed
    if out_queue is not None:
        out_queue.put('exit', increment=False)
    # Do not exit till all tasks are complete
    executor.wait_for_all()
    pbar.close()


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

    images = [os.path.join(dp, f) for dp, dn, fn in os.walk(d) for f in fn]
    for counter, img_path in enumerate(tqdm(images)):
        process_image(detector, classifier, img_path, writer, args.out_path)
        if counter % 500 == 0 and counter:
            writer.write_result(os.path.join(f'{args.out_path}', 'hor_crops_annotations.json'))
    
    writer.write_result(os.path.join(f'{args.out_path}', 'hor_crops_annotations.json'))
