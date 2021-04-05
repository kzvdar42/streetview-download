import cv2
import numpy as np

def pad_img(img, size):
    # Calculate new size
    scale_ratio = np.divide(size, np.max(img.shape[:2]), dtype=np.float32)
    new_size = np.round(img.shape[:2] * scale_ratio).astype(np.int32)
    
    # Calculate padding
    padding = size - new_size
    padding = [(d, d+m) for d, m in zip(*np.divmod(padding, 2))]
    padding.append((0, 0))
    
    # Resize and pad image
    img = cv2.resize(img, tuple(new_size[::-1]))
    img = np.pad(img, padding, mode='constant')
    return img


def draw_text(img, text, pos):
    cv2.rectangle(
        img,
        tuple(pos),
        (pos[0] + 22 * len(text), pos[1] - 22),
        (255, 255, 255),
        -1,
    )

    cv2.putText(
        img,
        str(text),
        tuple(pos),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 0, 0),
        2,
    )

def get_crop(img, bbox, padding, min_padding=4):
    """Bbox in x0,y0,w,h format."""

    img_h, img_w = img.shape[:2]

    w_padding = max(min_padding, bbox[2] * padding)
    h_padding = max(min_padding, bbox[3] * padding)

    bbox = np.array([
        max(0, bbox[0] - w_padding,
        max(0, bbox[1] - h_padding,
        min(img_w, bbox[0] + bbox[2] + w_padding),
        min(img_h, bbox[1] + bbox[3] + h_padding),
    ], dtype=np.int32)
    
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]