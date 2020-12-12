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