import logging
import os

import numpy as np
try:
    import cv2
except ImportError:
    logging.warn('Could not import cv2, please install it to enable camera viewer.')
    CV_FLAG = False
else:
    CV_FLAG = True

CV_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cv_cache')


def init_cv_window():
    cv2.namedWindow('RGB Image', cv2.WINDOW_NORMAL)


def close_cv_window():
    cv2.destroyAllWindows()


def show_image(image):
    cv2.imshow('RGB Image', image)
    cv2.waitKey(1)


def save_image(image):
    if not os.path.exists(CV_CACHE_DIR):
        os.makedirs(CV_CACHE_DIR)
    i = 0
    while os.path.exists(os.path.join(CV_CACHE_DIR, f'cv_cache_image_{i}.png')):
        i += 1
    image_path = os.path.join(CV_CACHE_DIR, f'cv_cache_image_{i}.png')
    cv2.imwrite(image_path, image)


def get_cam_intrinsic(fovy=45.0, width=320, height=240):
    aspect = width * 1.0 / height
    fovx = np.degrees(2 * np.arctan(aspect * np.tan(np.radians(fovy / 2))))

    cx = 0.5 * width
    cy = 0.5 * height
    fx = cx / np.tan(fovx * np.pi / 180 * 0.5)
    fy = cy / np.tan(fovy * np.pi / 180 * 0.5)

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K

