import cv2
import numpy as np


def get_cam_on_image(img, mask):
    heat_map = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heat_map = heat_map + np.float32(heat_map) / 255
    cam = heat_map + np.float32(img)
    cam = cam / np.max(cam)
    return cam
