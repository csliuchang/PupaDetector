import cv2
import numpy as np

from ..builder import PIPELINES
from engine.parallel import DataContainer as DC
import torch





@PIPELINES.register_module()
class Resize(object):
    """
        Resize images & rotated bbox
        Inherit Resize pipeline class to handle rotated bboxes
    """

    def __init__(self, img_scale):
        self.scale = img_scale
        self.resize_height, self.resize_width = self.scale

    def _resize_img(self, results):
        image = results['img_info']
        image = cv2.resize(image, [self.resize_width, self.resize_height], interpolation=cv2.INTER_LINEAR)
        results['img_info'] = image
        results['image_shape'] = [self.resize_width, self.resize_height]

    def _resize_bboxes(self, results):
        original_height, original_width = results['ori_image_shape']
        bboxes = results['ann_info']['bboxes']
        width_ratio = float(self.resize_width) / original_width
        height_ratio = float(self.resize_height) / original_height
        new_bbox = []
        for bbox in bboxes:
            bbox[0] = int(bbox[0] * width_ratio)
            bbox[2] = int(bbox[2] * width_ratio)
            bbox[4] = int(bbox[4] * width_ratio)
            bbox[6] = int(bbox[6] * width_ratio)
            bbox[1] = int(bbox[1] * height_ratio)
            bbox[3] = int(bbox[3] * height_ratio)
            bbox[5] = int(bbox[5] * height_ratio)
            bbox[7] = int(bbox[7] * height_ratio)
            new_bbox.append(bbox)
        new_bbox = np.array(new_bbox, dtype=np.float32)
        results['ann_info']['bboxes'] = new_bbox

    def __call__(self, results):
        self._resize_img(results)
        self._resize_bboxes(results)
        return results
