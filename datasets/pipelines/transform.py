import cv2
import numpy as np
from ..builder import PIPELINES





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
        image = cv2.resize(image, [self.resize_height, self.resize_width], interpolation=cv2.INTER_LINEAR)
        results['img_info'] = image
        results['image_shape'] = [self.resize_height, self.resize_width]

    def _resize_bboxes(self, results):
        original_height, original_width = results['ori_image_shape']
        width_ratio = float(self.resize_width) / original_width
        height_ratio = float(self.resize_height) / original_height
        if "bboxes" in results["ann_info"]:
            new_bbox = []
            for bbox in results["ann_info"]["bboxes"]:
                bbox[0] = int(bbox[0] * width_ratio)
                bbox[2] = int(bbox[2] * width_ratio)
                bbox[1] = int(bbox[1] * height_ratio)
                bbox[3] = int(bbox[3] * height_ratio)
                new_bbox.append(bbox)
            new_bbox = np.array(new_bbox, dtype=np.float32)
            results['ann_info']['bboxes'] = new_bbox
        elif "polylines" in results["ann_info"]:
            new_polylines = []
            for polyline in results["ann_info"]["polylines"]:
                new_polylines.append([[poly[0] * width_ratio, poly[1] * height_ratio] for poly in polyline])
            results['polygons'] = new_polylines

    def __call__(self, results):
        self._resize_img(results)
        self._resize_bboxes(results)
        return results


@PIPELINES.register_module()
class Rotate(object):
    def __init__(self):
        self.angle = self._get_angle(angle)

    def _get_angle(self):
        pass

    def __call__(self, *args, **kwargs):
        pass
