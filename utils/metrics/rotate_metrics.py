from shapely.geometry import Polygon


def combine_predicts_gt(predicts, metas, gt, network_type='segmentation'):
    if network_type == 'segmentation':
        return dict(img_metas=metas, predicts=predicts, gt_masks=gt)
    else:
        bboxes, scores = predicts[:, :8], predicts[:, 8]
        gt_bboxes, gt_labels, gt_masks = gt[0], gt[1], gt[2]
        return dict(img_meats=metas, pred_bboxes=bboxes, pred_score=scores,
                    gt_bboxes=gt_bboxes, gt_labels=gt_labels, gt_masks=gt_masks)


class RotateDetEval(object):
    def __init__(self):
        pass

    def val_per_measure(self, ):
        """
        Evaluate val datasets with batchz
        """
        pass

    def __call__(self, collection, box_thresh=0.4):
        pass
