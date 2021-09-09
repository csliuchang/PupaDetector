from tools import BaseRunner
from utils import save_checkpoint
import os.path as osp
from .build import Trainer
import os
import torch
from utils import save_checkpoint, mkdir_or_exist
from utils import load_checkpoint
import numpy as np
import cv2
from tqdm import tqdm
from visual.cam import ScoreCam
from visual.utils import basic_visualize


@Trainer.register_module()
class TrainDet(BaseRunner):
    def __init__(self, *args, **kwargs):
        super(TrainDet, self).__init__(*args, **kwargs)

    def _after_epoch(self, results):
        self.logger.info('finish %d epoch, train_loss: %f, time: %d ms, lr: %s' % (
            results['epoch'], results['train_loss'],
            results['time'] * 1000, results['lr']))
        model_save_dir = osp.join(self.save_pred_fn_path, 'checkpoints')
        net_save_path_best = osp.join(model_save_dir, 'model_best.pth')
        net_save_path_loss_best = osp.join(model_save_dir, 'model_best_loss.pth')
        # mkdir_or_exist(model_save_dir)
        assert self.val_dataloader is not None, "no val data in the dataset"
        precision, recall, mAP = self._eval(results['epoch'])
        if mAP >= self.metrics['mAP']:
            self.metrics['train_loss'] = results['train_loss']
            self.metrics['mAP'] = mAP
            self.metrics['precision'] = precision
            self.metrics['recall'] = recall
            self.metrics['best_model_epoch'] = results['epoch']
            save_checkpoint(self.model, net_save_path_best)
        elif results['train_loss'] <= self.metrics['train_loss']:
            self.metrics['train_loss'] = results['train_loss']
            self.metrics['best_model_epoch'] = results['epoch']
            save_checkpoint(self.model, net_save_path_loss_best)
        else:
            pass
        best_str = 'current best, '
        for k, v in self.metrics.items():
            best_str += '{}: {:.4f}, '.format(k, v)
        self.logger.info(best_str)
        self.logger.info('--' * 10 + f'finish {results["epoch"]} epoch training.' + '--' * 10)

    def _save_val_prediction(self, final_collections):
        """
        Detection prediction
        """
        pre_save_dir = self.save_pred_fn_path + '/predicts'
        for final_collection in final_collections:
            predictions = final_collection["predictions"]
            filename = final_collection['img_metas']['filename']
            filepath = os.path.join(pre_save_dir, 'val_' + filename)
            predictions = predictions.cpu().detach().numpy()
            mkdir_or_exist(osp.dirname(filepath))
            image_path = osp.join(self.data_root, 'images', filename)
            ori_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            image_shape = final_collection['img_metas']["image_shape"]
            cur_img = cv2.resize(ori_img, [image_shape[0], image_shape[1]], interpolation=cv2.INTER_LINEAR)
            predict_bboxes = predictions[:, :8]
            for predict_bbox in predict_bboxes:
                bbox = np.array([predict_bbox], np.float32)
                pts = np.array([bbox.reshape((4, 2))], dtype=np.int32)
                cv2.drawContours(cur_img, pts, 0, color=(0, 255, 0), thickness=2)
            cv2.imwrite(filepath, cur_img)

    def _generate_heat_map(self, final_collections):
        pre_save_dir = self.save_pred_fn_path + '/heatmaps'
        pass

    def _after_train(self):
        self.logger.info('all train epoch is finished, begin inference')
        if self.ge_heat_map:
            check_point = osp.join(self.save_pred_fn_path, 'checkpoints', 'model_best.pth')
            save_heat_maps_path = osp.join(self.save_pred_fn_path, 'heatmap')
            load_checkpoint(self.model, check_point, strict=True)
            mkdir_or_exist(save_heat_maps_path)
            self.model.train()
            for i, data in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader),
                                desc='begin inference'):
                _img, _ground_truth = data['images_collect']['img'], data['ground_truth']
                _img = _img.cuda()
                save_heat_map_path = osp.join(save_heat_maps_path, data['images_collect']['img_metas'][0]['filename'])
                # Construct the CAM object once, and then re-use it on many images:
                cam = ScoreCam(self.model)

                # If target_category is None, the highest scoring category
                # will be used for every image in the batch.
                # target_category can also be an integer, or a list of different integers
                # for every image in the batch.
                score_map = cam(_img)
                basic_visualize(_img.cpu(), score_map.type(torch.FloatTensor).cpu(), save_path=save_heat_map_path)
        else:
            pass




