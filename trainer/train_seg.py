import torch
from tools import BaseRunner
from utils import save_checkpoint, mkdir_or_exist
import numpy as np
import os.path as osp
import cv2
import os
from utils.visual import get_cam_on_image
import torch.distributed as dist
from .build import Trainer


@Trainer.register_module()
class TrainSeg(BaseRunner):
    """
    A simple segmentation trainer
    """
    def __init__(self, *args, **kwargs):
        super(TrainSeg, self).__init__(*args, **kwargs)

    def _after_epoch(self, results):
        self.logger.info('finish %d epoch, train_loss: %f, time: %d ms, lr: %s' % (
            results['epoch'], results['train_loss'],
            results['time'] * 1000, results['lr']))
        model_save_dir = osp.join(self.save_pred_fn_path, 'checkpoints')
        net_save_path_best = osp.join(model_save_dir, 'model_best.pth')
        net_save_path_loss_best = osp.join(model_save_dir, 'model_best_loss.pth')
        assert self.val_dataloader is not None, "no val data in the dataset"
        miou = self._eval(results['epoch'])
        if miou >= self.metrics['miou']:
            self.metrics['train_loss'] = results['train_loss']
            self.metrics['miou'] = miou
            self.metrics['best_model_epoch'] = results['epoch']
            save_checkpoint(self.model, net_save_path_best)
        elif results['train_loss'] <= self.metrics['train_loss']:
            self.metrics['train_loss'] = results['train_loss']
            if dist.get_rank() == 0:
                save_checkpoint(self.model, net_save_path_loss_best)
        else:
            pass
        best_str = 'current best, '
        for k, v in self.metrics.items():
            best_str += '{}: {:.6f}, '.format(k, v)
        self.logger.info(best_str)
        self.logger.info('--' * 10 + f'finish {results["epoch"]} epoch training.' + '--' * 10)

    def _save_val_prediction(self, final_collections):
        pre_save_dir = self.save_pred_fn_path + '/predicts'
        for final_collection in final_collections:
            predictions = final_collection['predicts']
            filename = final_collection['img_metas']['filename']
            filepath = os.path.join(pre_save_dir, 'val_' + filename)
            predict = torch.softmax(predictions, dim=0)
            predictions = predict.cpu().detach().numpy()
            predict_labels = np.argmax(predictions, axis=0).astype(np.uint8)
            if predictions.shape[0] == 2:
                max_scores = predictions[1, ...]
                predict_labels[max_scores >= self.min_score_threshold] = 1
                predict_labels[max_scores < self.min_score_threshold] = 0
            else:
                pass
            mkdir_or_exist(osp.dirname(filepath))
            img_file = final_collection['img_metas']['filename']
            image_path = osp.join(self.data_root, 'images', img_file)
            ori_img = cv2.imread(image_path, 0)
            predict_labels = np.expand_dims(predict_labels, axis=-1)
            predict_labels = cv2.resize(predict_labels, [ori_img.shape[0], ori_img.shape[1]],
                                        interpolation=cv2.INTER_LINEAR)
            merge_img = cv2.addWeighted(ori_img, 0.5, predict_labels * 255, 0.5, 0)
            cv2.imwrite(filepath, merge_img)

    def _generate_heat_map(self, final_collections):
        pre_save_dir = self.save_pred_fn_path + '/heatmaps'
        for final_collection in final_collections:
            predictions = final_collection['predicts']
            filename = final_collection['img_metas']['filename']
            image_path = osp.join(self.data_root, 'images', filename)
            filepath = os.path.join(pre_save_dir, 'cam_' + filename)
            mkdir_or_exist(osp.dirname(filepath))
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (predictions.shape[1], predictions.shape[2]), cv2.INTER_LINEAR)
            # predictions = predictions.resize_(predictions.shape[0], img.shape[0], img.shape[1])
            if self.ge_heat_map.mode == "score_cam":
                cam = get_cam_on_image(img, predictions, score_id=13)
            elif self.ge_heat_map.mode == "grad_cam":
                cam = None
            else:
                cam = img
            cv2.imwrite(filepath, cam)

    def _after_train(self):
        self.logger.info('all train epoch is finished')
