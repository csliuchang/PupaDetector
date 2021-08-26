import torch

from tools import BaseRunner
from utils import get_root_logger, save_checkpoint, mkdir_or_exist, tensor_to_device
import time
import numpy as np
import os.path as osp
from tqdm import tqdm
from utils.metrics.rotate_metrics import combine_predicts_gt
import cv2
import os
from utils.visual import get_cam_on_image


class Train(BaseRunner):
    def __init__(self, *args, **kwargs):
        super(Train, self).__init__(*args, **kwargs)

    def _train_epoch(self, epoch):
        all_losses = []
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        logger_batch = 0
        lr = self.optimizer.param_groups[0]['lr']
        for count, data in enumerate(self.train_dataloader):
            if count >= len(self.train_dataloader):
                break
            self.global_step += 1

            tensor_to_device(data, self.device)

            _img, _ground_truth = data['images_collect']['img'], data['ground_truth']
            batch = _img.shape[0]
            logger_batch += batch
            self.optimizer.zero_grad()
            if True:
                filepath = f'{self.checkpoint_dir}/{self.config.dataset.type}/{self.config.model.type}/' \
                         f'{self.time_str}/mask'
                filepath = osp.join(filepath, data['images_collect']['img_metas'][0]['filename'])
                mkdir_or_exist(osp.dirname(filepath))
                mask = _ground_truth['gt_masks'][0].cpu().detach().numpy()
                cv2.imwrite(filepath, mask*255)
            losses = self.model(_img, ground_truth=_ground_truth, return_metrics=True)
            losses = losses["loss"]
            losses.backward()
            self.optimizer.step()
            self.scheduler.step()

            losses = losses.detach().cpu().numpy()
            all_losses.append(losses)
            if self.global_step % self.log_iter == 0:
                batch_time = time.time() - batch_start
                self.logger.info(
                    'epochs=>[%d/%d], pers=>[%d/%d], training step: %d, running loss: %f, time/pers: %d ms' % (
                        epoch, self.epochs, (count + 1) * batch, len(self.train_dataloader.dataset), self.global_step,
                        np.array(all_losses).mean(), (batch_time * 1000) / logger_batch))
            if self.save_train_metrics_log:
                pass
            if self.save_train_predict_fn:
                pass

        return {'train_loss': sum(all_losses) / len(self.train_dataloader), 'lr': lr,
                'time': time.time() - epoch_start, 'epoch': epoch}

    def _after_epoch(self, results):
        self.logger.info('finish %d epoch, train_loss: %f, time: %d ms, lr: %s' % (
            results['epoch'], results['train_loss'],
            results['time'] * 1000, results['lr']))
        model_save_dir = f'{self.checkpoint_dir}/{self.config.dataset.type}/{self.config.model.type}/' \
                         f'{self.time_str}/checkpoints'
        net_save_path_best = osp.join(model_save_dir, 'model_best.pth')
        net_save_path_loss_best = osp.join(model_save_dir, f'model_best_loss.pth')
        assert self.val_dataloader is not None, "no val data in the dataset"
        if self.network_type == 'segmentation':
            miou = self._eval(results['epoch'])
            if miou >= self.metrics['miou']:
                self.metrics['train_loss'] = results['train_loss']
                self.metrics['miou'] = miou
                self.metrics['best_model_epoch'] = results['epoch']
                save_checkpoint(self.model, net_save_path_best)
            elif results['train_loss'] <= self.metrics['train_loss']:
                self.metrics['train_loss'] = results['train_loss']
                save_checkpoint(self.model, net_save_path_loss_best)
            else:
                pass
            best_str = 'current best, '
            for k, v in self.metrics.items():
                best_str += '{}: {:.6f}, '.format(k, v)
        else:
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
                best_str += '{}: {:.6f}, '.format(k, v)
        self.logger.info(best_str)
        self.logger.info('--' * 10 + f'finish {results["epoch"]} epoch training.' + '--' * 10)

    @torch.no_grad()
    def _eval(self, epoch):
        self.model.eval()
        final_collection = []
        total_frame = 0.0
        total_time = 0.0
        for i, data in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader),
                            desc='begin val mode'):
            tensor_to_device(data, self.device)
            start_time = time.time()
            _img, ground_truth = data['images_collect']['img'], data['ground_truth']
            cur_batch = _img.shape[0]
            total_frame += cur_batch
            predicts = self.model(_img)
            total_time += (time.time() - start_time)
            predict_gt_collection = combine_predicts_gt(predicts, data['images_collect']['img_metas'][0],
                                                        ground_truth)
            final_collection.append(predict_gt_collection)
        if self.save_val_pred:
            self.save_val_prediction(final_collection)
        if self.ge_heat_map:
            self.generate_heat_map(final_collection)
        metric = self.eval_method(final_collection, self.num_classes)
        self.logger.info('%2f FPS' % (total_frame / total_time))
        return metric

    def save_val_prediction(self, final_collections):
        pre_save_dir = f'{self.checkpoint_dir}/{self.config.dataset.type}/{self.config.model.type}/' \
                       f'{self.time_str}/predicts'
        if self.network_type == 'segmentation':
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
                merge_img = cv2.addWeighted(ori_img, 0.5, predict_labels * 255, 0.5, 0)
                cv2.imwrite(filepath, merge_img)
        else:
            pass

    def generate_heat_map(self, final_collections):
        pre_save_dir = f'{self.checkpoint_dir}/{self.config.dataset.type}/{self.config.model.type}/' \
                       f'{self.time_str}/heatmaps'
        if self.network_type == 'segmentation':
            for final_collection in final_collections:
                predictions = final_collection['predicts']
                filename = final_collection['img_metas']['filename']
                image_path = osp.join(self.data_root, 'images', filename)
                filepath = os.path.join(pre_save_dir, 'cam_' + filename)
                mkdir_or_exist(osp.dirname(filepath))
                img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                cam = get_cam_on_image(predictions, img)
                cv2.imwrite(filepath, cam)


    def _after_train(self):
        self.logger.info('all train epoch is finished')
