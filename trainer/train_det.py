from tools import BaseRunner
from utils import save_checkpoint
import os.path as osp
from .build import Trainer


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
        pass

    def _generate_heat_map(self, final_collections):
        pre_save_dir = self.save_pred_fn_path + '/heatmaps'
        pass

    def _after_train(self):
        self.logger.info('all train epoch is finished')
