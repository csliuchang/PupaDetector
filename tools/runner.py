import time
import torch
from datasets.builder import build_dataloader
from engine.optimizer import build_optimizer
from utils.metrics import RotateDetEval, SegEval
from utils.bar import ProgressBar
from utils.metrics.rotate_metrics import combine_predicts_gt
import torch.nn as nn
import numpy as np


class BaseRunner:
    """
    A base runner logic for detector
    """
    def __init__(self, cfg, datasets, model, meta, logger, distributed=False):
        # get config
        self.config = cfg
        self.distributed = distributed
        self.start_epoch, self.global_step = 0, 0
        self.epochs = cfg.total_epochs
        self.log_iter = cfg.log_iter
        self.network_type = cfg.network_type
        self.val_iter = 1
        self.logger = logger
        # set device
        if len(datasets) == 2:
            train_dataset, val_dataset = datasets
        else:
            train_dataset, val_dataset = datasets, None
        model = model.cuda()
        # show cam on models
        self.gradients = []
        self.activations = []
        self.model = nn.parallel.DistributedDataParallel(model,
                                                         device_ids=[cfg.local_rank, ],
                                                         output_device=cfg.local_rank,
                                                         find_unused_parameters=True
                                                         )
        self.data_root = cfg.dataset.data_root
        # get datasets dataloader
        self.train_dataloader = build_dataloader(train_dataset, cfg.dataloader.samples_per_gpu,
                                                 cfg.dataloader.workers_per_gpu,
                                                 len([cfg.local_rank, ]), dist=distributed, seed=cfg.seed,
                                                 drop_last=True)
        self.val_dataloader = build_dataloader(val_dataset, 1, cfg.dataloader.workers_per_gpu, len([cfg.local_rank, ]),
                                               dist=distributed,
                                               seed=cfg.seed
                                               )
        #  build optimizer scheduler
        self.optimizer = build_optimizer(cfg, model)
        self.scheduler = self._initialize('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
        self.save_train_metrics_log = cfg.save_train_metrics_log
        self.save_train_predict_fn = cfg.save_train_predict_fn
        self.checkpoint_dir = cfg.checkpoint_dir
        self.save_pred_fn_path = meta['work_dir']
        self.save_val_pred = cfg.save_val_pred
        self.min_score_threshold = 0.4
        self.ge_heat_map = cfg.ge_heat_map
        if self.network_type == 'segmentation':
            if isinstance(cfg.model.decode_head, dict):
                self.num_classes = cfg.model.decode_head.num_classes
            elif isinstance(cfg.model.decode_head, list):
                self.num_classes = cfg.model.decode_head[0].num_classes
            else:
                raise TypeError('not support decode head type =')
            self.eval_method = SegEval(num_classes=self.num_classes)
            self.metrics = {'miou': 0.}
        else:
            self.num_classes = cfg.model.bbox_head.num_classes
            self.eval_method = RotateDetEval(num_classes=self.num_classes)
            self.metrics = {'precision': 0., 'recall': 0., 'mAP': 0., 'train_loss': float('inf'), 'best_model_epoch': 0}

    def run(self):
        """
        running logic
        """
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            if self.distributed:
                pass
            ret_results = self._train_epoch(epoch)

            if epoch % self.val_iter == 0:
                self._after_epoch(ret_results)
        self._after_train()

    def _train_epoch(self, epoch):
        """
        epoch training logic
        """
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
            _img, _ground_truth = data['images_collect']['img'], data['ground_truth']
            _img = _img.cuda()
            for key, value in _ground_truth.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        _ground_truth[key] = value.cuda()
            batch = _img.shape[0]
            logger_batch += batch
            self.optimizer.zero_grad()
            # if True:
            #     filepath = osp.join(self.save_pred_fn_path, 'masks')
            #     filepath = osp.join(filepath, data['images_collect']['img_metas'][0]['filename'])
            #     mkdir_or_exist(osp.dirname(filepath))
            #     mask = _ground_truth['gt_masks'][0].cpu().detach().numpy()
            #     cv2.imwrite(filepath, mask*255)
            losses = self.model(_img, ground_truth=_ground_truth, return_metrics=True)
            losses = losses["loss"]
            losses.backward()
            activations, grads = self.activations, self.gradients
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

    @torch.no_grad()
    def _eval(self, epoch):
        """
        Eval logic
        """
        self.model.eval()
        final_collection = []
        total_frame = 0.0
        total_time = 0.0
        prog_bar = ProgressBar(len(self.val_dataloader))
        for i, data in enumerate(self.val_dataloader):
            _img, _ground_truth = data['images_collect']['img'], data['ground_truth']
            _img = _img.cuda()
            for key, value in _ground_truth.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        _ground_truth[key] = value.cuda()
            cur_batch = _img.shape[0]
            total_frame += cur_batch
            start_time = time.time()
            predicts = self.model(_img)
            total_time += (time.time() - start_time)
            predict_gt_collection = combine_predicts_gt(predicts, data['images_collect']['img_metas'][0],
                                                        _ground_truth, self.network_type)
            final_collection.append(predict_gt_collection)
            for _ in range(cur_batch):
                prog_bar.update()
        if self.save_val_pred:
            self._save_val_prediction(final_collection)
        if self.ge_heat_map.enable:
            self._generate_heat_map(final_collection)
        metric = self.eval_method(final_collection)
        print('\t %2f FPS' % (total_frame / total_time))
        return metric

    def _after_epoch(self, results):
        pass

    def _after_train(self):
        raise NotImplementedError

    def _save_activation(self, module, input, output):
        activation = output
        self.activations.append(activation.cpu().detach())

    def _save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        self.gradients = [grad.cpu().detach()] + self.gradients

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.config[name]['type']
        module_args = self.config[name]['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def _save_val_prediction(self, final_collections):
        raise NotImplementedError

    def _generate_heat_map(self, final_collections):
        raise NotImplementedError

    def save_activation(self, module, input, output):
        activation = output
        self.activations.append(activation.cpu().detach())
