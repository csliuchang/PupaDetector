import time

import torch
from datasets.builder import build_dataset, build_dataloader
from engine.optimizer import build_optimizer
from utils.metrics import RotateDetEval, SegEval
from utils import get_root_logger
import torch.nn as nn


class BaseRunner:
    def __init__(self, cfg, datasets, model, meta, distributed=False):
        # get config
        self.config = cfg
        self.distributed = distributed
        self.start_epoch = 0
        self.global_step = 0
        self.epochs = cfg.total_epochs
        self.log_iter = cfg.log_iter
        self.network_type = cfg.network_type
        self.val_iter = 1
        self.logger = get_root_logger()
        # set device
        if len(datasets) == 2:
            train_dataset, val_dataset = datasets
        else:
            train_dataset = datasets
            val_dataset = None
        model = model.cuda()
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
        self.time_str = meta['time_str']
        self.save_train_metrics_log = cfg.save_train_metrics_log
        self.save_train_predict_fn = cfg.save_train_predict_fn
        self.checkpoint_dir = cfg.checkpoint_dir
        self.save_pred_fn_path = f'{self.checkpoint_dir}/{self.config.dataset.type}/{self.config.model.type}/' \
                                 f'{self.time_str}'
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
            self.eval_method = SegEval()
            self.metrics = {'miou': 0.}
        else:
            self.eval_method = RotateDetEval()
            self.metrics = {'recall': 0., 'precision': 0., 'mAP': 0., 'train_loss': float('inf'), 'best_model_epoch': 0}

    def run(self):
        """
        running logic
        """
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            if self.distributed:
                pass
            ret_results = self._train_epoch(epoch)
            self.draw_lr(ret_results)
            if epoch % self.val_iter == 0:
                self._after_epoch(ret_results)
        self._after_train()

    def _train_epoch(self, epoch):
        """
        epoch training logic
        """
        raise NotImplementedError

    def _eval(self, epoch):
        """
        eval logic for an epoch
        """
        raise NotImplementedError

    def _after_epoch(self, results):
        pass

    def _after_train(self):
        raise NotImplementedError

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.config[name]['type']
        module_args = self.config[name]['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def draw_lr(self, results):
        lr = results['lr']
