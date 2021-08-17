import time

import torch
from datasets.builder import build_dataset, build_dataloader
from engine.optimizer import build_optimizer
from utils.metrics import RotateDetEval, SegEval
from tqdm import tqdm


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
        # set device
        torch.manual_seed(meta['seed'])  # set seed for cpu
        if torch.cuda.device_count() > 0 and torch.cuda.is_available():
            self.with_cuda = True
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(meta['seed'])
            torch.cuda.manual_seed_all(meta['seed'])
        if len(datasets) == 2:
            train_dataset, val_dataset = datasets
        else:
            train_dataset = datasets
            val_dataset = None
        self.model = model.to(self.device)
        self.data_root = cfg.dataset.data_root
        # get datasets dataloader
        self.train_dataloader = build_dataloader(train_dataset, cfg.dataloader.samples_per_gpu,
                                                 cfg.dataloader.workers_per_gpu,
                                                 len(cfg.gpu_ids), dist=distributed, seed=cfg.seed)
        self.val_dataloader = build_dataloader(val_dataset, 1, cfg.dataloader.workers_per_gpu, len(cfg.gpu_ids),
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
        self.save_val_pred = cfg.save_val_pred
        self.min_score_threshold = 0.4
        if self.network_type == 'segmentation':
            self.num_classes = cfg.model.decode_head.num_classes
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
            print('-' * 15 + f"Finish {ret_results['epoch']} epoch training" + '-' * 15)
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
