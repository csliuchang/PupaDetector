import argparse
from utils import Config, get_root_logger, model_info
from utils.dist_utils import _find_free_port
import os
from models import build_detector, build_segmentor
from trainer.train import Train
from datasets import build_dataset
import time
import torch
import random
import numpy as np
import logging
import os.path as osp
import copy
from utils import load_checkpoint
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--config', default='./config/segformer/train_segformer_hci.json', help='train config file path')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--local_rank',
        default=0,
        type=int)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args



def main():
    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))
    args = parse_args()
    cfg = Config.fromjson(args.config)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    work_dir = osp.join(cfg.checkpoint_dir, cfg.dataset.type, cfg.model.type, timestamp)
    logger = get_root_logger(log_file=work_dir,  log_level='INFO')
    # add network params
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.local_rank is not None:
        cfg.local_rank = args.local_rank
    else:
        cfg.local_rank = range(1) if args.gpus is None else range(args.gpus)
    # set random seeds
    meta = dict()
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['time_str'] = timestamp
    meta['work_dir'] = work_dir
    # cuda set
    rank = int(os.environ['LOCAL_RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    port = _find_free_port()
    dist.init_process_group(
        backend='nccl',
        init_method="tcp://127.0.0.1:33274",
        world_size=torch.cuda.device_count(),
        rank=args.local_rank

    )
    # build_dataset
    cfg.dataset.update({"stage": "train"})
    datasets = [build_dataset(cfg.dataset)]
    cfg.dataset.pop("stage")
    if len(cfg.workflow) == 2:
        cfg.dataset.update({"stage": "val"})
        val_dataset = copy.deepcopy(cfg.dataset)
        datasets.append(build_dataset(val_dataset))
    # build model
    network_type = cfg.network_type
    cfg.model.backbone.in_channels = cfg.input_channel
    cfg.model.backbone.input_size = (cfg.input_width, cfg.input_height)
    if network_type in ['rotate_detector', 'detector']:
        model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    else:
        if isinstance(cfg.model.decode_head, dict):
            cfg.model.decode_head.num_classes = cfg.num_classes
        elif isinstance(cfg.model.decode_head, list):
            for i in range(len(cfg.model.decode_head)):
                cfg.model.decode_head[i].num_classes = cfg.num_classes
        else:
            raise TypeError("decoder must be a dict or list")
        model = build_segmentor(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model_str = model_info(model)
    logger.info(model_str)
    logger.info(model)
    logger.info("Begin to train model")
    # build datasets
    if os.path.exists(cfg.pretrained):
        load_checkpoint(model, cfg.pretrained, map_location='cpu', strict=True)
        logger.info('pretrained checkpoint is loaded.')
    trainer = Train(
        cfg,
        datasets,
        model,
        meta,
        logger,
        distributed=False
    )
    trainer.run()


if __name__ == "__main__":
    import sys
    import pathlib
    main()