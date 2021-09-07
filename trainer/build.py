from utils import Registry, build_from_cfg


Trainer = Registry('trainer')


def build_trainer(cfg, datasets, model, meta, logger):
    dataset = build_from_cfg(cfg, Trainer, dict(datasets=datasets, model=model, meta=meta, logger=logger))
    return dataset
