import numpy as np
import oneflow as flow
import random
from functools import partial
from .collate_fn import collate
from ..utils.registry import Registry, build_from_config

DATASETS = Registry("DATASET")
TRANSFORM = Registry('transform')

def worker_init_fn(worker_id, seed):
    worker_seed = worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataset(split_cfg, cfg):
    return build_from_config(split_cfg, DATASETS, default_args=dict(cfg=cfg))

def build_dataloader(split_cfg, cfg, is_train=True,drop_last = True,distributed=False,**kwargs):

    if is_train:
        shuffle = True
    else:
        shuffle = False
    
    dataset = build_dataset(split_cfg, cfg)

    init_fn = partial(
            worker_init_fn, seed=cfg.seed)  

    if distributed:
        sampler = flow.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = flow.utils.data.RandomSampler(dataset)

    batch_sampler = flow.utils.data.BatchSampler(
        sampler, cfg.batch_size, drop_last=drop_last
    )

    samples_per_gpu = cfg.batch_size // cfg.num_gpus
    dataloader = flow.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.num_workers,
        collate_fn = partial(collate, samples_per_gpu=samples_per_gpu),
        worker_init_fn=init_fn,
        shuffle=shuffle)
    
    return dataloader