import math
import copy
import oneflow as flow

from ..utils.registry import Registry, build_from_config

LRSCHEDULERS = Registry("LRSCHEDULER")
OPTIMIZERS = Registry("OPTIMIZER")

def build_lr_scheduler(cfg, iters_per_epoch):
    # FIXME: if have a better way
    if cfg.name == 'CosineAnnealingLR':
        cfg.T_max *= iters_per_epoch
        return build_from_config(cfg, LRSCHEDULERS)
    elif cfg.name == 'MultiStepDecay':
        cfg.milestones = [x * iters_per_epoch for x in cfg.milestones]
        return build_from_config(cfg, LRSCHEDULERS)
    elif cfg.name == 'LinearWarmup':
        cfg.learning_rate = build_lr_scheduler(cfg.learning_rate,
                                               iters_per_epoch)
        cfg.warmup_steps *= iters_per_epoch
        return build_from_config(cfg, LRSCHEDULERS)
    elif cfg.name == 'CosineWarmup' or cfg.name == 'PolynomialDecay':
        return build_from_config(cfg, LRSCHEDULERS)
    else:
        raise NotImplementedError
    
def build_optimizer(cfg, lr_scheduler, model_list=None):
    cfg = copy.deepcopy(cfg)
    name = cfg.pop('name')
    if 'layer_decay' in cfg:
        layer_decay = cfg.pop('layer_decay')
    else:
        layer_decay = 1.0
    assert isinstance(layer_decay, float)

    parameters = sum([m.parameters()
                        for m in model_list], []) if model_list else None
    cfg['parameters'] = parameters

    return OPTIMIZERS.get(name)(lr_scheduler, **cfg)