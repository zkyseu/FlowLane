import math
import copy
import oneflow as flow

from ..utils.registry import Registry, build_from_config

LRSCHEDULERS = Registry("LRSCHEDULER")
OPTIMIZERS = Registry("OPTIMIZER")

#def build_lr_scheduler(cfg, iters_per_epoch):
#    # FIXME: if have a better way
#    if cfg.name == 'CosineAnnealingLR':
#        cfg.T_max *= iters_per_epoch
#        return build_from_config(cfg, LRSCHEDULERS)
#    elif cfg.name == 'MultiStepDecay':
#        cfg.milestones = [x * iters_per_epoch for x in cfg.milestones]
#        return build_from_config(cfg, LRSCHEDULERS)
#    elif cfg.name == 'LinearWarmup':
#        cfg.learning_rate = build_lr_scheduler(cfg.learning_rate,
#                                               iters_per_epoch)
#        cfg.warmup_steps *= iters_per_epoch
#        return build_from_config(cfg, LRSCHEDULERS)
#    elif cfg.name == 'CosineWarmup' or cfg.name == 'PolynomialLR':
#        return build_from_config(cfg, LRSCHEDULERS)
#    else:
#        raise NotImplementedError
    


def build_optimizer(cfg, net):
    params = []
    cfg_cp = cfg.optimizer.copy()
    cfg_type = cfg_cp.pop('name')

    if cfg_type not in dir(flow.optim):
        raise ValueError("{} is not defined.".format(cfg_type))

    _optim = getattr(flow.optim, cfg_type)
    return _optim(net.parameters(), **cfg_cp)

def build_lr_scheduler(cfg, optimizer):

    cfg_cp = cfg.lr_scheduler.copy()
    cfg_type = cfg_cp.pop('name')

    if cfg_type not in dir(flow.optim.lr_scheduler):
        raise ValueError("{} is not defined.".format(cfg_type))


    _scheduler = getattr(flow.optim.lr_scheduler, cfg_type) 


    return _scheduler(optimizer, **cfg_cp) 