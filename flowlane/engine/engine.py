import os
import math
import random
import logging
import numpy as np
from tqdm import tqdm
from collections import OrderedDict,deque

import oneflow

from ..hooks import build_hook, Hook
from ..utils.misc import AverageMeter
from ..datasets.builder import build_dataloader
from ..model import build_model
from ..solver import build_lr_scheduler, build_optimizer
from ..datasets import IterLoader
from ..hooks.checkpoint_hook import save_checkpoint

class Trainer:
    r"""
    # trainer calling logic:
    #
    #                build_model                               ||    model(BaseModel)
    #                     |                                    ||
    #               build_dataloader                           ||    dataloader
    #                     |                                    ||
    #               build_lr_scheduler                         ||    lr_scheduler
    #                     |                                    ||
    #               build_optimizer                            ||    optimizers
    #                     |                                    ||
    #               build_train_hooks                          ||    train hooks
    #                     |                                    ||
    #               build_custom_hooks                         ||    custom hooks
    #                     |                                    ||
    #                 train loop                               ||    train loop
    #                     |                                    ||
    #      hook(print log, checkpoint, evaluate, ajust lr)     ||    call hook
    #                     |                                    ||
    #                    end                                   \/

    """

    def __init__(self, cfg):
        #self.logger = logger
        cfg.num_gpus = oneflow.env.get_world_size()
        self.distributed = True if cfg.num_gpus > 1 else False
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.best_dir = cfg.best_di

        dp_rank = oneflow.env.get_rank()
        self.log_interval = cfg.log_config.interval

        # set seed
        seed = cfg.get('seed', False)
        if seed:
            seed += dp_rank
            oneflow.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        assert cfg['device'] in ['cpu', 'gpu']
        self.device = device = oneflow.device(cfg['device'])
        self.logger.info('train with oneflow {} on {} device'.format(
            oneflow.__version__, cfg['device']))
        
        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.inner_iter = 0
        self.batch_id = 0
        self.global_steps = 0
        self.metric = 0
        self.best_metric = 0
        self.best_epoch = 0
        self.save_model_list = deque()
        self.best_model_list = deque()

        self.epochs = cfg.get('epochs', None)
        self.timestamp = cfg.timestamp
        self.logs = OrderedDict()

        self.model = build_model(cfg).to(self.device)
        self.logger.info(self.model)

        n_parameters = sum(p.numel() for p in self.model.parameters()
                           if p.require_gradient).item()

        i = int(math.log(n_parameters, 10) // 3)
        size_unit = ['', 'K', 'M', 'B', 'T', 'Q']
        self.logger.info("Number of Parameters is {:.2f}{}.".format(
            n_parameters / math.pow(1000, i), size_unit[i]))
        
        # build train dataloader
        self.train_dataloader = build_dataloader(
            self.cfg.dataset.train, self.cfg, is_train=True, device=self.device, distributed=self.distributed)          
        self.iters_per_epoch = len(self.train_dataloader)

        # build learning rate
        self.lr_scheduler = build_lr_scheduler(cfg.lr_scheduler,
                                                self.iters_per_epoch)
        
        # build optimizer
        self.optimizer = build_optimizer(cfg.optimizer, self.lr_scheduler,
                                         [self.model])

        #distributed training
        if self.distributed:
            model = oneflow.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)
        
        # build hooks
        self.hooks = []

        self.add_train_hooks()
        self.add_custom_hooks()
        self.hooks = sorted(self.hooks, key=lambda x: x.priority)

        if self.epochs:
            self.total_iters = self.epochs * self.iters_per_epoch
            self.by_epoch = True
        else:
            self.by_epoch = False
            self.total_iters = cfg.total_iters

    def add_train_hooks(self):
        optim_cfg = self.cfg.get('optimizer_config', None)
        if optim_cfg is not None:
            self.add_hook(build_hook(optim_cfg))
        else:
            self.add_hook(build_hook({'name': 'OptimizerHook'}))

        timer_cfg = self.cfg.get('timer_config', None)
        if timer_cfg is not None:
            self.add_hook(build_hook(timer_cfg))
        else:
            self.add_hook(build_hook({'name': 'IterTimerHook'}))
        ckpt_cfg = self.cfg.get('checkpoint', None)
        if ckpt_cfg is not None:
            self.add_hook(build_hook(ckpt_cfg))
        else:
            self.add_hook(build_hook({'name': 'CheckpointHook'}))

        log_cfg = self.cfg.get('log_config', None)
        if log_cfg is not None:
            self.add_hook(build_hook(log_cfg))
        else:
            self.add_hook(build_hook({'name': 'LogHook'}))

        lr_cfg = self.cfg.get('lr_config', None)
        if lr_cfg is not None:
            self.add_hook(build_hook(lr_cfg))
        else:
            self.add_hook(build_hook({'name': 'LRSchedulerHook'}))

    def add_custom_hooks(self):
        custom_cfgs = self.cfg.get('custom_config', None)
        if custom_cfgs is None:
            return

        for custom_cfg in custom_cfgs:
            cfg_ = custom_cfg.copy()
            insert_index = cfg_.pop('insert_index', None)
            self.add_hook(build_hook(cfg_), insert_index)

    def add_hook(self, hook, insert_index=None):
        assert isinstance(hook, Hook)

        if insert_index is None:
            self.hooks.append(hook)
        elif isinstance(insert_index, int):
            self.hooks.insert(insert_index, hook)

    def call_hook(self, fn_name):
        for hook in self.hooks:
            getattr(hook, fn_name)(self)

    def train(self):
        """
        Training
        """
        self.mode = 'train'
        self.model.train()
        iter_loader = IterLoader(self.train_dataloader, self.current_epoch)
        self.call_hook('run_begin')

        while self.current_iter < (self.total_iters):
            if self.current_iter % self.iters_per_epoch == 0:
                self.call_hook('train_epoch_begin')
            self.inner_iter = self.current_iter % self.iters_per_epoch
            self.current_iter += 1
            self.current_epoch = iter_loader.epoch

            data = next(iter_loader)     

            self.call_hook('train_iter_begin') 

            self.outputs = self.model(data)

            self.call_hook('train_iter_end')

            if self.current_iter % self.iters_per_epoch == 0:
                self.call_hook('train_epoch_end')
                self.current_epoch += 1

        self.call_hook('run_end')  

    def val(self, **kargs):
        """
        Validation
        """
        if not hasattr(self, 'val_loader'):
            self.val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False,device=self.device,drop_last = False,distributed=self.distributed)        

        self.logger.info('start evaluate on epoch {} ..'.format(
            self.current_epoch + 1))
        world_size = oneflow.env.get_world_size()    

        total_samples = len(self.val_loader.dataset)
        self.logger.info('Evaluate total samples {}'.format(total_samples))
        
        self.model.eval()
        predictions = []

        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            with oneflow.no_grad():
                output = self.model(data,mode = 'test')
                if world_size > 1:
                    seg_list = []
                    if 'seg' in output.keys():
                        seg = output['seg']
                        oneflow.comm.all_gather(seg_list, seg)
                        seg = oneflow.cat(seg_list, 0)
                        output['seg'] = seg   
                    else:
                        seg = output['cls']
                        oneflow.comm.all_gather(seg_list, seg)
                        seg = oneflow.concat(seg_list, 0)
                        output['cls'] = seg                      
                    if 'exist' in output:
                        exists = output['exist']
                        exist_list = []
                        oneflow.comm.all_gather(exist_list, exists)
                        exists = oneflow.concat(exist_list, 0)
                        output['exist'] = exists
                    output = self.model._layers.get_lanes(output)  
                else:
                    output = self.model.get_lanes(output)             
                predictions.extend(output)
            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

        out = self.val_loader.dataset.evaluate(predictions, self.cfg.pred_save_dir) 
 

        if out > self.best_metric:
            self.best_metric = out
            self.best_epoch = self.current_epoch + 1
            filename = 'best_epoch_{}.pd'.format(self.current_epoch + 1)
            ckpt_path = os.path.join(self.output_dir, filename)
            self.best_model_list.append(ckpt_path)
            save_checkpoint(self.output_dir,self,filename_tmpl='best_epoch_{}.pd',save_optimizer=False,create_symlink=False)
            if len(self.best_model_list)>1:
                remove_model = self.best_model_list.popleft()
                os.unlink(remove_model)
                self.logger.info(f'remove the previous best model: {remove_model}')
        self.logger.info(f"best accuracy is {self.best_metric}") 
        self.logger.info(f"The epoch of best accuracy is {self.best_epoch}")

        self.model.train()

        self.metric = out

    def resume(self, checkpoint_path):
        """
        resume training
        """
        checkpoint = oneflow.load(checkpoint_path)
        if checkpoint.get('epoch', None) is not None:
            self.start_epoch = checkpoint['epoch']
            self.current_epoch = checkpoint['epoch']
            self.current_iter = (self.start_epoch - 1) * self.iters_per_epoch

        self.load(resume_weight=checkpoint['state_dict'])
#        self.model.set_state_dict(checkpoint['state_dict'])
        self.optimizer.set_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.set_state_dict(checkpoint['lr_scheduler'])

        self.logger.info('Resume training from {} success!'.format(
            checkpoint_path))
        
    def load(self, weight_path=None,resume_weight=None):
        """
        load the weight for inference
        """
        if weight_path is not None:
            self.logger.info('Loading pretrained model from {}'.format(weight_path))
            if os.path.exists(weight_path):
                para_state_dict = oneflow.load(weight_path)
                if 'state_dict' in para_state_dict:
                    para_state_dict = para_state_dict['state_dict']
        elif resume_weight is not None:
            para_state_dict = resume_weight
        else:
            raise ValueError('The weight is not invalid')

        model_state_dict = self.model.state_dict()
        keys = model_state_dict.keys()
        num_params_loaded = 0
        for k in keys:
            if k == "heads.prior_ys" and para_state_dict[k].dtype == oneflow.float64:
                para_state_dict[k] = para_state_dict[k].astype("float32")
            if k not in para_state_dict:
                self.logger.warning("{} is not in pretrained model".format(k))
            elif list(para_state_dict[k].shape) != list(model_state_dict[k]
                                                        .shape):
                self.logger.warning(
                    "[SKIP] Shape of pretrained params {} doesn't match.(Pretrained: {}, Actual: {})"
                    .format(k, para_state_dict[k].shape, model_state_dict[k]
                            .shape))
            else:
                model_state_dict[k] = para_state_dict[k]
                num_params_loaded += 1
        self.model.set_dict(model_state_dict)
        self.logger.info("There are {}/{} variables loaded into {}.".format(
            num_params_loaded,
            len(model_state_dict), self.model.__class__.__name__))