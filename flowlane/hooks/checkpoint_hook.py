import os
import copy
import oneflow as flow

from .hook import Hook
from .builder import HOOKS

def is_main_process():
    return flow.env.get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        flow.save(*args, **kwargs)

def save_checkpoint(out_dir,
                    trainer,
                    filename_tmpl='epoch_{}.pth',
                    save_optimizer=True,
                    create_symlink=True):
    filename = filename_tmpl.format(trainer.current_epoch + 1)
    filepath = os.path.join(out_dir, filename)
    optimizer = trainer.optimizer if save_optimizer else None
    lr_scheduler = trainer.lr_scheduler
    use_ema = trainer.use_ema
    if use_ema:
        model_weights = copy.deepcopy(trainer.model.state_dict())
        trainer.model.set_dict(trainer.ema.apply())
    else:
        model_weights = trainer.model.state_dict()

    if optimizer is not None:
        save_on_master({
            'epoch': trainer.current_epoch + 1,
            'state_dict': model_weights,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'ema_model':trainer.model.state_dict() if use_ema else None
        }, filepath)
    else:
        save_on_master({
            'epoch': trainer.current_epoch + 1,
            'state_dict': model_weights,
            'ema_model':trainer.model.state_dict() if use_ema else None
        }, filepath)        
    # in some environments, `os.symlink` is not supported, you may need to
    # set `create_symlink` to False
    if create_symlink:
        latest = os.path.join(out_dir, 'latest.pth')
        os.system('rm -rf %s' % latest)
        os.symlink(filename, latest)
    
@HOOKS.register()
class CheckpointHook(Hook):
    """Save checkpoints periodically.
    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``trainer.work_dir`` will be used by default.
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
    """

    def __init__(self,
                 interval=1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=5,
                 priority=1,
                 **kwargs):
        self.priority = priority
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.max_keep_ckpts = max_keep_ckpts
        self.out_dir = out_dir
        self.args = kwargs
        self.metric = 0.

    def train_epoch_end(self, trainer):
        if flow.env.get_rank() !=0:
            return

        if not self.by_epoch or not self.every_n_epochs(trainer, self.interval):
            return

        trainer.logger.info(
            f'Saving checkpoint at {trainer.current_epoch + 1} epochs')
        if not self.out_dir:
            self.out_dir = trainer.output_dir
        save_checkpoint(
            self.out_dir,
            trainer,
            save_optimizer=self.save_optimizer,
            **self.args)
        
        # remove other checkpoints
        filename_tmpl = self.args.get('filename_tmpl', 'epoch_{}.pth')
        ckpt_path = os.path.join(self.out_dir,
                                    filename_tmpl.format(trainer.current_epoch + 1))
        trainer.save_model_list.append(ckpt_path)
        if len(trainer.save_model_list) > self.max_keep_ckpts > 0:
            model_to_remove = trainer.save_model_list.popleft()
            os.unlink(model_to_remove)

    def train_iter_end(self, trainer):
        if flow.env.get_rank() !=0:
            return

        if self.by_epoch or not self.every_n_iters(trainer, self.interval):
            return

        trainer.logger.info(
            f'Saving checkpoint at {trainer.iter + 1} iterations')
        if not self.out_dir:
            self.out_dir = trainer.output_dir
        save_checkpoint(
            self.out_dir, trainer, save_optimizer=self.save_optimizer, **self.args)

        # remove other checkpoints
        filename_tmpl = self.args.get('filename_tmpl', 'epoch_{}.pth')
        ckpt_path = os.path.join(self.out_dir,
                                    filename_tmpl.format(trainer.current_epoch + 1))
        trainer.save_model_list.append(ckpt_path)
        if len(trainer.save_model_list) > self.max_keep_ckpts > 0:
            model_to_remove = trainer.save_model_list.popleft()
            os.unlink(model_to_remove)