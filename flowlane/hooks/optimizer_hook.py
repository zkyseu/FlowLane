from .hook import Hook
from .builder import HOOKS
from ..solver import build_optimizer

@HOOKS.register()
class OptimizerHook(Hook):
    def __init__(self, priority=1):
        self.priority = priority
        
    def train_iter_end(self, trainer):
        if 'Lars' in trainer.cfg['optimizer']['name']:
            trainer.optimizer.clear_gradients()
        else:
            trainer.optimizer.clear_grad()

        loss = 0
        loss = trainer.outputs['loss']
        
        if trainer.use_amp:
            scaled_loss = trainer.scaler.scale(loss)
            scaled_loss.backward()
            if 'lars' in trainer.optimizer.type:
                trainer.scaler.minimize(trainer.optimizer, scaled_loss)
            else:
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
        else:
            loss.backward()
            if 'lars' in trainer.optimizer.type:
                trainer.optimizer.minimize(loss)
            else:
                trainer.optimizer.step()

        if 'loss' not in trainer.outputs:
            trainer.outputs['loss'] = loss