from .hook import Hook
from .builder import HOOKS

@HOOKS.register()
class EvaluateHook(Hook):
    def __init__(self, init_eval=False, eval_kargs=None, priority=1):
        if eval_kargs is None:
            self.eval_kargs = {}
        else:
            self.eval_kargs = eval_kargs

        self.init_eval = init_eval
        self.priority = priority

    def run_begin(self, trainer):
        if self.init_eval:
            trainer.val(**self.eval_kargs)

    def train_epoch_end(self, trainer):
        trainer.val(**self.eval_kargs)