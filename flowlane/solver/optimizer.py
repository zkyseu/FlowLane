import oneflow
import oneflow as flow

from .builder import OPTIMIZERS

OPTIMIZERS.register(oneflow.optim.Adam)
OPTIMIZERS.register(oneflow.optim.AdamW)
OPTIMIZERS.register(oneflow.optim.SGD)
OPTIMIZERS.register(oneflow.optim.RMSprop)