from oneflow.optim.lr_scheduler import LinearLR
from oneflow.optim.lr_scheduler import CosineAnnealingLR
from oneflow.optim.lr_scheduler import MultiStepLR
from oneflow.optim.lr_scheduler import PolynomialLR
from .builder import LRSCHEDULERS

LRSCHEDULERS.register(CosineAnnealingLR)
LRSCHEDULERS.register(LinearLR)
LRSCHEDULERS.register(MultiStepLR)
LRSCHEDULERS.register(PolynomialLR)