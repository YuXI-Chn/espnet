import torch
import torch.nn as nn
from espnet2.kws.loss.abs_loss import AbsLoss

class SingleFrameDetectionLoss(AbsLoss):
    
    def __init__(self, name):
        self.name = name
    
    def forward(self, ):
        pass