from abc import ABC
import torch

class AbsLoss(torch.nn.Module, ABC):
    
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def forward(self, ):
        pass