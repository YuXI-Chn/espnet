
from abc import ABC 
from abc import abstractmethod
from typing import Tuple

import torch


class AbsTextEncoder(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, 
        xs: torch.tensor,
        xlens: torch.tensor,
        ys: torch.tensor, 
        ylens: torch.tensor,
    ) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def output_dim(self, ) -> int:
        raise NotImplementedError
        