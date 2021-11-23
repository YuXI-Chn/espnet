from abc import ABC 
from abc import abstractmethod
from typing import Tuple

import torch


class AbsIntraEncoder(torch.nn.Module, ABC):
    """Inter represents Interantion, which means using audio and text info interactively.

    Args:
        torch ([type]): [description]
        ABC ([type]): [description]

    Raises:
        NotImplementedError: [description]
        NotImplementedError: [description]
    """
        
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

    def make_pad_mask_2d(self, length_a, length_b):
        assert len(length_a) == len(length_b)
        batch_size = len(length_a)
        max_len_a, max_len_b = max(length_a), max(length_b)
        mask = torch.ones(batch_size, max_len_a, max_len_b).bool()
        for i in range(batch_size):
            mask[i, :length_a[i], :length_b[i]] = False
        return mask