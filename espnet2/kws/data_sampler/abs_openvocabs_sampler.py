from abc import ABC 
from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet2.kws.utils.field import Field

class AbsOpenVocabsSampler(torch.nn.Module, ABC):
    
    def __init__(self, sil_index: int = 0, pad_index: int = -1, ntokens: Optional[int] = None):
        self.sil = sil_index
        self.pad = pad_index
        if ntokens is None:
            raise ValueError('ntoken must be assigned.')
        self.ntokens = ntokens
    
    @abstractmethod
    def forward(
        self, 
        aligns: torch.tensor,
        align_lengths: torch.tensor,
        text: torch.tensor, 
        text_lengths: torch.tensor,
    ) -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError
    
    def sequence_padding(self, ):
        """ used to pad each keyword, for data-parallel."""
        pass
    
    @abstractmethod
    def batch_padding(self, ):
        raise NotImplementedError

    @abstractmethod
    def sampler(self, ):
        return NotImplementedError

    def online_phn_stamp(self, f_aligns: Field):
        """aligns: (B, Tmax)
           align_lengths: (B, )
           
           return: (1) time stamp for each word for each utt.
                   (2) token sequence (B, Lmax).
                   (3) token lengths. (B, )
        """
        aligns, align_lengths = f_aligns.tensor, f_aligns.length
        
        ret_stamp_list, ret_len_list, ret_seq_list = [], [], []
        batch_size = aligns.shape[0]
        i = 0
        while i < batch_size:
            stamp_list, stamp_nosil_list = [], []
            i_align = aligns[i, 0: align_lengths[i]]
            start_stamp, cur_token = 0, i_align[0]
            
            for index, token in enumerate(i_align[1:], 1):
                if token != cur_token:
                    stamp_list.append([cur_token, start_stamp, index-1])
                    start_stamp, cur_token = index, token
            stamp_list.append([i_align[-1], start_stamp, len(i_align)-1])

            seq = []
            for token, *stamp in stamp_list:
                if token != self.sil:
                    stamp_nosil_list.append(stamp)
                    seq.append(token)

            ret_stamp_list.append(
                    torch.tensor(stamp_nosil_list)
                )

            ret_seq_list.append(
                    torch.tensor(seq, dtype=torch.long)
                )

            ret_len_list.append(
                    len(stamp_nosil_list)
                )
            i += 1
        
        f_stamp = Field(
                    pad_list(xs = ret_stamp_list, pad_value = self.pad),
                    torch.tensor(ret_len_list, dtype=torch.long)
                )
        
        f_seq = Field(
                    pad_list(xs = ret_seq_list, pad_value=self.pad),
                    torch.tensor(ret_len_list, dtype=torch.long)
        )
        return f_stamp, f_seq