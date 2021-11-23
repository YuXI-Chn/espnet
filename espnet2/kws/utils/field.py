from typing import Optional
import torch

class Field():
    """An abstract definition for data structure describing a batched tensor of variable lengths

    Args:
    ¦   tensor (Tensor): the main payload
    ¦   length (LongTensor, optional): the length for each utterance

    """
    
    def __init__(self, tensor: torch.tensor, length: Optional[torch.tensor] = None):
        self.tensor = tensor
        self.length = length
        
    def __repr__(self,):
        return f"Field(tensor={self.tensor}, length={self.length})"
    
    def to(self, *args, **kwargs):
        """convert the tensor to the desired type.
        
        """
        return Field(self.tensor.to(*args, **kwargs), self.length)