import torch
import torch.nn as nn

from espnet2.kws.loss.abs_loss import AbsLoss

class FrameCrossEntropyLoss(AbsLoss):
    
    def __init__(self, ):
        super().__init__()
    
    def forward(self, f_logits, f_aligns, f_keywords, f_keyword_stamps):
        """[summary]

        Args:
            f_logits ([type]): [description]
            f_aligns ([type]): [description]
            f_keywords ([type]): [description]
            f_keyword_stamps ([type]): [description]
            
        Return:

        """
        loss = 0
        stats = {}
        
        stats[self.name+'_acc'] = 100
        return loss, stats
    