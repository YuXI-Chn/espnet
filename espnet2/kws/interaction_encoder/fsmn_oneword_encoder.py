import math

import numpy as np
import torch
import torch.nn as nn

from fsmn.dfsmn import DFSMNLayer
from espnet2.kws.interaction_encoder.abs_intra_encoder import AbsIntraEncoder
from espnet2.kws.utils.field import Field


class FSMNOneWordEncoder(AbsIntraEncoder):
    """FSMN + torch.embedding for single wake word."""
    def __init__(self, ninp, nhid, nproj, ntokens, ntinp, natt, dropout_rate, 
                 skip='res', nlayer=5, ndnn=0, lo=3, ro=0, ls=1, rs=1, kernel_res=True,
                 activation='relu', cross_layer = '-1',
                ):
        """cross_layer: which layers to exchange text and acoustic info. 
                        -1 means all layers, and 0 ~ nalyer-1 means the specific layer.
                        If you want to specify some layers to exchange info, you can 
                        split the number of layers by <space>, like cross_layer = "0 2 4".
                        If you want to make the model exchange info in each layer,
                        just specify cross_layer='-1'
        """
        super().init()
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError(f"{activation} is not supported now.")
        assert nlayer > 0 and ndnn >= 0, \
                                    f"{nlayer} must > 0 and {ndnn} must be >= 0"
        self.cross_layer = sorted(
                                  [int(i) for i in cross_layer.strip().split()])
        if -1 in self.cross_layer:
            assert len(self.cross_layer) == 1, \
                                f"the indexes of information exchanging layers are conflict!"
        else:
            assert max(self.cross_layer) < nlayer, \
                                f"the max index of information exchanging layer must smaller than nalyer: {nlayer}"
        kwargs = dict(
            ninp = ninp,
            nhid = nhid,
            nproj = nproj,
            tinp = ntinp,
            att_dim = natt,
            dp_rate = dropout_rate,
            skip = skip,
            lo = lo,
            ro = ro,
            ls = ls,
            rs = rs,
            kernel_res = kernel_res,
            activation = activation,
        )
        
        if -1 in self.cross_layer:
            model_list = [
                FSMNAttBlock(ninp = nproj if i > 0 else ninp, use_cross=True, **kwargs) if i == 0 else  
                FSMNAttBlock(ninp = nproj if i > 0 else ninp, use_cross=True, **kwargs) for i in range(0, nlayer)
            ]
        else:
            model_list = [
                FSMNAttBlock(ninp = ninp, use_cross=True, **kwargs) if 0 in self.cross_layer else
                FSMNAttBlock(ninp = ninp, use_cross=False, **kwargs)
            ]
            for i in range(1, nlayer):
                if i in self.cross_layer:
                    model_list.append(
                        FSMNAttBlock(ninp = nproj, use_cross=True, **kwargs)
                    )
                else:
                    model_list.append(
                        FSMNAttBlock(ninp = nproj, use_cross=False, **kwargs)
                    )
        self.model = nn.ModuleList(model_list)
    
        if ndnn > 0:
            output_in = nhid
            dnns = [nn.Linear(nproj, nhid), self.activation]
            for _ in range(ndnn - 1):
                dnns.extend([nn.Linear(nhid, nhid), self.activation])
        self.dnns = nn.Sequential(*dnns)
        
        output_in = nproj if ndnn == 0 else nhid
        self.classifer = nn.Linear(output_in, ntokens)

    def forward(self, f_speech, f_keywords):
        mask = self.make_pad_mask_2d(f_speech.length, f_keywords.length)
        
        f_embed = f_speech
        for layer in self.model:
            f_embed = layer(f_embed, f_keywords, mask)
        logits = self.classifer(self.dnns(f_embed.tensor))

        return Field(logits, f_embed.length)


class FSMNAttBlock(nn.Module):
    def __init__(self, use_cross: bool, ninp: int, nhid: int, nproj: int, ntinp: int,
                 natt: int, dp_rate: float, skip: str, lo: int, ro: int, ls: int, rs: int,
                 kernel_res: bool, activation: str, 
                 ):
        super().init()
        if use_cross:
            self.block = AttenBlock(
                ninp=ninp, nhid=nhid, nproj=nproj, ntinp=ntinp, natt=natt, dp_rate=dp_rate,
                skip=skip, lo=lo, ro=ro, ls=ls, rs=rs,
                kernel_res=kernel_res, activation=activation
            )
        else:
            self.block = DFSMNLayer(
                input_dim=ninp, hidden_dim=nhid, proj_dim=nproj,
                l_order=lo, r_order=ro, l_stride=ls, r_stride=rs, 
                skip_connection=skip, activation=activation, 
                kernel_res=kernel_res
            )
        self.use_cross = use_cross
        
    def forward(self, f_speech: Field, f_kws: Field, mask):
        if self.use_cross:
            return Field(self.block(f_speech, f_kws, mask), f_speech.length)
        else:
            return Field(self.block(f_speech.tensor, f_speech.length), f_speech.length)


class AttenBlock(nn.Module):
    def __init__(self, ninp: int, nhid:int, nproj: int, ntinp: int, natt: int,
                 dp_rate: float, skip: str, lo:int, ro:int, ls: int, rs:int,
                 kernel_res: bool, activation: str):
        super().init()
        self.fsmn = DFSMNLayer(
            input_dim = ninp,
            hidden_dim = nhid,
            proj_dim = nproj,
            l_order = lo,
            r_roder = ro,
            l_stride = ls,
            r_stride = rs,
            skip_connection=skip,
            activation = activation,
            kernel_res = kernel_res,
        )
        self.natt = natt
        self.dropout = nn.Dropout(p = dp_rate)
        self.query = nn.Linear(nproj, natt)
        self.key = nn.Linear(ntinp, natt)
        self.value = nn.Linear(ntinp, natt)
        self.linear_out = nn.Linear(natt, nproj)
        if activation == 'relu':
            self.activation = nn.ReLU()
        
    def forward(self, f_speech: Field, f_kws: Field, mask):
        audio = self.fsmn(f_speech.tensor, f_speech.length)
        q = self.query(audio)
        k = self.key(f_kws.tensor)
        v = self.key(f_kws.tensor)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.natt)
        fused_text = self.forward_attention(v, scores, mask)
        audio_text = audio + fused_text
        
        ff_result = self.linear_out(audio_text)
        ff_result = self.activation(ff_result)
        result = ff_result + audio_text
        return result
        
    def forward_attention(self, value, scores, mask):
        if mask is not None:
            min_value = float(
                np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            att_weight = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            att_weight = torch.softmax(scores, dim=-1)
        dp_att = self.dropout(att_weight)
        return torch.matmul(dp_att, value)
