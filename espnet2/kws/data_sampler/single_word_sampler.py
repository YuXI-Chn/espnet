import random
from typing import Optional
import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet2.kws.utils.field import Field
from espnet2.kws.data_sampler.abs_openvocabs_sampler import AbsOpenVocabsSampler

class SingleWordSampler(AbsOpenVocabsSampler):
    def __init__(self, 
                 ntokens: int,
                 ndim: int,
                 min_length: int = 3,
                 max_length: int = 9,
                 use_word_boundary: bool = False,
                 use_filler_node: bool = False,
                 filler_index: int = -1,
                 use_neg_samples: bool = False,
                 neg_prob: float = 0,
                ):
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length
        self.use_word_boundary = use_word_boundary
        self.use_filler_node = use_filler_node
        self.filler = filler_index
        self.use_neg_samples = use_neg_samples
        self.neg_prob = neg_prob
        if not use_filler_node:
            assert self.filler < 0, \
                    f"filler_index: {filler_index} > 0 only if use filler node."
        if use_neg_samples:
            assert 0 <= neg_prob <= 1, \
                    f"the prob of sampling neg samples must between 0 and 1, but get {neg_prob}"
            self.neg_prob = neg_prob
            
        # self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        #           use_filler  unuse_filler
        # tokens    10(0-9)     10(0-9)
        # filler    10          x
        # padding   11          10
        # num_emds  12          1
        self.embed = nn.Embedding(
            num_embeddings = ntokens+2 if use_filler_node else ntokens+1,
            embedding_dim = ndim,
            padding_idx = ntokens+1 if use_filler_node else ntokens,
        )
        
    def forward(self,
                f_aligns: Field,
                f_boundary: Optional[Field] = None):
        """aligns: (B, ALmax)
           align_lengths: (B, )
           texts: (B, TLmax)
           text_lengths: (B, )

           return: keyword index (B, Lmax)
        """
        
        f_stamp, f_seq = self.online_phn_stamp(f_aligns=f_aligns)
        
        #TODO 采样: 是否考虑单词边界, 是否考虑采样负例.
        f_keywords, f_keywords_stamp = self.naive_sampler(f_stamp=f_stamp,
                                                f_token=f_seq,
                                                f_boundary=f_boundary,
                                                )
        
        keyword_embeds = self.embed(f_keywords.tensor)
        f_keywords = Field(tensor=keyword_embeds, length=f_keywords.length)
        return f_keywords, f_keywords_stamp


    def naive_sampler(self, 
                f_stamp: Field, 
                f_token: Field, 
                f_boundary: Optional[Field] = None):
        """
        length <= self.min_length: pick all
        self.min_length < length < self.max_length: 
        length >= self.max_length: 
        
        """
        token_stamp = f_stamp.tensor
        token_lengths = f_stamp.length
        token_seq = f_token.tensor
        
        keywords, keywords_lengths, keywords_stamp = [], [], []
        if not self.use_word_boundary:
            for i, length in enumerate(token_lengths):
                if length >= self.max_length:
                    kws_length = random.randint(self.min_length, self.max_length)
                    kws_start = random.randint(0, length-self.max_length)
                    kws_end = kws_start + kws_length - 1
                elif self.min_length < length < self.max_length:
                    kws_length = min(random.randint(self.min_length, self.max_length), length)
                    kws_start = 0
                    kws_end = kws_length -1 
                else:
                    kws_start = 0
                    kws_end = token_lengths[i] - 1
                    
                if self.use_filler_node:
                    keywords.append(torch.cat(
                                    (token_seq[i][kws_start:kws_end+1], 
                                     torch.tensor([self.filler], dtype=torch.long))
                                    ), dim=0,
                                )
                else:
                    keywords.append(token_seq[i][kws_start:kws_end+1])
                
                keywords_lengths.append(kws_end-kws_start+1)    
                start_stamp = token_stamp[i][kws_start][0].item()
                end_stamp = token_stamp[i][kws_end][1].item()
                keywords_stamp.append([start_stamp, end_stamp]) # B, 2
        else:
            raise NotImplementedError
        
        f_keywords = Field(
            tensor = pad_list(
                xs=keywords, pad_value=self.filler+1 if self.use_filler_node else self.tokens
                ),
            length = torch.tensor(keywords_lengths, dtype=torch.long)
        )
        # not use filler: if ntokens = 412, means 412 modeling units. 
        #                 the indexes of them are 0~411, so the padding_value is ntokens=412,
        # use filler: the indexes of modeling units are also 0~411, and the filler is 412, so 
        #              the padding value is filler+1 = 413(which is also equal to ntokens+2)
        
        f_keyowrds_stmap = Field(
            tensor = torch.tensor(keywords_stamp),
            length = torch.tensor([2] * len(keywords_stamp, dtype=torch.long))
        )
        return f_keywords, f_keyowrds_stmap