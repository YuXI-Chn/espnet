from contextlib import contextmanager
from distutils.version import LooseVersion
from itertools import permutations
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.kws.data_sampler.abs_openvocabs_sampler import AbsOpenVocabsSampler
from espnet2.kws.interaction_encoder.abs_intra_encoder import AbsIntraEncoder
from espnet2.kws.audio_encoder.abs_audio_encoder import AbsAudioEncoder
from espnet2.kws.text_encoder.abs_text_encoder import AbsTextEncoder
from espnet2.kws.utils.field import Field
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


            
class ESPnetKwsModel(AbsESPnetModel):
    """keyword spotting or wake word detection model"""
    
    def __init__(self, 
                 encoder_type: str = "intra",
                 data_sampler: AbsOpenVocabsSampler = None,
                 intra_encoder: AbsIntraEncoder = None,
                 audio_encoder: AbsAudioEncoder = None,
                 text_encoder: AbsTextEncoder = None,
                 loss_modules: Dict = None,
                 ):
        """loss_module is a Dict which includes all losses.
        
           e.g.:
           loss_module = {'frameCE': (frameCELossModule, 0.7), 
                          'seqDet' : (seqDetLossModule, 0.3),}
           
           final_loss = 0.7 * frameCE + 0.3 * seqDet
        """
        assert check_argument_types()
        super().__init__()
        self.data_sampler = data_sampler
        self.intra_encoder = intra_encoder
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.encoder_type = encoder_type
        self.loss_modules = loss_modules
        
    def forward(self, 
                speech: torch.Tensor,
                speech_lengths: torch.tensor,
                text: torch.Tensor,
                text_lengths: torch.tensor,
                ) -> Tuple[torch.tensor, Dict[str, torch.Tensor], torch.tensor]:
        """Two paths to build the computational graphs:
                    1. use intra encoder, which means extract audio and text info interactively.
                    2. use audio encoder or audio+text encoders, which means extract audio
                       and text info respectively.

            Args:
                speech: (B, Tmax, ...)
                speech_text: (B, )
                text: (B, Lmax, )
                text_length: (B, )
        """

        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), f"Error: mismatch in the dim batch, check the input data." \
           f"speech shape: {speech.shape}" \
           f"speech_lens shape: {speech_lengths.shape}" \
           f"text shape: {text.shape}" \
           f"text_lens shape: {text_lengths.shape}"
        
        batch_size = speech.shape[0]
        
        f_aligns = Field(
            tensor = text[:, : text_lengths.max()], 
            length = text_lengths
        )
        f_boundary = None
        
        f_keywords, f_keyword_stamps = self.data_sampler(f_aligns, f_boundary)
        f_speech = Field(tensor=speech, length=speech_lengths.long())
       
        if self.encoder_type == "intra":
            f_logits = self.intra_encoder(f_speech, f_keywords)
        elif self.encoder_type == "inter":
            # TODO the params for text_encoder should be confirmed.
            # kws_embeddings = self.text_encoder(f_keywords)
            # sph_embeddings = self.audio_encoder(f_speech)
            pass
        elif self.encoder_type == "single":
            pass
        
        # loss part
        # {"frameCE": (FrameCELOSS, 0.5)}
        loss = 0
        loss_stats, other_stats = {}, {}
        for loss_name, loss_module in self.loss_modules.items():
            loss_model, weight = loss_module
            tmp_loss, tmp_stats = loss_model(f_logits, f_aligns, f_keywords, f_keyword_stamps)
            loss += (
                weight * tmp_loss
            )
            loss_stats[loss_name] = tmp_loss.detach()
        other_stats.update(tmp_stats)

        # stats part 
        stats = dict(
            loss = loss.detach(),
        )
        stats = dict(**stats, **loss_stats)
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight


    def collect_feats(self,
                      speech: torch.Tensor,
                      speech_lengths: torch.Tensor,
                      text: torch.Tensor,
                      text_lengths: torch.tensor) -> Dict[str, torch.Tensor]:
        speech = speech[:, : speech_lengths.max()]
        text = text[:, : text_lengths.max()]
        
        return {'feats': speech, 
                'feats_lengths': speech_lengths, 
                'text': text,
                'text_lengths': text_lengths, }