import argparse
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.kws.espnet_model import ESPnetKwsModel
from espnet2.kws.data_sampler.abs_openvocabs_sampler import AbsOpenVocabsSampler
from espnet2.kws.data_sampler.single_word_sampler import SingleWordSampler
from espnet2.kws.interaction_encoder.abs_intra_encoder import AbsIntraEncoder
from espnet2.kws.interaction_encoder.fsmn_oneword_encoder import FSMNOneWordEncoder
from espnet2.kws.loss.single_frame_cross_entropy import FrameCrossEntropyLoss
from espnet2.kws.loss.single_weighted_frame_ce import SingleWeightedFrameCrossEntropyLoss
from espnet2.kws.loss.single_frame_detection import SingleFrameDetectionLoss
from espnet2.kws.loss.single_seq_detection import SingleSeqDetectionLoss

from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none
from espnet2.tasks.abs_task import AbsTask
from espnet2.train.class_choices import ClassChoices

data_sampler_choices = ClassChoices(
    name='data_sampler',
    classes=dict(
        singleword=SingleWordSampler,
    ),
    type_check = AbsOpenVocabsSampler,
    default = '',
)
intra_encoder_choices = ClassChoices(
    name="intra_encoder",
    classes=dict(
        fsmn_oneWord=FSMNOneWordEncoder,
    ),
    type_check = AbsIntraEncoder,
    default='fsmn_oneWord',
)

audio_encoder_choices = ClassChoices(
    name="audio_encoder",
    classes=dict(),
    default='',
)

text_encoder_choices = ClassChoices(
    name="text_encoder",
    classes=dict(),
    default='',
)

# TODO Complete this part
loss_choices = ClassChoices(
    name="loss",
    classes=dict(
        frameCE = FrameCrossEntropyLoss,
        wFrameCE = SingleWeightedFrameCrossEntropyLoss,
        frameDet = SingleFrameDetectionLoss,
        seqDet = SingleSeqDetectionLoss,
    ),
    default = 'frameCE',
)

class KwsTask(AbsTask):
    num_optimizers: int = 1

    encoder_choices_list = [
        # intra_encoder and --intra_encoder_conf
        intra_encoder_choices,
        # --audio_encoder and --audio_encoder_conf
        audio_encoder_choices,
        # --text_encoder_choices and --text_encoder_conf
        text_encoder_choices,
    ]
    
    all_loss_list = [ 
        'frameCE',
        'wFrameCe',
        'frameDet',
        'seqDet',
    ]
    
    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="KWS Task related")
        
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer", 
                "xavier_uniform", 
                "xavier_normal", 
                "kaiming_uniform", 
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetKwsModel),
            help="The keyword arguments for model class.",
        )
        
        group = parser.add_argument_group(description="Preprocess related")
        
        group.add_argument(
            "--usr_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
        )
        
        for class_choices in cls.class_choices_list:
            class_choices.add_arguments(group)
            
    @classmethod
    def build_collate_fn(cls, 
                         args: argparse.Namespace,
                         train: bool
                        ) -> Callable[
                             [Collection[Tuple[str, Dict[str, np.ndarray]]]],
                             Tuple[List[str], Dict[str, torch.Tensor]],]:
                            
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)
    
    @classmethod
    def required_data_names(cls, 
                            train: bool = True,
                            inference: bool = False
                            ) -> Tuple[str, ...]:
        assert train and inference == False, \
                     f"bool 'train' and 'inference' must be different"
        if train:
            retval = ('speech', 'transcript_text')
        else:
            # detection mode
            retval = ('speech', 'keyword_text')
        return retval
    
    @classmethod
    def optional_data_names(cls, 
                            train: bool = True, 
                            inference: bool = False
                            ) -> Tuple[str, ...]:
        retval = ()
        assert check_return_type(retval)
        return retval
    
    @classmethod
    def build_model(cls,
                    args: argparse.Namespace,
                    ) -> ESPnetKwsModel:
        assert check_argument_types()
        
        # sampler
        data_sampler = data_sampler_choices.get_class(
                                        args.data_sampler)(**args.data_sampler_conf)
        # model arch
        encoder_type = args.encoder_type
        
        assert encoder_type in ("intra", "inter", "single"), \
                            f"encoder_type: {encoder_type} is not supported."
                            
        if encoder_type == "intra":
            audio_encoder, text_encoder = None, None
            intra_encoder = intra_encoder_choices.get_class(
                                            args.intra_encoder)(**args.intra_encoder_conf)
        elif encoder_type == "inter":
            intra_encoder = None
            audio_encoder = audio_encoder_choices.get_class(
                                            args.audio_encoder)(**args.audio_encoder_conf)
            text_encoder = text_encoder_choices.get_class(
                                            args.text_encoder)(**args.text_encoder_conf)
        elif encoder_type == "single":
            intra_encoder, text_encoder = None, None
            audio_encoder = audio_encoder_choices.get_class(
                                            args.audio_encoder)(**args.audio_encoder_conf)
        else:
            pass
        
        # loss type
        losses = args.loss_types.strip().split()
        loss_weights = [float(x) for x in args.loss_weights.strip().split()]
        assert sum(loss_weights) == 1 and len(loss_weights) == len(losses), \
                                     f"check the number or the sum of loss_weights"
        loss_modules = {}
        for i, loss in enumerate(losses):
            assert loss in cls.all_loss_list, f"the loss type: {loss} is not supported now."
            loss_modules[loss] = (loss_choices.get_class(loss)(**args.loss_confs[loss]), loss_weights[i])
            
        model = ESPnetKwsModel(encoder_type=encoder_type,
                               data_sampler=data_sampler,
                               intra_encoder=intra_encoder,
                               audio_encoder=audio_encoder,
                               text_encoder=text_encoder,
                               loss_modules=loss_modules
                               **args.model_conf
                               )
        if args.init is not None:
            initialize(model, args.init)
        assert check_return_type(model)
        return model