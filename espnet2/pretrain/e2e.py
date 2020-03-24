from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC

from espnet2.pretrain.encoder.abs_encoder import AbsEncoder
from espnet2.pretrain.decoder.abs_decoder import AbsDecoder
from espnet2.pretrain.frontend.abs_frontend import AbsFrontend
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_e2e import AbsE2E

from espnet2.utils.pretrain_utils import process_train_MAM_data



class PRETRAINE2E(AbsE2E):
    """Bert like Encoder  model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        normalize: Optional[AbsNormalize],
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        #ctc: CTC,
        #rnnt_decoder: None,
        #ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = False,
        report_wer: bool = False,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
    ):
        assert check_argument_types()
        #assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        #assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        #self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.normalize = normalize
        self.encoder = encoder
        self.decoder = decoder
        #if ctc_weight == 0.0:
        #    self.ctc = None
        #else:
        #    self.ctc = ctc
        #self.rnnt_decoder = rnnt_decoder
        
        
        # fix places which use this 
        #self.SpecHead = MockingjaySpecPredictionHead(output_dim=256,config=None)
        
        
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )              # figure out this (uk)
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape  # what's .dim() for?
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print(text.shape)
        #print(text[0,:])
        text = text[:, : text_lengths.max()]
        #print(text.shape)
        #print(text[2,:])


        #TO DO: modify code in transformer_encoder, modify code in transformer_decoder

        # 1. Encoder

        


        encoder_out, encoder_out_lens, spectrogram_out_lens,text_out_lens,feats_mask, text_mask,feats_gold,text_gold = self.encode(speech, speech_lengths,text,text_lengths)





        ## 2a. Attention-decoder branch
        #if self.ctc_weight == 1.0:
        #    loss_att, acc_att, cer_att, wer_att = None, None, None, None
        #else:
        #    loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
        #        encoder_out, encoder_out_lens, text, text_lengths
        #    )

        # 2b. CTC branch
        #if self.ctc_weight == 0.0:
        #    loss_ctc, cer_ctc = None, None
        #else:
        #    loss_ctc, cer_ctc = self._calc_ctc_loss(
        #        encoder_out, encoder_out_lens, text, text_lengths
        #    )

        # 2c. RNN-T branch
        #if self.rnnt_decoder is not None:
        #    _ = self._calc_rnnt_loss(encoder_out, encoder_out_lens, text, text_lengths)

        #if self.ctc_weight == 0.0:
        #    loss = loss_att
        #elif self.ctc_weight == 1.0:
        #    loss = loss_ctc
        #else:
        #    loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        # compute loss here

        loss = self._calc_reconstruction_loss(encoder_out,encoder_out_lens,spectrogram_out_lens, text_out_lens, feats_mask, text_mask, feats_gold, text_gold)


        stats = dict(
            loss=loss.detach(),
            #loss_att=loss_att.detach() if loss_att is not None else None,
            #loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            #acc=acc_att,
            #cer=cer_att,
            #wer=wer_att,
            #cer_ctc=cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
            self, speech: torch.Tensor, speech_lengths: torch.Tensor,text:torch.Tensor, text_lengths:torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_decode.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch,Length,...)
            text_lengths: (Batch, )
        """
        # 1. Extract feats


        # change this if you want to other features, cqcc
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)

        # 2. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        if self.normalize is not None:
            feats, feats_lengths = self.normalize(feats, feats_lengths)

        # 3. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        text = torch.unsqueeze(text,2)
        
        feats_gold = feats
        text_gold = text

        #print(text.shape)
        #print(feats.shape)
        #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        feats_masked,feats_mask = process_train_MAM_data(feats,20,config=None) # consecutive length is 20
        text_masked,text_mask = process_train_MAM_data(text,1,config=None) # TO DO: modify the function to fit text
        feats_mask = feats_mask.cuda()
        text_mask = text_mask.cuda()
        #text_masked = torch.squeeze(text_masked,-1)
        #text_mask = torch.squeeze(text_mask,-1)
        #text_masked, ys_out_pad = add_sos_eos(text_masked, self.sos, self.eos, self.ignore_id)
        # keep the vocab size same for text embedding layer
        #print(feats_mask.shape)
        #print(text_mask.shape)
        #print(text_masked.shape)
        #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        encoder_out, encoder_out_lens,speech_out_lens,text_out_lens = self.encoder(feats_masked, feats_lengths, text_masked,text_lengths)
        
        #print(f"{speech_out_lens}{speech_out_lens.shape}")
        #assert speech_out_lens + text_out_lens == encoder_out_lens
        

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        #AssertionError: (torch.Size([32, 284, 256]), tensor(283)), it's very wired
        #assert encoder_out.size(1) <= encoder_out_lens.max(), (
        #    encoder_out.size(),
        #    encoder_out_lens.max(),
        #)


        
        return encoder_out, encoder_out_lens, speech_out_lens, text_out_lens, feats_mask, text_mask, feats_gold, text_gold

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_reconstruction_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        feats_out_lens,
        text_out_lens, #useless for now 
        feats_mask,
        text_mask,
        feats_gold,
        text_gold,
        
    ):  
        speech_len = feats_gold.shape[1]
        text_len = text_gold.shape[1]
        assert speech_len + text_len == encoder_out.shape[1]


        # by Yuekai, TO DO: implemented this 
        pred_feats,pred_text = self.decoder(encoder_out,encoder_out_lens,speech_len,text_len)   
        loss = torch.nn.L1Loss()
        masked_spec_loss = loss(pred_feats.masked_select(feats_mask),
                feats_gold.masked_select(feats_mask))
         
        #print(pred_text.masked_select(text_mask).dtype)
        #print(text_gold.masked_select(text_mask).to(torch.float32).dtype)
        masked_text_loss = loss(pred_text.masked_select(text_mask),
                text_gold.masked_select(text_mask).to(torch.float32))
        
        print(f"The current spec_loss:{masked_spec_loss}, text_loss: {masked_text_loss}")
        return masked_spec_loss + 0.1 * masked_text_loss  # TO DO: tune this !!



    
    
    
    
    
    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        raise NotImplementedError
