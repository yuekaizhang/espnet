"""Transducer speech recognition model (pytorch)."""

from distutils.util import strtobool
import logging
import math

import chainer
from chainer import reporter
import torch

from espnet.utils.cli_utils import strtobool

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.attentions import att_for

#from espnet.nets.pytorch_backend.rnn.encoders import encoder_for
#from espnet.nets.pytorch_backend.tdnn.tdnn import encoder_for
from espnet.nets.pytorch_backend.contextnet.encoder import Encoder


from espnet.nets.pytorch_backend.transducer.initializer import initializer
from espnet.nets.pytorch_backend.transducer.loss import TransLoss
from espnet.nets.pytorch_backend.transducer.rnn_decoders import decoder_for
from espnet.nets.pytorch_backend.transducer.transformer_decoder import Decoder
from espnet.nets.pytorch_backend.transducer.utils import prepare_loss_inputs
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
#from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.mask import target_mask


class Reporter(chainer.Chain):
    """A chainer reporter wrapper for transducer models."""

    def report(self, loss, cer, wer):
        """Instantiate reporter attributes."""
        reporter.report({"cer": cer}, self)
        reporter.report({"wer": wer}, self)
        reporter.report({"loss": loss}, self)

        logging.info("loss:" + str(loss))


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    Args:
        idim (int): dimension of inputs
        odim (int): dimension of outputs
        args (Namespace): argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Extend arguments for transducer models.

        Both Transformer and RNN modules are supported.
        General options encapsulate both modules options.

        """
        group = parser.add_argument_group("context model setting")

        # Encoder - general
        group.add_argument(
            "--etype",
            default="blstmp",
            type=str,
            choices=[
                "transformer",
                "lstm",
                "blstm",
                "lstmp",
                "blstmp",
                "vgglstmp",
                "vggblstmp",
                "vgglstm",
                "vggblstm",
                "gru",
                "bgru",
                "grup",
                "bgrup",
                "vgggrup",
                "vggbgrup",
                "vgggru",
                "vggbgru",
                "tdnn",
                "contextnet"
            ],
            help="Type of encoder network architecture",
        )
        group.add_argument(
            "--elayers",
            default=4,
            type=int,
            help="Number of encoder layers (for shared recognition part "
            "in multi-speaker asr mode)",
        )
        group.add_argument(
            "--eunits",
            "-u",
            default=300,
            type=int,
            help="Number of encoder hidden units",
        )
        group.add_argument(
            "--dropout-rate",
            default=0.0,
            type=float,
            help="Dropout rate for the encoder",
        )
        # Encoder - RNN, TDNN
        group.add_argument(
            "--eprojs", default=320, type=int, help="Number of encoder projection units"
        ) 
        # For tdnn, using 320 now, original num of nnet output, e.g. 100 pdf-ids
        group.add_argument(
            "--subsample",
            default="1",
            type=str,
            help="Subsample input frames x_y_z means subsample every x frame "
            "at 1st layer, every y frame at 2nd layer etc.",
        )
        # Encoder - TDNN
        # TO DO: add TDNN papams
        group.add_argument(
            "--tdnn-hdim",
            default=[256,512,512,512,512],
            type=int,
            nargs='+',
            help="Output dimensions for each hidden layer ",
        )
        group.add_argument(
            "--kernel-sizes",
            default=[3,3,3,3],
            type=int,
            nargs='+',
            help="kernel sizes for each layers in TDNN",
        )
        group.add_argument(
            "--num-blocks",
            default=None,
            type=lambda s: [int(mod) for mod in s.split("_") if s!= ""],
        )
        group.add_argument(
            "--filters",
            default=None,
            type=lambda s: [int(mod) for mod in s.split("_") if s!= ""],
        )
        group.add_argument(
            "--num-convs",
            default=None,
            type=lambda s: [int(mod) for mod in s.split("_") if s!= ""],
        )
        group.add_argument(
            "--kernels",
            default=None,
            type=lambda s: [int(mod) for mod in s.split("_") if s!= ""],
        )
        group.add_argument(
            "--strides",
            default=None,
            type=lambda s: [int(mod) for mod in s.split("_") if s!= ""],
        )
        group.add_argument(
            "--dilations",
            default=None,
            type=lambda s: [int(mod) for mod in s.split("_") if s!= ""],
        )
        group.add_argument(
            "--dropouts",
            default=None,
            type=lambda s: [float(mod) for mod in s.split("_") if s!= ""],
        )
        # residual attention! FIX ME: yuekai
        group.add_argument(
            "--residuals",
            default=None,
            type=lambda s: [strtobool(mod) for mod in s.split("_") if s!= ""],
        )
        group.add_argument(
            "--activation",
            default="relu",
            type=str
        )
        group.add_argument(
            "--se-reduction-ratio",
            default=8,
            type=int
        )
        group.add_argument(
            "--context-window",
            default=-1,
            type=int
        )
        group.add_argument(
                "--interpolation-mode",
                default="nearest",
                type=str
        )
        #group.add_argument(
        #    "--dilations", 
        #    default=[1,1,3,3,3], 
        #    type=int, 
        #    nargs='+',
        #    help="dilations for tdnn"
        #)
        #group.add_argument(
        #    "--strides", 
        #    default=[1,1,1,1,3], 
        #    type=int, 
        #    nargs='+',
        #    help="strides for tdnn"
        #)
        # Attention - general
        group.add_argument(
            "--adim",
            default=320,
            type=int,
            help="Number of attention transformation dimensions",
        )
        group.add_argument(
            "--aheads",
            default=4,
            type=int,
            help="Number of heads for multi head attention",
        )
        group.add_argument(
            "--transformer-attn-dropout-rate-encoder",
            default=0.0,
            type=float,
            help="dropout in transformer decoder attention.",
        )
        group.add_argument(
            "--transformer-attn-dropout-rate-decoder",
            default=0.0,
            type=float,
            help="dropout in transformer decoder attention.",
        )
        # Attention - RNN
        group.add_argument(
            "--atype",
            default="location",
            type=str,
            choices=[
                "noatt",
                "dot",
                "add",
                "location",
                "coverage",
                "coverage_location",
                "location2d",
                "location_recurrent",
                "multi_head_dot",
                "multi_head_add",
                "multi_head_loc",
                "multi_head_multi_res_loc",
            ],
            help="Type of attention architecture",
        )
        group.add_argument(
            "--awin", default=5, type=int, help="Window size for location2d attention"
        )
        group.add_argument(
            "--aconv-chans",
            default=10,
            type=int,
            help="Number of attention convolution channels "
            "(negative value indicates no location-aware attention)",
        )
        group.add_argument(
            "--aconv-filts",
            default=100,
            type=int,
            help="Number of attention convolution filters "
            "(negative value indicates no location-aware attention)",
        )
        # Decoder - general
        group.add_argument(
            "--dtype",
            default="lstm",
            type=str,
            choices=["lstm", "gru", "transformer"],
            help="Type of decoder to use.",
        )
        group.add_argument(
            "--dlayers", default=1, type=int, help="Number of decoder layers"
        )
        group.add_argument(
            "--dunits", default=640, type=int, help="Number of decoder hidden units"
        )
        group.add_argument(
            "--dropout-rate-decoder",
            default=0.0,
            type=float,
            help="Dropout rate for the decoder",
        )
        # Decoder - RNN
        group.add_argument(
            "--dec-embed-dim",
            default=640,
            type=int,
            help="Number of decoder embeddings dimensions",
        )
        group.add_argument(
            "--dropout-rate-embed-decoder",
            default=0.0,
            type=float,
            help="Dropout rate for the decoder embeddings",
        )
        # Transformer
        group.add_argument(
            "--contextnet-warmup-steps",
            default=25000,
            type=int,
            help="optimizer warmup steps",
        )
        group.add_argument(
            "--contextnet-init",
            type=str,
            default="pytorch",
            choices=[
                "pytorch",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
            ],
            help="how to initialize transformer parameters",
        )
        group.add_argument(
            "--transformer-input-layer",
            type=str,
            default="conv2d",
            choices=["conv2d", "vgg2l", "linear", "embed"],
            help="transformer encoder input layer type",
        )
        group.add_argument(
            "--transformer-dec-input-layer",
            type=str,
            default="embed",
            choices=["linear", "embed"],
            help="transformer decoder input layer type",
        )
        group.add_argument(
            "--contextnet-lr",
            default=10.0,
            type=float,
            help="Initial value of learning rate",
        )
        # Transducer
        group.add_argument(
            "--trans-type",
            default="warp-transducer",
            type=str,
            choices=["warp-transducer"],
            help="Type of transducer implementation to calculate loss.",
        )
        group.add_argument(
            "--rnnt-mode",
            default="rnnt",
            type=str,
            choices=["rnnt", "rnnt-att"],
            help="Transducer mode for RNN decoder.",
        )
        group.add_argument(
            "--joint-dim",
            default=320,
            type=int,
            help="Number of dimensions in joint space",
        )
    
        group.add_argument(
            "--score-norm-transducer",
            type=strtobool,
            nargs="?",
            default=True,
            help="Normalize transducer scores by length",
        )

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1, blank_id=0):
        """Construct an E2E object for transducer model.

        Args:
            idim (int): dimension of inputs
            odim (int): dimension of outputs
            args (Namespace): argument Namespace containing options

        """
        torch.nn.Module.__init__(self)

        if args.etype == "transformer":
            self.subsample = get_subsample(args, mode="asr", arch="transformer")

            self.encoder = Encoder(
                idim=idim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.eunits,
                num_blocks=args.elayers,
                input_layer=args.transformer_input_layer,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate_encoder,
            )
        elif args.etype == "contextnet":
            self.subsample = get_subsample(args, mode="asr", arch="contextnet")
            self.encoder = Encoder(
                    idim=idim,
                    num_blocks=args.num_blocks,
                    filters=args.filters,
                    num_convs=args.num_convs,
                    kernels=args.kernels,
                    strides=args.strides,
                    dilations=args.dilations,
                    dropouts=args.dropouts,
                    residuals=args.residuals,
                    activation=args.activation,
                    se_reduction_ratio=args.se_reduction_ratio,
                    context_window=args.context_window,
                    interpolation_mode=args.interpolation_mode,
            )
        elif args.etype == "tdnn":
            self.subsample = get_subsample(args, mode="asr", arch="rnn-t")
            assert args.etype == "tdnn"
            # TO DO: organize args
            self.enc = encoder_for(args, idim, self.subsample)

        else:
            self.subsample = get_subsample(args, mode="asr", arch="rnn-t")
            self.enc = encoder_for(args, idim,self.subsample)

        if args.dtype == "transformer":
            self.decoder = Decoder(
                odim=odim,
                jdim=args.joint_dim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                input_layer=args.transformer_dec_input_layer,
                dropout_rate=args.dropout_rate_decoder,
                positional_dropout_rate=args.dropout_rate_decoder,
                attention_dropout_rate=args.transformer_attn_dropout_rate_decoder,
            )
    
        else:
            if args.etype == "transformer":
                args.eprojs = args.adim

            if args.rnnt_mode == "rnnt-att":
                self.att = att_for(args)
                self.dec = decoder_for(args, odim, self.att)
            else:
                self.dec = decoder_for(args, odim)
                # TO DO: keep this now, for tdnn

        self.etype = args.etype
        self.dtype = args.dtype
        self.rnnt_mode = args.rnnt_mode

        self.sos = odim - 1
        self.eos = odim - 1
        self.blank_id = blank_id
        self.ignore_id = ignore_id

        self.space = args.sym_space
        self.blank = args.sym_blank

        self.odim = odim
        self.adim = args.adim

        self.reporter = Reporter()

        self.criterion = TransLoss(args.trans_type, self.blank_id)

        self.default_parameters(args)

        if args.report_cer or args.report_wer:
            from espnet.nets.e2e_asr_common import ErrorCalculatorTrans

            if self.dtype == "transformer":
                self.error_calculator = ErrorCalculatorTrans(self.decoder, args)
            else:
                self.error_calculator = ErrorCalculatorTrans(self.dec, args)
        else:
            self.error_calculator = None

        self.logzero = -10000000000.0
        self.loss = None
        self.rnnlm = None

    def default_parameters(self, args):
        """Initialize/reset parameters for transducer."""
        initializer(self, args)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        Args:
            xs_pad (torch.Tensor): batch of padded source sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)

        Returns:
            loss (torch.Tensor): transducer loss value

        """
        # 1. encoder
        if self.etype == "transformer":
            xs_pad = xs_pad[:, : max(ilens)]
            src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

            hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        
        elif self.etype == "contextnet":
            xs_pad = xs_pad[:,:max(ilens)]
            hs_pad = self.encoder(xs_pad.transpose(1,2))
            hs_pad = hs_pad.transpose(1,2)
            # TO DO: fix mask yuekai
            hs_mask = torch.ones([xs_pad.size(0)]) * hs_pad.size(1)
        elif self.etype == "tdnn":
            #print(f"The shape of xs_pad, ilens, hs_pad, hs_mask")
            hs_pad, hs_mask = self.enc(xs_pad, ilens)

            #print(f"{hs_pad.shape} {hs_mask.shape} {xs_pad.shape} {ilens.shape}")
        else:
            hs_pad, hs_mask, _ = self.enc(xs_pad, ilens)
        self.hs_pad = hs_pad

        # 1.5. transducer preparation related
        ys_in_pad, target, pred_len, target_len = prepare_loss_inputs(ys_pad, hs_mask)

        # 2. decoder
        if self.dtype == "transformer":
            ys_mask = target_mask(ys_in_pad, self.blank_id)
            pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad)
        else:
            if self.rnnt_mode == "rnnt":
                pred_pad = self.dec(hs_pad, ys_in_pad)
                #print(f"shape for ys_in_pad, pred_pad: {ys_in_pad.shape} {pred_pad.shape}")
                #input()
            else:
                pred_pad = self.dec(hs_pad, ys_in_pad, pred_len)
        self.pred_pad = pred_pad

        # 3. loss computation
        loss = self.criterion(pred_pad, target, pred_len, target_len)

        self.loss = loss
        loss_data = float(self.loss)

        # 4. compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            cer, wer = self.error_calculator(hs_pad, ys_pad)

        if not math.isnan(loss_data):
            self.reporter.report(loss_data, cer, wer)
        else:
            logging.warning("loss (=%f) is not correct", loss_data)

        return self.loss

    def encode_transformer(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, attention_dim)

        """
        self.eval()

        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)

        return enc_output.squeeze(0)

    def encode_rnn(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, attention_dim)

        """
        self.eval()

        ilens = [x.shape[0]]

        x = x[:: self.subsample[0], :]
        h = to_device(self, to_torch_tensor(x).float())
        hs = h.contiguous().unsqueeze(0)

        h, _, _ = self.enc(hs, ilens)

        return h[0]

    def encode_tdnn(self,x):
        pass
        # TO DO
    def encode_contextnet(self,x):
        x = torch.as_tensor(x).unsqueeze(0)
        h = self.encoder(x.transpose(1,2))
        h = h.transpose(1,2).squeeze(0)
        return h
        # TO DO


    def recognize(self, x, recog_args, char_list=None, rnnlm=None):
        """Recognize input features.

        Args:
            x (ndarray): input acoustic feature (T, D)
            recog_args (namespace): argument Namespace containing options
            char_list (list): list of characters
            rnnlm (torch.nn.Module): language model module

        Returns:
            y (list): n-best decoding results

        """
        if self.etype == "transformer":
            h = self.encode_transformer(x)
        elif self.etype == "tdnn":
            h = self.encode_tdnn(x)
        elif self.etype == "contextnet":
            h = self.encode_contextnet(x)
        else:
            h = self.encode_rnn(x)
        params = [h, recog_args]

        if recog_args.beam_size == 1:
            if self.dtype == "transformer":
                nbest_hyps = self.decoder.recognize(*params)
            else:
                nbest_hyps = self.dec.recognize(*params)
        else:
            params.append(rnnlm)
            if self.dtype == "transformer":
                nbest_hyps = self.decoder.recognize_beam(*params)
            else:
                nbest_hyps = self.dec.recognize_beam(*params)

        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        Args:
            xs_pad (torch.Tensor): batch of padded input sequences (B, Tmax, idim)
            ilens (torch.Tensor): batch of lengths of input sequences (B)
            ys_pad (torch.Tensor):
                batch of padded character id sequence tensor (B, Lmax)

        Returns:
            ret (ndarray): attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).

        """
        if (
            self.etype == "transformer"
            and self.dtype != "transformer"
            and self.rnnt_mode == "rnnt-att"
        ):
            raise NotImplementedError(
                "Transformer encoder with rnn attention decoder" "is not supported yet."
            )
        elif self.etype != "transformer" and self.dtype != "transformer":
            if self.rnnt_mode == "rnnt":
                return []
            else:
                with torch.no_grad():
                    hs_pad, hlens = xs_pad, ilens
                    hpad, hlens, _ = self.enc(hs_pad, hlens)

                    ret = self.dec.calculate_all_attentions(hpad, hlens, ys_pad)
        else:
            with torch.no_grad():
                self.forward(xs_pad, ilens, ys_pad)

                ret = dict()
                for name, m in self.named_modules():
                    if isinstance(m, MultiHeadedAttention):
                        ret[name] = m.attn.cpu().numpy()

        return ret
