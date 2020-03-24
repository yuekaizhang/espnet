import numpy as np
import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.pretrain.decoder.abs_decoder import AbsDecoder


class LayerNorm(torch.nn.Module):
    def __init__(
        self,
        size: int = 1024,
        eps: float = 1e-12,
    ):
        super(LayerNorm, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(size)) # used to be hidden_size by Yuekai
        self.bias = torch.nn.Parameter(torch.zeros(size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class PretrainDecoder(AbsDecoder):
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        num_layers: int = 2,
        hidden_size: int = 1024,
        dropout: float = 0.0,
        num_encs: int = 1,
        downsample_rate: int = 3,
        layer_norm_eps: float = 1e-12,
    ):
    
        assert check_argument_types()

        super().__init__()
        self.eprojs = encoder_output_size
        #self.dtype = rnn_type # why keep this ? Yuekai
        self.dunits = hidden_size
        self.dlayers = num_layers
        self.odim = odim
        self.dropout = dropout
        self.num_encs = num_encs
        self.layer_norm_eps = layer_norm_eps

        self.dropout_linear = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()
        #self.activation = F.relu() this has bug Yuekai
        self.layer_norm = LayerNorm(size=self.dunits, eps=layer_norm_eps)

        self.decoder = torch.nn.ModuleList()
        self.decoder += [torch.nn.Linear(self.eprojs, self.dunits)]
        for _ in range(1, self.dlayers):
            self.decoder += [
                torch.nn.Linear(self.dunits, self.dunits)
            ]
        self.feats_decoder = torch.nn.Sequential(
                torch.nn.Linear(256, self.odim),
                torch.nn.LayerNorm(self.odim),
                torch.nn.Dropout(self.dropout),
                torch.nn.ReLU(),
        )
        self.text_decoder = torch.nn.Sequential(
                torch.nn.Linear(256, 1),
        )
        #self.output = torch.nn.Linear(self.dunits, self.odim * self.downsample_rate)


    def forward(self, encoder_out,encoder_out_lens,speech_len,text_len):
        #print(encoder_out.shape)
        

        #print(encoder_out_lens) # why some 314, others 313?
        
        #print("____________________________________________++++++++++++++++")
        #print(self.dropout)
        #print(text_out_lens)

        
        
        feats_output,text_output = encoder_out[:, :speech_len, :], \
                encoder_out[:,speech_len:,:]
        assert text_output.shape[1] == text_len

        #feats_decoder = torch.nn.Sequential(
        #        torch.nn.Linear(encoder_out.shape[-1], self.odim),
        #        torch.nn.LayerNorm(self.odim),
        #        torch.nn.Dropout(self.dropout),
        #        torch.nn.ReLU(),
        #)

        #text_decoder = torch.nn.Sequential(
        #        torch.nn.Linear(encoder_out.shape[-1], 1),
        #)


        pred_feats = self.feats_decoder(feats_output)
        pred_text = self.text_decoder(text_output)
        # for l in range(self.dlayers):
        #     hidden_states = self.layer_norm(
        #                         self.activation(
        #                             self.decoder[l](hidden_states)))
        # linear_output = self.output(hidden_states)
        # return linear_output, hidden_states
        # just keep the format (yuekai)
        return pred_feats, pred_text
