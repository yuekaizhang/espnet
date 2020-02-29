import numpy as np
import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder


class LayerNorm(torch.nn.Module):
    def __init__(
        self,
        size: int = 1024,
        eps: float = 1e-12,
    ):
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))
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
        self.dtype = rnn_type
        self.dunits = hidden_size
        self.dlayers = num_layers
        self.odim = odim
        self.dropout = dropout
        self.num_encs = num_encs
        self.layer_norm_eps = layer_norm_eps

        self.dropout_linear = torch.nn.Dropout(p=dropout)
        self.activation = F.relu()
        self.layer_norm = LayerNorm(size=self.dunits, eps=layer_norm_eps)

        self.decoder = torch.nn.ModuleList()
        self.decoder += [torch.nn.Linear(self.eprojs, self.dunits)]
        for _ in range(1, self.dlayers):
            self.decoder += [
                torch.nn.Linear(self.dunits, self.dunits)
            ]
        self.output = torch.nn.Linear(self.dunits, self.odim * self.downsample_rate)


    def forward(self, hidden_states):

        for l in range(self.dlayers):
            hidden_states = self.layer_norm(
                                self.activation(
                                    self.decoder[l](hidden_states)))
        linear_output = self.output(hidden_states)
        return linear_output, hidden_states
