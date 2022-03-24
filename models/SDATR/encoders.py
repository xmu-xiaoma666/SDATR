from torch.nn import functional as F
from models.transformer_ensemble.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer_ensemble.encoderAttention import MultiHeadAttention,ChannelAttention
import numpy as np
from numpy import math
from torch.autograd import Variable


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering

        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        
        self.chatt = ChannelAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)



        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)
        self.lnorm1=nn.LayerNorm(d_model)

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        #multi head attention & (add & norm)
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)

        #channel se attention & (add & norm)
        catt = self.chatt(queries)

        #add & norm
        att=self.lnorm1(queries+self.dropout1(att)+self.dropout2(catt))

        #ffd & (add & norm)
        ff = self.pwff(att)

        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        outs = []
        out = input
        for l in self.layers:
            out = l(out, out, out, attention_mask, attention_weights)
            # outs.append(out.unsqueeze(1))

        # outs = torch.cat(outs, 1)
        return out, attention_mask


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        rowPE = torch.zeros(max_len,max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        rowPE[ :,:,0::2] = torch.sin(position * div_term)
        rowPE[ :,:, 1::2] = torch.cos(position * div_term)
        colPE=rowPE.transpose(1, 0)
        rowPE = rowPE.unsqueeze(0)
        colPE = colPE.unsqueeze(0)
        self.rowPE=rowPE.cuda()
        self.colPE=colPE.cuda()

    def forward(self, x):
        feat=x
        bs,gs,dim=feat.shape
        feat=feat.view(bs,int(np.sqrt(gs)),int(np.sqrt(gs)),dim)
        feat = feat + self.rowPE[:, :int(np.sqrt(gs)), :int(np.sqrt(gs)),  :dim ]+ self.colPE[:,  :int(np.sqrt(gs)),  :int(np.sqrt(gs)),  :dim ]
        feat=feat.view(bs,-1,dim)
        return self.dropout(feat)


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.pe=PositionalEncoding(d_model=d_in,dropout=0)

    def forward(self, input, attention_weights=None):
        feat=self.pe(input)
        mask = (torch.sum(feat, dim=-1) == 0).unsqueeze(-1)
        out = F.relu(self.fc(feat))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.masked_fill(mask, 0)
        return super(TransformerEncoder, self).forward(out, attention_weights=attention_weights)