from torch import nn
from constants import *
from layers import *

import torch

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        
        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, d_model)

        # self.positional_encoding = PositionalEncoder(d_model = d_model, max_len = 5000)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output_linear = nn.Linear(d_model, trg_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src_seq, trg_seq, enc_mask=None, dec_mask=None):
        src_seq = self.src_embedding(src_seq) # (B, seq_len, d_model)
        trg_seq = self.trg_embedding(trg_seq) # (B, seq_len, d_model)

        # src_seq = self.positional_encoding(src_seq) # (B, seq_len, d_model)
        # trg_seq = self.positional_encoding(trg_seq) # (B, seq_len, d_model)

        src_seq = src_seq * math.sqrt(d_model)
        trg_seq = trg_seq * math.sqrt(d_model)

        src_enc_output = self.encoder(src_seq, enc_mask) # (B, seq_len, d_model)
        trg_dec_output = self.decoder(trg_seq, src_enc_output, enc_mask, dec_mask) # (B, seq_len, d_model)

        output = self.output_linear(trg_dec_output) # (B, seq_len, trg_vocab_size)

        return output # (B, seq_len, trg_vocab_size)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for i in range(num_layers)])
        self.layernorm = LayerNormalization()

    def forward(self, x, enc_mask = None):
        for i in range(num_layers):
            x = self.layers[i](x, enc_mask) #(B, seq_len, d_model)
        
        return self.layernorm(x) #(B, seq_len, d_model)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for i in range(num_layers)])
        self.layernorm = LayerNormalization()

    def forward(self, x, enc_output, enc_mask = None, dec_mask = None):
        for i in range(num_layers):
            x = self.layers[i](x, enc_output, enc_mask, dec_mask) #(B, seq_len, d_model)

        return self.layernorm(x) #(B, seq_len, d_model)