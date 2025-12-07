import torch
import torch.nn as nn

from constants import *
import math


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm1 = LayerNormalization()
        self.mha = MultiHeadAttentionLayer()
        self.dropout1 = nn.Dropout(drop_out_rate)

        self.layernorm2 = LayerNormalization()
        self.ffn = FeedForwardLayer()
        self.dropout2 = nn.Dropout(drop_out_rate)

    def forward(self, x, src_mask = None):
        x1 = self.layernorm1(x)
        x = x + self.dropout1(self.mha(x1, x1, x1, src_mask)) #(B, seq_len, d_model)
        x2 = self.layernorm2(x)
        x = x + self.dropout2(self.ffn(x2)) #(B, seq_len, d_model)

        return x # (B, seq_len, d_model)


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm1 = LayerNormalization()
        self.self_maksed_mha = MultiHeadAttentionLayer()
        self.dropout1 = nn.Dropout(drop_out_rate)

        self.layernorm2 = LayerNormalization()
        self.mha = MultiHeadAttentionLayer()
        self.dropout2 = nn.Dropout(drop_out_rate)

        self.layernorm3 = LayerNormalization()
        self.ffn = FeedForwardLayer()
        self.dropout3 = nn.Dropout(drop_out_rate)

    def forward(self, x, enc_output, enc_mask = None, dec_mask = None):
        x1 = self.layernorm1(x)
        x = x + self.dropout1(self.self_maksed_mha(x1, x1, x1, dec_mask)) #(B, seq_len, d_model)

        x2 = self.layernorm2(x)
        x = x + self.dropout2(self.mha(x2, enc_output, enc_output, enc_mask)) #(B, seq_len, d_model)

        x3 = self.layernorm3(x)
        x = x + self.dropout3(self.ffn(x3)) #(B, seq_len, d_model)

        return x # (B, seq_len, d_model)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inf = 1e9
        
        #W^q, W^k, W^v : linear layers to project input to q, k, v
        self.w_q = nn.Linear(d_model, d_model, bias = False)
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False)

        self.attn_softmax = nn.Softmax(dim = -1)
        self.attn_dropout = nn.Dropout(drop_out_rate)

        #w^o : linear layer to project concatenated output of all heads
        self.w_o = nn.Linear(d_model, d_model, bias = False)

    def forward(self, q, k, v, mask = None):
        input_shape = q.shape() # (batch_size, seq_len, d_model)

        q = self.w_q.view(input_shape[0], -1, num_heads, d_k) # (batch_size, seq_len, num_heads, d_k)
        k = self.w_k.view(input_shape[0], -1, num_heads, d_k) # (batch_size, seq_len, num_heads, d_k)
        v = self.w_v.view(input_shape[0], -1, num_heads, d_k) # (batch_size, seq_len, num_heads, d_k)

        #transpose to get dimensions (batch_size, num_heads, seq_len, d_k)
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)
        k = k.transpose(1, 2)  # (batch_size, num_heads , seq_len, d_k)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)

        attn_output = self.self_attention(q, k, v, mask = None)
        #concatenate heads and put back to original shape
        concat_output = attn_output.transpose(1, 2).contigous().view(input_shape[0], -1, d_model)   # (batch_size, seq_len, d_model)
        output = self.w_o(concat_output)  # (batch_size, seq_len, d_model)

        return output

    def self_attention(self, q, k, v, mask = None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) # (batch_size, num_heads, seq_len, seq_len) 
    
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, -1 * self.inf) # Apply the mask

        attn_distribs = self.attn_softmax(attn_scores) # (batch_size, num_heads, seq_len, seq_len)
        attn_distribs = self.attn_dropout(attn_distribs)

        attn_output = torch.matmul(attn_distribs, v)  # (batch_size, num_heads, seq_len, d_k)
        return attn_output


class FeedForwardLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer1 = nn.Linear(d_model, d_ff, bias = True)
        self.activation = nn.ReLU()
        self.linear_layer2 = nn.linear(d_ff, d_model, bias = True)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, x):
        x = self.activation(self.linear_layer1(x))
        x = self.dropout(x)
        x = self.linear_layer2(x)

        return x

class LayerNormalization(nn.Module):
    def __init__ (self, eps = 1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([d_model], elementwise_affine = True ,eps = self.eps)

    def forward(self, x):
        x = self.layer(x)
        return x
    
class PositionalEncoder(nn.Module):
    def __init__(self):
        super.__init__()
        #positional encoding with size (seq_len, d_model) becuase we add this to embedding of same size
        pe_matrix = torch.zeros(seq_len, d_model) # (seq_len, d_model)

        #calculate positional encoding values
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    pe_matrix[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                elif i % 2 == 1:
                    pe_matrix[pos, i] = math.cos(pos / (10000 ** ((2 * i)/d_model)))
                
        pe_matrix = pe_matrix.unsqueeze(0) # (1, seq_len, d_model)
        self.postional_encoding = pe_matrix.to(device = device).require_grad_(False)

    def forward(self, x):
        x = x * math.sqrt(d_model) #scale embedding
        x = x + self.positional__encoding[:, :x.size(1), :] #add positional encoding

        return x 
