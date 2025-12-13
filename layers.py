import torch
import torch.nn as nn

from constants import *
import math

#5th
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm1 = LayerNormalization()
        self.mha = MultiHeadAttentionLayer(use_rope=True)
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

#6th
class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm1 = LayerNormalization()
        self.self_maksed_mha = MultiHeadAttentionLayer(use_rope = True)
        self.dropout1 = nn.Dropout(drop_out_rate)

        self.layernorm2 = LayerNormalization()
        self.mha = MultiHeadAttentionLayer(use_rope=False)
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

#4th
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, use_rope = True):
        super().__init__()
        self.inf = 1e9

        # W^q, W^k, W^v
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.use_rope = use_rope
        if self.use_rope:
            self.rotary_emb = RotaryEmbedding(d_k)
        else:
            self.rotary_emb = None

        self.attn_softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(drop_out_rate)

        # W^o
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        #Define Shapes
        batch_sizee = q.size(0)

        # Use -1 to infer sequence length dynamically (safe for cross-attention)
        # Reshape: (B, Seq_Len, d_model) -> (B, Seq_Len, H, d_k)
        q = self.w_q(q).view(batch_sizee, -1, num_heads, d_k)
        k = self.w_k(k).view(batch_sizee, -1, num_heads, d_k)
        v = self.w_v(v).view(batch_sizee, -1, num_heads, d_k)

        #Transpose for Attention: (B, H, Seq_Len, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_rope:
            seq_length = q.size(2)
            # Get Cos/Sin for the current sequence length
            cos, sin = self.rotary_emb(v, seq_len=seq_length)
            # Apply rotation to Q and K
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = self.self_attention(q, k, v, mask=mask)


        concat_output = attn_output.transpose(1, 2).contiguous().view(batch_sizee, -1, d_model)

        output = self.w_o(concat_output)
        return output

    def self_attention(self, q, k, v, mask=None):
        # q, k, v are (Batch, Head, Len, d_k)

        # 1. Calculate Scores: (Batch, Head, Q_Len, K_Len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. Apply Mask
        # NEW (FP16 Safe Code)
        if mask is not None:
            # 2. Get the lowest possible number for this specific datatype (FP16 or FP32)
            min_val = torch.finfo(attn_scores.dtype).min

            # 3. Apply mask
            attn_scores = attn_scores.masked_fill(mask == 0, min_val)

        # 3. Softmax & Dropout
        attn_distribs = self.attn_softmax(attn_scores)
        attn_distribs = self.attn_dropout(attn_distribs)

        # 4. Context Vector
        attn_output = torch.matmul(attn_distribs, v)
        return attn_output

#3rd
class FeedForwardLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer1 = nn.Linear(d_model, d_ff, bias = True)
        self.activation = nn.ReLU()
        self.linear_layer2 = nn.Linear(d_ff, d_model, bias = True)
        self.dropout = nn.Dropout(drop_out_rate)

    def forward(self, x):
        x = self.activation(self.linear_layer1(x))
        x = self.dropout(x)
        x = self.linear_layer2(x)

        return x

#2nd
class LayerNormalization(nn.Module):
    def __init__ (self, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([d_model], elementwise_affine = True ,eps = self.eps)

    def forward(self, x):
        x = self.layer(x)
        return x

#1st
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=max_len):
        super().__init__()
        self.dim = dim

        # 1. Calculate Frequencies (The "Theta")
        # formula: 1 / (10000 ^ (2i / dim))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        # 2. Create Position indices
        t = torch.arange(max_seq_len).type_as(inv_freq)

        # 3. Outer Product to get angles
        freqs = torch.einsum('i,j->ij', t, inv_freq)  # (seq_len, dim/2)

        # 4. Concatenate to match dimension
        # (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        # 5. REGISTER BUFFER (Crucial for Accelerate/Multi-GPU)
        # We register cos and sin as buffers so they move to GPU automatically
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        # x shape: (Batch, Heads, Seq_Len, Head_Dim)
        if seq_len > self.cos_cached.shape[2]:
            # Optional: Resize cache if input is longer than max_seq_len
            # For now, we assume max_seq_len is big enough (5000)
            pass

        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

# Helper function to rotate vector
def rotate_half(x):
    # Split vector in half
    x1, x2 = x.chunk(2, dim=-1)
    # Return [-x2, x1]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: (Batch, Heads, Seq_Len, Head_Dim)
    # cos, sin: (1, 1, Seq_Len, Head_Dim)
    q_float = q.float()
    k_float = k.float()
    cos = cos.float()
    sin = sin.float()

    # Formula: (x * cos) + (rotate_half(x) * sin)
    q_embed = (q_float * cos) + (rotate_half(q_float) * sin)
    k_embed = (k_float * cos) + (rotate_half(k_float) * sin)
    return q_embed.type_as(q), k_embed.type_as(k)

#333
class PositionalEncoder(nn.Module):
    def __init__(self, d_model = d_model, max_len = 5000):
        # Pass d_model and max_len as args, don't rely on globals!
        super().__init__()

        # 1. Create Matrix (on CPU initially)
        pe = torch.zeros(max_len, d_model)

        # 2. Vectorized Calculation (Fast!)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Calculate the division term in log space for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply Sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply Cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # 3. THE MAGIC LINE
        # We register it as a "buffer".
        # - It is NOT a parameter (won't be updated by optimizer).
        # - It WILL be moved to GPU automatically by Accelerator.
        # - It WILL be saved in state_dict.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch_Size, Seq_Len, d_model)

        # Scale embedding (Standard Transformer practice)
        x = x * math.sqrt(x.size(-1))

        # Add PE
        # We slice self.pe to the length of the current input x
        # self.pe is already on the correct device!
        x = x + self.pe[:, :x.size(1), :]

        return x
