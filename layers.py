import math
import torch
from torch import nn
from constants import *
import constants  # <-- ADD

#5th
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm1 = LayerNormalization()
        self.mha = MultiHeadAttentionLayer(use_rope=constants.USE_ROPE)  # <-- CHANGE
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
        self.self_maksed_mha = MultiHeadAttentionLayer(use_rope=constants.USE_ROPE)  # <-- CHANGE
        self.dropout1 = nn.Dropout(drop_out_rate)

        self.layernorm2 = LayerNormalization()
        self.mha = MultiHeadAttentionLayer(use_rope=False)  # cross-attn keep False
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
    def __init__(self, use_rope=True):
        super().__init__()
        self.inf = 1e9
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.use_rope = use_rope
        self.rope = RotaryEmbedding(dim=d_k, max_seq_len=max_len) if use_rope else None

        self.attn_softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(drop_out_rate)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)

        q = self.w_q(q).view(B, -1, num_heads, d_k).transpose(1, 2)  # (B,H,Lq,D)
        k = self.w_k(k).view(B, -1, num_heads, d_k).transpose(1, 2)  # (B,H,Lk,D)
        v = self.w_v(v).view(B, -1, num_heads, d_k).transpose(1, 2)  # (B,H,Lk,D)

        # Apply RoPE only when Lq == Lk (self-attn). Your cross-attn already uses use_rope=False.
        if self.use_rope:
            cos, sin = self.rope(q, seq_len=q.size(-2))
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output = self.self_attention(q, k, v, mask=mask)
        concat = attn_output.transpose(1, 2).contiguous().view(B, -1, d_model)
        return self.w_o(concat)

    def self_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask should broadcast to (B,1,Lq,Lk) or (B,H,Lq,Lk)
            attn_scores = attn_scores.masked_fill(mask == 0, torch.finfo(attn_scores.dtype).min)

        attn = self.attn_softmax(attn_scores)
        attn = self.attn_dropout(attn)
        return torch.matmul(attn, v)

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
    """
    Rotary Position Embedding (RoPE) - Fixed implementation
    Uses the standard LLaMA/GPT-NeoX style rotation (split-half method)
    """
    def __init__(self, dim, max_seq_len=max_len, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies: theta_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cos and sin for all positions
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        # freqs shape: (seq_len, dim/2)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # emb shape: (seq_len, dim) - duplicate for both halves
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Shape: (1, 1, seq_len, dim) for broadcasting
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: (B, H, L, D)
        if seq_len is None:
            seq_len = x.size(-2)
        
        # Extend cache if needed
        if seq_len > self.cos_cached.size(-2):
            self._set_cos_sin_cache(seq_len)

        cos = self.cos_cached[..., :seq_len, :].to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached[..., :seq_len, :].to(device=x.device, dtype=x.dtype)
        return cos, sin


def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    For input [..., d], splits into [..., d/2] and [..., d/2], 
    then returns [-x2, x1] concatenated
    """
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply rotary positional embeddings to q and k tensors.
    
    The rotation formula: 
        q_rot = q * cos + rotate_half(q) * sin
    
    This encodes relative position information into the attention scores.
    """
    # Keep computation stable in fp32, then cast back
    orig_dtype = q.dtype
    q_f = q.float()
    k_f = k.float()
    cos_f = cos.float()
    sin_f = sin.float()

    q_embed = (q_f * cos_f) + (rotate_half(q_f) * sin_f)
    k_embed = (k_f * cos_f) + (rotate_half(k_f) * sin_f)
    
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

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
        # Note: Scaling is already done in Transformer.forward(), don't scale again here!

        # Add PE
        # We slice self.pe to the length of the current input x
        # self.pe is already on the correct device!
        x = x + self.pe[:, :x.size(1), :]

        return x
