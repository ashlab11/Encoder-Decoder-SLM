"""
Module for RoPE (Rotary Positional Embeddings) Attention Layer
This module implements a cross-attention and self-attention layer with RoPE.
"""

import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings 

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, max_seq_len = 1024, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.max_seq_len = max_seq_len
        
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        
        self.rotary_emb = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, causal=True, key_padding_mask=None, **kwargs):
        B, L, D = x.size()  # [batch_size, seq_length, embed_dim]
        
        # Self-attention
        query = self.query_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, L, head_dim]
        key = self.key_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, L, head_dim]
        value = self.value_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, L, head_dim]
        
        # Apply RoPE
        query = self.rotary_emb(query)
        key = self.rotary_emb(key)

        # Build causal mask (L×L), then fold in padding
        if causal:
            causal_part = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
            causal_part = causal_part.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        else:
            causal_part = None

        if key_padding_mask is not None:
            # key_padding_mask: B×L boolean, True=pad → we want mask=True where pad
            pad_part = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            pad_part = pad_part.expand(B, 1, L, L)  # [B, 1, L, L]
        else:
            pad_part = None

        # Combine: mask out either causal positions or padding positions
        if causal_part is not None and pad_part is not None:
            full_mask = pad_part | causal_part
        elif causal_part is not None:
            full_mask = causal_part
        else:
            full_mask = pad_part

        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=full_mask,
            dropout_p=self.dropout.p if isinstance(self.dropout, nn.Dropout) else self.dropout)

        output = output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, embed_dim]
        output = self.out_proj(output)
        return output
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, enc_max_seq_len = 1024, dec_max_seq_len = 1024, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.enc_max_seq_len = enc_max_seq_len
        self.dec_max_seq_len = dec_max_seq_len
        
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
        
        # Separate rotary embeddings for decoder (query) and encoder (key) sequences
        self.query_rotary = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=dec_max_seq_len)
        self.key_rotary = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=enc_max_seq_len)
        
    def forward(self, x, encoder_inputs, key_padding_mask=None, **kwargs):
        B, L, D = x.size()  # [batch_size, decoder_seq_length, embed_dim]
        seq_len_enc = encoder_inputs.size(1)  # [batch_size, encoder_seq_length, embed_dim]
        
        # Cross-attention
        query = self.query_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, L, head_dim]
        key = self.key_proj(encoder_inputs).reshape(B, seq_len_enc, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, seq_len_enc, head_dim]
        value = self.value_proj(encoder_inputs).reshape(B, seq_len_enc, self.num_heads, self.head_dim).transpose(1, 2) # [B, num_heads, seq_len_enc, head_dim]
        
        # Apply RoPE separately to decoder queries and encoder keys
        query = self.query_rotary(query)  # Use decoder positions
        key = self.key_rotary(key)    # Use encoder positions
        
        # Build padding mask for cross-attn: B×S
        if key_padding_mask is not None:
            # expand to B×1×L×S so every decoder position uses same pad mask
            full_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]
            full_mask = full_mask.expand(B, 1, L, seq_len_enc)  # [B, 1, L, S]
        else:
            full_mask = None

        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=full_mask,
            dropout_p=self.dropout if not isinstance(self.dropout, nn.Dropout) else self.dropout.p)
        
        output = output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, embed_dim]
        output = self.out_proj(output)
        return output
        