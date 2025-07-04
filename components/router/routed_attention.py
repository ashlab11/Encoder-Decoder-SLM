"""
Module for RoPE (Rotary Positional Embeddings) Attention Layer
This module implements a cross-attention and self-attention layer with RoPE.
"""

import torch
import torch.nn as nn
from torchtune.modules import RotaryPositionalEmbeddings 
import torch.nn.functional as F

class RoutedSelfAttention(nn.Module):
    def __init__(self, dim, num_attentions, num_heads, max_seq_len = 1024, dropout=0.1):
        super(RoutedSelfAttention, self).__init__()
        self.dim = dim
        self.num_attentions = num_attentions
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.max_seq_len = max_seq_len
        
        # Stack all projections into single tensors for more efficient computation
        self.query_projs = nn.Parameter(torch.randn(num_attentions, dim, dim))
        self.key_projs = nn.Parameter(torch.randn(num_attentions, dim, dim))
        self.value_projs = nn.Parameter(torch.randn(num_attentions, dim, dim))
        self.out_projs = nn.Parameter(torch.randn(num_attentions, dim, dim))
        
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.router = nn.Linear(dim, num_attentions) # Router used for attention
        
        self.rotary_emb = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x, causal=True, key_padding_mask=None, **kwargs):
        B, L, D = x.size()  # [batch_size, seq_length, embed_dim]
        
        # Pooled router
        logits = self.router(x.mean(dim=1)) # [B, num_attentions]
        attn_probs = F.softmax(logits, dim=-1) # [B, num_attention]
        routes = attn_probs.argmax(dim=-1) # [B] -- what attention block each goes to
        
        # Get the projection matrices for each batch item
        query_weights = self.query_projs[routes]  # [B, D, D]
        key_weights = self.key_projs[routes]      # [B, D, D]
        value_weights = self.value_projs[routes]  # [B, D, D]
        
        # Batch matrix multiplication for all projections at once
        query = torch.bmm(x, query_weights)  # [B, L, D]
        key = torch.bmm(x, key_weights)      # [B, L, D]
        value = torch.bmm(x, value_weights)  # [B, L, D]
        
        # Reshape for multi-head attention
        query = query.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        key = key.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)      # [B, num_heads, L, head_dim]
        value = value.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, L, head_dim]
        
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
        
        # Apply output projection using the same batched approach
        out_weights = self.out_projs[routes]  # [B, D, D]
        output = torch.bmm(output, out_weights)  # [B, L, D]
        
        return output, attn_probs
    