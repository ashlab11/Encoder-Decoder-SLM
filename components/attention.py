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

    def forward(self, x, causal=True, attention_mask=None, **kwargs):
        B, L, D = x.size()  # [batch_size, seq_length, embed_dim]
        
        # Self-attention
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        
        # Apply RoPE
        query = self.rotary_emb(query)
        key = self.rotary_emb(key)

        # Create causal mask
        if causal:
            causal_mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
            causal_mask = causal_mask.to(x.device)
            
            # Combine with attention mask if provided
            if attention_mask is not None:
                causal_mask = causal_mask | attention_mask

        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=causal_mask if causal else attention_mask,
            dropout_p=self.dropout if self.training else 0.0)
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
        
    def forward(self, x, encoder_inputs, attention_mask=None, **kwargs):
        B, L, D = x.size()  # [batch_size, decoder_seq_length, embed_dim]
        
        # Cross-attention
        query = self.query_proj(x)
        key = self.key_proj(encoder_inputs)
        value = self.value_proj(encoder_inputs)
        
        # Apply RoPE separately to decoder queries and encoder keys
        query = self.query_rotary(query)  # Use decoder positions
        key = self.key_rotary(key)    # Use encoder positions
        
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0)
        output = output.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, embed_dim]
        output = self.out_proj(output)
        return output
        