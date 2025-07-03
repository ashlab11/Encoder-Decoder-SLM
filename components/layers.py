"""
Module for Encoder-Decoder Transformer
"""

import torch
import torch.nn as nn
from .attention import SelfAttention, CrossAttention
#NOTE: fix use of max_seq_len for RoPE

class EncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, enc_seq_len, mlp_dim=None, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.dim = dim
        mlp_dim = mlp_dim or 4 * dim 
        
        # Self-attention block
        self.self_attn = SelfAttention(dim=dim, num_heads=num_heads, max_seq_len=enc_seq_len, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x, enc_key_padding_mask=None):
        # Self-attention with residual
        residual = x
        x = self.norm1(x)  # Pre-norm architecture
        x = self.self_attn(x, causal=False, key_padding_mask=enc_key_padding_mask)
        x = residual + x
        
        # MLP with residual
        residual = x
        x = self.norm2(x)  # Pre-norm architecture
        x = self.mlp(x)
        x = residual + x
        
        return x

class DecoderLayerWithCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, dec_seq_len, enc_seq_len, mlp_dim=None, dropout=0.1):
        super(DecoderLayerWithCrossAttention, self).__init__()
        self.dim = dim
        mlp_dim = mlp_dim or 4 * dim  # Default to 4x if not specified
        
        # Self-attention block
        self.self_attn = SelfAttention(dim=dim, num_heads=num_heads, max_seq_len=dec_seq_len, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        
        # Cross-attention block
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads, enc_max_seq_len=enc_seq_len, dec_max_seq_len=dec_seq_len, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self, x, encoder_output, dec_key_padding_mask=None, cross_key_padding_mask=None):
        # Self-attention with residual
        residual = x
        x = self.norm1(x)  # Pre-norm architecture
        x = self.self_attn(x, causal=True, key_padding_mask=dec_key_padding_mask)  # Always causal in decoder
        x = residual + x
        
        # Cross-attention with residual
        residual = x
        x = self.norm2(x)  # Pre-norm architecture
        x = self.cross_attn(x, encoder_output, key_padding_mask=cross_key_padding_mask)
        x = residual + x
        
        # MLP with residual
        residual = x
        x = self.norm3(x)  # Pre-norm architecture
        x = self.mlp(x)
        x = residual + x
        
        return x

