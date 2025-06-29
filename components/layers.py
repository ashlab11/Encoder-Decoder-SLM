"""
Module for Encoder-Decoder Transformer
"""

import torch
import torch.nn as nn
from .attention import SelfAttention, CrossAttention
#NOTE: fix use of max_seq_len for RoPE

class EncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim=None, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.dim = dim
        mlp_dim = mlp_dim or 4 * dim 
        
        # Self-attention block
        self.self_attn = SelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
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
        
    def forward(self, x, attention_mask=None):
        # Self-attention with residual
        residual = x
        x = self.norm1(x)  # Pre-norm architecture
        x = self.self_attn(x, causal=False, attention_mask=attention_mask)
        x = residual + x
        
        # MLP with residual
        residual = x
        x = self.norm2(x)  # Pre-norm architecture
        x = self.mlp(x)
        x = residual + x
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.dim = dim
        mlp_dim = mlp_dim or 4 * dim  # Default to 4x if not specified
        
        # Self-attention block
        self.self_attn = SelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        
        # Cross-attention block
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads, dropout=dropout)
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
        
    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        # Self-attention with residual
        residual = x
        x = self.norm1(x)  # Pre-norm architecture
        x = self.self_attn(x, causal=True, attention_mask=self_attention_mask)  # Always causal in decoder
        x = residual + x
        
        # Cross-attention with residual
        residual = x
        x = self.norm2(x)  # Pre-norm architecture
        x = self.cross_attn(x, encoder_output, attention_mask=cross_attention_mask)
        x = residual + x
        
        # MLP with residual
        residual = x
        x = self.norm3(x)  # Pre-norm architecture
        x = self.mlp(x)
        x = residual + x
        
        return x

