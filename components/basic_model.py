import torch
import torch.nn as nn
from .layers import EncoderLayer, DecoderLayerWithCrossAttention

class BasicEDModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        num_heads: int = 9,
        num_encoder_layers: int = 14,
        num_decoder_layers: int = 7,
        enc_seq_len: int = 1024,
        dec_seq_len: int = 1024,
        mlp_dim: int = None,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super(BasicEDModel, self).__init__()
        self.dim = dim
        self.pad_token_id = pad_token_id
        
        # Token embeddings
        print(vocab_size, dim)
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dim=dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout, enc_seq_len=enc_seq_len)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayerWithCrossAttention(dim=dim, num_heads=num_heads, dec_seq_len=dec_seq_len, enc_seq_len=enc_seq_len, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights between input and output embeddings
        self.output_projection.weight = self.embedding.weight
        
    def create_attention_mask(self, input_ids):
        """Create attention mask for padding tokens"""
        return (input_ids == self.pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def encode(self, input_ids, attention_mask=None):
        # Create padding mask if not provided
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)
            
        # Embed input tokens
        x = self.embedding(input_ids)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x, attention_mask=attention_mask)
            
        return x
    
    def decode(self, input_ids, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        # Create padding mask if not provided
        if self_attention_mask is None:
            self_attention_mask = self.create_attention_mask(input_ids)
            
        # Embed input tokens
        x = self.embedding(input_ids)
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(
                x,
                encoder_output,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask
            )
            
        # Project to vocabulary
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        return logits
    
    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attention_mask=None,
        decoder_attention_mask=None,
        cross_attention_mask=None
    ):
        # Encode
        encoder_output = self.encode(
            encoder_input_ids,
            attention_mask=encoder_attention_mask
        )
        
        # Decode
        logits = self.decode(
            decoder_input_ids,
            encoder_output,
            self_attention_mask=decoder_attention_mask,
            cross_attention_mask=cross_attention_mask
        )
        
        return logits
    