import torch
import torch.nn as nn
from .layers import EncoderLayer, DecoderLayerWithCrossAttention
import torch.nn.functional as F

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
    
    def encode(self, input_ids, enc_key_padding_mask=None):
        # Create padding mask if not provided
        if enc_key_padding_mask is None:
            enc_key_padding_mask = self.create_attention_mask(input_ids)
            
        # Embed input tokens
        x = self.embedding(input_ids)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x, enc_key_padding_mask=enc_key_padding_mask)
            
        return x
    
    def decode(self, input_ids, encoder_output, dec_key_padding_mask=None, cross_key_padding_mask=None):
        # Create padding mask if not provided
        if dec_key_padding_mask is None:
            dec_key_padding_mask = self.create_attention_mask(input_ids)
            
        # Embed input tokens
        x = self.embedding(input_ids)
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(
                x,
                encoder_output,
                dec_key_padding_mask=dec_key_padding_mask,
                cross_key_padding_mask=cross_key_padding_mask
            )
            
        # Project to vocabulary
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        return logits
    
    def forward(
        self,
        input_ids,
        decoder_input_ids,
        labels,
        enc_key_padding_mask=None,
        dec_key_padding_mask=None,
        cross_key_padding_mask=None
    ):
        # Encode
        encoder_output = self.encode(
            input_ids,
            enc_key_padding_mask=enc_key_padding_mask
        )
        
        # Decode
        logits = self.decode(
            decoder_input_ids,
            encoder_output,
            dec_key_padding_mask=dec_key_padding_mask,
            cross_key_padding_mask=cross_key_padding_mask
        )
        
        # Computing loss here
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.label_pad_token_id
        )
        return {"loss": loss, "logits": logits}
    