import torch
import torch.nn as nn
from .layers import EncoderLayer, DecoderLayer

class EncoderDecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int,
        dim: int = 256,
        num_heads: int = 9,
        num_encoder_layers: int = 14,
        num_decoder_layers: int = 7,
        enc_max_seq_len: int = 1024,
        dec_max_seq_len: int = 1024,
        mlp_dim: int = None,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super(EncoderDecoderModel, self).__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dim=dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout, enc_max_seq_len=enc_max_seq_len)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(dim=dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
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
    
    def generate(
        self,
        encoder_input_ids,
        decoder_start_token_id,
        max_length=None,
        encoder_attention_mask=None,
        temperature=1.0,
        top_k=None,
        top_p=None
    ):
        batch_size = encoder_input_ids.shape[0]
        max_length = max_length or self.max_seq_length
        device = encoder_input_ids.device
        
        # Encode input sequence
        encoder_output = self.encode(
            encoder_input_ids,
            attention_mask=encoder_attention_mask
        )
        
        # Initialize decoder input with start token
        decoder_input_ids = torch.full(
            (batch_size, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=device
        )
        
        # Generate tokens auto-regressively
        for _ in range(max_length - 1):
            # Get predictions for next token
            logits = self.decode(decoder_input_ids, encoder_output)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append next token to decoder input
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # Stop if all sequences have reached the end token
            if (next_token == self.pad_token_id).all():
                break
                
        return decoder_input_ids 