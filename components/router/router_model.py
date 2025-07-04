import torch
import torch.nn as nn
from ..base.layers import EncoderLayer, DecoderLayerWithCrossAttention
import torch.nn.functional as F

class RouterModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 14,
        num_decoder_layers: int = 7,
        enc_seq_len: int = 1024,
        dec_seq_len: int = 1024,
        mlp_dim: int = None,
        dropout: float = 0.1,
        pad_token_id: int = 0, 
        label_pad_token_id: int = -100
    ):
        super(RouterModel, self).__init__()
        self.dim = dim
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
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
        
        # Initialize weights using Xavier initialization for Linear layers
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Apply Xavier initialization to all Linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
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
    
    def generate(self, input, max_new_tokens, tokenizer, temperature=1.0, top_k=50, top_p=0.9, device=torch.device('cpu')):
        """Given an input of a prompt, generate a response"""
        eos_token_id = tokenizer.token_to_id("</s>") 
        input_ids = tokenizer.encode(input).ids # [L]
        encoder_output = self.encode(input_ids) # [L, D]
        decoder_input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
        
        while decoder_input_ids[:, -1] != eos_token_id and len(decoder_input_ids) < max_new_tokens:
            logits = self.decode(decoder_input_ids, encoder_output)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                indices_to_remove = logits < values[:, [-1]]
                logits[indices_to_remove] = -float('inf')
            if top_p > 0:
                sorted_logits, _ = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
                logits[sorted_indices_to_remove] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
        return tokenizer.decode(decoder_input_ids[0].tolist())
        
        