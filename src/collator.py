from dataclasses import dataclass
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollatorForSpanCorrection:
    """
    Data-collator for T5-style span corruption / span correction.

    Args
    ----
    tokenizer: a Hugging Face tokenizer that **contains <extra_id_0>, <extra_id_1> …**.
    noise_density: percentage of tokens to mask (T5-Base uses 0.15).
    mean_noise_span_length: average length of each masked span (T5 uses 3.0).
    pad_to_multiple_of: if given, pads length to nearest multiple (e.g. 8 for tensor cores).

    Returns from __call__
    ---------------------
    {
        "input_ids":         encoder_input_ids,
        "attention_mask":    encoder_attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "labels":            decoder_input_ids,        # identical (no shift here – letting model wrapper do it)
    }
    All tensors are left-padded with `tokenizer.pad_token_id`.
    """
    
    tokenizer: PreTrainedTokenizerBase
    mlm_prob: float = 0.15
    mean_span_length: float = 3.0
    
    def __precompute_sentinels__(self):
        self.sentinel_ids = [
            self.tokenizer.encode(f"<sentinel_{i}>")
            for i in range(self.tokenizer.num_special_tokens)
        ]
        self.sentinel_ids = self.sentinel_ids[::-1] # reverse order to pop from the end
    
    # ----- HELPER FUNCTIONS -----
    def _random_spans_mask(self, length: int):
        """Returns a bool mask where True == keep, False == mask"""
        num_spans = max(1, int(length * self.noise_density / self.mean_span_length))