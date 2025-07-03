import datasets
from datasets import load_dataset, IterableDataset, Dataset
import itertools, sentencepiece as spm
import torch
import pandas as pd
import sentencepiece as spm
import re


def pack_iter(ds, tokenizer, ctx_len, eos_id=0): 
    """Yield packed lists of token-ids <= ctx_len."""
    
    current = []
    for doc in ds:
        # First normalize using the same normalization as training
        ids = tokenizer.encode(doc["text"]).ids
        while ids: #ids may even be twice as long as the ctx length
            if len(ids) + len(current) > ctx_len - 1:
                current += ids[:(ctx_len - len(current))]
                yield {"input_ids": torch.tensor(current, dtype=torch.long)}
                current = []
                ids = ids[ctx_len - len(current):]
            else:
                current += ids
                current.append(eos_id)  # Only add EOS when we finish a document
                ids = []
    if current:  # Only yield remaining tokens if we have any
        yield {"input_ids": torch.tensor(current, dtype=torch.long)}

def build_packed_dataset(ds, tokenizer, ctx_len, eos_id=0):
    # ds is any hf streaming dataset with a "text" column
    def generator():
        yield from pack_iter(ds, tokenizer, ctx_len)

    return IterableDataset.from_generator(generator)

