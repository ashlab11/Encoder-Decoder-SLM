import datasets
from datasets import load_dataset, IterableDataset
import itertools, sentencepiece as spm
import torch
import pandas as pd
import sentencepiece as spm
import re


def pack_iter(ds, tokenizer, ctx_len): 
    """Yield packed lists of token-ids <= ctx_len."""
    
    print("Starting to pack??")
    EOS_ID = tokenizer.eos_id()
    current = []
    for doc in ds:
        # First normalize using the same normalization as training
        ids = tokenizer.encode(doc["text"])
        if ids:
            print("Doc text:", doc["text"])
            print("Tokenized, ids are", ids)
        while ids: #ids may even be twice as long as the ctx length
            print("Checking if we need to yield")
            if len(ids) + len(current) > ctx_len - 1:
                print("Yielding!")
                current += ids[:(ctx_len - len(current))]
                print("Yielding!")
                yield {"input_ids": current}
                current = []
                ids = ids[ctx_len - len(current):]
            else:
                print("Adding to current")
                current += ids
                current.append(EOS_ID)  # Only add EOS when we finish a document
                ids = []
    if current:  # Only yield remaining tokens if we have any
        yield {"input_ids": current}

def build_packed_dataset(ds, tokenizer, ctx_len):
    # ds is any hf streaming dataset with a "text" column
    def generator():
        yield from pack_iter(ds, tokenizer, ctx_len)

    return IterableDataset.from_generator(generator)

