import datasets
from datasets import load_dataset, IterableDataset
import itertools, sentencepiece as spm
import torch
import pandas as pd
import sentencepiece as spm

def pack_iter(text_iter, tokenizer, ctx_len): 
    """Yield packed lists of token-ids <= ctx_len."""
    
    EOS_ID = tokenizer.eos_id()
    current = []
    for doc in text_iter:
        ids = tokenizer.encode(doc)
        while ids: #ids may even be twice as long as the ctx length
            if len(ids) + len(current) > ctx_len - 1:
                current += ids[:(ctx_len - len(current))]
                yield current
                current = []
                ids = ids[ctx_len - len(current):]
            else:
                current += ids
                current.append(EOS_ID)  # Only add EOS when we finish a document
                ids = []
    if current:  # Only yield remaining tokens if we have any
        yield current

def build_packed_dataset(ds, tokenizer, ctx_len):
    # ds is any hf streaming dataset with a "text" column
    def generator(): 
        for row in ds:
            yield from pack_iter([row["text"]], tokenizer, ctx_len)
    return IterableDataset.from_generator(
        lambda: ({"input_ids": seq} for seq in generator()),
        features=datasets.Features({
            "input_ids": datasets.Sequence(feature=datasets.Value("int32"))
        }),
    )

