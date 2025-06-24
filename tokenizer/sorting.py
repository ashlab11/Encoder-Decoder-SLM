#!/usr/bin/env python3
# sort_by_token_len.py
import os, tempfile, heapq, pickle, itertools, gc
from pathlib import Path
from tqdm import tqdm
import sentencepiece as spm

TOKENIZER_PATH = 'tokenizer/tokenizer.model'
IN_FILE  = Path('data/slimpajama_chunk4_10G.txt')
OUT_FILE = Path('data/slimpajama_sample_sorted.txt')
CHUNK_SIZE = 1_000_000        # lines per in-RAM chunk; adjust to taste

tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)

##
## Phase 1 – produce sorted chunk files
##
tmp_files = []
with IN_FILE.open('r', encoding='utf-8') as fin:
    pbar = tqdm(total=None, desc='Chunking+Sorting')
    while True:
        pairs = []
        for _ in range(CHUNK_SIZE):
            line = fin.readline()
            if not line:
                break
            pairs.append((len(tokenizer.encode_as_ids(line)), line))
        if not pairs:
            break

        pairs.sort(key=lambda x: x[0])          # in-RAM sort
        fd, tmp_path = tempfile.mkstemp(suffix='.pkl')
        with os.fdopen(fd, 'wb') as fout:
            # dump only the *bytes*; no Python-object overhead during merge
            pickle.dump(pairs, fout, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_files.append(tmp_path)
        pbar.update(len(pairs))
        del pairs                               # free RAM
        gc.collect()
    pbar.close()

##
## Phase 2 – k-way merge stream
##
def pair_stream(tmp_path):
    with open(tmp_path, 'rb') as f:
        for length, line in pickle.load(f):
            yield length, line

streams = [pair_stream(p) for p in tmp_files]
merged  = heapq.merge(*streams, key=lambda x: x[0])  # lazy generator

with OUT_FILE.open('w', encoding='utf-8') as fout, \
     tqdm(total=None, desc='Merging+Writing') as pbar:
    for _, line in merged:
        fout.write(line)
        pbar.update()

##
## Phase 3 – cleanup
##
for p in tmp_files:
    os.remove(p)
print('✓ Done.  Sorted file written to', OUT_FILE)
