#!/usr/bin/env python
# sample_100M.py  —  build a 100 M-token high-quality corpus
# ------------------------------------------------------------------
# Outputs four *.jsonl files (and an optional merged file) in ./MiniHQ_100M
# Token-count uses a quick whitespace split; swap in your tokenizer later
# if you want an exact count.

from datasets import load_dataset
from pathlib import Path
import json, itertools, re

# -------- settings -------------------------------------------------
TARGETS = {
    # repo_id / subset : (split, target_tokens)
    ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup"): ("train", 60_000_000),
    ("permutans/fineweb-bbc-news", "sample-350BT"):        ("train", 20_000_000),
    ("roneneldan/TinyStories", None):                     ("train", 15_000_000),
    ("m-a-p/FineFineWeb", None):                ("train",  5_000_000),
}

OUT_DIR = Path("MiniHQ_100M")
OUT_DIR.mkdir(exist_ok=True)

TOKENIZER = re.compile(r"\S+")        # simple ≈word splitter; 10 × faster than Python split

def n_tokens(txt: str) -> int:
    return len(TOKENIZER.findall(txt))

def extract_text(example):
    for k in ("text", "content", "story"):          # handle field-name differences
        if k in example and example[k]:
            return example[k].strip()
    return None

# -------- main loop ------------------------------------------------
for (repo, subset), (split, target) in TARGETS.items():
    label = f"{repo.split('/')[-1]}_{subset or 'nosub'}"
    path  = OUT_DIR / f"{label}.jsonl"
    tok_so_far, doc_so_far = 0, 0

    print(f"▶ Sampling {label}  —  target {target/1e6:.0f} M tokens")
    ds = load_dataset(repo, subset, split=split, streaming=True)

    with path.open("w", encoding="utf-8") as f:
        for ex in ds:
            if tok_so_far >= target:
                break
            txt = extract_text(ex)
            if not txt:
                continue
            ntok = n_tokens(txt)
            f.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
            tok_so_far += ntok
            doc_so_far += 1

    print(f"  ✓ {doc_so_far:,} docs | {tok_so_far:,} tokens → {path}")

# -------- optional: concatenate into a single file -----------------
merged = OUT_DIR / "MiniHQ_100M.jsonl"
with merged.open("w", encoding="utf-8") as fout:
    for part in OUT_DIR.glob("*.jsonl"):
        if part.name.endswith("MiniHQ_100M.jsonl"):
            continue
        with part.open() as fin:
            for line in fin:
                fout.write(line)
print(f"\nAll done!  Merged corpus at {merged}")
