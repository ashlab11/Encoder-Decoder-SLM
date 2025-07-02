#!/usr/bin/env python
# Retrieve 100M lines from SlimPajama CommonCrawl
# ------------------------------------------------------------------

from datasets import load_dataset
from pathlib import Path
import json
import sys
import re

TOKENIZER = re.compile(r"\S+")        # simple ≈word splitter; 10 × faster than Python split

def n_tokens(txt: str) -> int:
    return len(TOKENIZER.findall(txt))

# -------- settings -------------------------------------------------
OUT_DIR = Path("data/MiniHQ_100M")
OUT_DIR.mkdir(exist_ok=True)


token_counts = {
    "RedPajamaCommonCrawl": {"count": 50_000_000, "so_far": 0},
    "RedPajamaC4": {"count": 20_000_000, "so_far": 0},
    "RedPajamaGithub": {"count": 15_000_000, "so_far": 0},
    "RedPajamaArXiv": {"count": 10_000_000, "so_far": 0},
    "RedPajamaBook": {"count": 5_000_000, "so_far": 0}
}


# -------- main loop ------------------------------------------------
print("▶ Retrieving 100M lines from SlimPajama")

try:
    ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

tok_so_far = 0
target_tokens = 100_000_000
output_file = OUT_DIR / "slimpajama_cc_100M.jsonl"

print(f"Writing to {output_file}")

with output_file.open("w", encoding="utf-8") as f:
    try:
        for ex in ds:
            if tok_so_far >= target_tokens:
                break
            
            set_name = ex.get('meta', {}).get('redpajama_set_name')
            # Take enough examples from each! 
            if set_name not in token_counts.keys() or token_counts[set_name]["so_far"] >= token_counts[set_name]["count"]:
                continue
                
            f.write(json.dumps({"text": ex["text"]}, ensure_ascii=False) + "\n")
            tok_so_far += n_tokens(ex["text"])
            token_counts[set_name]["so_far"] += n_tokens(ex["text"])
            
            if tok_so_far % 1_000_000 == 0:
                print(f"  ✓ {tok_so_far:,} lines written")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
    finally:
        print(f"\nProcessed lines: {tok_so_far:,}")

if tok_so_far >= target_tokens:
    print(f"\nSuccess! {tok_so_far:,} tokens written to {output_file}")
else:
    print(f"\nWarning: Only {tok_so_far:,} tokens written to {output_file} (target was {target_tokens:,})")
