#!/usr/bin/env python
# Retrieve 10M tokens from Ultrachat
# ------------------------------------------------------------------

from datasets import load_dataset
from pathlib import Path
import json
import sys
import re
import unicodedata
from tokenizers import Tokenizer

TOKENIZER = re.compile(r"\S+")        # simple ≈word splitter; 10× faster than Python split
def n_tokens(txt: str) -> int:
    return len(TOKENIZER.findall(txt))


# -------- settings -------------------------------------------------
OUT_DIR = Path("data/SFT")
OUT_DIR.mkdir(exist_ok=True)


# -------- main loop ------------------------------------------------
print("▶ Retrieving 10M tokens from Ultrachat")

try:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

tok_so_far    = 0
target_tokens = 10_000_000
output_file   = OUT_DIR / "ultrachat_200k.jsonl"

print(f"Writing to {output_file}")

with output_file.open("w", encoding="utf-8") as f:
    try:
        for ex in ds:
            if tok_so_far >= target_tokens:
                break
            messages = ex["messages"]
            context  = ""
            for message in messages:
                if message['role'] == 'user':
                    if context:
                        context += " " # prepend a space so segments don't mash together, only after the first
                    context += f"<user> <s> {message['content']} </s>"
                else:
                    f.write(json.dumps({
                        "input_text":  context,
                        "output_text": "<assistant> <s> " + message['content'] + " </s>"
                    }, ensure_ascii=False) + "\n")
                    tok_so_far += n_tokens(message['content']) + n_tokens(context)
                    # add this assistant turn into context for next rounds
                    context += " " + f"<assistant> <s> {message['content']} </s>"
                    if tok_so_far >= target_tokens:
                        break  # stop inner loop once we've hit our budget

            # log every 100k tokens, but not at zero
            if tok_so_far and tok_so_far % 100_000 == 0:
                print(f"  ✓ {tok_so_far:,} tokens written")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
    finally:
        print(f"\nProcessed tokens: {tok_so_far:,}")

if tok_so_far >= target_tokens:
    print(f"\nSuccess! {tok_so_far:,} tokens written to {output_file}")
else:
    print(f"\nWarning: Only {tok_so_far:,} tokens written (target was {target_tokens:,})")
