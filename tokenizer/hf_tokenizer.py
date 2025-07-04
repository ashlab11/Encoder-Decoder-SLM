from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.normalizers import NFKC, Sequence, Strip
import os

# Constants
NUM_SENTINELS = 100
VOCAB_SIZE = 8192
CORPUS_FILE = "data/Pretrain/slimpajama_100M.txt"
MODEL_PREFIX = "tokenizer/tokenizer"


def create_tokenizer():
    # 1) Initialize a BPE tokenizer with an unk token
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # 2) Normalization: NFKC + strip
    tokenizer.normalizer = Sequence([NFKC(), Strip()])

    # 3) Pre-tokenization: byte-level (whitespace+punctuation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # 4) Special tokens (trainer will assign IDs automatically at the top of the vocab)
    sentinel_tokens = [f"<sentinel_{i}>" for i in range(NUM_SENTINELS)]
    special_tokens = [
        "<pad>", "<unk>", "<s>", "</s>",
        "<user>", "<assistant>",
        *sentinel_tokens
    ]

    # 5) Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        show_progress=True
    )

    if not os.path.exists(CORPUS_FILE):
        raise FileNotFoundError(f"Corpus file not found: {CORPUS_FILE}")

    tokenizer.train([CORPUS_FILE], trainer)

    # 6) Post-processing: add <s>…</s> around single/pair sequences
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ]
    )

    # 7) Decoder (optional)
    tokenizer.decoder = decoders.ByteLevel()

    # 8) Padding
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("<pad>"),
        pad_token="<pad>"
    )

    # 9) (Optional) Truncation
    # tokenizer.enable_truncation(max_length=1024)

    tokenizer.save(f"{MODEL_PREFIX}.json")
    return tokenizer


def test_tokenizer(tokenizer):
    """Test the tokenizer with various inputs"""
    print("\nTesting sentinel token encoding:")
    for i in range(5):
        token = f"<sentinel_{i}>"
        output = tokenizer.encode(token)
        print(f"{token}: ids={output.ids}, tokens={output.tokens}")

    print("\nTesting special token encoding:")
    test_inputs = ["<sentinel_3>", "<user>", "<assistant>", "Hello, world!"]
    for text in test_inputs:
        output = tokenizer.encode(text)
        print(f"{text} -> ids={output.ids}, tokens={output.tokens}")

    print("\nTesting sentence pair encoding:")
    output = tokenizer.encode("Hello", "World")
    print(f"Pair encoding: ids={output.ids}, tokens={output.tokens}")

if __name__ == "__main__":
    tokenizer = create_tokenizer()
    test_tokenizer(tokenizer)
