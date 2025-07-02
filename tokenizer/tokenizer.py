import sentencepiece as spm
import os

# Paths and parameters
CORPUS_FILE   = "data/MiniHQ_100M/slimpajama_cc_100M.jsonl"
MODEL_DIR     = "tokenizer"
MODEL_PREFIX  = os.path.join(MODEL_DIR, "tokenizer")
VOCAB_SIZE    = 8192
CHAR_COVERAGE = 0.9995  # Lower coverage for English-only text
#MIN_SENTENCE_LENGTH = 50
MAX_SENTENCE_LENGTH = 10000
SAMPLE_SIZE = 10000000  

# Ensure output directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

NUM_SENTINELS = 100
sentinel_tokens = [f"<sentinel_{i}>" for i in range(NUM_SENTINELS)]

# Define special tokens
special_tokens = [
    "<context>", "<answer>", *sentinel_tokens
]

# Basic ASCII punctuation and control symbols
control_symbols = [".", ",", "!", "?", "-", "'", '"', "(", ")", "[", "]", "{", "}", ":", ";", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "~", "`"]

# Train the SentencePiece BPE tokenizer
spm.SentencePieceTrainer.Train(
    input=CORPUS_FILE,
    model_prefix=MODEL_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type='bpe',
    character_coverage=CHAR_COVERAGE,
    pad_id=3,
    user_defined_symbols=special_tokens,
    control_symbols=control_symbols,
    normalization_rule_name='nmt_nfkc',  # Normalize unicode characters
    max_sentence_length=MAX_SENTENCE_LENGTH,
    input_sentence_size=SAMPLE_SIZE,
    shuffle_input_sentence=True
)

print(f"Trained tokenizer saved as {MODEL_PREFIX}.model and {MODEL_PREFIX}.vocab")
