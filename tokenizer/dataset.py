# stream_dataset.py
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm

class StreamTextDataset(IterableDataset):
    """
    Streams a sorted-by-length text file **line-by-line**, tokenises on the fly,
    and yields 1-D int tensors.  Nothing except a small read-ahead buffer ever
    lives in RAM.

    Args
    ----
    file_path : str
        Path to the (sorted) text file.
    tokenizer_model : str
        Path to a SentencePiece `.model` file.
    min_chars : int, optional
        Skip lines shorter than this (default: 50).
    """

    def __init__(self, file_path: str, tokenizer_model: str):
        super().__init__()
        self.file_path  = file_path
        self.tokenizer  = spm.SentencePieceProcessor(model_file=tokenizer_model)
    # --------------------------------------------------------------------- #
    # Required by IterableDataset                                           #
    # --------------------------------------------------------------------- #
    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Tokenise the line and convert to tensor
                tokens = self.tokenizer.encode(line.strip(), out_type=int)
                if tokens:
                    yield torch.tensor(tokens, dtype=torch.int64)