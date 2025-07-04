import random
import torch
from scipy.stats import poisson

class ConversationCollator:
    """
    Data collator for conversations.
    Takes in a batch of conversations and returns a batch of padded input and label sequences.
    """
    def __init__(
        self,
        label_pad_token_id=-100,
        pad_token_id=3
    ):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch):
        """
        batch: List[Dict[str, List[int]]]  each dict must have an "input_ids" and "output_ids" key.
        Returns a dict with:
          - "input_ids": padded input sequences
          - "labels":   padded label sequences
        """
        inputs = []
        labels = []

        for example in batch:
            inputs.append(example["input_ids"])
            labels.append(example["output_ids"])

        return self._pad_sequences(inputs, labels)

    def _pad_sequences(self, input_ids, labels):
        """
        Pads input and label sequences and returns:
          - padded_inputs
          - padded_labels
          - encoder_key_padding_mask
          - decoder_key_padding_mask
          - cross_key_padding_mask
        """
        # 1) Pad encoder inputs
        max_input_len = max(len(seq) for seq in input_ids)
        padded_inputs = [
            seq + [self.pad_token_id] * (max_input_len - len(seq))
            for seq in input_ids
        ]
        # True = pad, False = real token
        encoder_key_padding_mask = [
            [False] * len(seq) + [True]  * (max_input_len - len(seq))
            for seq in input_ids
        ] # [B, L]

        # 2) Pad decoder labels
        max_label_len = max(len(seq) for seq in labels)
        padded_labels = [
            seq + [self.label_pad_token_id] * (max_label_len - len(seq))
            for seq in labels
        ] 
        
        decoder_key_padding_mask = [
            [False] * len(seq) + [True]  * (max_label_len - len(seq))
            for seq in labels
        ] # [B, L]
        
        # 3) Shift labels right to create decoder inputs (replace -100 with pad)
        decoder_input_ids = [
            [self.bos_token_id] +
            [
                (token if token != self.label_pad_token_id else self.pad_token_id)
                for token in label_seq[:-1]
            ]
            for label_seq in padded_labels
        ]
        
        return {
            "input_ids":                 torch.tensor(padded_inputs, dtype=torch.long),
            "decoder_input_ids":         torch.tensor(decoder_input_ids, dtype=torch.long),
            "labels":                    torch.tensor(padded_labels, dtype=torch.long),
            "enc_key_padding_mask":      torch.tensor(encoder_key_padding_mask, dtype=torch.bool),
            "dec_key_padding_mask":      torch.tensor(decoder_key_padding_mask, dtype=torch.bool),
            "cross_key_padding_mask":    torch.tensor(encoder_key_padding_mask, dtype=torch.bool) # same as encoder mask
        }
