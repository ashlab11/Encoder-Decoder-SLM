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
        bos_token_id=2,
        pad_token_id=3, 
        assistant_token_id=4
    ):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.bos_token_id = bos_token_id
        self.assistant_token_id = assistant_token_id
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

        # Example: [5, 2, 10, 3] but max length 5
        # Labels pt 1: [5, 2, 10, 3, -100]
        # Decoder: [5, 2, 10, 3]
        # Labels 2: [2, 10, 3, -100]
        
        # 2) Creating decoder inputs + labels
        max_label_len = max(len(seq) for seq in labels)
        padded_labels = [
            seq + [self.label_pad_token_id] * (max_label_len - len(seq))
            for seq in labels
        ] #later we push labels by one but now we don't
        
        decoder_input_ids = [[
            (token if token != self.label_pad_token_id else self.pad_token_id)
            for token in label_seq[:-1]
        ] for label_seq in padded_labels]
        
        # Shift labels left by one to align with decoder inputs (length = max_label_len - 1)
        padded_labels = [
            seq[1:]
            for seq in padded_labels
        ]
        
        #Mask out the <assistant> tag
        padded_labels = [
            [
                (token if token != self.assistant_token_id else self.label_pad_token_id)
                for token in seq
            ]
            for seq in padded_labels
        ]
        
        decoder_key_padding_mask = [
            [tok_id == self.pad_token_id for tok_id in seq]
            for seq in decoder_input_ids
        ]
        
        return {
            "input_ids":                 torch.tensor(padded_inputs, dtype=torch.long),
            "decoder_input_ids":         torch.tensor(decoder_input_ids, dtype=torch.long),
            "labels":                    torch.tensor(padded_labels, dtype=torch.long),
            "enc_key_padding_mask":      torch.tensor(encoder_key_padding_mask, dtype=torch.bool),
            "dec_key_padding_mask":      torch.tensor(decoder_key_padding_mask, dtype=torch.bool),
            "cross_key_padding_mask":    torch.tensor(encoder_key_padding_mask, dtype=torch.bool) # same as encoder mask
        }
