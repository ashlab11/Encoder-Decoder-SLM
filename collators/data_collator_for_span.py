import random
import torch
from scipy.stats import poisson

class SpanCorruptionCollator:
    """
    Data collator for span‐corruption (T5 style).  
    Inserts sentinel tokens in the input and builds corresponding labels.
    """
    def __init__(
        self,
        sentinel_ids,
        span_length=3,
        mask_probability=0.15,
        label_pad_token_id=-100,
        pad_token_id=3,
        bos_token_id=2,
        eos_token_id=4
    ):
        self.sentinel_ids = sentinel_ids
        self.span_length = span_length
        self.mask_probability = mask_probability
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.poisson = poisson(mu=span_length)

    def __call__(self, batch):
        """
        batch: List[Dict[str, List[int]]]  each dict must have an "input_ids" key.
        Returns a dict with:
          - "input_ids": masked & padded input sequences
          - "labels":   padded label sequences (with sentinel spans)
        """
        corrupted_inputs = []
        label_sequences = []

        for example in batch:
            input_ids = example["input_ids"]
            masked_input, labels = self._corrupt_sequence(input_ids)
            labels.append(self.eos_token_id) #Teaching model to predict eos
            corrupted_inputs.append(masked_input)
            label_sequences.append(labels)

        return self._pad_sequences(corrupted_inputs, label_sequences)

    def _corrupt_sequence(self, input_ids):
        """
        Walks through input_ids and with probability mask_probability
        replaces spans of length span_length with unique sentinel tokens.
        Returns (masked_input, labels) for a single sequence.
        """
        masked_input = []
        labels = []
        sentinel_iterator = iter(self.sentinel_ids)
        sentinels_left = len(self.sentinel_ids)
        position = 0

        while position < len(input_ids):
            if sentinels_left > 0 and random.random() < self.mask_probability / self.span_length:
                sentinel_token = next(sentinel_iterator)
                masked_input.append(sentinel_token)
                labels.append(sentinel_token)

                span_end = position + self.poisson.rvs()
                original_span = input_ids[position:span_end]
                labels.extend(original_span)

                position = span_end
                sentinels_left -= 1
            else:
                masked_input.append(input_ids[position])
                position += 1

        return masked_input, labels
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
