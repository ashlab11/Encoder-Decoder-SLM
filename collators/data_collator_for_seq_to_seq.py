import random
import torch
from scipy.stats import uniform

class SelfSeq2SeqCollator:
    """
    Data collator for self-supervised seq2seq tasks.
    Takes in a batch of sequences and returns a batch of padded input and label sequences.
    """
    def __init__(
        self,
        max_coverage = 0.8, 
        bos_token_id=2,
        pad_token_id=3,
        label_pad_token_id=-100,
    ):
        self.bos_token_id = bos_token_id
        self.label_pad_token_id = label_pad_token_id
        self.max_coverage = max_coverage
        self.pad_token_id = pad_token_id
        self.uniform_dist = uniform(1 - self.max_coverage, self.max_coverage)
        
    def __call__(self, batch):
        """
        batch: List[Dict[str, List[int]]]  each dict must have an "input_ids" key.
        Returns a dict with:
          - "input_ids": padded input sequences
          - "labels":   padded label sequences
          - "enc_key_padding_mask":  mask for padding tokens in the encoder
          - "dec_key_padding_mask":  mask for padding tokens in the decoder
          - "cross_key_padding_mask": mask for padding tokens in the cross-attention
        """
        inputs = []

        for example in batch:
            inputs.append(example["input_ids"])
            
        inputs, labels = self._create_seq2seq(inputs)

        return self._pad_sequences(inputs, labels)


    def _create_seq2seq(self, input_ids):
        """
        Creates seq2seq inputs and labels.
        """
        inputs = []
        labels = []
        for ids in input_ids:
            #Select percent that will be input and percent that will be output
            coverage = self.uniform_dist.rvs()
            input_len = len(ids)
            encoder_len = min(1024, int(input_len * coverage))
            decoder_len = min(512, int(input_len * coverage))
            
            input_ids = ids[:encoder_len]
            output_ids = ids[encoder_len:encoder_len + decoder_len]
            output_ids = [self.bos_token_id] + output_ids #add <bos> token
            inputs.append(input_ids)
            labels.append(output_ids)
            
        return inputs, labels

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
