# Code to train the model using span correction loss
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup, DataCollatorForLanguageModeling
from transformers import Pipeline
from components.basic_model import BasicEDModel
import json
from torch.cuda.amp import autocast
import os
from transformers import AutoTokenizer
import sentencepiece as spm
from datasets import load_dataset

def train_model(model, 
                train_dataloader, 
                num_epochs, 
                output_dir,
                lr = 3e-4, 
                device = 'mps'):
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=10000)
    

def main():
    tokenizer = spm.SentencePieceProcessor(model_file='tokenizer/tokenizer.model') # Loading the tokenizer
    model = BasicEDModel(
        vocab_size=tokenizer.vocab_size(),
        dim=256, 
        num_heads = 8, 
        num_encoder_layers = 6, 
        num_decoder_layers = 3, 
        enc_seq_len = 1024, 
        dec_seq_len = 512, 
    )
    device = torch.device("mps")
    model.to(device)

        
    
if __name__ == "__main__":
    main()