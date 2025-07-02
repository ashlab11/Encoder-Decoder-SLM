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
from torch.nn.utils.rnn import pad_sequence


#NEED TO CREATE DATALOADER! THIS IS WEIRD IT'S T5 STYLE!

def train_model(model, 
                num_epochs, 
                output_dir,
                lr = 3e-4, 
                device = 'mps'):
    
    ds = load_dataset("json", data_files="data/MiniHQ_100M.jsonl")

    batch_size = 2
    loader = DataLoader(ds, batch_size = batch_size, num_workers = 0, shuffle = False, collate_fn=lambda x: x, 
                        prefetch_factor=1)
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=10000)
    
    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(loader):
            if idx % 1000 == 0:
                #Save model
                print(f"Saving model at epoch {epoch} and batch {idx}")
                torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                
        #Forward pass, T5 style
        
            
    

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
    
    train_model(model, num_epochs=1, output_dir='models/base/', lr = 3e-4)
    


if __name__ == "__main__":
    main()