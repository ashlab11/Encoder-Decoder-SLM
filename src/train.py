# Code to train the model using span correction loss
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup
from transformers import Pipeline
from components.model import EncoderDecoderModel
import json
from torch.cuda.amp import autocast
import os

def train_model(model, 
                train_dataloader, 
                num_epochs=1, 
                lr = 3e-4, 
                device = 'mps'):
    pass
