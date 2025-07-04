from tokenizers import Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import torch
from components.base.basic_model import BasicEDModel
from components.router.router_model import RouterModel
from data.pack_datasets import build_packed_dataset
from collators.data_collator_for_span import SpanCorruptionCollator
import os
from tqdm import tqdm

def main():
    # 1) tokenizer & sentinel ids
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    # Token to id doesn't add <s>
    sentinel_ids = [tokenizer.token_to_id(f"<sentinel_{i}>") for i in range(100)]
    pad_token_id = tokenizer.token_to_id("<pad>")
    bos_token_id = tokenizer.token_to_id("<s>")
    eos_token_id = tokenizer.token_to_id("</s>") 
 
    # 2) model -- for now we use base model, will change eventually
    model = BasicEDModel(
        vocab_size=tokenizer.get_vocab_size(),
        dim=256,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=3,
        enc_seq_len=1024,
        dec_seq_len=512,
        pad_token_id=pad_token_id
    )
 
    # 3) dataset pipeline (streaming ok)
    ds = load_dataset("json",
                    data_files="data/Pretrain/slimpajama_100M.jsonl",
                    split="train",
                    streaming=True)
    ds = ds.shuffle(buffer_size=10_000, seed=42)
    ds = build_packed_dataset(ds, tokenizer, 1024, eos_id=eos_token_id)
    
    eval_ds = load_dataset("json",
                        data_files="data/eval_sample.jsonl",
                        split="train")
    eval_ds = build_packed_dataset(eval_ds, tokenizer, 1024, eos_id=eos_token_id)
    
    # 4) collator
    collator = SpanCorruptionCollator(
        sentinel_ids=sentinel_ids,
        span_length=3,
        mask_probability=0.15,
        pad_token_id=pad_token_id,
        label_pad_token_id=-100,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id
    )

    batch_size = 8
    grad_accum_steps = 4
    logging_steps = 100
    eval_steps = 500
    save_steps = 1000
    
    max_steps = int(100_000_000 / 1024 / batch_size / grad_accum_steps)

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=collator,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )
    
    eval_dataloader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        collate_fn=collator,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    device = "cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=max_steps,
    )
    scaler = torch.amp.GradScaler(device='cuda') if torch.cuda.is_available() else None

    model.train()
    os.makedirs("models/base", exist_ok=True)
    optimizer.zero_grad()
    step = 0
    for micro_step, batch in tqdm(enumerate(dataloader), total=max_steps * grad_accum_steps, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast(device_type=device, dtype=torch.float16):
            outputs = model(**batch)
            loss = outputs["loss"] / grad_accum_steps
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (micro_step + 1) % grad_accum_steps == 0:
            if scaler:
                # Unscale gradients before clipping when using mixed precision
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Clip gradients to a global norm of 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % logging_steps == 0:
                print(f"Step {step}/{max_steps} - loss: {loss.item() * grad_accum_steps:.4f}")

            #FIX PERPLEXITY CALCULATIONS!
            if step % eval_steps == 0:
                model.eval()
                with torch.no_grad():
                    eval_loss = 0
                    total_tokens = 0
                    for eval_batch in eval_dataloader:
                        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                        with torch.amp.autocast(device_type=device, dtype=torch.float16):
                            eval_outputs = model(**eval_batch)
                            num_tokens = (eval_batch['labels'] != -100).sum()
                            eval_loss = eval_outputs["loss"] * num_tokens
                        eval_loss += eval_loss
                        total_tokens += num_tokens
                        
                    eval_loss /= total_tokens
                    eval_ppl = torch.exp(eval_loss)
                    print(f"Step {step}/{max_steps} - eval perplexity: {eval_ppl:.4f}")
                    model.train()
                        
            if step % save_steps == 0:
                torch.save(model.state_dict(), f"models/base/checkpoint_{step}.pt")


    torch.save(model.state_dict(), "models/base/final.pt")


if __name__ == "__main__":
    main()
