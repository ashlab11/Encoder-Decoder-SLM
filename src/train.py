from transformers import Trainer, TrainingArguments
from tokenizers import Tokenizer
from datasets import load_dataset
from components.basic_model import BasicEDModel
from data.pack_datasets import build_packed_dataset
from src.data_collator_for_span import SpanCorruptionCollator
import torch
import os
def main():
    torch.set_float32_matmul_precision("high")

    # 1) tokenizer & sentinel ids
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    # Token to id doesn't add <s>
    sentinel_ids = [tokenizer.token_to_id(f"<sentinel_{i}>") for i in range(100)]
    pad_token_id = tokenizer.token_to_id("<pad>")
    bos_token_id = tokenizer.token_to_id("<s>")
    eos_token_id = tokenizer.token_to_id("</s>") 

    # 2) model
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

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    # 3) dataset pipeline (streaming ok)
    ds = load_dataset(
        "json",
        data_files="data/MiniHQ_100M/slimpajama_100M.jsonl",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(buffer_size=10_000, seed=42)
    ds = build_packed_dataset(ds, tokenizer, 1024, eos_id=eos_token_id)

    eval_raw = load_dataset(
        "json",
        data_files="data/eval_sample.jsonl",
        split="train",
    )
    eval_ds = build_packed_dataset(eval_raw, tokenizer, 1024, eos_id=eos_token_id)
    
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
    max_steps = int(100_000_000 / 1024 / 8 / 4) # 100M tokens / 1024 tokens per seq / 8 batches / 4 accumulation ≈ 3052
    os.environ['ACCELERATE_DISPATCH_BATCHES'] = "false"
    # 5) TrainingArguments
    training_args = TrainingArguments(
        output_dir="models/base",
        per_device_train_batch_size=8,
        max_steps=max_steps,        # computed steps
        learning_rate=3e-4,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        save_steps=1000,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=50,
        save_total_limit=2,
        fp16=False,                     # use fp16 on nvidia gpu
        push_to_hub=False,
        remove_unused_columns=False,   # because our model doesn't expect extra cols
        #dataloader_num_workers=4,      # parallelize data loading
        dataloader_pin_memory=False,   # disable for MPS
        dataloader_drop_last=True,     # drop incomplete batches
        gradient_accumulation_steps=4,  # accumulate gradients over 4 batches (effective batch size = 32)
        ddp_find_unused_parameters=False  # disable parameter sync check
    )

    # 6) Trainer
    def compute_metrics(eval_preds):
        import math
        logits, labels = eval_preds
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        perplexity = torch.exp(loss)
        return {"perplexity": perplexity.item(), "loss": loss.item()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # 7) Kick off training
    trainer.train()
    trainer.evaluate()
    trainer.save_model("models/base/final")

if __name__ == "__main__":
    main()
