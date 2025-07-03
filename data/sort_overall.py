from datasets import load_dataset, concatenate_datasets

#Not necessary for span correction -- instead we just cram the data together until we reach our batch size -- however, it's VERY 
#Useful for SFT later on!

bucket_size = 2**12 # 4096, so this works with all batch sizes

ds = load_dataset('json', data_files="data/MiniHQ_100M.jsonl", split="train")
ds = ds.filter(lambda x: x["text"].strip() != "")
ds = ds.shuffle(seed=42)

# Add text length column
ds = ds.map(lambda x: {"text_length": len(x["text"])})

# Chunk into buckets of 1000 and sort each bucket
buckets = [
    ds.select(range(i, min(i + bucket_size, len(ds)))).sort("text_length")
    for i in range(0, len(ds), bucket_size)
]

# Reassemble into one dataset
sorted_bucketted_ds = concatenate_datasets(buckets)

# Save as JSON
sorted_bucketted_ds.to_json("data/MiniHQ_100M.jsonl", orient="records", lines=True)