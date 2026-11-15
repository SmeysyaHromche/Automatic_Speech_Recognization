import datasets

dataset = datasets.load_dataset(
    "openslr/librispeech_asr",
    "clean",
    split="train.100",
    streaming=True,
)

dataset = dataset.cast_column(
    "audio",
    datasets.Audio(decode=False),
)

for idx, sample in enumerate(dataset):
    print(f"Batch {idx}: {sample['text']}", flush=True)

    if idx >= 4:
        break

print("Program finished", flush=True)