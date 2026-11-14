
from src.model import CNN_LSTM_model
from pathlib import Path
from src.data import AudioDataset, collate_fn
from src.utils import get_train_config
from torch.utils.data import DataLoader
import torch

path_to_config = Path("/home/xkukht01/Dev/SFC/configs/train.yaml")  # TODO: pargs args
config = get_train_config(path_to_config)

device = torch.device(
    config.device if torch.cuda.is_available() else "cpu"
)

dataset = AudioDataset(config.data)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=config.batch_size,
    collate_fn=collate_fn,
    num_workers=0
)

model = CNN_LSTM_model(
    encoder_output_dim=256,
    hidden_decoder_dim=256,
    vocab_size=31
).to(device)

model.train()

for epoch in range(config.epochs):
    for idx_batch, (audios, texts, lengths) in enumerate(dataloader):

        audios = audios.to(device)

        logits = model(audios)

        print("Input:", audios.shape)
        print("Output:", logits.shape)
        print("Lengths:", lengths)
        print("Texts:", texts[:2])

        if idx_batch >= 10:
            break

    break
