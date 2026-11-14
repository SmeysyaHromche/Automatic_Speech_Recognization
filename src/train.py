from src.data import AudioDataset
from src.utils import get_train_config
from pathlib import Path
from torch.utils.data import DataLoader
import torch

path_to_config = Path("/home/xkukht01/Dev/SFC/configs/train.yaml")  # TODO: pargs args
config = get_train_config(path_to_config)

device = torch.device(config.device if torch.cuda.is_available else "cpu")

dataset = AudioDataset(config.data)
dataloader = DataLoader(
    dataset=dataset, 
    batch_size=config.batch_size
)

for epoch in range(config.epochs):
    print(epoch)
