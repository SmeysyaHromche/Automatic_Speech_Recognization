from pathlib import Path
from src.config import TrainConfig
import argparse
import torch
import yaml

def get_train_config() -> TrainConfig:
    path_to_config = get_config_path()
    file = load_yaml(path_to_config)
    config = TrainConfig.model_validate(file)
    return config

def load_yaml(path:Path):
    with open(path, mode="r", encoding="UTF-8") as file:
        return yaml.safe_load(file)

def get_config_path():
    parser = argparse.ArgumentParser(prog="SFC", description="ASR")  # TODO: more official text
    parser.add_argument("-c", "--config", required=True)

    args = parser.parse_args()
    path_to_config = Path(args.config)

    return path_to_config

def extract_experiment_metadata(config:TrainConfig):
    return  {
        "device": config.device,
        "epochs": config.epochs,
        "lr": config.lr,
        "batch_size": config.batch_size,
        "audio_augmentation": config.train_data.augmentation.enable,
        "log_mel_augmentation": config.train_data.augmentation.mel_enable,
    }

def get_optimizer(config:TrainConfig, model:torch.nn.Module)->tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR | None]:
    optimizer = None
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params=model.params, lr=config.lr)
    else:
        raise ValueError(f"Unkown optimizer:{config.optimizer}! Use: adam, adamw")
    
    scheduler = None
    if config.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6
        )

    return optimizer, scheduler
