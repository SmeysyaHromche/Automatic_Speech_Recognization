from pathlib import Path
from src.config import TrainConfig
import argparse
import wandb
import yaml

WANDB_PROJECT = "SFC"

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
        "batch_size": config.batch_size
    }

def wandb_init_train(config:TrainConfig):
    config_wandb = extract_experiment_metadata(config)

    run = wandb.init(
        project = config.logging_config.wandb_project_name,
        name=config.logging_config.wandb_experiment_name,
        config=config_wandb
    )

    return run