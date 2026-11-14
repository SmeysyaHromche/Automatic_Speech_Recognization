from pathlib import Path
from src.config import TrainConfig
import yaml

def get_train_config(path:Path) -> TrainConfig:
    file = load_yaml(path)
    return TrainConfig.model_validate(file)

def load_yaml(path:Path):
    with open(path, mode="r", encoding="UTF-8") as file:
        return yaml.safe_load(file)