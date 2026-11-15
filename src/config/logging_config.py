from pydantic import BaseModel

class LoggingConfig:
    log_in_cli:bool
    wandb_enable:bool
    wandb_project_name:str
    wandb_experiment_name:str
