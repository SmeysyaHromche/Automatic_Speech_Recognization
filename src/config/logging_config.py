from pydantic import BaseModel

class LoggingConfig(BaseModel):
    path_log_file:str
    log_in_cli:bool
    log_in_cli_per_step:bool
    wandb_enable:bool
    wandb_project_name:str
    wandb_experiment_name:str
    logging_each_n_step: int
    random_gen_output:int
