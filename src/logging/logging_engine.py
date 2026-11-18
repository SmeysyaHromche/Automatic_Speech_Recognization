from pathlib import Path
from src.config import TrainConfig
import torch
import wandb

GLOBAL_STEP_DIM = "global_step"
EPOCH_DIM = "epoch"

TRAIN_STEP_METRIC = "train/step_"
TRAIN_EPOCH_METRIC = "train/epoch_"
VALID_METRIC = "valid/"

def extract_experiment_metadata(config:TrainConfig):
    return  {
        "device": config.device,
        "epochs": config.epochs,
        "lr": config.lr,
        "batch_size": config.batch_size
    }

def wandb_init_train(config:TrainConfig) -> wandb.Run:
    config_wandb = extract_experiment_metadata(config)

    run = wandb.init(
        project = config.logging_config.wandb_project_name,
        name=config.logging_config.wandb_experiment_name,
        config=config_wandb
    )

    run.define_metric(GLOBAL_STEP_DIM)
    run.define_metric(EPOCH_DIM)

    run.define_metric(f"{TRAIN_STEP_METRIC}*", step_metric=GLOBAL_STEP_DIM)
    run.define_metric(f"{TRAIN_EPOCH_METRIC}*", step_metric=EPOCH_DIM)
    run.define_metric(F"{VALID_METRIC}*", step_metric=EPOCH_DIM)

    return run


def log_wandb_one_train_step(
    run:wandb.Run,
    global_step:int,
    loss:float,
    grad_norm:float
) -> None:
    run.log({
        GLOBAL_STEP_DIM: global_step,
        f"{TRAIN_STEP_METRIC}loss": loss,
        f"{TRAIN_STEP_METRIC}grad_norm": grad_norm,
    })

def log_cli_one_train_step(
    epoch: int,
    epochs: int,
    idx_batch:int,
    global_step:int,
    loss:float,
    grad_norm:float      
) -> None:
    print(f"Train per step: {epoch}/{epochs} epoch, {idx_batch} batch, {global_step} glob step, {loss:.4f} loss, {grad_norm:.4f} grad")

def log_wandb_one_train_epoch(
    run:wandb.Run,
    epoch:int,
    avg_loss:float,
    avg_grad_norm:float
) -> None:
    run.log({
        EPOCH_DIM: epoch,
        f"{TRAIN_EPOCH_METRIC}avg_loss": avg_loss,
        f"{TRAIN_EPOCH_METRIC}avg_grad_norm": avg_grad_norm,
    })

def log_cli_one_train_epoch(
    epoch:int,
    epochs:int,
    avg_loss:float,
    avg_grad_norm:float      
) -> None:
    print(f"Train per epoch: {epoch}/{epochs} epoch, {avg_loss:.4f} avg loss, {avg_grad_norm:.4f} avg grad norm")

def log_wandb_validation(
    run:wandb.Run,
    epoch:int,
    validation_prefix:str, 
    avg_loss:float, 
    cer:float,
    wer:float
)->None:
    run.log({
        EPOCH_DIM: epoch,
        f"{validation_prefix}avg_loss": avg_loss,
        f"{validation_prefix}cer": cer,
        f"{validation_prefix}wer": wer,
    })

def log_cli_validation(
    epoch:int,
    epochs:int,
    avg_loss:float,
    cer:float, 
    wer:float  
) -> None:
    print(f"Eval per epoch: {epoch}/{epochs} epoch, {avg_loss:.4f} avg loss, {cer:.4f} cer, {wer:.4f} wer")


def save_model(
    path_to_save:Path,
    epoch:int,
    global_step:int,
    model:torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    validation_cer:float,
    validation_wer:float,
    validation_loss:float,
):
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_cer": validation_cer,
            "validation_wer": validation_wer,
            "validation_loss": validation_loss,
        },
        path_to_save
    )