from src.config import TrainConfig
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
    loss_value:float,
    grad_norm_value:float
) -> None:
    run.log({
        GLOBAL_STEP_DIM: global_step,
        f"{TRAIN_STEP_METRIC}loss": loss_value,
        f"{TRAIN_STEP_METRIC}grad_norm": grad_norm_value,
    })

def log_cli_one_train_step(
    epoch: int,
    epochs: int,
    idx_batch:int,
    global_step:int,
    loss_value:float,
    grad_norm_value:float      
) -> None:
    print(f"Train per step: {epoch}/{epochs} epoch, {idx_batch} batch, {global_step} glob step, {loss_value:.4f} loss, {grad_norm_value:.4f} grad")

def log_wandb_one_train_epoch(
    run:wandb.Run,
    epoch:int,
    avg_loss_value:float,
    avg_grad_norm_value:float
) -> None:
    run.log({
        EPOCH_DIM: epoch,
        f"{TRAIN_EPOCH_METRIC}avg_loss": avg_loss_value,
        f"{TRAIN_EPOCH_METRIC}avg_grad_norm": avg_grad_norm_value,
    })

def log_cli_one_train_epoch(
    epoch:int,
    epochs:int,
    avg_loss_value:float,
    avg_grad_norm_value:float      
) -> None:
    print(f"Train per epoch: {epoch}/{epochs} epoch, {avg_loss_value:.4f} avg loss, {avg_grad_norm_value:.4f} avg grad norm")

def log_wandb_validation(
    run:wandb.Run,
    epoch:int,
    validation_prefix:str, 
    avg_loss_value:float, 
    wer_value:float
)->None:
    run.log({
        EPOCH_DIM: epoch,
        f"{validation_prefix}avg_loss": avg_loss_value,
        f"{validation_prefix}wer": wer_value,
    })

def log_cli_validation(
    epoch:int,
    epochs:int,
    avg_loss_value:float, 
    wer_value:float  
) -> None:
    print(f"Eval per epoch: {epoch}/{epochs} epoch, {avg_loss_value:.4f} avg loss, {wer_value:.4f} wer")
