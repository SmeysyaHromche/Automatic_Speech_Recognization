from pathlib import Path
from src.config import TrainConfig
import random
import torch
import wandb

GLOBAL_STEP_DIM = "global_step"
EPOCH_DIM = "epoch"

TRAIN_STEP_METRIC = "train/step_"
TRAIN_EPOCH_METRIC = "train/epoch_"
VALID_METRIC = "valid/"

def write_to_log_file(path_to_log_file:Path, log:str):
    with open(path_to_log_file, "a", encoding="utf-8") as f:
        f.write(f"{log}\n")

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
    grad_norm:float,
    path_to_log_file:Path|None=None
) -> None:
    log = f"Train per step: {epoch}/{epochs} epoch, {idx_batch} batch, {global_step} glob step, {loss:.4f} loss, {grad_norm:.4f} grad"
    print(log)
    if path_to_log_file is not None:
        write_to_log_file(path_to_log_file, log)

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
    avg_grad_norm:float,
    path_to_log_file:Path|None=None      
) -> None:
    log = f"Train per epoch: {epoch}/{epochs} epoch, {avg_loss:.4f} avg loss, {avg_grad_norm:.4f} avg grad norm"
    print(log)
    if path_to_log_file is not None:
        write_to_log_file(path_to_log_file, log)

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
    wer:float,
    path_to_log_file:Path|None=None    
) -> None:
    log = f"Eval per epoch: {epoch}/{epochs} epoch, {avg_loss:.4f} avg loss, {cer:.4f} cer, {wer:.4f} wer"
    print(log)
    if path_to_log_file is not None:
        write_to_log_file(path_to_log_file, log)


def log_random_predicitions(
    all_predictions: list[str],
    all_targets: list[str],
    random_gen_output:int,
    path_to_log_file:Path|None=None
) -> None:
    num_samples = min(random_gen_output, len(all_predictions))

    if num_samples > 0:
        random_gen_ids = random.sample(
            range(len(all_predictions)),
            k=num_samples
        )

        for idx in random_gen_ids:
            log_predicted = f"<{idx}> Predicted: {all_predictions[idx]}"
            log_target = f"<{idx}> Target: {all_targets[idx]}"
            print(log_predicted)
            print(log_target)

            if path_to_log_file is not None:
                write_to_log_file(path_to_log_file, log_predicted)
                write_to_log_file(path_to_log_file, log_target)

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

def lig_in_cli_shape_output_batch(model, lengths, logits, input_lengths, target_lengths):
    predicted_ids = logits.argmax(dim=-1)

    print("Logits shape:", logits.shape)
    print("Original lengths:", lengths[:5].tolist())
    print("CTC input lengths:", input_lengths[:5].tolist())
    print("Target lengths:", target_lengths[:5].tolist())

    for i in range(min(3, logits.size(0))):
        valid_ids = predicted_ids[i, :int(input_lengths[i])]

        blank_ratio = (valid_ids == 0).float().mean().item()

        unique_ids, counts = torch.unique(valid_ids, return_counts=True)

        print(f"Sample {i}")
        print("Blank ratio:", round(blank_ratio, 4))
        print("Unique predicted IDs:", unique_ids.tolist())
        print("Counts:", counts.tolist())
        print("Decoded:", model.vocab.ctc_decode(valid_ids.cpu().tolist()))
