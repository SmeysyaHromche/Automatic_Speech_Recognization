from src.config import TrainConfig
from torch.utils.data import DataLoader
import src.logging as logging
import torch
import torch.nn as nn
import torch.optim as optim
import wandb as wb

def train_one_epoch(
        config: TrainConfig,
        epoch:int,
        global_step:int,
        dataloader:DataLoader,
        model:nn.Module,
        device:torch.device,
        optimizer: optim.Optimizer,
        ctc_loss:nn.CTCLoss,
        run:wb.Run|None=None
    ) -> tuple[int, float, float]:
    model.train()

    total_loss = 0.0
    total_norm_grad = 0.0
    num_batches = 0
     
    for idx_batch, (audios, texts, lengths) in enumerate(dataloader):

        audios = audios.to(device)
        
        target_sequences = [
            torch.tensor(model.vocab.encode(text), dtype=torch.long)
            for text in texts
        ]
        target_lengths = torch.tensor(
            [len(seq) for seq in target_sequences],
            dtype=torch.long
        )

        targets = torch.cat(target_sequences).to(device)

        logits, input_lengths = model(audios, lengths)

        log_probs = logits.log_softmax(dim=-1)
        log_probs = log_probs.permute(1, 0, 2)

        loss = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch={epoch}, "
                f"batch={idx_batch}, step={global_step}"
            )
        
        optimizer.zero_grad(set_to_none=True)

        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True
        )

        optimizer.step()

        loss_value = loss.item()

        grad_norm_value = grad_norm.item()

        if global_step + 1 % config.logging_config.logging_each_n_step == 0:
            if config.logging_config.log_in_cli and config.logging_config.log_in_cli_per_step:
                logging.log_cli_one_train_step(
                    epoch=epoch,
                    idx_batch=idx_batch,
                    global_step=global_step,
                    loss_value=loss_value,
                    grad_norm_value=grad_norm_value
                )
            if config.logging_config.wandb_enable:
                logging.log_wandb_one_train_step(
                    run=run, 
                    global_step=global_step, 
                    loss_value=loss_value, 
                    grad_norm_value=grad_norm_value
                )

        global_step += 1
        total_loss += loss_value
        total_norm_grad += grad_norm_value
        num_batches += 1

        if config.max_steps_per_epoch > 0 and idx_batch + 1 >= config.max_steps_per_epoch:
            break 

    if num_batches == 0:
        raise RuntimeError("No training batches processed")
    
    avg_loss = total_loss / num_batches
    avg_norm_grad = total_norm_grad / num_batches

    return global_step, avg_loss, avg_norm_grad
