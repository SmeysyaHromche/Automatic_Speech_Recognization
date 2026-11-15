
from src.model import CNN_LSTM_model
from src.data import get_train_valid_dataloaders, Vocabulary
from src.train import train_one_epoch
import torch
import src.utils as utils
import src.logging as logging

vocab = Vocabulary()

config = utils.get_train_config()

print(f"Experiment {config.logging_config.wandb_experiment_name} setup:")
print(utils.extract_experiment_metadata(config))

run = logging.wandb_init_train(config) if config.logging_config.wandb_enable else None

device = torch.device(
    config.device if torch.cuda.is_available() else "cpu"
)

train_dataloader, valid_dataloader = get_train_valid_dataloaders(config)

model = CNN_LSTM_model(
    encoder_output_dim=256,
    hidden_decoder_dim=256,
    vocab=vocab
).to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
ctc_loss = torch.nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)

global_step = 0

for epoch in range(config.epochs):

    global_step, avg_loss, avg_grad_norm = train_one_epoch(
        config=config, epoch=epoch, global_step=global_step, dataloader=train_dataloader,
        model=model, device=device, run=run, optimizer=optimizer, ctc_loss=ctc_loss
    )

    if config.logging_config.log_in_cli and not config.logging_config.log_in_cli_per_step:
        logging.log_cli_one_epoch(
            epoch=epoch, 
            avg_loss_value=avg_loss, 
            avg_grad_norm_value=avg_grad_norm)
    if config.logging_config.wandb_enable:
        logging.log_wandb_one_epoch(
            run=run, 
            epoch=epoch, 
            avg_loss_value=avg_loss,
            avg_grad_norm_value=avg_grad_norm)

if config.logging_config.wandb_enable:
    run.finish()