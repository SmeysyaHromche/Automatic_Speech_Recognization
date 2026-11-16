
import torch
import src.data as data
import src.logging as logging
import src.model as model
import src.utils as utils
import src.train as train
import src.evaluate as evaluate

vocab = data.Vocabulary()

config = utils.get_train_config()

print(f"Experiment {config.logging_config.wandb_experiment_name} setup:")
print(utils.extract_experiment_metadata(config))

run = logging.wandb_init_train(config) if config.logging_config.wandb_enable else None

device = torch.device(config.device if torch.cuda.is_available() else "cpu")

train_dataloader, valid_dataloader = data.get_train_valid_dataloaders(config)

model = model.CNN_LSTM_model(
    encoder_output_dim=256,
    hidden_decoder_dim=256,
    vocab=vocab
).to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
ctc_loss = torch.nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)

global_step = 0

print("The training was started...")
for epoch in range(config.epochs):

    global_step, avg_loss, avg_grad_norm = train.train_one_epoch(
        config=config, epoch=epoch, global_step=global_step, dataloader=train_dataloader,
        model=model, device=device, run=run, optimizer=optimizer, ctc_loss=ctc_loss
    )

    if config.logging_config.log_in_cli and not config.logging_config.log_in_cli_per_step:
        logging.log_cli_one_train_epoch(
            epoch=epoch, 
            epochs=config.epochs,
            avg_loss_value=avg_loss, 
            avg_grad_norm_value=avg_grad_norm
        )
    if config.logging_config.wandb_enable:
        logging.log_wandb_one_train_epoch(
            run=run, 
            epoch=epoch, 
            avg_loss_value=avg_loss,
            avg_grad_norm_value=avg_grad_norm
        )

    validation_avg_loss, validation_wer = evaluate.evaluate(
        config=config, epoch=epoch, dataloader=valid_dataloader,
        model=model, device=device, ctc_loss=ctc_loss
    )
    
    if config.logging_config.log_in_cli:
        logging.log_cli_validation(
            epoch=epoch,
            epochs=config.epochs,
            avg_loss_value=validation_avg_loss,
            wer_value=validation_wer
        )
    if config.logging_config.wandb_enable and run is not None:
        logging.log_wandb_validation(
            run=run,
            epoch=epoch,
            validation_prefix=logging.VALID_METRIC,
            avg_loss_value=validation_avg_loss,
            wer_value=validation_wer
        )
print("The training was successfully completed =)")
if config.logging_config.wandb_enable:
    run.finish()