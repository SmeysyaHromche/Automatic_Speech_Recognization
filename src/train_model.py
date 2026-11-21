from pathlib import Path
import torch
import src.data as data
import src.evaluate as evaluate
import src.logging as logging
import src.model as models
import src.utils as utils
import src.train as train

vocab = data.Vocabulary()

config = utils.get_train_config()

print(f"Experiment {config.logging_config.wandb_experiment_name} setup:")
print(utils.extract_experiment_metadata(config))

run = logging.wandb_init_train(config) if config.logging_config.wandb_enable else None

device = torch.device(config.device if torch.cuda.is_available() else "cpu")

train_dataloader, valid_dataloader = data.get_train_valid_dataloaders(config)

model = models.CNN_RNN_model(
    encoder_config=config.encoder,
    decoder_config=config.decoder,
    vocab=vocab
).to(device)

optimizer, scheduler = utils.get_optimizer(config, model)
ctc_loss = torch.nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)

global_step = 0

best_validation_wer = float("inf")
best_epoch = -1

log_path = config.logging_config.path_to_log_file
path_to_log_file = Path(log_path) if log_path else None
if path_to_log_file is not None:
    path_to_log_file.parent.mkdir(parents=True, exist_ok=True)
    path_to_log_file.touch()
    print(f"loggs will be in {path_to_log_file}")

path_to_best_model = Path(config.logging_config.path_to_best_model)
path_to_best_model.parent.mkdir(parents=True, exist_ok=True)
path_to_last_model = Path(config.logging_config.path_to_last_model)
path_to_last_model.parent.mkdir(parents=True, exist_ok=True)

print("The training was started...")
for epoch in range(config.epochs):
    epoch += 1
    global_step, avg_loss, avg_grad_norm = train.train_one_epoch(
        config=config, epoch=epoch, global_step=global_step, dataloader=train_dataloader,
        model=model, device=device, run=run, optimizer=optimizer, scheduler=scheduler,
        ctc_loss=ctc_loss, path_to_log_file=path_to_log_file
    )

    if config.logging_config.log_in_cli and not config.logging_config.log_in_cli_per_step:
        logging.log_cli_one_train_epoch(
            epoch=epoch, 
            epochs=config.epochs,
            avg_loss=avg_loss, 
            avg_grad_norm=avg_grad_norm,
            path_to_log_file=path_to_log_file
        )
    if config.logging_config.wandb_enable:
        logging.log_wandb_one_train_epoch(
            run=run, 
            epoch=epoch, 
            avg_loss=avg_loss,
            avg_grad_norm=avg_grad_norm
        )

    validation_avg_loss, validation_cer, validation_wer = evaluate.evaluate(
        config=config, epoch=epoch, dataloader=valid_dataloader,
        model=model, device=device, ctc_loss=ctc_loss
    )
    
    if config.logging_config.log_in_cli:
        logging.log_cli_validation(
            epoch=epoch,
            epochs=config.epochs,
            avg_loss=validation_avg_loss,
            cer=validation_cer,
            wer=validation_wer,
            path_to_log_file=path_to_log_file,
        )
    if config.logging_config.wandb_enable and run is not None:
        logging.log_wandb_validation(
            run=run,
            epoch=epoch,
            validation_prefix=logging.VALID_METRIC,
            avg_loss=validation_avg_loss,
            cer=validation_cer,
            wer=validation_wer
        )

    if validation_wer < best_validation_wer:
        best_validation_wer = validation_wer
        best_epoch = epoch

        logging.save_model(
            path_to_save=path_to_best_model,
            epoch=epoch,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            validation_cer=validation_cer,
            validation_wer=validation_wer,
            validation_loss=validation_avg_loss
        )

        print(
            f"New best model saved! "
            f"Epoch: {epoch}, "
            f"CER: {validation_cer:.4f}, "
            f"WER: {best_validation_wer:.4f}"
        )

    logging.save_model(
        path_to_save=path_to_last_model,
        epoch=epoch,
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        validation_cer=validation_cer,
        validation_wer=validation_wer,
        validation_loss=validation_avg_loss
    )

print("The training was successfully completed =)")
if run is not None:
    run.finish()