from pathlib import Path
import src.data as data
import src.evaluate as evaluate
import src.logging as logging
import src.model as models
import src.utils as utils
import torch

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
    search_config=config.search,
    vocab=vocab
).to(device)

model.load_state_dict(torch.load(config.path_to_weights), weights_only=True)

ctc_loss = torch.nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)

epoch = 1

log_path = config.logging_config.path_to_log_file
path_to_log_file = Path(log_path) if log_path else None
if path_to_log_file is not None:
    path_to_log_file.parent.mkdir(parents=True, exist_ok=True)
    path_to_log_file.touch()
    print(f"loggs will be in {path_to_log_file}")

print("The testing was started...")

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

print("The testing was successfully completed =)")
