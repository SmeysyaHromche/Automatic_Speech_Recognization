
from src.model import CNN_LSTM_model
from src.data import AudioDataset, collate_fn, Vocabulary
from torch.utils.data import DataLoader
import torch
import src.utils as utils

vocab = Vocabulary()

config = utils.get_train_config()

if config.logging_config.log_in_cli:
    print(f"Experiment {config.logging_config.wandb_experiment_name} setup:")
    print(utils.extract_experiment_metadata(config))

if config.logging_config.wandb_enable:
    run = utils.wandb_init_train(config)

device = torch.device(
    config.device if torch.cuda.is_available() else "cpu"
)

dataset = AudioDataset(config.data)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=config.batch_size,
    collate_fn=collate_fn,
    num_workers=0
)

model = CNN_LSTM_model(
    encoder_output_dim=256,
    hidden_decoder_dim=256,
    vocab_size=len(vocab)
).to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr)
ctc_loss = torch.nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)

model.train()
global_step = 0

for epoch in range(config.epochs):
    for idx_batch, (audios, texts, lengths) in enumerate(dataloader):
        audios = audios.to(device)
        
        target_sequences = [
            torch.tensor(vocab.encode(text), dtype=torch.long)
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
        
        optimizer.zero_grad(set_to_none=True)

        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        loss_value = loss.item()
        grad_norm_value = grad_norm.item()

        batch_epoch_log = {
            "train/epoch": epoch,
            "train/batch": idx_batch,
            "train/global_step": global_step,
            "train/loss": loss_value,
            "train/grad": grad_norm_value
        }

        if config.logging_config.log_in_cli:
            print(f"{epoch} epoch, {idx_batch} batch, {global_step} glob step, {loss_value:4f} loss, {grad_norm_value:4f} grad")
        if config.logging_config.wandb_enable:
            run.log(
                batch_epoch_log
            )
        
        if idx_batch >= 10:
            break

        global_step += 1
    break

if config.logging_config.wandb_enable:
    run.finish()