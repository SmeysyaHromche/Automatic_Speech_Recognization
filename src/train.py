
from src.model import CNN_LSTM_model
from pathlib import Path
from src.data import AudioDataset, collate_fn, Vocabulary
from src.utils import get_train_config
from torch.utils.data import DataLoader
import torch

vocab = Vocabulary()

path_to_config = Path("/home/xkukht01/Dev/SFC/configs/train.yaml")  # TODO: pargs args
config = get_train_config(path_to_config)

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

model.train()
ctc_loss = torch.nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)

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

        targets = torch.cat(target_sequences)

        logits, input_lengths = model(audios, lengths)

        log_probs = logits.log_softmax(dim=-1)
        log_probs = log_probs.permute(1, 0, 2)

        loss = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths
        )
        
        print(f"{epoch} epoch, {idx_batch} batch, {loss:3f} loss")
        
        if idx_batch >= 10:
            break

    break
