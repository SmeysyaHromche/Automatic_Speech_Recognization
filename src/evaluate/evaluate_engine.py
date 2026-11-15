from src.config import TrainConfig
from torch.utils.data import DataLoader
import src.metrics as metrics
import torch
import torch.nn as nn
import random

@torch.inference_mode()
def evaluate(
    config: TrainConfig,
    epoch: int,
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    ctc_loss: nn.CTCLoss
):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_targets = []

    for idx_batch, (audios, texts, lengths) in enumerate(dataloader):
        audios = audios.to(device)

        target_sequences = [
            torch.tensor(model.vocab.encode(text), dtype=torch.long)
            for text in texts
        ]

        target_lengths = torch.tensor(
            [len(seq) for seq in target_sequences],
            dtype=torch.long,
        )

        targets = torch.cat(target_sequences).to(device)

        logits, input_lengths = model(audios, lengths)

        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)

        loss = ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss at epoch={epoch}, batch={idx_batch}"
            )

        loss_value = loss.item()

        predicted_ids = logits.argmax(dim=-1).cpu()
        input_lengths_cpu = input_lengths.cpu()

        for predicted_sequence, output_length in zip(predicted_ids, input_lengths_cpu):
            predicted_sequence = predicted_sequence[:int(output_length)]

            predicted_text = model.vocab.ctc_decode(predicted_sequence.tolist())

            all_predictions.append(predicted_text)

        all_targets.extend(texts)

        total_loss += loss_value
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("No validation batches processed")

    avg_loss = total_loss / num_batches

    wer = metrics.compute_wer(all_targets=all_targets, all_predictions=all_predictions)

    if (config.logging_config.random_gen_output > 0 and 
        config.logging_config.random_gen_output > len(all_predictions)):
        random_gen_ids = random.sample(
            range(0, len(all_predictions)), 
            config.logging_config.random_gen_output
        )
        for id in random_gen_ids:
            print(f"<{id}> Predicted: {all_predictions[id]}")
            print(f"<{id}> Target: {all_targets[id]}")
            print()
        pass

    return avg_loss, wer