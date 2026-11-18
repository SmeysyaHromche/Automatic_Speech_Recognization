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
) -> tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    num_batches = 0

    all_predictions = []
    all_targets = []

    for idx_batch, (log_mels, texts, lengths) in enumerate(dataloader):
        log_mels = log_mels.to(device)

        target_sequences = [
            torch.tensor(model.vocab.encode(text), dtype=torch.long)
            for text in texts
        ]

        target_lengths = torch.tensor(
            [len(seq) for seq in target_sequences],
            dtype=torch.long,
        )

        targets = torch.cat(target_sequences).to(device)

        logits, input_lengths = model(log_mels, lengths)

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

    cer, wer = metrics.compute_cer_wer(all_targets=all_targets, all_predictions=all_predictions)

    num_samples = min(config.logging_config.random_gen_output, len(all_predictions))

    if num_samples > 0:
        random_gen_ids = random.sample(
            range(len(all_predictions)),
            k=num_samples
        )

        for idx in random_gen_ids:
            print(f"<{idx}> Predicted: {all_predictions[idx]}")
            print(f"<{idx}> Target: {all_targets[idx]}")
            print()

    return avg_loss, cer, wer