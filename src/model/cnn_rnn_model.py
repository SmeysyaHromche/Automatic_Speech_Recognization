from .cnn_encoder import CNNEncoder
from .rnn_decoder import RNNDecoder
from pyctcdecode import build_ctcdecoder
from src.data import Vocabulary
from src.config import EncoderConfig, DecoderConfig, SearchConfig
import torch

class CNN_RNN_model(torch.nn.Module):
    def __init__(
            self, 
            encoder_config:EncoderConfig, 
            decoder_config:DecoderConfig,
            search_config:SearchConfig,
            vocab:Vocabulary
        ):
        super().__init__()
    
        self.encoder = CNNEncoder(encoder_config)
        self.decoder = RNNDecoder(self.encoder.out_dim, decoder_config, len(vocab))
        self.vocab = vocab
        self.search_config = search_config

        labels = self.vocab.tokens.copy()
        labels[self.vocab.blank_id] = ""

        self.beam_decoder = build_ctcdecoder(labels=labels)

    def forward(self, x, lengths):
        x = self.encoder(x)

        output_lengths = self.encoder.get_output_lengths(lengths)

        logits = self.decoder(x, output_lengths)

        return logits, output_lengths

    @torch.inference_mode()
    def generate(self, x, lengths)->list[str]:
        logits, output_lengths = self(x, lengths)
        return self.decode(logits, output_lengths)


    def decode(self, logits:torch.Tensor, output_lengths:torch.Tensor) -> list[str]:
        all_predictions = []
        output_lengths_cpu = output_lengths.cpu()

        if self.search_config.search_policy == "greedy":
            predicted_ids = logits.argmax(dim=-1).cpu()

            for predicted_sequence, output_length in zip(predicted_ids, output_lengths_cpu):
                predicted_sequence = predicted_sequence[:int(output_length)]

                predicted_text = self.vocab.ctc_decode(predicted_sequence.tolist())

                all_predictions.append(predicted_text)
        elif self.search_config.search_policy == "beam":
            assert self.search_config.beam_width > 1

            logits = logits.float().cpu().numpy()
            logits = [sample[:length] for sample, length in zip(logits, output_lengths_cpu)]

            all_predictions = self.beam_decoder.decode_batch(
                pool=None,
                logits_list=logits,
                beam_width=self.search_config.beam_width
            )
        else:
            raise ValueError(f"Unknown search policy: {self.search_config.search_policy}! Use greedy or beam")

        return all_predictions
