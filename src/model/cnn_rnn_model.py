from .cnn_encoder import CNNEncoder
from .rnn_decoder import RNNDecoder
from src.data import Vocabulary
from src.config import EncoderConfig, DecoderConfig
import torch

class CNN_RNN_model(torch.nn.Module):
    def __init__(
            self, 
            encoder_config:EncoderConfig, 
            decoder_config:int,
            vocab:Vocabulary
        ):
        super().__init__()
        
        self.encoder = CNNEncoder(encoder_config)
        self.decoder = RNNDecoder(self.encoder.out_dim, decoder_config, len(vocab))
        self.vocab = vocab


    def forward(self, x, lengths):
        x = self.encoder(x)

        output_lengths = self.encoder.get_output_lengths(lengths)

        logits = self.decoder(x, output_lengths)

        return logits, output_lengths

    @torch.inference_mode()
    def generate(self, x, lengths):
        logits, output_lengths = self(x, lengths)

        token_ids = logits.argmax(dim=-1)

        predictions = []

        for tokens, length in zip(token_ids, output_lengths):
            tokens = tokens[:length].tolist()

            text = self.vocab.ctc_decode(tokens)
            predictions.append(text)

        return predictions



