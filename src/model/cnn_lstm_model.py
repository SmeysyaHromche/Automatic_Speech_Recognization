from .cnn_encoder import CNNEncoder
from .lstm_decoder import LSTMDecoder
from src.data import Vocabulary
import torch

class CNN_LSTM_model(torch.nn.Module):
    def __init__(
            self, 
            encoder_output_dim:int, 
            hidden_decoder_dim:int,
            vocab:Vocabulary
        ):
        super().__init__()
        
        self.encoder = CNNEncoder(encoder_output_dim)
        self.decoder = LSTMDecoder(encoder_output_dim, hidden_decoder_dim, len(vocab))
        self.vocab = vocab


    def forward(self, x, lengths):
        x = self.encoder(x)

        output_lengths = (lengths + 1) // 2
        output_lengths = (output_lengths + 1) // 2

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



