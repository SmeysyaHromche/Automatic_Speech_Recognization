from .cnn_encoder import CNNEncoder
from .lstm_decoder import LSTMDecoder
from torch import nn

class CNN_LSTM_model(nn.Module):
    def __init__(
            self, 
            encoder_output_dim:int, 
            hidden_decoder_dim:int,
            vocab_size:int
        ):
        super().__init__()
        
        self.encoder = CNNEncoder(encoder_output_dim)
        self.decoder = LSTMDecoder(encoder_output_dim, hidden_decoder_dim, vocab_size)


    def forward(self, x, lengths):
        x = self.encoder(x)

        output_lengths = (lengths + 1) // 2
        output_lengths = (output_lengths + 1) // 2

        logits = self.decoder(x, output_lengths)

        return logits, output_lengths
