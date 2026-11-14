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
        
        self.model = nn.Sequential(
            CNNEncoder(encoder_output_dim),
            LSTMDecoder(encoder_output_dim, hidden_decoder_dim, vocab_size)
        )

    def forward(self, x):
        logits = self.model(x)
        return logits
