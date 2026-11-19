from src.config import DecoderConfig
from torch import nn

class LSTMDecoder(nn.Module):
    def __init__(self, input_size:int, config:DecoderConfig, vocab_size:int):
        super().__init__()

        self.hidden_size = config.hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layer,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.projection = nn.Linear(
            in_features=2 * config.hidden_size,  # 2 beacuse BiLSTM
            out_features=vocab_size
        )

    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=x.size(1))
        logits = self.projection(output)
        
        return logits