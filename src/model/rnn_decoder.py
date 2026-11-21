from src.config import DecoderConfig
from torch import nn


def get_rnn(input_size:int, config:DecoderConfig) -> nn.LSTM | nn.GRU:
    if config.rnn == "lstm":
        return nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layer,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
    elif config.rnn == "gru":
        return nn.GRU(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layer,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

    raise ValueError(f"Unknown rnn type: {config.rnn}! Use: lstm, gru")

def get_projection(config:DecoderConfig, vocab_size:int) -> nn.Sequential:
    if config.projection == "linear":
        return nn.Sequential(
            nn.Linear(
                in_features=2 * config.hidden_size,  # 2 beacuse BiRNN
                out_features=vocab_size
            )
        )
    elif config.projection == "unlinear":
        return nn.Sequential(
            nn.Linear(
                in_features=2 * config.hidden_size,
                out_features=256
            ),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(
                in_features=256,
                out_features=vocab_size
            )
        )

    raise ValueError(f"Unknown projection: {config.projection}! Use: linear, unlinear")


class RNNDecoder(nn.Module):
    def __init__(self, input_size:int, config:DecoderConfig, vocab_size:int):
        super().__init__()

        self.config = config

        self.hidden_size = config.hidden_size
        
        self.rnn = get_rnn(input_size, config)

        self.norm = (
            nn.Sequential(
                nn.LayerNorm(2 * config.hidden_size),
                nn.Dropout(0.2),
            )
            if config.norm
            else nn.Identity()
        )

        self.projection = get_projection(config, vocab_size)

    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_x)
        
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=x.size(1))
        output = self.norm(output)
        
        logits = self.projection(output)
        
        return logits