from torch import nn

class LSTMDecoder(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, vocab_size:int):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.projection = nn.Linear(
            in_features=2 * hidden_size,  # 2 beacuse BiLSTM
            out_features=vocab_size
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        logits = self.projection(output)
        
        return logits