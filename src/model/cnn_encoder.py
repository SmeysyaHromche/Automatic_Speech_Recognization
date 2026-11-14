from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )

        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.layers(x))

# TODO: ReLU vs GeLU
class CNNEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.cnn = nn.Sequential(
            # [B, 1, 80, T]
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [B, 32, 40, T/2]

            ResidualBlock(32),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [B, 64, 20, T/4]

            ResidualBlock(64)
        )

        self.projection = nn.Linear(64 * 20, d_model)

    def forward(self, x):
        x = self.cnn(x)

        B, C, F, T = x.shape

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B, T, C * F)

        # [B, T', d_model]
        return self.projection(x)