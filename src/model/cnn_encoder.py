from src.config import EncoderConfig
from torch import nn


def get_activation(activation_str:str):
    if activation_str == "gelu":
        return nn.GELU()
    elif activation_str == "relu":
        return nn.ReLU()
    elif activation_str == "selu":
        return nn.SELU()

    raise ValueError(f"Unknown activation fn: {activation_str}")

def get_normalization_layer(normalization_str: str, channels: int):
    if normalization_str == "batch":
        return nn.BatchNorm2d(channels)

    elif normalization_str == "group":
        if channels % 8 != 0:
            raise ValueError("Channels must be divisible by 8")
        return nn.GroupNorm(8, channels)

    raise ValueError(f"Unknown normalization: {normalization_str}")

class ResidualBlock(nn.Module):
    def __init__(self, channels:int, activation_fn:str, norm_blck:str):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            get_normalization_layer(normalization_str=norm_blck, channels=channels),
            get_activation(activation_fn),
            nn.Conv2d(channels, channels, 3, padding=1),
            get_normalization_layer(normalization_str=norm_blck, channels=channels)
        )

        self.activation = get_activation(activation_fn)

    def forward(self, x):
        return self.activation(x + self.layers(x))

class ClassicConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, norm_blck:str, activation_fn:str):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            get_normalization_layer(normalization_str=norm_blck, channels=out_channels),
            get_activation(activation_fn)
        )

    def forward(self, x):
        return self.block(x)


# TODO: ReLU vs GeLU
class CNNEncoder(nn.Module):
    def __init__(self, config:EncoderConfig):
        super().__init__()

        layers = []
        in_channels = 1
        freq = config.n_mels
        for out_channels, num_blocks in zip(config.channels, config.res_blocks_per_stage,):
            layers.append(
                ClassicConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    norm_blck=config.normalization,
                    activation_fn=config.activation_fn
            ))

            for _ in range(num_blocks):
                layers.append(
                    ResidualBlock(
                        channels=out_channels,
                        activation_fn=config.activation_fn,
                        norm_blck=config.normalization,
                    )
                )

            in_channels = out_channels
            freq = (freq + 1) // 2

        self.cnn = nn.Sequential(*layers)

        self.projection = nn.Linear(in_channels * freq, config.d_model)

    def forward(self, x):
        x = self.cnn(x)

        B, C, F, T = x.shape

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B, T, C * F)

        # [B, T', d_model]
        return self.projection(x)