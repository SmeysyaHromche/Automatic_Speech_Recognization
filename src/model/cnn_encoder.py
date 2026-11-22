from src.config import EncoderConfig
from torch import nn


def get_activation(activation_str:str):
    if activation_str == "gelu":
        return nn.GELU()
    elif activation_str == "relu":
        return nn.ReLU()
    elif activation_str == "selu":
        return nn.SELU()

    raise ValueError(f"Unknown activation fn: {activation_str}! Use: gelu, relu, selu")

def get_normalization_layer(normalization_str: str, channels: int):
    if normalization_str == "batch":
        return nn.BatchNorm2d(channels)

    elif normalization_str == "group":
        if channels % 8 != 0:
            raise ValueError("Channels must be divisible by 8")
        return nn.GroupNorm(8, channels)

    raise ValueError(f"Unknown normalization: {normalization_str}! Use: batch, group")

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
    def __init__(self, in_channels:int, out_channels:int, stride:int, norm_blck:str, activation_fn:str):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            get_normalization_layer(normalization_str=norm_blck, channels=out_channels),
            get_activation(activation_fn)
        )

    def forward(self, x):
        return self.block(x)

class CNNEncoder(nn.Module):
    def __init__(self, config:EncoderConfig):
        super().__init__()

        layers = []
        in_channels = 1
        freq = config.n_mel
        for out_channels, num_blocks, stride in zip(config.channels, config.res_blocks_per_stage, config.strides):
            layers.append(
                ClassicConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
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
            freq = (freq + stride + 1) // 2

        self.cnn = nn.Sequential(*layers)
        self.cnn_out_channel = in_channels
        self.out_dim = config.d_model

        self.projection = nn.Linear(self.cnn_out_channel * freq, self.out_dim)

    def get_output_lengths(self, lengths):
        for module in self.cnn.modules():
            if isinstance(module, nn.Conv2d):
                kernel = module.kernel_size[1]
                stride = module.stride[1]
                padding = module.padding[1]
                dilation = module.dilation[1]

                lengths = (lengths + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

        return lengths

    def forward(self, x):
        x = self.cnn(x)

        B, C, F, T = x.shape

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(B, T, C * F)

        # [B, T', d_model]
        return self.projection(x)