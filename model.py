import torch
import torch.nn as nn


# 1. Double Convolution Block
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# 2. Encoder Block
class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)

        return features, pooled

# 3. Bottleneck Block
class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(x)

# 4. Decoder Block
class Decoder(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )

        self.conv = DoubleConv(
            in_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):
        x = self.up(x)

        # Combine decoder output with encoder features
        x = torch.cat([skip, x], dim=1)

        return self.conv(x)

# 5. Complete U-Net Model
class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.encoder1 = Encoder(in_channels, 32)
        self.encoder2 = Encoder(32, 64)

        # Bottleneck
        self.bottleneck = Bottleneck(64, 128)

        # Decoder
        self.decoder2 = Decoder(128, 64, 64)
        self.decoder1 = Decoder(64, 32, 32)

        # Output
        self.output = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):

        # Encoder 1
        skip1, x = self.encoder1(x)

        # Encoder 2
        skip2, x = self.encoder2(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder 2
        x = self.decoder2(x, skip2)

        # Decoder 1
        x = self.decoder1(x, skip1)

        # Output
        return self.output(x)



# Temporary test
if __name__ == "__main__":

    model = UNet()

    x = torch.randn(1, 1, 256, 256)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    print("Total parameters:", sum(
        p.numel() for p in model.parameters()
    ))