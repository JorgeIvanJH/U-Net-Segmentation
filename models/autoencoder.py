import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet_model import ConvBlock, UpsamplingBlock

class Autoencoder(nn.Module):
    def __init__(self, input_channels=3, n_filters=64):
        super(Autoencoder, self).__init__()

        # Encoder (same as U-Net’s encoder)
        self.cblock1 = ConvBlock(n_in_channels=input_channels, n_filters=n_filters* 1)
        self.cblock2 = ConvBlock(n_in_channels=n_filters * 1, n_filters=n_filters * 2)
        self.cblock3 = ConvBlock(n_in_channels=n_filters * 2, n_filters=n_filters * 4)
        self.cblock4 = ConvBlock(n_in_channels=n_filters * 4, n_filters=n_filters * 8) # We might want to set droput to 0 since here overfit is beneficial
        self.cblock5 = ConvBlock(n_in_channels=n_filters * 8, n_filters=n_filters * 16, max_pooling=False)

        # Decoder (reconstructs image without skip connections)
        self.ublock6 = UpsamplingBlock(n_in_channels=n_filters * 16, n_filters=n_filters * 8, for_autoencoder = True)
        self.ublock7 = UpsamplingBlock(n_in_channels=n_filters * 8, n_filters=n_filters * 4, for_autoencoder = True)
        self.ublock8 = UpsamplingBlock(n_in_channels=n_filters * 4, n_filters=n_filters * 2, for_autoencoder = True)
        self.ublock9 = UpsamplingBlock(n_in_channels=n_filters * 2, n_filters=n_filters, for_autoencoder = True)

        # Final reconstruction layer (output's #channels is the same as input)
        self.final_conv = nn.Conv2d(n_filters, input_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x, _ = self.cblock1(x)
        x, _ = self.cblock2(x)
        x, _ = self.cblock3(x)
        x, _ = self.cblock4(x)
        x, _ = self.cblock5(x)

        # Decoder
        x = self.ublock6(x)  # Skip connections removed
        x = self.ublock7(x)
        x = self.ublock8(x)
        x = self.ublock9(x)

        # Final reconstruction
        x = self.final_conv(x)
        return x