import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, n_in_channels = 3, n_filters=64, dropout_prob=0, max_pooling=True):
        """
        Convolutional downsampling block

        Arguments:
            n_in_channels -- Number of input channels
            n_filters -- Number of filters for the convolutional layers
            dropout_prob -- Dropout probability
            max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
        """
        super(ConvBlock, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=n_in_channels, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)

        # Dropout Layer (Optional)
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else None

        # Max Pooling Layer (Optional)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2) if max_pooling else None

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        if self.dropout:
            x = self.dropout(x)

        skip_connection = x  # Save for the U-Net like architectures

        if self.max_pooling:
            x = self.max_pooling(x)

        return x, skip_connection
    
class UpsamplingBlock(nn.Module):
    def __init__(self, n_in_channels = 64, n_filters=64, for_autoencoder = False):
        """
        Convolutional upsampling block

        Arguments:
            n_in_channels -- Number of input channels
            n_filters -- Number of filters for the convolutional layers
        """
        super(UpsamplingBlock, self).__init__()

        # Transposed Convolution for upsampling
        self.upconv = nn.ConvTranspose2d(in_channels=n_in_channels, out_channels=n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Convolutional layers after concatenation
        self.conv1 = nn.Conv2d(in_channels=n_filters * 2, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        self.for_autoencoder = for_autoencoder

    def forward(self, expansive_input, contractive_input = None):
        if not self.for_autoencoder:
            # Upsample the input
            up = self.upconv(expansive_input)

            # Ensure spatial sizes match before concatenation
            diffY = contractive_input.size()[2] - up.size()[2]
            diffX = contractive_input.size()[3] - up.size()[3]
            up = F.pad(up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            # Concatenate along the channel dimension
            merge = torch.cat([up, contractive_input], dim=1)

            # Apply convolutions
            conv = F.relu(self.conv1(merge))
            conv = F.relu(self.conv2(conv))

            return conv
        else:
            # Upsample the input
            up = self.upconv(expansive_input)

            return up

class UNet(nn.Module):
    def __init__(self, input_channels=3, n_filters=64, n_classes=4):
        """
        UNet model

        Arguments:
            input_channels -- Number of input channels (default: 3 for RGB images)
            n_filters -- Number of filters for the convolutional layers
            n_classes -- Number of output classes
        """
        super(UNet, self).__init__()

        self.n_classes = n_classes

        # Contracting Path (Encoder)
        self.cblock1 = ConvBlock(n_in_channels=input_channels, n_filters=n_filters* 1)
        self.cblock2 = ConvBlock(n_in_channels=n_filters * 1, n_filters=n_filters * 2)
        self.cblock3 = ConvBlock(n_in_channels=n_filters * 2, n_filters=n_filters * 4)
        self.cblock4 = ConvBlock(n_in_channels=n_filters * 4, n_filters=n_filters * 8, dropout_prob=0.3)
        self.cblock5 = ConvBlock(n_in_channels=n_filters * 8, n_filters=n_filters * 16, dropout_prob=0.3, max_pooling=False)

        # Expanding Path (Decoder)
        self.ublock6 = UpsamplingBlock(n_in_channels=n_filters * 16, n_filters=n_filters * 8)
        self.ublock7 = UpsamplingBlock(n_in_channels=n_filters * 8, n_filters=n_filters * 4)
        self.ublock8 = UpsamplingBlock(n_in_channels=n_filters * 4, n_filters=n_filters * 2)
        self.ublock9 = UpsamplingBlock(n_in_channels=n_filters * 2, n_filters=n_filters)


        # Final Convolutional Layers
        self.conv9 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=n_filters, out_channels=n_classes, kernel_size=1)

    def forward(self, x):

        # Encoder
        cblock1, skip1 = self.cblock1(x)
        cblock2, skip2 = self.cblock2(cblock1)
        cblock3, skip3 = self.cblock3(cblock2)
        cblock4, skip4 = self.cblock4(cblock3)
        cblock5, _ = self.cblock5(cblock4)  # No skip connection for last layer
        # Decoder
        ublock6 = self.ublock6(cblock5, skip4)
        ublock7 = self.ublock7(ublock6, skip3)
        ublock8 = self.ublock8(ublock7, skip2)
        ublock9 = self.ublock9(ublock8, skip1)

        # Final convolutions
        conv9 = F.relu(self.conv9(ublock9))
        conv10 = self.conv10(conv9)  # No activation (typically softmax is applied externally)

        return conv10
    


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