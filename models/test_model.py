import torch
import torch.nn as nn
import torch.nn.functional as F

class TestModel(nn.Module):
    def __init__(self, input_channels=3, n_filters=32, dropout_prob=0, n_classes=4):
        """
        Mock model to test the training pipeline for segmentation tasks.

        Arguments:
            input_channels -- Number of input channels
            n_filters -- Number of filters for the convolutional layers
            dropout_prob -- Dropout probability
            n_classes -- Number of output classes
        """
        super(TestModel, self).__init__()

        # Convolutions
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)

        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else None

        # Deconvolution
        self.upconv1 = nn.ConvTranspose2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(in_channels=n_filters, out_channels=n_filters, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(in_channels=n_filters, out_channels=n_classes, kernel_size=1)

    def forward(self, x):

        # Downsampling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        if self.dropout:
            x = self.dropout(x)
        
        # Upsampling
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))

        # Final Convolution
        x = self.final_conv(x)
        return x


        
        