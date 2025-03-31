import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pytorch_mlp_framework.model_architectures import (
    ConvolutionalProcessingBlockBatchNormalizedResConn,
    ConvolutionalDimensionalityReductionBlockBatchNormalized,
    ConvolutionalProcessingBlockBatchNormalized,
)
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F

# Random Params
input_shape = (1, 3, 32, 32)
num_filters = 16
kernel_size = 3
padding = 1
bias = True
dilation = 1
reduction_factor = 2

# Initialize layers
block_cpBNRC = ConvolutionalProcessingBlockBatchNormalizedResConn(
    input_shape=input_shape,
    num_filters=num_filters,
    kernel_size=kernel_size,
    padding=padding,
    bias=bias,
    dilation=dilation
)

block_cpBN = ConvolutionalProcessingBlockBatchNormalized(
    input_shape=input_shape,
    num_filters=num_filters,
    kernel_size=kernel_size,
    padding=padding,
    bias=bias,
    dilation=dilation
)

block_cdrbbn = ConvolutionalDimensionalityReductionBlockBatchNormalized(
    input_shape=input_shape,
    num_filters=num_filters,
    kernel_size=kernel_size,
    padding=padding,
    bias=bias,
    dilation=dilation,
    reduction_factor=reduction_factor

)

# Create dummy input
x = torch.randn(input_shape)

# Forward pass
out_cpBNRC = block_cpBNRC(x)
out__cpBN = block_cpBN(x)
out_cdrbbn = block_cdrbbn(x)

# Check output
print("Input shape:", x.shape)
print("Output shape out_cpBNRC:", out_cpBNRC.shape)
print("Output shape out__cpBN:", out__cpBN.shape)
print("Output shape out_cdrbbn:", out_cdrbbn.shape)
