import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_model import UpsamplingBlock

import clip

class CLIPResNetFeatureExtractor(nn.Module):
    def __init__(self, clip_encoder):
        super().__init__()
        self.clip_encoder = clip_encoder
        self.feature_maps = {}

        def save_hook(name):
            def hook(module, input, output):
                self.feature_maps[name] = output
            return hook

        # Register hooks to grab skip features
        self.clip_encoder.layer1.register_forward_hook(save_hook("layer1"))  # → [B, 256, 56, 56]
        self.clip_encoder.layer2.register_forward_hook(save_hook("layer2"))  # → [B, 512, 28, 28]
        self.clip_encoder.layer3.register_forward_hook(save_hook("layer3"))  # → [B, 1024, 14, 14]
        self.clip_encoder.layer4.register_forward_hook(save_hook("layer4"))  # → [B, 2048, 7, 7]

    def forward(self, x):
        _ = self.clip_encoder(x)
        return self.feature_maps

class PromptUNet(nn.Module):
    def __init__(self, device, n_classes):
        super().__init__()
        self.n_classes = n_classes

        clip_model, _ = clip.load("RN101", device=device)
        clip_encoder = clip_model.visual
        self.encoder = CLIPResNetFeatureExtractor(clip_encoder)

        for param in self.encoder.parameters():
            param.requires_grad = False

        # Heatmap downsampling to match spatial size of layer4 (7x7)
        self.heatmap_down = nn.AdaptiveAvgPool2d((7, 7))

        # Decoder using UpsamplingBlock
        self.ublock1 = UpsamplingBlock(n_in_channels=2049, n_filters=1024)  # 2048 + 1 from heatmap
        self.ublock2 = UpsamplingBlock(n_in_channels=1024, n_filters=512)   # skip: layer3
        self.ublock3 = UpsamplingBlock(n_in_channels=512, n_filters=256)    # skip: layer2
        self.ublock4 = UpsamplingBlock(n_in_channels=256, n_filters=128)    # skip: layer1

        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, n_classes, kernel_size=1)
        )

    def forward(self, x, heatmap):
        features = self.encoder(x)
        f1 = features["layer1"]  # [B, 256, 56, 56]
        f2 = features["layer2"]  # [B, 512, 28, 28]
        f3 = features["layer3"]  # [B, 1024, 14, 14]
        f4 = features["layer4"]  # [B, 2048, 7, 7]

        # Preprocess heatmap
        if heatmap.shape[1] == 3:  # RGB → grayscale
            heatmap = heatmap.mean(dim=1, keepdim=True)

        heatmap_down = self.heatmap_down(heatmap)  # [B, 1, 7, 7]

        # Concatenate heatmap with deepest feature map
        f4_combined = torch.cat([f4, heatmap_down], dim=1)  # [B, 2049, 7, 7]

        x = self.ublock1(f4_combined, f3)  # → 14x14
        x = self.ublock2(x, f2)            # → 28x28
        x = self.ublock3(x, f1)            # → 56x56
        x = self.ublock4(x)                # → 112x112

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.final_conv(x)             # → [B, n_classes, 224, 224]

        return x
