import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_model import UpsamplingBlock

import clip

"""
class CLIPResNetFeatureExtractor(nn.Module):
    def __init__(self, clip_encoder):
        super().__init__()
        self.clip_encoder = clip_encoder
        self.feature_map = None

        def save_features(module, input, output):
            self.feature_map = output

        # Find last conv layer before attention pool
        self.clip_encoder.layer4.register_forward_hook(save_features)

    def forward(self, x):
        _ = self.clip_encoder(x)  # run through entire CLIP encoder
        return self.feature_map  # output from layer4 (shape [B, 2048, 7, 7])



class CLIPResnetSegmentationModel(nn.Module):
    def __init__(self, device, n_classes):
        super().__init__()

        self.n_classes = n_classes
        
        clip_model, preprocess = clip.load("RN101", device=device)
        clip_encoder = clip_model.visual

        self.encoder = CLIPResNetFeatureExtractor(clip_encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1),  # 7 → 14
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),   # 14 → 28
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),    # 28 → 56
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),    # 56 → 112
            nn.ReLU(),
            nn.ConvTranspose2d(128, n_classes, 4, 2, 1)  # 112 → 224
        )

    def forward(self, x):
        features = self.encoder(x)  # [B, 2048, 7, 7]
        return self.decoder(features)

"""
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

class CLIPResnetSegmentationModel(nn.Module):
    def __init__(self, device, n_classes):
        super().__init__()
        self.n_classes = n_classes
        clip_model, _ = clip.load("RN101", device=device)
        clip_encoder = clip_model.visual

        self.encoder = CLIPResNetFeatureExtractor(clip_encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Decoder: use UpsamplingBlocks with CLIP feature dims
        self.ublock1 = UpsamplingBlock(n_in_channels=2048, n_filters=1024)  # input + layer3 (1024)
        self.ublock2 = UpsamplingBlock(n_in_channels=1024, n_filters=512)   # input + layer2 (512)
        self.ublock3 = UpsamplingBlock(n_in_channels=512, n_filters=256)    # input + layer1 (256)
        self.ublock4 = nn.Sequential(  # Final upsampling (no skip conn)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

        self.conv_final1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv_final2 = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)

    def forward(self, x):
        features = self.encoder(x)
        f1 = features["layer1"]  # [B, 256, 56, 56]
        f2 = features["layer2"]  # [B, 512, 28, 28]
        f3 = features["layer3"]  # [B, 1024, 14, 14]
        f4 = features["layer4"]  # [B, 2048, 7, 7]

        x = self.ublock1(f4, f3)  # → [B, 1024, 14, 14]
        x = self.ublock2(x, f2)   # → [B, 512, 28, 28]
        x = self.ublock3(x, f1)   # → [B, 256, 56, 56]
        x = self.ublock4(x)       # → [B, 128, 112, 112]

        x = F.relu(self.conv_final1(x))        # → [B, 128, 112, 112]
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.conv_final2(x)                # → [B, n_classes, 224, 224]
        return x



