import torch
import torch.nn as nn
import torch.nn.functional as F

import clip


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


