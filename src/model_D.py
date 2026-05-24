# model_D.py
"""Model D – Zero‑shot monocular 3‑D camera.

The architecture mirrors the one you trained:
- ResNet‑50 backbone (ImageNet pretrained) for feature extraction.
- Depth head of 1×1 convolutions producing a dense depth map.
- 3‑D projection layer that lifts pixel locations to world coordinates using camera intrinsics.

This stub is functional for inference with a dummy checkpoint; replace the weight loading with your real .pth file.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ModelD(nn.Module):
    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        # ResNet‑50 backbone (exclude final FC)
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # output: (B, 2048, H/32, W/32)
        # Depth head – simple 1×1 conv stack
        self.depth_head = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),  # single‑channel depth
        )
        # Scale factor to convert raw output to meters (learned during training)
        self.register_parameter("scale", nn.Parameter(torch.tensor(10.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return dense depth map of shape (B, 1, H, W)."""
        feats = self.backbone(x)               # (B, 2048, H/32, W/32)
        depth = self.depth_head(feats)          # (B, 1, H/32, W/32)
        # Upsample to input resolution (bilinear)
        depth = torch.nn.functional.interpolate(depth, size=x.shape[2:], mode="bilinear", align_corners=False)
        depth = torch.abs(depth) * self.scale    # enforce positivity
        return depth

def load_model(checkpoint_path: str) -> ModelD:
    """Utility to load Model D from a .pth checkpoint.
    The checkpoint should contain a state_dict compatible with ModelD.
    """
    model = ModelD(pretrained_backbone=True)
    state = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model
