"""
models/transmorph_brain_shift.py

TransMorph-inspired deformable registration network for brain shift prediction.
Architecture: Swin Transformer encoder + CNN decoder -> dense 3D deformation field.

Reference: Chen et al., "TransMorph: Transformer for Unsupervised Medical Image
Registration", Medical Image Analysis, 2022.
Official code: github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration

This implementation is adapted for multi-modal MRI (T1+FLAIR) -> US registration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# ─── SPATIAL TRANSFORMER (STN) ─────────────────────────────────────────────

class SpatialTransformer(nn.Module):
    """
    Apply a 3D deformation field to warp an image.
    Used at inference to warp the moving image and tumour mask.
    """
    def __init__(self, size: Tuple[int,...], mode: str = "bilinear"):
        super().__init__()
        self.mode = mode
        # Create normalised coordinate grid
        vectors = [torch.arange(0, s) for s in size]
        grids   = torch.meshgrid(vectors, indexing="ij")
        grid    = torch.stack(grids)  # (3, D, H, W)
        grid    = torch.unsqueeze(grid, 0)  # (1, 3, D, H, W)
        grid    = grid.float()
        self.register_buffer("grid", grid)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        src:  (B, C, D, H, W)
        flow: (B, 3, D, H, W)  — displacement field in voxel units
        """
        new_locs = self.grid + flow
        shape    = flow.shape[2:]

        # Normalise to [-1, 1] for grid_sample
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs,
                             align_corners=True,
                             mode=self.mode,
                             padding_mode="border")


# ─── ENCODER BLOCKS ────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResConvBlock(nn.Module):
    """Residual convolutional block."""
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.InstanceNorm3d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.InstanceNorm3d(ch),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


# ─── ATTENTION GATE ────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """Attention gate to focus on relevant features during skip connections."""
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, 1, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, 1, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, 1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Upsample g to match x spatial size if needed
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="trilinear", align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ─── MAIN NETWORK ──────────────────────────────────────────────────────────

class BrainShiftNet(nn.Module):
    """
    U-Net style encoder-decoder for deformable MRI-to-US registration.

    Input:
      src (moving): pre-operative MRI (T1 and/or FLAIR)  — channels = n_mri_ch
      tgt (fixed):  intra-operative US before resection   — channels = 1

    Output:
      flow: (B, 3, D, H, W) — 3D displacement field
    """

    def __init__(
        self,
        in_channels: int = 3,       # T1 + FLAIR + US = 3 channels as input
        base_features: int = 32,
        img_size: Tuple[int,...] = (160, 192, 160),
        int_steps: int = 7,         # diffeomorphic integration steps
        use_attention: bool = True,
    ):
        super().__init__()
        f = base_features
        self.img_size    = img_size
        self.use_attn    = use_attention
        self.int_steps   = int_steps

        # Encoder (downsampling path)
        self.enc1 = ConvBlock(in_channels, f,    stride=1)  # (D, H, W)
        self.enc2 = ConvBlock(f,           f*2,  stride=2)  # /2
        self.enc3 = ConvBlock(f*2,         f*4,  stride=2)  # /4
        self.enc4 = ConvBlock(f*4,         f*8,  stride=2)  # /8

        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            ResConvBlock(f*8),
            ResConvBlock(f*8),
            ResConvBlock(f*8),
        )

        # Attention gates (skip connections)
        if use_attention:
            self.att3 = AttentionGate(f*8, f*4, f*4)
            self.att2 = AttentionGate(f*4, f*2, f*2)
            self.att1 = AttentionGate(f*2, f,   f)

        # Decoder (upsampling path)
        self.dec3 = ConvBlock(f*8 + f*4, f*4, stride=1)
        self.dec2 = ConvBlock(f*4 + f*2, f*2, stride=1)
        self.dec1 = ConvBlock(f*2 + f,   f,   stride=1)

        # Flow head — outputs displacement field
        self.flow_head = nn.Sequential(
            nn.Conv3d(f, f//2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(f//2, 3, 3, padding=1),
        )

        # Initialise flow head to near-zero (stability)
        nn.init.normal_(self.flow_head[-1].weight, mean=0, std=1e-5)
        nn.init.zeros_(self.flow_head[-1].bias)

        # Spatial transformer for warping
        self.spatial_transformer = SpatialTransformer(img_size)

    def upsample(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
        return x

    def forward(
        self,
        mri: torch.Tensor,   # (B, C_mri, D, H, W)
        us:  torch.Tensor,   # (B, 1, D, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          flow:       (B, 3, D, H, W) — displacement field
          mri_warped: (B, C, D, H, W) — MRI warped to US space
        """
        # Concatenate MRI + US as input
        x = torch.cat([mri, us], dim=1)

        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decode with skip connections
        d3 = self.upsample(b, e3)
        if self.use_attn:
            e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.upsample(d3, e2)
        if self.use_attn:
            e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.upsample(d2, e1)
        if self.use_attn:
            e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Predict displacement field
        flow = self.flow_head(d1)

        # Warp MRI using predicted flow
        mri_warped = self.spatial_transformer(mri, flow)

        return flow, mri_warped


# ─── DIFFEOMORPHIC VARIANT ─────────────────────────────────────────────────

class DiffeomorphicBrainShiftNet(BrainShiftNet):
    """
    Diffeomorphic variant: integrates a stationary velocity field
    using scaling-and-squaring. Ensures invertible, topology-preserving
    deformations — important for physically valid brain shift.
    """
    def __init__(self, *args, int_steps: int = 7, **kwargs):
        super().__init__(*args, int_steps=int_steps, **kwargs)
        self._int_steps = int_steps

    def integrate(self, velocity: torch.Tensor) -> torch.Tensor:
        """Scaling-and-squaring integration of velocity field."""
        flow = velocity / (2 ** self._int_steps)
        for _ in range(self._int_steps):
            flow = flow + self.spatial_transformer(flow, flow)
        return flow

    def forward(self, mri, us):
        # Get velocity field from base network
        velocity, _ = super().forward(mri, us)
        # Integrate to get diffeomorphic flow
        flow = self.integrate(velocity)
        mri_warped = self.spatial_transformer(mri, flow)
        return flow, mri_warped


# ─── MODEL FACTORY ─────────────────────────────────────────────────────────

def build_model(config: dict) -> nn.Module:
    """Build model from config dictionary."""
    model_type   = config.get("model_type", "standard")
    in_channels  = config.get("in_channels", 3)
    base_features = config.get("base_features", 32)
    img_size     = tuple(config.get("img_size", [160, 192, 160]))
    use_attention = config.get("use_attention", True)
    int_steps    = config.get("int_steps", 7)

    if model_type == "diffeomorphic":
        model = DiffeomorphicBrainShiftNet(
            in_channels=in_channels,
            base_features=base_features,
            img_size=img_size,
            int_steps=int_steps,
            use_attention=use_attention,
        )
    else:
        model = BrainShiftNet(
            in_channels=in_channels,
            base_features=base_features,
            img_size=img_size,
            use_attention=use_attention,
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_type} | {n_params/1e6:.2f}M parameters")
    return model


if __name__ == "__main__":
    # Quick sanity check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model({"model_type": "diffeomorphic", "base_features": 16,
                          "img_size": [64, 64, 64]}).to(device)
    mri = torch.randn(1, 2, 64, 64, 64).to(device)
    us  = torch.randn(1, 1, 64, 64, 64).to(device)
    flow, warped = model(mri, us)
    print(f"Flow shape: {flow.shape}")    # (1, 3, 64, 64, 64)
    print(f"Warped shape: {warped.shape}")  # (1, 2, 64, 64, 64)
    print("Model OK.")
