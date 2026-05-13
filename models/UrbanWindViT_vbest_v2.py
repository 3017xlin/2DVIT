"""V_best_v2 model: V8 + FiLM modulation on Uinf inside each ViT layer.

Single change vs V_best: each TransformerBlock applies a FiLM-style
gamma/beta modulation derived from Uinf [2] AFTER the FFN residual.
This re-injects flow conditions (free-stream velocity / equivalently
Re + AoA) at every depth of the ViT, rather than only at the encoder
entry and decoder identity.

The FiLM head is small (Linear(2, 2*latent_dim) per layer), adding only
~10K params total over 5 layers.
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.UrbanWindViT_vbest import (
    PointNetEncoder,
    RMSNorm,
    MultiHeadAttention,
    SwiGLUFFN,
    FourierFeatures,
    Decoder,
)


class TransformerBlock_FiLM(nn.Module):
    """V_best_v2 TransformerBlock with FiLM conditioning on Uinf."""

    def __init__(self, dim=256, num_heads=8, ffn_hidden=1024, dropout=0.0, uinf_dim=2):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_hidden, dropout=dropout)
        # FiLM head: 2 -> 2*dim. Per-layer affine modulation on token features.
        self.film = nn.Linear(uinf_dim, 2 * dim)
        # Initialise to identity transform: gamma=0, beta=0 -> x * 1 + 0 = x
        # so the layer starts as a no-op and gradually learns to condition.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, uinf, cos_x, sin_x, cos_y, sin_y):
        # x: [B, T, D], uinf: [B, 2]
        x = x + self.attn(self.norm1(x), cos_x, sin_x, cos_y, sin_y)
        x = x + self.ffn(self.norm2(x))
        # FiLM: (1 + gamma) * x + beta, broadcast over tokens.
        gamma_beta = self.film(uinf)              # [B, 2*D]
        gamma, beta = gamma_beta.chunk(2, dim=-1) # [B, D], [B, D]
        x = x * (1.0 + gamma).unsqueeze(1) + beta.unsqueeze(1)
        return x


class ViTProcessor_FiLM(nn.Module):
    """V_best_v2 ViTProcessor — passes Uinf through to each FiLM layer."""

    def __init__(self, grid_size=64, patch_size=2, dim=256, num_layers=5, num_heads=8,
                 ffn_hidden=1024, dropout=0.0, uinf_dim=2):
        super().__init__()
        assert grid_size % patch_size == 0
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.tokens_per_side = grid_size // patch_size
        self.num_tokens = self.tokens_per_side ** 2

        patch_flat = dim * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_flat, dim)
        self.unpatch_embed = nn.Linear(dim, patch_flat)

        self.layers = nn.ModuleList([
            TransformerBlock_FiLM(
                dim=dim, num_heads=num_heads, ffn_hidden=ffn_hidden,
                dropout=dropout, uinf_dim=uinf_dim,
            )
            for _ in range(num_layers)
        ])

        # Same U-Net skip pattern as V8
        self.skip_pairs = [(i, num_layers - 1 - i) for i in range(num_layers // 2)]
        self.skip_projections = nn.ModuleDict({
            str(dst): nn.Linear(2 * dim, dim) for _, dst in self.skip_pairs
        })

        self._rope_cache = None
        self._rope_device = None
        self._rope_dtype = None

    def _get_rope(self, device, dtype):
        from models.UrbanWindViT_vbest import _build_rope_freqs
        if (self._rope_cache is None
                or self._rope_device != device
                or self._rope_dtype != dtype):
            head_dim = self.dim // self.num_heads
            self._rope_cache = _build_rope_freqs(
                head_dim, self.tokens_per_side, self.tokens_per_side, device, dtype
            )
            self._rope_device = device
            self._rope_dtype = dtype
        return self._rope_cache

    def patchify(self, x):
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                      p1=self.patch_size, p2=self.patch_size)
        return self.patch_embed(x)

    def unpatchify(self, x):
        x = self.unpatch_embed(x)
        return rearrange(
            x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
            h=self.tokens_per_side, w=self.tokens_per_side,
            p1=self.patch_size, p2=self.patch_size, c=self.dim,
        )

    def forward(self, x, uinf):
        # x: [B, dim, H, W], uinf: [B, 2]
        cos_x, sin_x, cos_y, sin_y = self._get_rope(x.device, x.dtype)
        x = self.patchify(x)

        skip_dict = {dst: src for src, dst in self.skip_pairs}
        cache = {}
        for i, layer in enumerate(self.layers):
            if i in skip_dict:
                src = skip_dict[i]
                x = self.skip_projections[str(i)](torch.cat([x, cache[src]], dim=-1))
            x = layer(x, uinf, cos_x, sin_x, cos_y, sin_y)
            cache[i] = x

        return self.unpatchify(x)


class UrbanWindViT(nn.Module):
    def __init__(
        self,
        grid_size=64,
        grid_x_range=(-2.0, 4.0),
        grid_y_range=(-1.5, 1.5),
        pointnet_scales=((0.15, 32), (0.5, 64)),
        pointnet_hidden=32,
        pointnet_out_per_scale=64,
        latent_dim=256,
        patch_size=2,
        num_layers=5,
        num_heads=8,
        ffn_hidden=1024,
        fourier_freqs=10,
        pos_hidden=256,
        pos_out=512,
        pred_hidden=256,
        out_dim=4,
        dropout=0.1,
    ):
        super().__init__()
        self.__name__ = 'UrbanWindViT_vbest_v2'

        self.grid_size = grid_size
        self.grid_x_range = tuple(grid_x_range)
        self.grid_y_range = tuple(grid_y_range)

        self.pointnet = PointNetEncoder(
            scales=pointnet_scales,
            hidden_dim=pointnet_hidden,
            out_dim_per_scale=pointnet_out_per_scale,
        )
        encoder_in_dim = self.pointnet.out_dim + 1 + 2 + 2
        self.encoder_proj = nn.Linear(encoder_in_dim, latent_dim)

        self.processor = ViTProcessor_FiLM(
            grid_size=grid_size,
            patch_size=patch_size,
            dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_hidden=ffn_hidden,
            dropout=dropout,
            uinf_dim=2,
        )

        self.decoder = Decoder(
            grid_dim=latent_dim,
            fourier_freqs=fourier_freqs,
            pos_hidden=pos_hidden,
            pos_out=pos_out,
            pred_hidden=pred_hidden,
            out_dim=out_dim,
            grid_x_min=self.grid_x_range[0],
            grid_x_max=self.grid_x_range[1],
            grid_y_min=self.grid_y_range[0],
            grid_y_max=self.grid_y_range[1],
        )

        self.register_buffer('grid_coords', self._build_grid_coords(), persistent=False)

    def _build_grid_coords(self):
        import numpy as np
        x = torch.linspace(self.grid_x_range[0], self.grid_x_range[1], self.grid_size)
        y = torch.linspace(self.grid_y_range[0], self.grid_y_range[1], self.grid_size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    def forward(self, data):
        device = data.pos.device
        N = data.pos.shape[0]
        H = W = self.grid_size

        grid_coords = self.grid_coords
        if grid_coords.device != device:
            grid_coords = grid_coords.to(device)

        # Precomputed analytic geometry from preprocess.py v2.
        grid_sdf = data.grid_sdf
        grid_sdf_grad = data.grid_sdf_grad
        if grid_sdf.device != device:
            grid_sdf = grid_sdf.to(device)
            grid_sdf_grad = grid_sdf_grad.to(device)

        pn_feats = self.pointnet(grid_coords, data.pos)

        uinf = data.uinf.to(device)                # [2]  (case-level)
        uinf_grid = uinf[None, :].expand(H * W, -1)

        encoder_in = torch.cat(
            [pn_feats, grid_sdf[:, None], grid_sdf_grad, uinf_grid], dim=-1
        )
        latent = self.encoder_proj(encoder_in)
        latent = latent.reshape(H, W, -1).permute(2, 0, 1).unsqueeze(0)  # [1, dim, H, W]

        # V2: pass uinf through ViT for per-layer FiLM modulation.
        uinf_batch = uinf.unsqueeze(0)              # [1, 2]
        processed = self.processor(latent, uinf_batch)

        out = self.decoder(
            processed,
            data.pos,
            uinf[None, :].expand(N, -1),   # [N, 2]
            data.sdf[:, None],              # [N, 1]
            data.sdf_grad,                  # [N, 2]
        )
        return out
