"""V14 model: V8 + auxiliary (Cd, Cl) prediction head.

Single change vs V8: a small 2-output MLP attached after the ViT processor
that mean-pools the latent grid and predicts (Cd_proxy, Cl_proxy) — i.e.
the integrated pressure-projection scalars. The forward returns a tuple
(field_pred, coef_pred); train_v14 supervises both terms.

Why: V8 has no global-integration signal. Per-point MSE on p doesn't
penalize systematic front-back asymmetry, but Cd integrates exactly that.
Adding a head that predicts Cd directly forces the encoder's latent to
retain Cd-relevant structure.
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.UrbanWindViT_v8 import (
    PointNetEncoder,
    RMSNorm,
    MultiHeadAttention,
    SwiGLUFFN,
    TransformerBlock,
    ViTProcessor,
    FourierFeatures,
    Decoder,
)


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
        aux_hidden=128,
    ):
        super().__init__()
        self.__name__ = 'UrbanWindViT_v14'

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

        self.processor = ViTProcessor(
            grid_size=grid_size,
            patch_size=patch_size,
            dim=latent_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_hidden=ffn_hidden,
            dropout=dropout,
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

        # V14 auxiliary head: mean-pool latent over space -> 2 scalars (Cd_proxy, Cl_proxy).
        self.cd_cl_head = nn.Sequential(
            nn.Linear(latent_dim, aux_hidden),
            nn.ReLU(),
            nn.Linear(aux_hidden, 2),
        )

        self.register_buffer('grid_coords', self._build_grid_coords(), persistent=False)

    def _build_grid_coords(self):
        x = torch.linspace(self.grid_x_range[0], self.grid_x_range[1], self.grid_size)
        y = torch.linspace(self.grid_y_range[0], self.grid_y_range[1], self.grid_size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    @staticmethod
    def _signed_distance_2d(grid, polygon):
        dist = torch.cdist(grid, polygon).min(dim=1).values
        p1 = polygon
        p2 = torch.roll(polygon, -1, dims=0)
        gx = grid[:, 0:1]
        gy = grid[:, 1:2]
        cond_y = (p1[:, 1][None, :] > gy) != (p2[:, 1][None, :] > gy)
        denom = p2[:, 1][None, :] - p1[:, 1][None, :]
        x_intersect = p1[:, 0][None, :] + (gy - p1[:, 1][None, :]) * (
            p2[:, 0][None, :] - p1[:, 0][None, :]
        ) / (denom + 1e-12)
        cond_x = x_intersect > gx
        crossings = (cond_y & cond_x).sum(dim=1)
        inside = (crossings % 2) == 1
        return torch.where(inside, -dist, dist)

    @staticmethod
    def _sdf_gradient_2d(sdf, dx, dy):
        H, W = sdf.shape
        grad_x = torch.zeros_like(sdf)
        grad_y = torch.zeros_like(sdf)
        if W >= 3:
            grad_x[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) / (2.0 * dx)
        if H >= 3:
            grad_y[1:-1, :] = (sdf[2:, :] - sdf[:-2, :]) / (2.0 * dy)
        if W >= 2:
            grad_x[:, 0] = (sdf[:, 1] - sdf[:, 0]) / dx
            grad_x[:, -1] = (sdf[:, -1] - sdf[:, -2]) / dx
        if H >= 2:
            grad_y[0, :] = (sdf[1, :] - sdf[0, :]) / dy
            grad_y[-1, :] = (sdf[-1, :] - sdf[-2, :]) / dy
        return torch.stack([grad_x, grad_y], dim=-1)

    def forward(self, data):
        x = data.x
        device = x.device
        N = x.shape[0]
        H = W = self.grid_size

        grid_coords = self.grid_coords
        if grid_coords.device != device:
            grid_coords = grid_coords.to(device)

        airfoil_pos = data.airfoil_pos
        if airfoil_pos.device != device:
            airfoil_pos = airfoil_pos.to(device)
        airfoil_pos = airfoil_pos.to(grid_coords.dtype)

        sdf = self._signed_distance_2d(grid_coords, airfoil_pos)
        sdf_2d = sdf.reshape(H, W)
        dx = (self.grid_x_range[1] - self.grid_x_range[0]) / (W - 1)
        dy = (self.grid_y_range[1] - self.grid_y_range[0]) / (H - 1)
        sdf_grad_2d = self._sdf_gradient_2d(sdf_2d, dx, dy)
        sdf_grad = sdf_grad_2d.reshape(H * W, 2)

        points = data.pos
        if points.device != device:
            points = points.to(device)
        pn_feats = self.pointnet(grid_coords, points)

        uinf = x[0, 2:4]
        uinf_grid = uinf[None, :].expand(H * W, -1)

        encoder_in = torch.cat([pn_feats, sdf[:, None], sdf_grad, uinf_grid], dim=-1)
        latent = self.encoder_proj(encoder_in)
        latent = latent.reshape(H, W, -1).permute(2, 0, 1).unsqueeze(0)  # [1, dim, H, W]

        processed = self.processor(latent)  # [1, dim, H, W]

        # V14 aux: mean-pool processed latent across spatial dims -> [1, dim] -> [1, 2] -> [2]
        pooled = processed.mean(dim=(2, 3))  # [1, dim]
        coef_aux = self.cd_cl_head(pooled).squeeze(0)  # [2]

        sdf_grad_grid = sdf_grad_2d.permute(2, 0, 1).unsqueeze(0)
        field_out = self.decoder(
            processed,
            sdf_grad_grid,
            data.pos,
            x[:, 2:4],
            x[:, 4:5],
            x[:, 5:7],
        )

        return field_out, coef_aux
