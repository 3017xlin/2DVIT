"""V_best_v4 model: V8 with deeper decoder (pos_mlp + pred_head 2 -> 4 layers each).

Single change vs V_best: the Decoder's two MLPs each grow from 2 linear
layers to 4 linear layers (with ReLU in between). pos_mlp now does
Linear(189->256)->ReLU->Linear(256->256)->ReLU->Linear(256->256)->ReLU->Linear(256->512),
and pred_head similarly. Hidden width unchanged.

Hypothesis: V_best's bottleneck is at the "latent -> physical field" decode
step (particularly for nut, which has highly non-linear relation to latent).
Deeper decoder MLPs let the model learn more expressive mappings without
increasing encoder/ViT capacity (which previous attempts showed harmful).
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
)


class Decoder_Deep(nn.Module):
    """V_best_v4 Decoder with deeper pos_mlp and pred_head (2 -> 4 layers each)."""

    _oob_warned = False

    def __init__(
        self,
        grid_dim=256,
        fourier_freqs=10,
        pos_hidden=256,
        pos_out=512,
        pred_hidden=256,
        out_dim=4,
        grid_x_min=-2.0,
        grid_x_max=4.0,
        grid_y_min=-1.5,
        grid_y_max=1.5,
    ):
        super().__init__()
        self.fourier = FourierFeatures(num_freqs=fourier_freqs)
        identity_dim = 9
        fourier_dim = identity_dim * self.fourier.out_dim_factor

        # V_best_v4: 4 layers (3 hidden + 1 output) instead of V8's 2 layers.
        self.pos_mlp = nn.Sequential(
            nn.Linear(fourier_dim, pos_hidden),
            nn.ReLU(),
            nn.Linear(pos_hidden, pos_hidden),
            nn.ReLU(),
            nn.Linear(pos_hidden, pos_hidden),
            nn.ReLU(),
            nn.Linear(pos_hidden, pos_out),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(grid_dim + pos_out, pred_hidden),
            nn.ReLU(),
            nn.Linear(pred_hidden, pred_hidden),
            nn.ReLU(),
            nn.Linear(pred_hidden, pred_hidden),
            nn.ReLU(),
            nn.Linear(pred_hidden, out_dim),
        )

        self.grid_x_min = grid_x_min
        self.grid_x_max = grid_x_max
        self.grid_y_min = grid_y_min
        self.grid_y_max = grid_y_max

    def physical_to_norm(self, pos):
        x_norm = 2.0 * (pos[:, 0] - self.grid_x_min) / (self.grid_x_max - self.grid_x_min) - 1.0
        y_norm = 2.0 * (pos[:, 1] - self.grid_y_min) / (self.grid_y_max - self.grid_y_min) - 1.0
        if self.training and not Decoder_Deep._oob_warned:
            with torch.no_grad():
                if bool(((x_norm.abs() > 1.0) | (y_norm.abs() > 1.0)).any()):
                    warnings.warn(
                        "Query points outside grid domain — clamped to border.",
                        stacklevel=2,
                    )
                    Decoder_Deep._oob_warned = True
        return torch.stack([x_norm.clamp(-1.0, 1.0), y_norm.clamp(-1.0, 1.0)], dim=-1)

    def forward(self, processed_grid, sdf_grad_grid, query_pos,
                query_uinf, query_sdf, query_normals):
        N = query_pos.shape[0]
        norm_query = self.physical_to_norm(query_pos)
        sample_grid = norm_query.reshape(1, N, 1, 2)

        local_geo = F.grid_sample(
            processed_grid, sample_grid, mode='bilinear', align_corners=True, padding_mode='border'
        ).squeeze(-1).squeeze(0).permute(1, 0)

        sdf_grad_at_query = F.grid_sample(
            sdf_grad_grid, sample_grid, mode='bilinear', align_corners=True, padding_mode='border'
        ).squeeze(-1).squeeze(0).permute(1, 0)

        identity = torch.cat([
            query_pos,
            query_uinf,
            query_sdf,
            sdf_grad_at_query,
            query_normals,
        ], dim=-1)

        fourier_feats = self.fourier(identity)
        pos_encoding = self.pos_mlp(fourier_feats)

        context = torch.cat([local_geo, pos_encoding], dim=-1)
        return self.pred_head(context)


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
        self.__name__ = 'UrbanWindViT_vbest_v4'

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

        self.decoder = Decoder_Deep(
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
        x = torch.linspace(self.grid_x_range[0], self.grid_x_range[1], self.grid_size)
        y = torch.linspace(self.grid_y_range[0], self.grid_y_range[1], self.grid_size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    @staticmethod
    def _signed_distance_2d(grid, polygon):
        from models.UrbanWindViT_v8 import UrbanWindViT as _v8
        return _v8._signed_distance_2d(grid, polygon)

    @staticmethod
    def _sdf_gradient_2d(sdf, dx, dy):
        from models.UrbanWindViT_v8 import UrbanWindViT as _v8
        return _v8._sdf_gradient_2d(sdf, dx, dy)

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
        latent = latent.reshape(H, W, -1).permute(2, 0, 1).unsqueeze(0)

        processed = self.processor(latent)

        sdf_grad_grid = sdf_grad_2d.permute(2, 0, 1).unsqueeze(0)
        out = self.decoder(
            processed,
            sdf_grad_grid,
            data.pos,
            x[:, 2:4],
            x[:, 4:5],
            x[:, 5:7],
        )
        return out
