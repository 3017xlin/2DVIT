"""
UrbanWindViT: Encode unstructured points onto a 2D latent grid via PointNet,
process the grid with a Vision Transformer, decode at arbitrary query points.

Pipeline:
    Input points [N, 7] + airfoil geometry [M, 2]
        -> ENCODER (build 64x64 grid -> circle query + PointNet -> SDF + grad
                    -> concat with Uinf -> [1, 256, 64, 64])
        -> PROCESSOR (2x2 patch -> 1024 tokens -> 5-layer ViT -> unpatch
                      -> [1, 256, 64, 64])
        -> DECODER (bilinear interp at queries + FourierMLP + MLP -> [N, 4])
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Encoder pieces
# ---------------------------------------------------------------------------

class PointNetEncoder(nn.Module):
    """Multi-scale circle-query PointNet encoding from input points onto a grid.

    For each grid point, gather neighbours within radius (capped at max_k), apply
    a shared MLP independently per neighbour on the relative coordinates, then
    max-pool over neighbours. Grid points with no neighbours within radius output
    zeros.
    """

    def __init__(self, scales, hidden_dim=32, out_dim_per_scale=64):
        super().__init__()
        self.scales = list(scales)
        self.out_dim_per_scale = out_dim_per_scale
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim_per_scale),
            )
            for _ in self.scales
        ])
        self.out_dim = out_dim_per_scale * len(self.scales)

    def forward(self, grid, points, chunk_size=512):
        feats = []
        for (radius, max_k), mlp in zip(self.scales, self.mlps):
            feats.append(self._encode_one_scale(grid, points, radius, max_k, mlp, chunk_size))
        return torch.cat(feats, dim=-1)

    def _encode_one_scale(self, grid, points, radius, max_k, mlp, chunk_size):
        G = grid.shape[0]
        N = points.shape[0]
        out = torch.zeros(G, self.out_dim_per_scale, device=grid.device, dtype=grid.dtype)

        k = min(max_k, N)
        if k == 0:
            return out

        for s in range(0, G, chunk_size):
            e = min(s + chunk_size, G)
            grid_chunk = grid[s:e]
            dists = torch.cdist(grid_chunk, points)  # [g, N]

            top_dists, top_idx = dists.topk(k, dim=1, largest=False)
            valid = top_dists < radius

            top_pos = points[top_idx]
            rel_pos = top_pos - grid_chunk[:, None, :]

            # Zero out invalid relative positions before MLP so values stay finite.
            rel_pos_safe = torch.where(valid[..., None], rel_pos, torch.zeros_like(rel_pos))
            feats_local = mlp(rel_pos_safe)

            # Mask out invalid neighbours so max-pool ignores them.
            neg_inf = torch.full_like(feats_local, float('-inf'))
            masked = torch.where(valid[..., None], feats_local, neg_inf)
            pooled = masked.max(dim=1).values

            no_valid = ~valid.any(dim=1)
            pooled = torch.where(no_valid[:, None], torch.zeros_like(pooled), pooled)

            out[s:e] = pooled
        return out


# ---------------------------------------------------------------------------
# Transformer pieces (RMSNorm + 2D RoPE + SwiGLU)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def _build_rope_freqs(head_dim, grid_h, grid_w, device, dtype):
    """Pre-compute axial 2D RoPE rotation factors.

    head_dim is split half-and-half across x- and y-axes. Each half holds
    head_dim/4 (sin, cos) pairs. Returns four tensors of shape [grid_h*grid_w, head_dim/4].
    """
    half_dim = head_dim // 2
    pair_dim = half_dim // 2

    # freqs = 1/10000^(2i/half_dim) for i in [0, pair_dim)
    inv_freqs = 1.0 / (10000.0 ** (torch.arange(0, half_dim, 2, dtype=dtype, device=device) / half_dim))

    h = torch.arange(grid_h, device=device, dtype=dtype)
    w = torch.arange(grid_w, device=device, dtype=dtype)
    h_grid, w_grid = torch.meshgrid(h, w, indexing='ij')
    h_idx = h_grid.flatten()
    w_idx = w_grid.flatten()

    angles_x = w_idx[:, None] * inv_freqs[None, :]
    angles_y = h_idx[:, None] * inv_freqs[None, :]
    return torch.cos(angles_x), torch.sin(angles_x), torch.cos(angles_y), torch.sin(angles_y)


def apply_rope_2d(x, cos_x, sin_x, cos_y, sin_y):
    """Apply 2D axial RoPE on the last (head) dimension.

    x: [B, H, T, D]; cos/sin tensors are [T, D/4].
    Lower half of D is rotated by x-axis (column index), upper half by y-axis.
    """
    B, H, T, D = x.shape
    half_d = D // 2
    pair_d = half_d // 2

    x_xhalf = x[..., :half_d]
    x_yhalf = x[..., half_d:]

    x_x_pairs = x_xhalf.reshape(B, H, T, pair_d, 2)
    x_y_pairs = x_yhalf.reshape(B, H, T, pair_d, 2)

    cos_xb = cos_x.reshape(1, 1, T, pair_d)
    sin_xb = sin_x.reshape(1, 1, T, pair_d)
    cos_yb = cos_y.reshape(1, 1, T, pair_d)
    sin_yb = sin_y.reshape(1, 1, T, pair_d)

    x0, x1 = x_x_pairs[..., 0], x_x_pairs[..., 1]
    rot_x = torch.stack([x0 * cos_xb - x1 * sin_xb,
                         x0 * sin_xb + x1 * cos_xb], dim=-1).reshape(B, H, T, half_d)

    y0, y1 = x_y_pairs[..., 0], x_y_pairs[..., 1]
    rot_y = torch.stack([y0 * cos_yb - y1 * sin_yb,
                         y0 * sin_yb + y1 * cos_yb], dim=-1).reshape(B, H, T, half_d)

    return torch.cat([rot_x, rot_y], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # 2D axial RoPE splits head_dim into halves (x-axis / y-axis), each into pairs.
        # Requires head_dim divisible by 4 to avoid length mismatch in apply_rope_2d.
        assert self.head_dim % 4 == 0, (
            f"head_dim must be divisible by 4 for 2D axial RoPE (got head_dim={self.head_dim})"
        )
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos_x, sin_x, cos_y, sin_y):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope_2d(q, cos_x, sin_x, cos_y, sin_y)
        k = apply_rope_2d(k, cos_x, sin_x, cos_y, sin_y)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.out(F.silu(self.gate(x)) * self.value(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8, ffn_hidden=1024, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_hidden, dropout=dropout)

    def forward(self, x, cos_x, sin_x, cos_y, sin_y):
        x = x + self.attn(self.norm1(x), cos_x, sin_x, cos_y, sin_y)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ViT processor (patch -> transformer w/ U-Net skips -> unpatch)
# ---------------------------------------------------------------------------

class ViTProcessor(nn.Module):
    def __init__(self, grid_size=64, patch_size=2, dim=256, num_layers=5, num_heads=8, ffn_hidden=1024, dropout=0.0):
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
            TransformerBlock(dim=dim, num_heads=num_heads, ffn_hidden=ffn_hidden, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Symmetric U-Net skips: layer i <-> layer (n-1-i) for i < n//2.
        # Middle layer (n//2 if n is odd) has no skip in.
        self.skip_pairs = [(i, num_layers - 1 - i) for i in range(num_layers // 2)]
        self.skip_projections = nn.ModuleDict({
            str(dst): nn.Linear(2 * dim, dim) for _, dst in self.skip_pairs
        })

        self._rope_cache = None
        self._rope_device = None
        self._rope_dtype = None

    def _get_rope(self, device, dtype):
        # Rebuild cache when device OR dtype changes — under AMP/autocast the dtype
        # of activations can switch (fp32 <-> bf16/fp16), and stale fp32 freqs would
        # mismatch the q/k tensors at the rotation step.
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
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
        return self.patch_embed(x)

    def unpatchify(self, x):
        x = self.unpatch_embed(x)
        return rearrange(
            x, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
            h=self.tokens_per_side, w=self.tokens_per_side,
            p1=self.patch_size, p2=self.patch_size, c=self.dim,
        )

    def forward(self, x):
        cos_x, sin_x, cos_y, sin_y = self._get_rope(x.device, x.dtype)

        x = self.patchify(x)

        skip_dict = {dst: src for src, dst in self.skip_pairs}
        cache = {}
        for i, layer in enumerate(self.layers):
            if i in skip_dict:
                src = skip_dict[i]
                x = self.skip_projections[str(i)](torch.cat([x, cache[src]], dim=-1))
            x = layer(x, cos_x, sin_x, cos_y, sin_y)
            cache[i] = x

        return self.unpatchify(x)


# ---------------------------------------------------------------------------
# Decoder (bilinear interp + FourierMLP + prediction head)
# ---------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    """NeRF-style sinusoidal positional encoding: freq = 2^k * pi for k in [0, L)."""

    def __init__(self, num_freqs=10):
        super().__init__()
        freqs = (2.0 ** torch.arange(num_freqs, dtype=torch.float32)) * math.pi
        self.register_buffer('freqs', freqs)
        self.num_freqs = num_freqs

    @property
    def out_dim_factor(self):
        # raw + 2*L (sin and cos for each freq)
        return 1 + 2 * self.num_freqs

    def forward(self, x):
        scaled = x[..., None] * self.freqs  # [..., D, L]
        sin = torch.sin(scaled)
        cos = torch.cos(scaled)
        encoded = torch.cat([sin, cos], dim=-1).reshape(*x.shape[:-1], -1)
        return torch.cat([x, encoded], dim=-1)


class Decoder(nn.Module):
    # Class-level flag so we only complain once per process when query points fall
    # outside the grid domain; otherwise a noisy training would print thousands.
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
        # Fourier on normalized position only (2 dims); concat the other 5
        # physics features raw (uinf 2 + sdf 1 + sdf_grad 2). Normals dropped:
        # sdf_grad carries the same outward-direction info on every node
        # without the 99%-zero sparsity that normals had on volume nodes.
        fourier_pos_dim = 2 * self.fourier.out_dim_factor
        other_feat_dim = 2 + 1 + 2
        combined_dim = fourier_pos_dim + other_feat_dim

        self.pos_mlp = nn.Sequential(
            nn.Linear(combined_dim, pos_hidden),
            nn.ReLU(),
            nn.Linear(pos_hidden, pos_out),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(grid_dim + pos_out, pred_hidden),
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
        # Warn (once per process) if any query point lies outside the grid domain so
        # debug doesn't have to chase silent boundary clamping.
        if self.training and not Decoder._oob_warned:
            with torch.no_grad():
                if bool(((x_norm.abs() > 1.0) | (y_norm.abs() > 1.0)).any()):
                    warnings.warn(
                        "Query points outside grid domain — clamped to border. "
                        "(This warning fires once per process.)",
                        stacklevel=2,
                    )
                    Decoder._oob_warned = True
        return torch.stack([x_norm.clamp(-1.0, 1.0), y_norm.clamp(-1.0, 1.0)], dim=-1)

    def forward(self, processed_grid, query_pos,
                query_uinf, query_sdf, query_sdf_grad):
        N = query_pos.shape[0]
        norm_query = self.physical_to_norm(query_pos)
        sample_grid = norm_query.reshape(1, N, 1, 2)

        local_geo = F.grid_sample(
            processed_grid, sample_grid, mode='bilinear',
            align_corners=True, padding_mode='border'
        ).squeeze(-1).squeeze(0).permute(1, 0)  # [N, dim]

        # Fourier on normalized position [-1, 1] only; raw concat for others.
        # query_sdf_grad is the analytic per-node eikonal gradient from
        # preprocess.py — no more grid_sample(sdf_grad_grid) at query time.
        fourier_pos = self.fourier(norm_query)                        # [N, 42]
        other_feats = torch.cat([
            query_uinf,       # [N, 2] z-scored
            query_sdf,        # [N, 1] z-scored
            query_sdf_grad,   # [N, 2] unit magnitude (raw)
        ], dim=-1)                                                     # [N, 5]
        pos_encoding = self.pos_mlp(torch.cat([fourier_pos, other_feats], dim=-1))

        context = torch.cat([local_geo, pos_encoding], dim=-1)
        return self.pred_head(context)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

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
        self.__name__ = 'UrbanWindViT_vbest'

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

        self.register_buffer('grid_coords', self._build_grid_coords(), persistent=False)

    def _build_grid_coords(self):
        x = torch.linspace(self.grid_x_range[0], self.grid_x_range[1], self.grid_size)
        y = torch.linspace(self.grid_y_range[0], self.grid_y_range[1], self.grid_size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        # Row-major: index i*W + j -> physical (x[j], y[i]).
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

        # PointNet circle-query encoding on the latent grid.
        pn_feats = self.pointnet(grid_coords, data.pos)

        # Uinf is constant per case; broadcast to every grid cell.
        uinf = data.uinf.to(device)
        uinf_grid = uinf[None, :].expand(H * W, -1)

        encoder_in = torch.cat(
            [pn_feats, grid_sdf[:, None], grid_sdf_grad, uinf_grid], dim=-1
        )
        latent = self.encoder_proj(encoder_in)
        latent = latent.reshape(H, W, -1).permute(2, 0, 1).unsqueeze(0)  # [1, dim, H, W]

        processed = self.processor(latent)

        out = self.decoder(
            processed,
            data.pos,
            uinf[None, :].expand(N, -1),   # [N, 2]
            data.sdf[:, None],              # [N, 1]
            data.sdf_grad,                  # [N, 2]
        )
        return out
