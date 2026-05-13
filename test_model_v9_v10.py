"""Smoke test for V9 (bigger PointNet) and V10 (grid 96)."""

import math
import torch
from torch_geometric.data import Data

from models.UrbanWindViT_v9 import UrbanWindViT as UrbanWindViT_v9
from models.UrbanWindViT_v10 import UrbanWindViT as UrbanWindViT_v10


def _make_mock_data(N=1000, M=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.empty(N, 2)
    pos[:, 0] = torch.rand(N, generator=g) * 6.0 - 2.0
    pos[:, 1] = torch.rand(N, generator=g) * 3.0 - 1.5
    x = torch.randn(N, 7, generator=g)
    x[:, 2:4] = x[0, 2:4][None, :]
    y = torch.randn(N, 4, generator=g)
    surf = torch.zeros(N, dtype=torch.bool)
    surf[:50] = True
    theta = torch.linspace(0, 2 * math.pi, M)
    airfoil_pos = torch.stack([0.5 * torch.cos(theta), 0.05 * torch.sin(theta)], dim=-1)
    return Data(pos=pos, x=x, y=y, surf=surf, airfoil_pos=airfoil_pos)


def test(name, model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data = _make_mock_data()
    model.train()
    out = model(data)
    assert out.shape == (data.pos.shape[0], 4), f"{name}: wrong shape {out.shape}"
    assert not torch.isnan(out).any(), f"{name}: NaN"
    loss = out.pow(2).mean()
    loss.backward()
    print(f"{name}: params={n_params:,}, loss={loss.item():.4f}, OK")


if __name__ == '__main__':
    print("--- V9 (bigger PointNet) ---")
    m9 = UrbanWindViT_v9(
        pointnet_scales=((0.10, 64), (0.30, 96), (1.0, 128)),
        pointnet_hidden=64,
        pointnet_out_per_scale=96,
        dropout=0.1,
    )
    test("V9", m9)

    print("--- V10 (grid 96) ---")
    m10 = UrbanWindViT_v10(
        grid_size=96,
        dropout=0.1,
    )
    test("V10", m10)

    print("All tests passed")
