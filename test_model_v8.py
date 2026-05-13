"""Smoke test for UrbanWindViT_v8 (with dropout)."""

import math
import torch
from einops import rearrange
from torch_geometric.data import Data

from models.UrbanWindViT_v8 import UrbanWindViT


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


def test_v8_forward_backward():
    data = _make_mock_data()
    model = UrbanWindViT(dropout=0.1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"V8 model params: {n_params:,}")
    assert model.__name__ == 'UrbanWindViT_v8'

    # train mode (dropout active)
    model.train()
    out_train = model(data)
    assert out_train.shape == (data.pos.shape[0], 4)
    assert not torch.isnan(out_train).any()
    loss = out_train.pow(2).mean()
    loss.backward()

    # eval mode (dropout off)
    model.eval()
    with torch.no_grad():
        out_eval = model(data)
    assert out_eval.shape == (data.pos.shape[0], 4)
    print(f"V8 forward+backward OK, loss={loss.item():.4f}")


if __name__ == '__main__':
    test_v8_forward_backward()
    print("V8 tests passed")
