"""Smoke test for UrbanWindViT.

Run from the AirfRANS root: ``python test_model.py``.
"""

import math

import torch
from einops import rearrange
from torch_geometric.data import Data

from models.UrbanWindViT import PointNetEncoder, UrbanWindViT, ViTProcessor


def _make_mock_data(N=1000, M=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.empty(N, 2)
    pos[:, 0] = torch.rand(N, generator=g) * 6.0 - 2.0   # in [-2, 4]
    pos[:, 1] = torch.rand(N, generator=g) * 3.0 - 1.5   # in [-1.5, 1.5]
    x = torch.randn(N, 7, generator=g)
    # Make Uinf constant per case (column 2:4).
    x[:, 2:4] = x[0, 2:4][None, :]
    y = torch.randn(N, 4, generator=g)
    surf = torch.zeros(N, dtype=torch.bool)
    surf[:50] = True

    theta = torch.linspace(0, 2 * math.pi, M)
    airfoil_pos = torch.stack([0.5 * torch.cos(theta), 0.05 * torch.sin(theta)], dim=-1)
    return Data(pos=pos, x=x, y=y, surf=surf, airfoil_pos=airfoil_pos)


def test_forward_backward():
    data = _make_mock_data()
    model = UrbanWindViT()
    out = model(data)
    assert out.shape == (data.pos.shape[0], 4), f"unexpected output shape {out.shape}"
    assert not torch.isnan(out).any(), "output contains NaN"
    loss = out.pow(2).mean()
    loss.backward()
    grads_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert grads_ok, "no parameter received a gradient"
    print(f"forward+backward OK | output shape {tuple(out.shape)} | loss {loss.item():.4f}")


def test_patch_unpatch_identity():
    # einops rearrange-based patch + unpatch must be a pure permutation,
    # independent of the linear projections inside ViTProcessor.
    x = torch.randn(1, 256, 64, 64)
    patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=2, p2=2)
    recon = rearrange(patches, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                      h=32, w=32, p1=2, p2=2, c=256)
    assert torch.allclose(x, recon)
    # Sanity-check the ViTProcessor patch/unpatch shapes wire up.
    proc = ViTProcessor(grid_size=64, patch_size=2, dim=256, num_layers=1, num_heads=8, ffn_hidden=256)
    z = torch.randn(1, 256, 64, 64)
    tokens = proc.patchify(z)
    assert tokens.shape == (1, 1024, 256)
    back = proc.unpatchify(tokens)
    assert back.shape == (1, 256, 64, 64)
    print("patch/unpatch identity OK")


def test_empty_neighborhood():
    encoder = PointNetEncoder(scales=[(0.15, 32), (0.5, 64)], hidden_dim=32, out_dim_per_scale=64)
    grid = torch.randn(100, 2)
    far_point = torch.tensor([[1000.0, 1000.0]])
    out = encoder(grid, far_point)
    assert out.shape == (100, 128)
    assert torch.all(out == 0), f"expected all zeros, max abs {out.abs().max().item()}"
    assert not torch.isnan(out).any()
    print("empty-neighborhood handling OK")


if __name__ == '__main__':
    test_patch_unpatch_identity()
    test_empty_neighborhood()
    test_forward_backward()
    print("all tests passed")
