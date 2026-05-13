"""V_best_v3 model: V8 with patch_size=1 (no patchify).

Single change vs V_best: the ViT processor uses patch_size=1, so each cell
of the 64x64 latent grid becomes one token (4096 tokens total instead of
1024). This gives the ViT full per-cell resolution, at the cost of 16x
more attention FLOPs (4096^2 vs 1024^2 dot-product pairs per head per layer).

Hypothesis: 2x2 patchify averages out boundary-layer detail. patch=1
preserves it. Trade-off is wall time + memory.
"""

from models.UrbanWindViT_v8 import (
    PointNetEncoder,
    RMSNorm,
    MultiHeadAttention,
    SwiGLUFFN,
    TransformerBlock,
    ViTProcessor,
    FourierFeatures,
    Decoder,
    UrbanWindViT as _UrbanWindViT_v8,
)


class UrbanWindViT(_UrbanWindViT_v8):
    """Same as V8 model, but patch_size defaults to 1."""

    def __init__(self, patch_size=1, **kwargs):
        # Force patch_size=1 unless caller explicitly overrides
        super().__init__(patch_size=patch_size, **kwargs)
        self.__name__ = 'UrbanWindViT_vbest_v3'
