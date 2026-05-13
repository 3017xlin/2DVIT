"""V_best model: architecturally identical to V8.

The V_best name signifies the V8 architecture with the per-variable loss
weights restored to uniform [1, 1, 1, 1] (handled in train_vbest.py, not in
the model itself). The model code here is identical to UrbanWindViT_v8.

Used by:
  - main_vbest.py     (default radii: ((0.15, 32), (0.5, 64)))
  - main_vbest_v1.py  (radii override: ((0.08, 32), (0.20, 48)))
  - main_vbest_v1b.py (radii override: ((0.05, 32), (0.30, 64)))
"""

# Re-export V8 model under the V_best name. Identical architecture.
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name__ = 'UrbanWindViT_vbest'
