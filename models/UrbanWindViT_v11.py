"""V11 model: identical architecture to V8.

The only V11 difference is at the data layer (target nut is in log-space)
and at the loss layer (per-variable weights restored to balanced because
log-space already balances the dynamic range). The model itself doesn't
change — same 4-channel linear output, same dropout, same everything.
"""

# Re-export V8 model under the V11 name for clarity in checkpointing.
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
        self.__name__ = 'UrbanWindViT_v11'
