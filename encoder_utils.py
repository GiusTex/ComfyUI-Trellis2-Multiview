import torch
import torch.nn as nn
import importlib
from typing import *
import torch.nn.functional as F

from comfy.ldm.trellis2.vae import SparseTensor, SparseConv3d, SparseSpatial2Channel
from comfy.ldm.modules.diffusionmodules.util import zero_module


def manual_cast(tensor, dtype):
    """
    Cast if autocast is not enabled.
    """
    if dtype == torch.float8_e4m3fn:
        dtype = torch.bfloat16

    if not torch.is_autocast_enabled():
        return tensor.type(dtype)
    return tensor


class LayerNorm32(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = 65536

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = manual_cast(x, torch.float32)
        o = super().forward(x)
        return manual_cast(o, x_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)


class SparseResBlockS2C3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        chunk_size: int = 65536,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.chunk_size = chunk_size
        
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = SparseConv3d(channels, self.out_channels // 8, 3)
        self.conv2 = zero_module(SparseConv3d(self.out_channels, self.out_channels, 3))
        self.skip_connection = lambda x: x.replace(x.feats.reshape(x.feats.shape[0], out_channels, channels * 8 // out_channels).mean(dim=-1))
        self.updown = SparseSpatial2Channel(2)

    def _forward(self, x: SparseTensor) -> SparseTensor:
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = self.updown(h)
        x = self.updown(x)
        h = h.replace(self.norm2(h.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h
    
    def forward(self, x: SparseTensor) -> SparseTensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)

_backends = {}
CONV = 'flex_gemm' 

def set_conv_backend(backend: Literal['none', 'spconv', 'torchsparse', 'flex_gemm']):
    global CONV
    CONV = backend

class SparseInverseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, indice_key=None):
        super(SparseInverseConv3d, self).__init__()
        if CONV not in _backends:
            _backends[CONV] = importlib.import_module(f'..conv_{CONV}', __name__)
        _backends[CONV].sparse_inverse_conv3d_init(self, in_channels, out_channels, kernel_size, stride, dilation, bias, indice_key)

    def forward(self, x: SparseTensor) -> SparseTensor:
        return _backends[CONV].sparse_inverse_conv3d_forward(self, x)