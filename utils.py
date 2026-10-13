import torch
import numpy as np
from PIL import Image


def tensor2pil(image: torch.Tensor) -> Image.Image:
    """
    Accepts either:
      - (H,W,C)
      - (1,H,W,C)
    Returns a PIL RGB/RGBA image depending on channels.
    """
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.ndim == 4:
            # Expect (B,H,W,C); allow only B==1 here
            if t.shape[0] != 1:
                raise ValueError(f"tensor2pil expects batch of 1, got batch={t.shape[0]}")
            t = t[0]
        elif t.ndim != 3:
            raise ValueError(f"tensor2pil expects (H,W,C) or (1,H,W,C), got shape={tuple(t.shape)}")

        arr = (t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    raise TypeError(f"tensor2pil expected torch.Tensor, got {type(image)}")
