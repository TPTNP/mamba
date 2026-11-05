import torch
from functools import partial
from typing import Callable

from mamba_ssm.utils.device import autocast_device_type

def custom_amp_decorator(dec: Callable, cuda_amp_deprecated: bool):
    def decorator(*args, **kwargs):
        if cuda_amp_deprecated:
            # The autocast a custom function records and replays: CUDA's, or an Intel GPU's where
            # PyTorch has one and no CUDA.
            kwargs["device_type"] = autocast_device_type()
        return dec(*args, **kwargs)
    return decorator


if hasattr(torch.amp, "custom_fwd"): # type: ignore[attr-defined]
    deprecated = True
    from torch.amp import custom_fwd, custom_bwd # type: ignore[attr-defined]
else:
    deprecated = False
    from torch.cuda.amp import custom_fwd, custom_bwd

custom_fwd = custom_amp_decorator(custom_fwd, deprecated)
custom_bwd = custom_amp_decorator(custom_bwd, deprecated)
