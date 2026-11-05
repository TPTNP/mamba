# Copyright (c) 2026, TPTNP.
"""Which device a Triton kernel launches on: CUDA (or ROCm), or an Intel GPU through PyTorch's XPU backend.

Mamba2's kernels are Triton, and Intel's Triton backend compiles them.  What tied them to CUDA was
the code around the launches: the ``torch.cuda.device`` context each launch runs in, which a
PyTorch built without CUDA refuses; the streaming-multiprocessor count some kernels size their
grids by; and the device type the custom autograd functions record autocast for.  These helpers
answer each by the tensor's own device, so one code path serves both.
"""
import os

import torch


def _xpu_available():
    xpu = getattr(torch, "xpu", None)
    return xpu is not None and xpu.is_available()


def device_context(device):
    """The context that makes ``device`` current for a kernel launch."""
    device = torch.device(device)
    if device.type == "xpu":
        return torch.xpu.device(device.index)
    return torch.cuda.device(device.index)


def core_count(device):
    """The device's compute units, as kernels that split their work over them count them: streaming
    multiprocessors on CUDA, compute units on ROCm, Xe cores on an Intel GPU."""
    device = torch.device(device)
    if device.type == "xpu":
        return torch.xpu.get_device_properties(device).gpu_subslice_count
    return torch.cuda.get_device_properties(device).multi_processor_count


def norm_backward_programs(device):
    """How many programs the norms' backward kernels split their rows over.

    One per streaming multiprocessor suits an NVIDIA card.  An Intel GPU runs each program on a
    few of an Xe core's threads, so one program per Xe core leaves most of it idle: on an Aurora
    tile (Max 1550, 64 Xe cores), eight per core took the plain RMSNorm backward over 49k rows of
    512 from 1.50 to 0.64 ms and the gated one from 2.01 to 0.86 ms, the gradients equal to
    rounding (the weight gradient's partial sums are added in more pieces).  MAMBA_XPU_NORM_GRID
    sets the factor; 1 restores one program per core.
    """
    count = core_count(device)
    if torch.device(device).type == "xpu":
        count *= max(1, int(os.environ.get("MAMBA_XPU_NORM_GRID", "8")))
    return count


def autocast_device_type():
    """The device type the custom autograd functions record and replay autocast for: CUDA's, unless
    this PyTorch has an Intel GPU and no CUDA."""
    if not torch.cuda.is_available() and _xpu_available():
        return "xpu"
    return "cuda"
