# Copyright (c) 2026, TPTNP.
"""The norm writing the dtype its output is cast to anyway, as before a matmul under autocast.

The kernel rounds as it stores, as the cast would, and its backward reads the gradient in that
dtype into float32 exactly, so output and gradients are the cast's to the bit -- at one kernel
configuration.  Tuned, Triton picks the kernels' warp counts per dtype they read and write, and a
row's sum runs in that count's order: then they agree to a float32 ulp, as one process's tuning
and another's already do.
"""
import unittest
from unittest.mock import patch

import torch

from mamba_ssm.ops.triton import layer_norm
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, rms_norm_fn


def _accelerator():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return "xpu"
    return None


def pinned(test):
    """Both norm kernels at their first configuration, untuned, for the rest of ``test``."""
    from triton.runtime.autotuner import Autotuner
    for name in ("_layer_norm_fwd_1pass_kernel", "_layer_norm_bwd_kernel"):
        tuner = getattr(layer_norm, name)
        while not isinstance(tuner, Autotuner):
            tuner = tuner.fn
        patcher = patch.object(tuner, "configs", tuner.configs[:1])
        patcher.start()
        test.addCleanup(patcher.stop)
        kept = dict(tuner.cache)
        tuner.cache.clear()
        test.addCleanup(lambda tuner=tuner, kept=kept: (tuner.cache.clear(), tuner.cache.update(kept)))


@unittest.skipUnless(_accelerator() is not None, "the norms' Triton kernels run on a GPU")
class TheNormWritesTheCastDtype(unittest.TestCase):
    def setUp(self):
        pinned(self)
        self.device = _accelerator()

    def _compare(self, norm, with_bias):
        torch.manual_seed(1)
        weight = (torch.randn(48, device=self.device) * .2 + 1).requires_grad_()
        bias = (torch.randn(48, device=self.device) * .1).requires_grad_() if with_bias else None
        value = torch.randn(5, 33, 48, device=self.device, requires_grad=True)
        upstream = torch.randn(5, 33, 48, device=self.device, dtype=torch.bfloat16)
        found = {}
        for direct in (True, False):
            for tensor in (value, weight, bias):
                if tensor is not None:
                    tensor.grad = None
            out = norm(value, weight, bias, out_dtype=torch.bfloat16) if direct else norm(value, weight, bias).to(torch.bfloat16)
            out.backward(upstream)
            found[direct] = [out, value.grad.clone(), weight.grad.clone()] + ([bias.grad.clone()] if with_bias else [])
        self.assertEqual(found[True][0].dtype, torch.bfloat16)
        for name, ours, theirs in zip(("output", "input gradient", "weight gradient", "bias gradient"), found[True], found[False]):
            self.assertTrue(torch.equal(ours, theirs), name)

    def test_rms_norm(self):
        self._compare(rms_norm_fn, with_bias=False)

    def test_layer_norm(self):
        self._compare(layer_norm_fn, with_bias=True)

    def test_without_it_the_output_keeps_the_input_dtype(self):
        value = torch.randn(4, 48, device=self.device)
        self.assertEqual(rms_norm_fn(value, torch.ones(48, device=self.device), None).dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
