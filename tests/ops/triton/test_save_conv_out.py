# Copyright (c) 2026, TPTNP.
"""The fused convolution and scan keeping the convolution's output for the backward.

By default the backward runs the convolution forward again to get x, B and C; with
MAMBA_SAVE_CONV_OUT=1 (or save_conv_out=True) the forward keeps them instead.  The convolution's
forward has no atomic adds, so the backward sees the same x, B and C either way and every
gradient must come out the same: to the bit in deterministic mode (MAMBA_DETERMINISTIC=1), except
the convolution's own weight and bias gradients, which causal_conv1d sums with atomic adds in any
mode; otherwise to the bit except every gradient that is summed with atomic adds.
"""
import copy
import os
import unittest
from unittest.mock import patch

import torch

from mamba_ssm.ops.triton import ssd_combined


def _accelerator():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return "xpu"
    return None


# Summed with atomic adds: by causal_conv1d whatever the mode, and by the scan unless deterministic.
CONVOLUTION = ("conv1d.weight", "conv1d.bias")
ATOMIC = ("dt_bias", "A_log", "D") + CONVOLUTION


class TheChoice(unittest.TestCase):
    def test_off_unless_asked_for(self):
        with patch.dict(os.environ):
            os.environ.pop("MAMBA_SAVE_CONV_OUT", None)
            self.assertFalse(ssd_combined._save_conv_out())
        with patch.dict(os.environ, {"MAMBA_SAVE_CONV_OUT": "0"}):
            self.assertFalse(ssd_combined._save_conv_out())
        with patch.dict(os.environ, {"MAMBA_SAVE_CONV_OUT": "1"}):
            self.assertTrue(ssd_combined._save_conv_out())


@unittest.skipUnless(_accelerator() is not None, "the fused path's Triton kernels run on a GPU")
class AgainstTheRecompute(unittest.TestCase):
    def setUp(self):
        self.device = _accelerator()
        self.convolutions = 0
        original = ssd_combined.causal_conv1d_fwd_function

        def counted(*args, **kwargs):
            self.convolutions += 1
            return original(*args, **kwargs)
        patcher = patch.object(ssd_combined, "causal_conv1d_fwd_function", counted)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _layer(self, **options):
        from mamba_ssm.modules.mamba2 import Mamba2
        torch.manual_seed(0)
        return Mamba2(d_model=64, d_state=32, headdim=16, use_mem_eff_path=True, **options).to(self.device)

    def _run(self, layer, source, save, deterministic):
        """Output, input gradient and every parameter gradient, and the convolution forwards run."""
        layer = copy.deepcopy(layer)
        environment = {"MAMBA_SAVE_CONV_OUT": "1" if save else "0", "MAMBA_DETERMINISTIC": "1" if deterministic else "0"}
        before = self.convolutions
        with patch.dict(os.environ, environment):
            value = source.clone().requires_grad_()
            with torch.autocast(self.device, dtype=torch.bfloat16):
                output = layer(value)
            output.float().square().mean().backward()
        return [output.detach(), value.grad] + [parameter.grad for parameter in layer.parameters()], self.convolutions - before

    def _same(self, **options):
        layer = self._layer(**options)
        source = torch.randn(3, 200, 64, device=self.device)
        names = ["output", "input gradient"] + [name for name, _ in layer.named_parameters()]
        for deterministic in (True, False):
            recomputed, recomputes = self._run(layer, source, False, deterministic)
            kept, keeps = self._run(layer, source, True, deterministic)
            self.assertEqual(recomputes, 2, "the recompute path runs the convolution forward again")
            self.assertEqual(keeps, 1, "the kept output spares the backward its convolution forward")
            atomic = CONVOLUTION if deterministic else ATOMIC
            for name, a, b in zip(names, recomputed, kept):
                with self.subTest(name=name, deterministic=deterministic):
                    if name in atomic:
                        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-5)
                    else:
                        self.assertTrue(torch.equal(a, b), f"{name} differs: {(a.float() - b.float()).abs().max().item()}")

    def test_gated_rms_norm(self):
        self._same()

    def test_without_the_norm(self):
        self._same(rmsnorm=False)

    def test_with_mlp_channels(self):
        self._same(d_ssm=32)

    def test_the_argument_wins_over_the_environment(self):
        layer = self._layer()
        source = torch.randn(2, 64, 64, device=self.device, requires_grad=True)
        with patch.dict(os.environ, {"MAMBA_SAVE_CONV_OUT": "1"}):
            out = ssd_combined.mamba_split_conv1d_scan_combined(
                layer.in_proj(source), layer.conv1d.weight.squeeze(1), layer.conv1d.bias, layer.dt_bias,
                -torch.exp(layer.A_log.float()), layer.D, chunk_size=layer.chunk_size, headdim=layer.headdim,
                ngroups=layer.ngroups, rmsnorm_weight=layer.norm.weight, outproj_weight=layer.out_proj.weight,
                save_conv_out=False)
            before = self.convolutions
            out.square().mean().backward()
        self.assertEqual(self.convolutions - before, 1, "save_conv_out=False recomputes despite the environment")


if __name__ == "__main__":
    unittest.main()
