# Copyright (c) 2026, TPTNP.
"""The Triton stand-ins for causal_conv1d's functions, which let Mamba2 take its fused path without
that package, as on an Intel GPU.

They must compute what causal_conv1d computes, in its calling convention: (batch, channels,
length) in channels-last memory, dx written into the buffer the fused path hands in.
"""
import unittest
import warnings
from unittest.mock import patch

import torch
import torch.nn.functional as F

from mamba_ssm.ops.triton import causal_conv1d_triton as causal_conv


def reference(x, weight, bias, silu):
    """Mamba2's unfused convolution: a padded depthwise Conv1d cut to length."""
    out = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1, groups=weight.shape[0])[..., :x.shape[-1]]
    return F.silu(out) if silu else out


def reference_gradients(x, weight, bias, dout, silu):
    leaves = [value.detach().clone().requires_grad_() for value in (x, weight, bias)]
    return torch.autograd.grad(reference(*leaves, silu), leaves, dout)


def channels_last(batch, channels, length, dtype=torch.float64, device="cpu", wider=0):
    """(batch, channels, length) with channels innermost, as a slice of a ``wider`` projection when asked."""
    return torch.randn(batch, length, channels + wider, dtype=dtype, device=device)[..., wider:].transpose(1, 2)


def one_tile(test):
    """Tune among the first candidate only: a full tuning is a dozen compilations per channel count."""
    for kernel in (causal_conv._forward_kernel, causal_conv._backward_kernel, causal_conv._backward_kernel4):
        patcher = patch.object(kernel, "configs", kernel.configs[:1]); patcher.start(); test.addCleanup(patcher.stop)


def _accelerator():
    if torch.cuda.is_available():return "cuda"
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():return "xpu"
    return None


class InPyTorch(unittest.TestCase):
    """On the CPU the PyTorch version runs, against PyTorch's own depthwise convolution in float64."""

    def setUp(self):
        torch.manual_seed(0)
        self.x = channels_last(2, 6, 9); self.weight = torch.randn(6, 4, dtype=torch.float64); self.bias = torch.randn(6, dtype=torch.float64)

    def test_the_forward(self):
        for silu in (False, True):
            with self.subTest(silu=silu):
                ours = causal_conv.causal_conv1d_fwd_function(self.x, self.weight, self.bias, None, None, None, silu)
                self.assertEqual(ours.shape, self.x.shape)
                torch.testing.assert_close(ours, reference(self.x, self.weight, self.bias, silu))

    def test_the_backward_writes_dx_into_the_buffer_it_is_given(self):
        dout = channels_last(2, 6, 9)
        for silu in (False, True):
            with self.subTest(silu=silu):
                buffer = channels_last(2, 6, 9)
                dx, dweight, dbias, dinitial = causal_conv.causal_conv1d_bwd_function(
                    self.x, self.weight, self.bias, dout, None, None, None, buffer, False, silu)
                self.assertIs(dx, buffer)
                self.assertIsNone(dinitial)
                for ours, theirs in zip((dx, dweight, dbias), reference_gradients(self.x, self.weight, self.bias, dout, silu)):
                    torch.testing.assert_close(ours, theirs)

    def test_carried_states_are_refused_rather_than_ignored(self):
        with self.assertRaises(NotImplementedError):
            causal_conv.causal_conv1d_fwd_function(self.x, self.weight, self.bias, torch.zeros(2, 9, dtype=torch.int32), None, None, True)


@unittest.skipUnless(torch.cuda.is_available() or _accelerator() == "xpu", "the Triton kernels run on a GPU")
class TritonKernels(unittest.TestCase):
    """The kernels against the float64 reference, reading a strided slice as the fused path hands them."""

    def setUp(self):
        self.device = _accelerator()
        self.assertTrue(causal_conv._triton["working"], "Triton is not importable here")
        if self._testMethodName != "test_bfloat16_as_training_runs_it":one_tile(self)

    def _case(self, batch, channels, length, width, dtype, bias=True):
        x = channels_last(batch, channels, length, dtype, self.device, wider=24)
        dout = channels_last(batch, channels, length, dtype, self.device, wider=8)
        weight = torch.randn(channels, width, device=self.device) * 0.5
        return x, weight, (torch.randn(channels, device=self.device) if bias else None), dout

    def _check(self, x, weight, bias, dout, silu, tolerance):
        wide = lambda value: None if value is None else value.double()
        expected_out = reference(wide(x), wide(weight), wide(bias), silu)
        out = causal_conv.causal_conv1d_fwd_function(x, weight, bias, None, None, None, silu)
        self.assertEqual(out.dtype, x.dtype)
        torch.testing.assert_close(out.double(), expected_out, **tolerance)
        buffer = torch.zeros_like(x.transpose(1, 2)).transpose(1, 2)
        dx, dweight, dbias, _ = causal_conv.causal_conv1d_bwd_function(x, weight, bias, dout, None, None, None, buffer, False, silu)
        self.assertIs(dx, buffer)
        expected = reference_gradients(wide(x), wide(weight), wide(bias) if bias is not None else torch.zeros_like(wide(weight[:, 0])), wide(dout), silu)
        torch.testing.assert_close(dx.double(), expected[0], **tolerance)
        torch.testing.assert_close(dweight.double(), expected[1], rtol=1e-4, atol=1e-3)
        self.assertEqual(dweight.dtype, weight.dtype)
        if bias is not None:torch.testing.assert_close(dbias.double(), expected[2], rtol=1e-4, atol=1e-3)
        else:self.assertIsNone(dbias)
        self.assertTrue(causal_conv._triton["working"], "the kernels fell back to PyTorch")

    def test_float32_over_ragged_tiles_and_widths(self):
        torch.manual_seed(2)
        for width in (2, 3, 4):
            for silu in (False, True):
                with self.subTest(width=width, silu=silu):
                    self._check(*self._case(3, 70, 131, width, torch.float32), silu, dict(rtol=1e-5, atol=1e-5))

    def test_without_a_bias(self):
        torch.manual_seed(3)
        self._check(*self._case(2, 40, 77, 4, torch.float32, bias=False), True, dict(rtol=1e-5, atol=1e-5))

    def test_bfloat16_as_training_runs_it(self):
        """With the whole tuning, once, at a width the tile choice matters for."""
        torch.manual_seed(4)
        self._check(*self._case(4, 160, 300, 4, torch.bfloat16), True, dict(rtol=2e-2, atol=2e-2))

    def test_every_tile_the_tuner_may_choose(self):
        torch.manual_seed(5)
        x, weight, bias, dout = self._case(2, 150, 75, 4, torch.float32)
        expected_out = reference(x.double(), weight.double(), bias.double(), True)
        expected = reference_gradients(x.double(), weight.double(), bias.double(), dout.double(), True)
        for tile in causal_conv.FORWARD_TILES:
            with self.subTest(forward=tile):
                torch.testing.assert_close(causal_conv._kernel_forward(x, weight, bias, True, tile).double(), expected_out, rtol=1e-5, atol=1e-5)
        # Width 4's kernel walks each sequence, so a length that is no multiple of any step it takes.
        for tile in causal_conv.BACKWARD_TILES4:
            with self.subTest(backward4=tile):
                buffer = torch.zeros_like(x.transpose(1, 2)).transpose(1, 2)
                dx, dweight, dbias = causal_conv._kernel_backward(x, weight, bias, dout, buffer, True, tile)
                torch.testing.assert_close(dx.double(), expected[0], rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(dweight.double(), expected[1], rtol=1e-4, atol=1e-3)
                torch.testing.assert_close(dbias.double(), expected[2], rtol=1e-4, atol=1e-3)
        x3, weight3, bias3, dout3 = self._case(2, 150, 75, 3, torch.float32)
        expected3 = reference_gradients(x3.double(), weight3.double(), bias3.double(), dout3.double(), True)
        for tile in causal_conv.BACKWARD_TILES:
            with self.subTest(backward=tile):
                dx, dweight, dbias = causal_conv._kernel_backward(x3, weight3, bias3, dout3, None, True, tile)
                torch.testing.assert_close(dx.double(), expected3[0], rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(dweight.double(), expected3[1], rtol=1e-4, atol=1e-3)
                torch.testing.assert_close(dbias.double(), expected3[2], rtol=1e-4, atol=1e-3)

    def test_width_four_takes_the_kernel_that_loads_each_tile_once(self):
        """Mamba2's width goes to the width-4 kernel, any other to the general one; both against float64."""
        torch.manual_seed(6)
        for width, expected in ((4, "_backward_kernel4"), (3, "_backward_kernel")):
            with self.subTest(width=width):
                x, weight, bias, dout = self._case(2, 70, 75, width, torch.float32)
                taken = []
                originals = {name: getattr(causal_conv, name) for name in ("_backward_kernel", "_backward_kernel4")}
                class Spy:
                    def __init__(self, name):self.name = name
                    def __getitem__(self, grid):taken.append(self.name);return originals[self.name][grid]
                    def __getattr__(self, attribute):return getattr(originals[self.name], attribute)
                with patch.object(causal_conv, "_backward_kernel", Spy("_backward_kernel")), patch.object(causal_conv, "_backward_kernel4", Spy("_backward_kernel4")):
                    dx, dweight, dbias = causal_conv._kernel_backward(x, weight, bias, dout, None, True)
                self.assertEqual(taken, [expected])
                expected_grads = reference_gradients(x.double(), weight.double(), bias.double(), dout.double(), True)
                torch.testing.assert_close(dx.double(), expected_grads[0], rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(dweight.double(), expected_grads[1], rtol=1e-4, atol=1e-3)
                torch.testing.assert_close(dbias.double(), expected_grads[2], rtol=1e-4, atol=1e-3)

    def test_a_failing_kernel_falls_back_to_pytorch(self):
        x, weight, bias, _ = self._case(2, 16, 20, 4, torch.float32)
        def broken(*arguments):raise RuntimeError("no kernel today")
        with patch.dict(causal_conv._triton, {"working": True}), patch.object(causal_conv, "_kernel_forward", broken):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = causal_conv.causal_conv1d_fwd_function(x, weight, bias, None, None, None, True)
            self.assertFalse(causal_conv._triton["working"])
        self.assertTrue(any("Triton kernel failed" in str(warning.message) for warning in caught))
        torch.testing.assert_close(out, reference(x, weight, bias, True), rtol=1e-5, atol=1e-5)


def _cuda_kernels():
    try:
        from causal_conv1d.cpp_functions import causal_conv1d_bwd_function, causal_conv1d_fwd_function
        return causal_conv1d_fwd_function, causal_conv1d_bwd_function
    except Exception:
        return None


@unittest.skipUnless(torch.cuda.is_available() and _cuda_kernels() is not None, "compared with causal_conv1d's own kernels on the card")
class AgainstCausalConv1d(unittest.TestCase):
    def setUp(self):
        one_tile(self)

    def test_the_functions(self):
        real_fwd, real_bwd = _cuda_kernels()
        torch.manual_seed(1)
        x = channels_last(3, 64, 200, torch.bfloat16, "cuda"); dout = channels_last(3, 64, 200, torch.bfloat16, "cuda")
        weight = torch.randn(64, 4, device="cuda"); bias = torch.randn(64, device="cuda")
        torch.testing.assert_close(causal_conv.causal_conv1d_fwd_function(x, weight, bias, None, None, None, True),
                                   real_fwd(x, weight, bias, None, None, None, True))
        ours = causal_conv.causal_conv1d_bwd_function(x, weight, bias, dout, None, None, None, None, False, True)
        theirs = real_bwd(x, weight, bias, dout, None, None, None, None, False, True)
        for left, right in zip(ours[:3], theirs[:3]):torch.testing.assert_close(left.float(), right.float(), rtol=2e-2, atol=2e-2)

    def test_a_fused_layer_computes_what_it_does_with_the_real_kernels(self):
        from mamba_ssm.modules.mamba2 import Mamba2
        from mamba_ssm.ops.triton import ssd_combined
        torch.manual_seed(0)
        layer = Mamba2(d_model=64, d_state=64, headdim=16, use_mem_eff_path=True).cuda()
        source = torch.randn(3, 200, 64, device="cuda")
        answers = []
        for functions in (_cuda_kernels(), (causal_conv.causal_conv1d_fwd_function, causal_conv.causal_conv1d_bwd_function)):
            with patch.object(ssd_combined, "causal_conv1d_fwd_function", functions[0]), \
                 patch.object(ssd_combined, "causal_conv1d_bwd_function", functions[1]):
                layer.zero_grad(); value = source.clone().requires_grad_()
                output = layer(value); output.square().mean().backward()
                answers.append((output.detach(), value.grad, layer.in_proj.weight.grad.clone(), layer.conv1d.weight.grad.clone()))
        self.assertTrue(causal_conv._triton["working"], "the kernels fell back to PyTorch")
        for ours, theirs in zip(answers[1], answers[0]):torch.testing.assert_close(ours, theirs, rtol=1e-3, atol=1e-4)


class TheFusedPathTakesThem(unittest.TestCase):
    """Where causal_conv1d is installed the fused path keeps it unless MAMBA_TRITON_CONV=1; without
    it, these.  The choice is made as ssd_combined is imported, so each case imports it afresh."""

    def _taken(self, environment):
        import os, subprocess, sys
        code = ("from mamba_ssm.ops.triton import ssd_combined, causal_conv1d_triton as ours;"
                "print(ssd_combined.causal_conv1d_fwd_function is ours.causal_conv1d_fwd_function,"
                " ssd_combined.causal_conv1d_bwd_function is ours.causal_conv1d_bwd_function)")
        result = subprocess.run([sys.executable, "-c", code], env={**os.environ, **environment}, capture_output=True, text=True, check=True)
        return result.stdout.split()

    def test_by_the_package_and_the_switch(self):
        try:
            import causal_conv1d  # noqa: F401
            installed = True
        except ImportError:
            installed = False
        self.assertEqual(self._taken({"MAMBA_TRITON_CONV": "1"}), ["True", "True"])
        self.assertEqual(self._taken({"MAMBA_TRITON_CONV": "0"}), ["False", "False"] if installed else ["True", "True"])


if __name__ == "__main__":
    unittest.main()
