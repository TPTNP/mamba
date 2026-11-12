# Copyright (c) 2026, TPTNP.
"""The fused convolution and scan keeping the states passed between chunks for the backward.

By default the backward works the states out again (a chunk-state and a state-passing kernel);
with MAMBA_SAVE_SCAN_STATES=1 (or save_scan_states=True) the forward keeps the ones it handed to
the scan, in that dtype.  The backward's dC kernel reads them in that dtype either way, so dx, dB,
dC and dD come out the same to the bit; the gradient of each chunk's decay sums the states times
their gradient, and reads the kept states rounded where the recomputed ones were float32, so dt,
A and dt_bias move by that rounding.
"""
import copy
import math
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


class TheChoice(unittest.TestCase):
    def test_off_unless_asked_for(self):
        with patch.dict(os.environ):
            os.environ.pop("MAMBA_SAVE_SCAN_STATES", None)
            self.assertFalse(ssd_combined._save_scan_states())
        with patch.dict(os.environ, {"MAMBA_SAVE_SCAN_STATES": "1"}):
            self.assertTrue(ssd_combined._save_scan_states())


class _Counting(unittest.TestCase):
    def setUp(self):
        self.device = _accelerator()
        self.states_worked_out = 0
        original = ssd_combined._chunk_state_fwd

        def counted(*args, **kwargs):
            self.states_worked_out += 1
            return original(*args, **kwargs)
        patcher = patch.object(ssd_combined, "_chunk_state_fwd", counted)
        patcher.start()
        self.addCleanup(patcher.stop)


@unittest.skipUnless(_accelerator() is not None, "the scan's Triton kernels run on a GPU")
class TheScan(_Counting):
    """_mamba_chunk_scan_combined_bwd given the forward's states, against working them out."""

    def _inputs(self, batch, seqlen):
        torch.manual_seed(0)
        r = lambda *shape, scale=1.0: (torch.randn(*shape, device=self.device) * scale).to(torch.bfloat16)
        nheads, headdim, dstate, ngroups = 8, 32, 64, 2
        # Mamba2's dt initialisation (softplus(dt_bias) log-uniform in [1e-3, 1e-1]): a chunk's decay
        # stays well above zero, so the gradient through each chunk's decay -- the one that reads the
        # states' rounding -- is not lost to underflow.
        dt0 = torch.exp(torch.rand(nheads, device=self.device) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        return dict(x=r(batch, seqlen, nheads, headdim), dt=r(batch, seqlen, nheads, scale=0.1),
                    A=-torch.exp(torch.rand(nheads, device=self.device) * math.log(16)),
                    B=r(batch, seqlen, ngroups, dstate, scale=0.3), C=r(batch, seqlen, ngroups, dstate, scale=0.3),
                    D=torch.randn(nheads, device=self.device), dt_bias=dt0 + torch.log(-torch.expm1(-dt0)),
                    dout=r(batch, seqlen, nheads, headdim))

    def test_same_gradients(self):
        for batch, seqlen, chunk in ((4, 1000, 128), (3, 300, 64)):
            v = self._inputs(batch, seqlen)
            with patch.dict(os.environ, {"MAMBA_DETERMINISTIC": "1"}):
                out, _, _, dA_cumsum, states, _ = ssd_combined._mamba_chunk_scan_combined_fwd(
                    v["x"], v["dt"], v["A"], v["B"], v["C"], chunk, D=v["D"], dt_bias=v["dt_bias"],
                    dt_softplus=True, return_final_states=False)
                self.assertGreater(torch.exp(dA_cumsum[..., -1]).max().item(), 0.5, "some chunk keeps memory")
                self.assertIsNotNone(states)
                self.assertEqual(states.dtype, torch.bfloat16, "kept as the scan read them")
                before = self.states_worked_out
                args = (v["dout"], v["x"], v["dt"], v["A"], v["B"], v["C"], out, chunk)
                options = dict(D=v["D"], dt_bias=v["dt_bias"], dt_softplus=True)
                worked_out = ssd_combined._mamba_chunk_scan_combined_bwd(*args, **options)
                self.assertEqual(self.states_worked_out - before, 1)
                kept = ssd_combined._mamba_chunk_scan_combined_bwd(*args, **options, states=states)
                self.assertEqual(self.states_worked_out - before, 1, "the kept states spare the recompute")
            names = ["dx", "ddt", "dA", "dB", "dC", "dD", "dz", "ddt_bias"]
            for name, a, b in zip(names, worked_out, kept):
                if a is None:
                    continue
                with self.subTest(shape=(batch, seqlen, chunk), name=name):
                    if name in ("ddt", "dA", "ddt_bias"):
                        torch.testing.assert_close(b, a, rtol=2e-2, atol=2e-2 * a.abs().max().item())
                    else:
                        self.assertTrue(torch.equal(a, b), f"{name}: {(a.float() - b.float()).abs().max().item()}")


@unittest.skipUnless(_accelerator() is not None, "the fused path's Triton kernels run on a GPU")
class AMamba2Layer(_Counting):
    def _run(self, layer, source, keep):
        layer = copy.deepcopy(layer)
        before = self.states_worked_out
        with patch.dict(os.environ, {"MAMBA_SAVE_SCAN_STATES": "1" if keep else "0"}):
            value = source.clone().requires_grad_()
            with torch.autocast(self.device, dtype=torch.bfloat16):
                output = layer(value)
            output.float().square().mean().backward()
        grads = [value.grad] + [parameter.grad for parameter in layer.parameters()]
        return output.detach(), grads, self.states_worked_out - before

    def test_to_rounding(self):
        from mamba_ssm.modules.mamba2 import Mamba2
        for options in (dict(), dict(rmsnorm=False)):
            torch.manual_seed(0)
            layer = Mamba2(d_model=64, d_state=64, headdim=16, ngroups=2, chunk_size=64, use_mem_eff_path=True, **options).to(self.device)
            source = torch.randn(3, 300, 64, device=self.device)
            names = ["input"] + [name for name, _ in layer.named_parameters()]
            out_a, grads_a, count_a = self._run(layer, source, False)
            out_b, grads_b, count_b = self._run(layer, source, True)
            self.assertEqual((count_a, count_b), (2, 1), "forward and recompute, then the forward alone")
            self.assertTrue(torch.equal(out_a, out_b), "the forward is untouched")
            for name, a, b in zip(names, grads_a, grads_b):
                with self.subTest(options=options, name=name):
                    torch.testing.assert_close(b, a, rtol=2e-2, atol=2e-2 * a.abs().max().item())

    def test_one_chunk_has_no_states(self):
        from mamba_ssm.modules.mamba2 import Mamba2
        torch.manual_seed(0)
        layer = Mamba2(d_model=64, d_state=64, headdim=16, ngroups=2, chunk_size=256, use_mem_eff_path=True).to(self.device)
        source = torch.randn(3, 200, 64, device=self.device)
        out_a, grads_a, count_a = self._run(layer, source, False)
        out_b, grads_b, count_b = self._run(layer, source, True)
        self.assertEqual((count_a, count_b), (0, 0))
        self.assertTrue(torch.equal(out_a, out_b))


if __name__ == "__main__":
    unittest.main()
