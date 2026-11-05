# Copyright (c) 2026, TPTNP.
"""Mamba2's scan for sequences of one chunk, without the states between chunks.

It leaves out only work whose every contribution is zero there -- the states, their passing, the
state term in the scan's forward and dx kernels, the gradients through the states -- so it has to
answer what the full scan answers, outputs and every gradient.  In deterministic mode
(MAMBA_DETERMINISTIC=1) that is to the bit, except the convolution's gradients where the
causal_conv1d package computes them, with atomic adds the mode does not reach; otherwise to the
bit except where either sums with atomic adds, which differ in their last bits between two runs.
"""
import contextlib
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


ATOMIC = ("dt_bias", "A_log", "D", "conv1d.weight", "conv1d.bias")


def pin_configurations(test):
    """Every autotuned scan kernel at its smallest configuration, for the rest of ``test``.

    The kernels with and without states are tuned separately (HAS_STATES is a tuning key), and a
    different tile sums dD's partial sums in another order; at one configuration each, the two
    paths must agree to the bit.
    """
    import sys
    from triton.runtime.autotuner import Autotuner

    def size(config):
        product = 1
        for value in config.kwargs.values():
            product *= value if isinstance(value, int) else 1
        return (product, config.num_warps)
    for name, module in list(sys.modules.items()):
        if not name.startswith("mamba_ssm.ops.triton") or module is None:
            continue
        for value in list(vars(module).values()):
            for _ in range(4):
                if isinstance(value, Autotuner):
                    patcher = patch.object(value, "configs", [min(value.configs, key=size)])
                    patcher.start()
                    test.addCleanup(patcher.stop)
                    kept = dict(value.cache)
                    value.cache.clear()
                    test.addCleanup(lambda tuner=value, kept=kept: (tuner.cache.clear(), tuner.cache.update(kept)))
                    break
                value = getattr(value, "fn", None)
                if value is None:
                    break
# causal_conv1d's backward sums the weight and bias gradients with atomic adds whatever the mode.
CONVOLUTION = ("conv1d.weight", "conv1d.bias")


class TheChoice(unittest.TestCase):
    def test_one_chunk_without_initial_states_or_seq_idx(self):
        with patch.dict(os.environ, {"MAMBA_SINGLE_CHUNK": "1"}):
            self.assertTrue(ssd_combined._single_chunk(100, 128, None, None))
            self.assertTrue(ssd_combined._single_chunk(128, 128, None, None))
            self.assertFalse(ssd_combined._single_chunk(129, 128, None, None), "two chunks")
            self.assertFalse(ssd_combined._single_chunk(100, 128, torch.zeros(1), None), "an initial state")
            self.assertFalse(ssd_combined._single_chunk(100, 128, None, torch.zeros(1)), "packed sequences")
        with patch.dict(os.environ, {"MAMBA_SINGLE_CHUNK": "0"}):
            self.assertFalse(ssd_combined._single_chunk(100, 128, None, None), "switched off")


@unittest.skipUnless(_accelerator() is not None, "the scan's Triton kernels run on a GPU")
class AgainstTheFullScan(unittest.TestCase):
    def setUp(self):
        self.device = _accelerator()
        pin_configurations(self)
        self.states_computed = 0
        original = ssd_combined._chunk_state_fwd

        def counted(*args, **kwargs):
            self.states_computed += 1
            return original(*args, **kwargs)
        patcher = patch.object(ssd_combined, "_chunk_state_fwd", counted)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _layer(self, d_state, ngroups, fused):
        from mamba_ssm.modules.mamba2 import Mamba2
        torch.manual_seed(0)
        return Mamba2(d_model=64, d_state=d_state, headdim=16, ngroups=ngroups, use_mem_eff_path=fused).to(self.device)

    def _run(self, layer, source, single, deterministic):
        """Output and every gradient, with the single-chunk path allowed or not."""
        layer = copy.deepcopy(layer)
        environment = {"MAMBA_SINGLE_CHUNK": "1" if single else "0", "MAMBA_DETERMINISTIC": "1" if deterministic else "0"}
        with patch.dict(os.environ, environment):
            value = source.clone().requires_grad_()
            with torch.autocast(self.device, dtype=torch.bfloat16):
                output = layer(value)
            output.float().square().mean().backward()
        return [output.detach(), value.grad] + [parameter.grad for parameter in layer.parameters()]

    def _same(self, seqlen, chunk_size, d_state=64, ngroups=1, fused=True):
        """Compare the two in both modes; returns how many times the states were worked out on the single side."""
        layer = self._layer(d_state, ngroups, fused)
        layer.chunk_size = chunk_size
        source = torch.randn(3, seqlen, 64, device=self.device)
        names = ["output", "input gradient"] + [name for name, _ in layer.named_parameters()]
        computed = 0
        for deterministic in (True, False):
            full = self._run(layer, source, False, deterministic)
            before = self.states_computed
            single = self._run(layer, source, True, deterministic)
            computed += self.states_computed - before
            for name, a, b in zip(names, single, full):
                exact = name not in CONVOLUTION if deterministic else name not in ATOMIC
                torch.testing.assert_close(a, b, rtol=0 if exact else 1e-5, atol=0 if exact else 1e-7,
                                           msg=f"{name}, deterministic {deterministic}")
        return computed

    def test_the_fused_path(self):
        for d_state, ngroups in ((64, 2), (256, 1)):
            with self.subTest(d_state=d_state, ngroups=ngroups):
                self.assertEqual(self._same(100, 128, d_state, ngroups), 0, "a single chunk worked out its states")

    def test_a_chunk_of_512_not_a_power_of_two_long(self):
        self.assertEqual(self._same(300, 320, d_state=256), 0)
        self.assertEqual(self._same(300, 512, d_state=64), 0)

    def test_the_unfused_path(self):
        self.assertEqual(self._same(64, 64, fused=False), 0)

    def test_several_chunks_take_the_full_scan(self):
        self.assertGreater(self._same(200, 64), 0)

    def test_final_states_when_asked_for_them(self):
        """A caller that reads the final state gets it, from the full scan."""
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
        torch.manual_seed(1)
        x = torch.randn(2, 50, 4, 16, device=self.device)
        dt = torch.rand(2, 50, 4, device=self.device)
        A = -torch.rand(4, device=self.device)
        B = torch.randn(2, 50, 1, 32, device=self.device)
        C = torch.randn_like(B)
        out, final = mamba_chunk_scan_combined(x, dt, A, B, C, 64, return_final_states=True)
        self.assertEqual(final.shape, (2, 4, 16, 32))
        self.assertEqual(self.states_computed, 1)
        with patch.dict(os.environ, {"MAMBA_SINGLE_CHUNK": "0"}):
            out_full, final_full = mamba_chunk_scan_combined(x, dt, A, B, C, 64, return_final_states=True)
        self.assertTrue(torch.equal(out, out_full) and torch.equal(final, final_full))


if __name__ == "__main__":
    unittest.main()
