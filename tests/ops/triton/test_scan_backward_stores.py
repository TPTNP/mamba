# Copyright (c) 2026, TPTNP.
"""Three ways the scan stopped doing work whose result it already had.

- The dx backward stores ddt and dD where a head's channels fit one tile, since each position then
  has one writer: the same numbers as the deterministic mode's stores, which sum in a fixed order.
- CB is computed causally: the tiles above the diagonal, which every kernel reading CB masks away,
  are left unwritten, so whatever they hold -- NaN here -- changes nothing.
- dcb's sum over head splits is skipped when there is one split.
"""
import os
import unittest
from unittest.mock import patch

import torch

from mamba_ssm.ops.triton import ssd_chunk_scan, ssd_combined
from mamba_ssm.ops.triton.ssd_bmm import _bmm_chunk_fwd
from mamba_ssm.ops.triton.ssd_chunk_scan import _chunk_scan_bwd_dcb, _chunk_scan_fwd
from mamba_ssm.ops.triton.ssd_chunk_state import _chunk_cumsum_fwd
from mamba_ssm.ops.triton.ssd_combined import _chunk_scan_chunk_state_bwd_dx


def _accelerator():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return "xpu"
    return None


@unittest.skipUnless(_accelerator() is not None, "the scan's Triton kernels run on a GPU")
class TheStores(unittest.TestCase):
    def setUp(self):
        self.device = _accelerator()
        torch.manual_seed(0)
        batch, seqlen, nheads, headdim, ngroups, dstate, self.chunk = 3, 96, 8, 32, 2, 64, 128
        self.x = torch.randn(batch, seqlen, nheads, headdim, device=self.device, dtype=torch.bfloat16)
        self.dout = torch.randn_like(self.x)
        dt = torch.rand(batch, seqlen, nheads, device=self.device)
        A = -torch.rand(nheads, device=self.device)
        self.B = torch.randn(batch, seqlen, ngroups, dstate, device=self.device, dtype=torch.bfloat16)
        self.C = torch.randn_like(self.B)
        self.D = torch.randn(nheads, device=self.device)
        self.dA_cumsum, self.dt = _chunk_cumsum_fwd(dt, A, self.chunk, dt_softplus=True)
        self.ngroups = ngroups

    def test_ddt_and_dD_as_the_deterministic_mode_stores_them(self):
        CB = _bmm_chunk_fwd(self.C, self.B, self.chunk, output_dtype=torch.float32)
        found = {}
        for mode in ("0", "1"):
            with patch.dict(os.environ, {"MAMBA_DETERMINISTIC": mode}):
                found[mode] = _chunk_scan_chunk_state_bwd_dx(self.x, self.dt, self.dA_cumsum, self.B, CB, self.dout, None, D=self.D)
        for name, single, deterministic in zip(("dx", "ddt", "dD"), found["0"], found["1"]):
            self.assertTrue(torch.equal(single, deterministic), name)

    def test_causal_cb_is_the_full_one_where_it_is_read(self):
        full = _bmm_chunk_fwd(self.C, self.B, self.chunk, output_dtype=torch.float32)
        causal = _bmm_chunk_fwd(self.C, self.B, self.chunk, causal=True, output_dtype=torch.float32)
        lower = torch.ones(self.chunk, self.chunk, dtype=torch.bool, device=self.device).tril()
        self.assertTrue(torch.equal(causal[..., lower], full[..., lower]))
        poisoned = full.clone()
        poisoned[..., ~lower] = float("nan")
        out_full, _ = _chunk_scan_fwd(full, self.x, self.dt, self.dA_cumsum, self.C, None, D=self.D)
        out_poisoned, _ = _chunk_scan_fwd(poisoned, self.x, self.dt, self.dA_cumsum, self.C, None, D=self.D)
        self.assertTrue(torch.equal(out_full, out_poisoned), "the upper triangle is never read")
        dx_full = _chunk_scan_chunk_state_bwd_dx(self.x, self.dt, self.dA_cumsum, self.B, full, self.dout, None, D=self.D)
        dx_poisoned = _chunk_scan_chunk_state_bwd_dx(self.x, self.dt, self.dA_cumsum, self.B, poisoned, self.dout, None, D=self.D)
        for name, a, b in zip(("dx", "ddt", "dD"), dx_full, dx_poisoned):
            self.assertTrue(torch.equal(a, b), name)
        ddA_full = ssd_chunk_scan._chunk_scan_bwd_ddAcs_stable(self.x, self.dt, self.dA_cumsum, self.dout, full)
        ddA_poisoned = ssd_chunk_scan._chunk_scan_bwd_ddAcs_stable(self.x, self.dt, self.dA_cumsum, self.dout, poisoned)
        self.assertTrue(torch.equal(ddA_full, ddA_poisoned))

    def test_dcb_of_one_split_is_the_sum_of_several(self):
        """One program per group of heads against one per head, summed afterwards: equal to rounding."""
        found = {}
        for cores in (1, 10**6):
            with patch.object(ssd_chunk_scan, "core_count", return_value=cores):
                found[cores] = _chunk_scan_bwd_dcb(self.x, self.dt, self.dA_cumsum, self.dout, ngroups=self.ngroups)
        self.assertEqual(found[1].shape, found[10**6].shape)
        torch.testing.assert_close(found[1], found[10**6], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
