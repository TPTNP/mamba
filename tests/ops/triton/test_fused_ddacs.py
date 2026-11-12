# Copyright (c) 2026, TPTNP.
"""The scan backward's ddA_cumsum worked out by the dcb kernel (MAMBA_FUSED_DDACS=1, the default).

ddA_cumsum[k] sums, over the strictly lower triangle of a chunk, the pairs (m, n) with m >= k > n of
(dout x^T)[m, n] dt[n] CB[m, n] exp(dA_cs[m] - dA_cs[n]).  The separate kernel took a cumsum along
every row of that triangle; the sum telescopes to ddA_cumsum[k] - ddA_cumsum[k-1] = colsum[k-1] -
rowsum[k-1], so the dcb kernel, which forms the same decayed dout x^T tiles for dCB, stores its tiles'
row and column sums and one cumsum over the chunk finishes it.  dCB, and with it dx, dB, dC and dD,
come out the same to the bit; dt, A and dt_bias move by fp32 summation order.
"""
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
    def test_on_unless_turned_off(self):
        with patch.dict(os.environ):
            os.environ.pop("MAMBA_FUSED_DDACS", None)
            self.assertTrue(ssd_combined._fused_ddacs())
        with patch.dict(os.environ, {"MAMBA_FUSED_DDACS": "0"}):
            self.assertFalse(ssd_combined._fused_ddacs())


@unittest.skipUnless(_accelerator() is not None, "the scan's Triton kernels run on a GPU")
class TheScan(unittest.TestCase):
    """_mamba_chunk_scan_combined_bwd with ddA_cumsum from tile sums, against the release's kernel."""

    def _inputs(self, batch, seqlen, dtype):
        device = _accelerator()
        torch.manual_seed(0)
        r = lambda *shape, scale=1.0: (torch.randn(*shape, device=device) * scale).to(dtype)
        nheads, headdim, dstate, ngroups = 8, 32, 64, 2
        dt0 = torch.exp(torch.rand(nheads, device=device) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        return dict(x=r(batch, seqlen, nheads, headdim), dt=r(batch, seqlen, nheads, scale=0.1),
                    A=-torch.exp(torch.rand(nheads, device=device) * math.log(16)),
                    B=r(batch, seqlen, ngroups, dstate, scale=0.3), C=r(batch, seqlen, ngroups, dstate, scale=0.3),
                    D=torch.randn(nheads, device=device), dt_bias=dt0 + torch.log(-torch.expm1(-dt0)),
                    dout=r(batch, seqlen, nheads, headdim))

    # The three ways to ddA_cumsum: the dcb kernel's tile sums (the default), the separate kernel
    # with row and column sums (MAMBA_FUSED_DDACS=0), and the separate kernel's own per-row cumsum
    # as the release has it (MAMBA_FUSED_DDACS=0 MAMBA_DDACS_SUMS=0), the one both are held to.
    PATHS = (("fused", {"MAMBA_FUSED_DDACS": "1"}),
             ("sums", {"MAMBA_FUSED_DDACS": "0", "MAMBA_DDACS_SUMS": "1"}),
             ("release", {"MAMBA_FUSED_DDACS": "0", "MAMBA_DDACS_SUMS": "0"}))

    def _each_path(self, v, chunk):
        out, *_ = ssd_combined._mamba_chunk_scan_combined_fwd(
            v["x"], v["dt"], v["A"], v["B"], v["C"], chunk, D=v["D"], dt_bias=v["dt_bias"], dt_softplus=True)
        args = (v["dout"], v["x"], v["dt"], v["A"], v["B"], v["C"], out, chunk)
        options = dict(D=v["D"], dt_bias=v["dt_bias"], dt_softplus=True)
        # The fused path first, so that the dcb kernel tunes inside it: the autotuner runs every
        # config on the same partial-sum buffers, and the sums must read only the chosen one's.
        results = {}
        for path, environment in self.PATHS:
            with patch.dict(os.environ, environment):
                results[path] = ssd_combined._mamba_chunk_scan_combined_bwd(*args, **options)
        return results

    def test_same_gradients(self):
        # several chunks, a ragged last chunk, and one chunk alone (the single-chunk path)
        # chunk sizes no other test uses, so each tunes afresh here
        for batch, seqlen, chunk in ((4, 1000, 96), (3, 300, 64), (3, 200, 224), (2, 160, 160)):
            for dtype in (torch.bfloat16, torch.float32):
                results = self._each_path(self._inputs(batch, seqlen, dtype), chunk)
                names = ["dx", "ddt", "dA", "dB", "dC", "dD", "dz", "ddt_bias"]
                for path in ("fused", "sums"):
                    for name, a, b in zip(names, results["release"], results[path]):
                        if a is None:
                            continue
                        with self.subTest(path=path, shape=(batch, seqlen, chunk), dtype=dtype, name=name):
                            if name in ("ddt", "dA", "ddt_bias"):
                                # fp32 sums in another order; a bf16 ddt rounds them to its own 8 bits,
                                # so an element may land one ulp (2^-8 to 2^-7 relative) to either side.
                                rtol = 2 ** -7 if a.dtype == torch.bfloat16 else 1e-4
                                torch.testing.assert_close(b, a, rtol=rtol, atol=1e-5 * a.abs().max().item())
                            else:
                                self.assertTrue(torch.equal(a, b), f"{name}: {(a.float() - b.float()).abs().max().item()}")


if __name__ == "__main__":
    unittest.main()
