# Copyright (c) 2026, TPTNP.
"""Intel GPU support: the context each launch runs in, the core count grids are sized by, the norms'
backward grid, and the autocast the custom functions record -- each chosen by the tensor's device.

These run on any machine, the Intel GPU parts of torch stood in for by mocks.
"""
import os
import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from mamba_ssm.utils import device as device_utils


class Properties:
    def __init__(self, **values):
        self.__dict__.update(values)


class TheLaunchContext(unittest.TestCase):
    def test_an_intel_gpu_tensor_launches_in_its_own_context(self):
        xpu = MagicMock()
        xpu.device.return_value = "xpu context"
        with patch.object(torch, "xpu", xpu, create=True):
            self.assertEqual(device_utils.device_context(torch.device("xpu", 1)), "xpu context")
        xpu.device.assert_called_once_with(1)

    def test_a_cuda_tensor_keeps_cudas(self):
        with patch.object(torch.cuda, "device", return_value="cuda context") as cuda_device:
            self.assertEqual(device_utils.device_context(torch.device("cuda", 0)), "cuda context")
        cuda_device.assert_called_once_with(0)


class TheCoreCount(unittest.TestCase):
    def test_xe_cores_on_an_intel_gpu_and_multiprocessors_on_cuda(self):
        xpu = MagicMock()
        xpu.get_device_properties.return_value = Properties(gpu_subslice_count=64)
        with patch.object(torch, "xpu", xpu, create=True), \
                patch.object(torch.cuda, "get_device_properties", return_value=Properties(multi_processor_count=170)):
            self.assertEqual(device_utils.core_count(torch.device("xpu", 0)), 64)
            self.assertEqual(device_utils.core_count(torch.device("cuda", 0)), 170)

    def test_the_norms_split_their_rows_eight_ways_finer_on_an_intel_gpu(self):
        xpu = MagicMock()
        xpu.get_device_properties.return_value = Properties(gpu_subslice_count=64)
        with patch.object(torch, "xpu", xpu, create=True), \
                patch.object(torch.cuda, "get_device_properties", return_value=Properties(multi_processor_count=170)):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MAMBA_XPU_NORM_GRID", None)
                self.assertEqual(device_utils.norm_backward_programs(torch.device("xpu", 0)), 512)
                self.assertEqual(device_utils.norm_backward_programs(torch.device("cuda", 0)), 170, "CUDA unchanged")
            with patch.dict(os.environ, {"MAMBA_XPU_NORM_GRID": "1"}):
                self.assertEqual(device_utils.norm_backward_programs(torch.device("xpu", 0)), 64)


class TheAutocastDevice(unittest.TestCase):
    def test_xpu_only_where_there_is_an_intel_gpu_and_no_cuda(self):
        xpu = MagicMock()
        xpu.is_available.return_value = True
        with patch.object(torch, "xpu", xpu, create=True):
            with patch.object(torch.cuda, "is_available", return_value=False):
                self.assertEqual(device_utils.autocast_device_type(), "xpu")
            with patch.object(torch.cuda, "is_available", return_value=True):
                self.assertEqual(device_utils.autocast_device_type(), "cuda")
        xpu.is_available.return_value = False
        with patch.object(torch, "xpu", xpu, create=True), patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(device_utils.autocast_device_type(), "cuda")


class NoLaunchIsTiedToCuda(unittest.TestCase):
    def test_the_triton_ops_take_the_helpers(self):
        """A launch written the CUDA way again would fail on an Intel GPU without anything here noticing."""
        root = Path(device_utils.__file__).resolve().parents[1] / "ops" / "triton"
        offenders = []
        for path in sorted(root.glob("*.py")):
            text = path.read_text()
            for pattern in (r"torch\.cuda\.device\(", r"\.multi_processor_count"):
                offenders += [f"{path.name}: {pattern}" for _ in re.finditer(pattern, text)]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
