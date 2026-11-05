__version__ = "2.3.2.post1"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
try:
    from mamba_ssm.modules.mamba3 import Mamba3
except ImportError:  # its kernels' dependencies may be missing, as on an Intel GPU; Mamba and Mamba2 do not need them
    Mamba3 = None
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
except ImportError:  # its generation helpers want names newer transformers releases removed
    MambaLMHeadModel = None
