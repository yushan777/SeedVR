# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

"""
Single GPU optimized functions - bypasses distributed operations.
"""

import torch
from datetime import timedelta


def get_global_rank() -> int:
    """Get the global rank - always 0 for single GPU."""
    return 0


def get_local_rank() -> int:
    """Get the local rank - always 0 for single GPU."""
    return 0


def get_world_size() -> int:
    """Get the world size - always 1 for single GPU."""
    return 1


def get_device() -> torch.device:
    """Get current device - always cuda:0 for single GPU."""
    return torch.device("cuda:0")


def barrier_if_distributed(*args, **kwargs):
    """No-op for single GPU."""
    pass


def init_torch(cudnn_benchmark=True, timeout=timedelta(seconds=600)):
    """Single GPU PyTorch initialization - no distributed setup."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.cuda.set_device(0)
    # Skip distributed initialization


def convert_to_ddp(module: torch.nn.Module, **kwargs) -> torch.nn.Module:
    """Return module as-is for single GPU - no DDP wrapping."""
    return module


# Advanced distributed functions - all return single GPU defaults
def get_data_parallel_group():
    """No data parallel group for single GPU."""
    return None


def get_sequence_parallel_group():
    """No sequence parallel group for single GPU."""
    return None


def get_sequence_parallel_cpu_group():
    """No sequence parallel CPU group for single GPU."""
    return None


def get_data_parallel_rank() -> int:
    """Data parallel rank - always 0 for single GPU."""
    return 0


def get_data_parallel_world_size() -> int:
    """Data parallel world size - always 1 for single GPU."""
    return 1


def get_sequence_parallel_rank() -> int:
    """Sequence parallel rank - always 0 for single GPU."""
    return 0


def get_sequence_parallel_world_size() -> int:
    """Sequence parallel world size - always 1 for single GPU."""
    return 1


def init_sequence_parallel(sequence_parallel_size: int):
    """No-op for single GPU - no sequence parallel initialization."""
    pass


def get_sequence_parallel_global_ranks():
    """Return single rank list for single GPU."""
    return [0]


def get_next_sequence_parallel_rank() -> int:
    """Next sequence parallel rank - always 0 for single GPU."""
    return 0


def get_prev_sequence_parallel_rank() -> int:
    """Previous sequence parallel rank - always 0 for single GPU."""
    return 0


# Single GPU optimized ops - bypass all distributed operations
def sync_data(data, sp_idx, name="tmp"):
    """Return data as-is for single GPU - no synchronization needed."""
    return data


def slice_inputs(x: torch.Tensor, dim: int, padding: bool = True):
    """Return tensor as-is for single GPU - no slicing needed."""
    return x


def gather_outputs(x: torch.Tensor, **kwargs):
    """Return tensor as-is for single GPU - no gathering needed."""
    return x


def gather_seq_scatter_heads_qkv(qkv_tensor: torch.Tensor, **kwargs):
    """Return tensor as-is for single GPU - no sequence parallel operations."""
    return qkv_tensor


def gather_heads_scatter_seq(x: torch.Tensor, head_dim: int, seq_dim: int) -> torch.Tensor:
    """Return tensor as-is for single GPU - no head scattering needed."""
    return x


def gather_seq_scatter_heads(x: torch.Tensor, seq_dim: int, head_dim: int) -> torch.Tensor:
    """Return tensor as-is for single GPU - no sequence scattering needed."""
    return x


def scatter_heads(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Return tensor as-is for single GPU - no head scattering needed."""
    return x


def gather_heads(x: torch.Tensor, dim: int, grad_scale=False) -> torch.Tensor:
    """Return tensor as-is for single GPU - no head gathering needed."""
    return x


def remove_seqeunce_parallel_padding(x: torch.Tensor, dim: int, unpad_dim_size: int):
    """Return tensor as-is for single GPU - no padding removal needed."""
    return x
