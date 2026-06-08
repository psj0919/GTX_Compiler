# Copyright 2025 Supergate.cc, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from torch import nn
from enum import unique, Enum

try:
    from packaging.version import Version as _Version
except Exception:  # pragma: no cover - fallback for older environments
    from distutils.version import LooseVersion as _Version


@unique
class CmpFlag(Enum):
    """
    Enum for comparison flags
    """

    EQUAL = 0
    LESS = 1
    LESS_EQUAL = 2
    GREATER = 3
    GREATER_EQUAL = 4
    NOT_EQUAL = 5


def compare_torch_version(*args):
    """Compare the installed ``torch.__version__`` against ``version``.

    Accepts the two arguments in either order, since callers in this codebase
    use both ``compare_torch_version(CmpFlag.LESS, "1.11.0")`` and
    ``compare_torch_version("1.6.0", CmpFlag.GREATER_EQUAL)``.
    """
    if len(args) != 2:
        raise TypeError(
            f"compare_torch_version expects 2 arguments, got {len(args)}"
        )

    compare_type, version = args
    if isinstance(version, CmpFlag):
        compare_type, version = version, compare_type
    if not isinstance(compare_type, CmpFlag):
        raise TypeError("one argument must be a CmpFlag")

    current = _Version(torch.__version__)
    target = _Version(str(version))

    if compare_type == CmpFlag.EQUAL:
        return current == target
    if compare_type == CmpFlag.LESS:
        return current < target
    if compare_type == CmpFlag.LESS_EQUAL:
        return current <= target
    if compare_type == CmpFlag.GREATER:
        return current > target
    if compare_type == CmpFlag.GREATER_EQUAL:
        return current >= target
    if compare_type == CmpFlag.NOT_EQUAL:
        return current != target
    raise ValueError(f"unknown comparison flag {compare_type!r}")


def strip_parallel(model):
    if isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    ):
        return model.module
    return model
