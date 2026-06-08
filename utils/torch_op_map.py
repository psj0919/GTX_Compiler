#
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
#

from glob import glob
from shared.base import OP
from shared.utils import ScreenLogger

_OP_2_TORCH_OP = {}  # Dict[str, str]
_TORCH_OP_2_OP = {}  # Dict[str, str]


def add_mapping_item(op_name: str, torch_op_name: str):
    global _OP_2_TORCH_OP
    if op_name not in _OP_2_TORCH_OP:
        _OP_2_TORCH_OP[op_name] = torch_op_name
    global _TORCH_OP_2_OP
    if torch_op_name not in _TORCH_OP_2_OP:
        _TORCH_OP_2_OP[torch_op_name] = op_name


def get_OP_2_TORCH_OP_map():
    global _OP_2_TORCH_OP
    if len(_OP_2_TORCH_OP) == 0:
        raise Exception("please build the op -> torch_op map")
    return _OP_2_TORCH_OP


def get_TORCH_OP_2_OP_map():
    global _TORCH_OP_2_OP
    if len(_TORCH_OP_2_OP) == 0:
        raise Exception("please build the torch_op -> op map")
    return _TORCH_OP_2_OP


def get_torch_op_type(op_type):

    # if op_type in TORCH_UNSUPPORTED_OPS:
    #   return op_type

    op_map = get_OP_2_TORCH_OP_map()
    # export 는 op-type 을 대문자('CONV2D')로 내보내지만 맵 키는 OP 값('conv2d')이다.
    # 정확히 일치하지 않으면 소문자로 정규화하여 재시도한다.
    if op_map.get(op_type, None) is not None:
        return op_map[op_type]
    lowered = op_type.lower() if isinstance(op_type, str) else op_type
    if op_map.get(lowered, None) is not None:
        return op_map[lowered]
    raise Exception('please register the operator:"{}"'.format(op_type))


def get_op_type(torch_op_type):

    if get_TORCH_OP_2_OP_map().get(torch_op_type, None) is None:
        # raise Exception('please register the operator:"{}"'.format(torch_op_type))
        ScreenLogger().warning(
            'There is no "{}" layer in this model, please remove "{}" \
configuration in config file'.format(
                torch_op_type, torch_op_type
            )
        )
        return
    else:
        return get_TORCH_OP_2_OP_map()[torch_op_type]
