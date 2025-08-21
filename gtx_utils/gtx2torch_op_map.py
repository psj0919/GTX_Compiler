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
from gtx_shared.base import GTX_OP
from gtx_shared.utils import GtxScreenLogger

_GTX_OP_2_TORCH_OP = {}  # Dict[str, str]
_TORCH_OP_2_GTX_OP = {}  # Dict[str, str]


def add_mapping_item(gtx_op_name: str, torch_op_name: str):
    global _GTX_OP_2_TORCH_OP
    if gtx_op_name not in _GTX_OP_2_TORCH_OP:
        _GTX_OP_2_TORCH_OP[gtx_op_name] = torch_op_name
    global _TORCH_OP_2_GTX_OP
    if torch_op_name not in _TORCH_OP_2_GTX_OP:
        _TORCH_OP_2_GTX_OP[torch_op_name] = gtx_op_name


def get_GTX_OP_2_TORCH_OP_map():
    global _GTX_OP_2_TORCH_OP
    if len(_GTX_OP_2_TORCH_OP) == 0:
        raise Exception("please build the gtx_op -> torch_op map")
    return _GTX_OP_2_TORCH_OP


def get_TORCH_OP_2_GTX_OP_map():
    global _TORCH_OP_2_GTX_OP
    if len(_TORCH_OP_2_GTX_OP) == 0:
        raise Exception("please build the torch_op -> gtx_op map")
    return _TORCH_OP_2_GTX_OP


def get_torch_op_type(gtx_op_type):

    # if gtx_op_type in TORCH_UNSUPPORTED_GTXOPS:
    #   return gtx_op_type

    if get_GTX_OP_2_TORCH_OP_map().get(gtx_op_type, None) is None:
        raise Exception('please register the operator:"{}"'.format(gtx_op_type))
    else:
        return get_GTX_OP_2_TORCH_OP_map()[gtx_op_type]


def get_gtx_op_type(torch_op_type):

    if get_TORCH_OP_2_GTX_OP_map().get(torch_op_type, None) is None:
        # raise Exception('please register the operator:"{}"'.format(torch_op_type))
        GtxScreenLogger().warning(
            'There is no "{}" layer in this model, please remove "{}" \
configuration in config file'.format(
                torch_op_type, torch_op_type
            )
        )
        return
    else:
        return get_TORCH_OP_2_GTX_OP_map()[torch_op_type]
