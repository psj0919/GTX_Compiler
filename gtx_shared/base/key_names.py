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

import sys


class GlobalMap(object):
    globalmap = {}

    def set_map(self, key, value):
        self.globalmap[key] = value

    def set(self, **keys):
        try:
            for key_, value_ in keys.items():
                self.globalmap[key_] = str(value_)
                print(key_ + ":" + str(value_))
        except BaseException as msg:
            print(msg)
            raise msg

    def del_map(self, key):
        try:
            del self.globalmap[key]
            return self.globalmap
        except KeyError:
            pass
            # print("key:'" + str(key) + "'  not found!")

    def get_ele(self, key):
        if key in self.globalmap:
            return self.globalmap[key]
        return None

    def get(self, *args):
        try:
            dic = {}
            for key in args:
                if len(args) == 1:
                    dic = self.globalmap[key]
                    print(key + ":" + str(dic))
                elif len(args) == 1 and args[0] == "all":
                    dic = self.globalmap
                else:
                    dic[key] = self.globalmap[key]
            return dic
        except KeyError:
            print("key:'" + str(key) + "'  not found!")
            return "Null_"


class gtx_KEYS(object):
    # basic names
    INFO_FLAG = "gtx_NOTE"
    WARN_FLAG = "gtx_WARN"
    DEBUG_FLAG = "gtx_DEBUG"
    ERROR_FLAG = "gtx_ERROR"
    VERBOSE_LEVEL = "gtx_verbose_lvl"
    LOG_LEVEL = "gtx_log_lvl"
    LOGGER = "gtx_logger"
    SUFFIX_CONNECT = "SUFFIX"

    # for debug
    COMPILER = "gtx_compiler"
    OUTPUT_TO_NODE_MAP = "output_to_node_map"
    NODE_TO_OUTPUT_MAP = "node_to_output_map"

    # for Xgraph & Xnode
    XMODEL_SUFFIX = ".xmodel"
    XMODEL_IMAGE_SUFFIX = ".svg"
    XPARAM_SUFFIX = ".xparams"
    XPATTERN_SUFFIX = ".xpattern"
    XBLOBS_SUFFIX = ".xblobs"

    # for parsing/exporting
    TORCH_REFLECT_OPS_MAP = "torch_reflect_ops_map"
    TORCH_PARSER_MAP = "torch_parser_map"
    TORCH_SUPPORT_OPS_MAP = "torch_support_ops_map"
    TORCH_PARAM_MAP = "torch_parameters_name_map"
    TORCH_IR_ATTRS_MAP = "torch_ir_attrs_map"
    TORCH_SCHEMA_OP_TABLE = "torch_schema_op_table"
    NODE_CALLER_MAP = "node_caller_map"
    CUSTOM_OP_ATTRS_MAP = "custom_op_attrs_map"
    CUSTOM_TO_ISS_LIST = "custom_to_iss_list"
    DEVICE = "device"
    TORCH_SCRIPT_MODEL = "torch_script_model"
    # for quantization module:
    QUANT_MODE = "quant_mode"
    QUANTIZER = "gtx_quantizer"
    QUANT_SUFFIX = "_quant.json"
    QUANT_DEVICE = "quant_device"
    QUANT_CONFIG = "quant_config"

    PARAM_SCAN_SCOPE = "ParamScan"
    BLOB_SCAN_SCOPE = "BlobScan"

    QUANT_PARAMSCAN_OPS_COLLECTION = "qaunt_paramscan_ops_collection"

    BLOB_PREFFIX = "Blob"
    MAX_SCAN_SUFFIX = SUFFIX_CONNECT + "maxscan"
    MIN_SCAN_SUFFIX = SUFFIX_CONNECT + "minscan"
    DIFFS_SCAN_SUFFIX = SUFFIX_CONNECT + "diffs"

    QUANTTABLE_VAR_SUFFIX = SUFFIX_CONNECT + "QuantTableVar"

    # for load module
    gtx_LOADER = "gtx_loader"
    LOAD_FLAG = "load_flag"
    ORGVARS_SUFFIX = "_OrgVars.json"
    ORGKERASMODEL_SUFFIX = "_OrgKerasModel.json"

    # for modification process
    MODIFIER = "gtx_modifier"
    TRANS_SCOPE = "TransScp"

    # for graph export
    IR_GRAPH = "gtx_ir_graph"
    IR_NAME = "gtx_ir_name"
    IR_EXPORT_TYPE = "ir_export_type"

    # for training and controlling
    NRS_COLLECTION = "non_restorable_collection"
    NGTS_COLLECTION = "non_grad_tensor_collection"
    DEBUG_COLLECTION = "gtx_debug_collection"

    # for compile
    PARAMETER_FILE = "GtxParameter"
    ISTRUCTION_FILE = "GtxInstruction"
    WORKSPACE_PATH = "GtxWorkspace"
    INPUT_FILE = "GtxInput"
    DEVOP_PREFFIX = "fpga_op_"
    FIX_OP_SUFFIX = "_fix"
    PRE_FIX_OP_SUFFIX = "_pre_fix"
    TRANSPOSE_OP_SUFFIX = "_t"
    # deploy
    DEPLOY_CHECK_DATA_FOLDER = "deploy_check_data"

    # dynamo
    WEGO_DYNAMO_SCRIPTER = "wego_dynamo_scripter"
    GRAPH_COUNTER = "graph_counter"


class GTX_OP(object):
    ADAPTIVEAVGPOOL2D = "gtx_adaptive_avg_pool2d"
    ADD = "gtx_elemwise_add"
    ADDMM = "gtx_addmm"
    ANGLE = "gtx_angle"
    ARANGE = "gtx_arange"
    ARGMAX = "gtx_argmax_no_dim"
    ARGMAX_DIM = "gtx_argmax_dim"
    AVG_POOL = "gtx_avgpool"
    BASIC_GRU = "gtx_basic_gru"
    BASIC_LSTM = "gtx_basic_lstm"
    BATCH_NORM = "gtx_batch_norm"
    BATCH_TO_SPACE_ND = "gtx_batch_to_space_nd"
    BIAS_ADD = "gtx_bias_add"
    BIDIRECTIONAL_RNN = "gtx_bidirectional_rnn"
    BMM = "gtx_bmm"
    BUFFER_GET_NEXT = "gtx_buffer_get_next"
    BLOCK = "gtx_block"
    CAST = "gtx_cast"
    CALL_FUNCTION = "gtx_call_function"
    CALL_MODULE = "gtx_call_module"
    CALL_METHOD = "gtx_call_method"
    CEIL = "gtx_ceil"
    CHANNEL_SCALE = "gtx_channel_scale"
    CORRELATION1D_ELEMWISE = "gtx_correlation1d_elemwise"
    CORRELATION2D_ELEMWISE = "gtx_correlation2d_elemwise"
    COST_VOLUME = "gtx_cost_volume"
    CHUNK = "gtx_chunk"
    CLAMP = "gtx_clamp"
    COMPLEX_ABS = "gtx_complex_abs"
    CONCAT = "gtx_concat"
    CONSTANT_WITH_RESHAPE = "constant_with_reshape"
    CONST = "gtx_const"
    CONTIGUOUS = "gtx_contiguous"
    CONV1D = "gtx_conv1d"
    CONV2D = "gtx_conv2d"
    CONV3D = "gtx_conv3d"
    CONVTRANSPOSE2D = "gtx_conv_transpose_2d"
    CONVTRANSPOSE3D = "gtx_conv_transpose_3d"
    DENSE = "gtx_dense"
    DEPTHWISE_CONV1D = "gtx_depthwise_conv1d"
    DEPTHWISE_CONV2D = "gtx_depthwise_conv2d"
    DEPTHWISE_CONV3D = "gtx_depthwise_conv3d"
    DEPTHWISE_CONVTRANSPOSE2D = "gtx_depthwise_conv_transpose_2d"
    DEPTHWISE_CONVTRANSPOSE3D = "gtx_depthwise_conv_transpose_3d"
    DEQUANT_STUB = "gtx_dequant_stub"
    DERIVE_LOOP_INDEX = "gtx_derive_loop_index"
    DETACH = "gtx_detach"
    DEVICE = "gtx_device"
    DTYPE = "gtx_dtype"
    DIV = "gtx_elemwise_div"
    DROPOUT = "gtx_dropout"
    ELU = "gtx_elu"
    EMBEDDING = "gtx_embedding"
    EMBEDDING_BAG = "gtx_embedding_bag"
    EMPTY = "gtx_empty"
    EQUAL = "gtx_equal"
    EXP = "gtx_elemwise_exp"
    EXPAND = "gtx_expand"
    EXPAND_AS = "gtx_expand_as"
    EXPONENTIAL = "gtx_exponential"
    FLATTEN = "gtx_flatten"
    FLOOR = "gtx_floor"
    FLOOR_DIV = "gtx_floor_divide"
    FIX = "gtx_fix"
    FPGA_OP = "gtx_fpga_op"
    GATHER = "gtx_gather"
    GELU = "gtx_GELU"
    GENERIC = "gtx_generic"
    GRID_SAMPLE = "gtx_grid_sample"
    GROUP_NORM = "gtx_group_norm"
    GRU = "gtx_gru"
    HARDTANH = "gtx_hardtanh"
    HSIGMOID = "gtx_hsigmoid"
    HSWISH = "gtx_hswish"
    IDENTITY = "gtx_identity"
    IF = "gtx_if"
    INDEX = "gtx_index"
    INDEX_INPUT_INPLACE = "gtx_index_put_inplace"
    INPLACE_COPY = "gtx_copy_"
    INPUT = "gtx_input"
    INPUT_WITH_DEFAULT = "gtx_input_with_default"
    INSTANCE_NORM = "gtx_instance_norm"
    INT = "gtx_int"
    INTERPOLATE = "gtx_interpolate"
    IRFFT = "gtx_irfft"
    ITER_GET_NEXT = "gtx_iter_get_next"
    LAYER_NORM = "gtx_layer_norm"
    LEAKY_RELU = "gtx_leaky_relu"
    LENGTH = "gtx_len"
    LINEAR = "gtx_linear"
    LINEAR = "gtx_linear"
    LIST = "gtx_list"
    LIST_ADD = "gtx_list_add"
    LOG = "gtx_log"
    LOG_SOFTMAX = "gtx_log_softmax"
    LOOP = "gtx_loop"
    LSTM = "gtx_lstm"
    LSTM_CELL = "gtx_lstm_cell"
    MATMUL = "gtx_matmul"
    MAX = "gtx_max"
    MAX_POOL = "gtx_maxpool"
    MAX_POOL1D = "gtx_maxpool1d"
    MEAN = "gtx_mean"
    MERGE = "gtx_merge"
    MIN = "gtx_min"
    MISH = "gtx_mish"
    # e.g tf.math.multiply x * y, y can be a num or a tensor with same shape with x
    MULTIPLY = "gtx_elemwise_mul"
    # e.g tf.keras.layers.multiply
    # Takes a list of tensors, all of the same shape, and returns a single tensor (same shape).
    MULTIPLYLAYER = "gtx_multiply_layer"
    NANQUANTILE = "gtx_nanquantile"
    NEG = "gtx_neg"
    NOOP = "gtx_noop"
    NORM = "gtx_normalize"
    NOT_EQUAL = "gtx_not_equal"
    NON_TENSOR_SUB = "gtx_non_tensor_sub"
    ONE_HOT = "gtx_one_hot"
    PACK = "gtx_pack"
    PAD = "gtx_pad"
    PAD_ND = "gtx_pad_nd"
    PERMUTE = "gtx_permute"
    PIXEL_SHUFFLE = "gtx_pixel_shuffle"
    PIXEL_UNSHUFFLE = "gtx_pixel_unshuffle"
    PLACEHOLDER = "gtx_placeholder"
    PRELU = "gtx_prelu"
    QUANT_NEURON = "gtx_quant_neuron"
    QUANT_STUB = "gtx_quant_stub"
    QUANTILE = "gtx_quantile"
    RANDOM_UNIFORM = "gtx_random_uniform"
    RANGE = "gtx_range"
    REALDIV = "gtx_real_div"
    RELU = "gtx_relu"
    RELU6 = "gtx_relu6"
    RELUK = "gtx_reluk"
    REORG = "gtx_reorg"
    REPEAT = "gtx_repeat"
    RESHAPE = "gtx_reshape"
    RESCALING = "gtx_rescaling"
    RESIZE = "gtx_resize"
    RESIZE_3D = "gtx_resize_3d"
    RESIZE_NEAREST_3D = "gtx_resize_nearest_3d"
    RETURN = "gtx_return"
    RFFT = "gtx_rfft"
    RNN = "gtx_rnn"
    RNN_LAYER = "gtx_rnn_layer"
    RSQRT = "gtx_rsqrt"
    RSUB = "gtx_rsub"
    REMAINDER = "gtx_remainder"
    SCALAR_ADD = "gtx_add"
    SCALAR_EQUAL = "gtx_scalar_equal"
    SCALAR_LESS_THAN = "gtx_scalar_lt"
    SCALAR_MUL = "gtx_mul"
    SCALAR_SUB = "gtx_sub"
    SCALAR_REMAINDER = "gtx_scalar_remainder"
    SELECT = "gtx_select"
    SELU = "gtx_selu"
    SEPARABLECONV2D = "gtx_separableconv2D"
    SHAPE = "gtx_shape"
    SHAPE_AS_TENSOR = "gtx_shape_as_tensor"
    SIGMOID = "gtx_sigmoid"
    SIMPLE_RNN = "gtx_simple_rnn"
    SLICE = "gtx_slice"
    SLICE_TENSOR_INPLACE_COPY = "gtx_slice_tensor_inplace_copy"
    SOFTMAX = "gtx_softmax"
    SOFTPLUS = "gtx_softplus"
    SOFTSIGN = "gtx_softsign"
    SPACE_TO_BATCH_ND = "gtx_space_to_batch_nd"
    SPARSE_SOFTMAX_CROSS_ENTROPY = "gtx_sparse_softmax_cross_entropy_with_logits"
    SPLIT = "gtx_split"
    SQRT = "gtx_sqrt"
    SQUARE = "gtx_square"
    SQUEEZE = "gtx_squeeze"
    STACK = "gtx_stack"
    STACKED_RNN_CELLS = "gtx_stacked_rnn_cells"
    STFT = "gtx_stft"
    STRIDED_SLICE = "gtx_strided_slice"
    STRIDED_SLICE_INPLACE_COPY = "gtx_strided_slice_inplace_copy"
    SUB = "gtx_elementwise_sub"
    SUM = "gtx_sum"
    SWISH = "gtx_swish"
    TANH = "gtx_tanh"
    TENSOR = "gtx_tensor"
    TENSOR_ARRAY_GATHER = "gtx_tensor_array_gather"
    TENSOR_TO_SCALAR = "gtx_tensor_to_scalar"
    THRESHOLD = "gtx_threshold"
    TILE = "gtx_tile"
    TRANSPOSE = "gtx_transpose"
    TUPLE = "gtx_tuple"
    TUPLE_INPUT = "gtx_tuple_input"
    TUPLE_INDEX = "gtx_tuple_index"
    TUPLE_UNPACK = "gtx_tuple_unpack"
    UNSQUEEZE = "gtx_unsqueeze"
    UP_SAMPLING = "gtx_up_sampling"
    ZEROS = "gtx_zeros"
    UNIQUE_DIM = "gtx_unique_dim"
    _UNIQUE2 = "gtx_unique2"
    _UNIQUE = "gtx_unique"


class gtx_PARAM(object):
    WEIGHT = "weights"
    BIAS = "bias"
    GAMMA = "gamma"
    BETA = "beta"
    VAR = "var"
    MEAN = "mean"


class FrameworkType(object):
    # Frontend types
    TORCH = "torch"
    TENSORFLOW = "tensorflow"

    # GTX as a bridge
    GTX = "gtx"


class gtx_CONSTANT(object):
    INT_MAX = 2**31 - 1
