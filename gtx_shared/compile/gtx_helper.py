import re
from collections import defaultdict
from .gtx_graph import _XMODEL_NAME_PATTERN


class GTX_Helper(object):

    @classmethod
    def find_iss_ops_from_gtx_node(cls, gtx_node, xmodel):
        iss_op_lst = []
        formal_name = re.sub(_XMODEL_NAME_PATTERN, "_", gtx_node.name)
        for iss_op in cls.get_xmodel_ops(xmodel):
            if cls.get_iss_op_type(iss_op) in [
                "download",
                "upload",
                "fix2float",
                "float2fix",
                "transpose",
                "fix",
                "data-fix",
            ]:
                continue
            if formal_name in cls.get_iss_op_name(iss_op):
                iss_op_lst.append(iss_op)

        return iss_op_lst

    @staticmethod
    def get_iss_op_device_type(iss_op):
        if iss_op.has_attr("device"):
            return iss_op.get_attr("device")
        else:
            return None

    @staticmethod
    def get_iss_op_name(iss_op):
        return iss_op.get_name()

    @staticmethod
    def get_iss_op_template_name(op_template):
        return op_template.get_name()

    @staticmethod
    def get_iss_op_template_types(op_template):
        return op_template.get_types()

    @staticmethod
    def get_xmodel_ops(xmodel):
        return xmodel.get_ops()

    @staticmethod
    def get_iss_op_type(iss_op):
        return iss_op.get_type()

    @staticmethod
    def get_input_iss_ops(iss_op):
        return iss_op.get_input_ops()["input"]

    @staticmethod
    def get_op_partition_msg(iss_op):
        msg = ""
        if iss_op and iss_op.has_attr("partition_msg"):
            msg = iss_op.get_attr("partition_msg")
        elif iss_op and iss_op.has_attr("error_msg"):
            msg = iss_op.get_attr("error_msg")
        return msg

    @classmethod
    def is_dpu_pattern(cls, xmodel):
        for iss_op in cls.get_xmodel_ops(xmodel):
            if cls.get_iss_op_device_type(iss_op) == "CPU":
                if cls.get_iss_op_type(iss_op) == "reshape-fix":
                    input_op = cls.get_input_iss_ops(iss_op)[0]
                    if cls.get_iss_op_type(input_op) not in ["data", "data-fix"]:
                        return False
                elif cls.get_iss_op_type(iss_op) not in ["fix2float", "download"]:
                    return False
            elif cls.get_iss_op_device_type(iss_op) is None:
                return False
        return True

    @classmethod
    def get_pattern_partition_msg(cls, xmodel):
        msg = ""
        for iss_op in cls.get_xmodel_ops(xmodel):
            msg += cls.get_op_partition_msg(iss_op)
        return msg

    @classmethod
    def is_valid_compiled_pattern(cls, xmodel):
        for iss_op in cls.get_xmodel_ops(xmodel):
            if iss_op is None or iss_op.has_attr("error_msg"):
                return False
        if any(
            [cls.get_iss_op_device_type(iss_op) is None for iss_op in cls.get_xmodel_ops(xmodel)]
        ):
            return False
        return True
