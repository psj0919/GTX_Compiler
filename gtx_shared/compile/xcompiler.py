try:
    import xcompiler
    import gtx
except ModuleNotFoundError:
    raise Exception("please install xcompiler package")
from gtx_shared.base import gtx_KEYS
from gtx_shared.utils import GtxOption, GtxScreenLogger


class XCompiler(object):

    @staticmethod
    def compile(xmodel_file, target):
        output_file = "_".join([xmodel_file, target]) + gtx_KEYS.XMODEL_SUFFIX
        args = [
            "-i",
            f"{xmodel_file}_int{gtx_KEYS.XMODEL_SUFFIX}",
            "-o",
            f"{output_file}",
            "-t",
            f"{target}",
            "--inspector",
        ]
        xcompiler.xcompiler([xmodel_file] + args)
        return output_file

    @classmethod
    def compile_and_reload(cls, xmodel_file, target):
        output_file = cls.compile(xmodel_file, target)
        graph = gtx.Graph.deserialize(output_file)
        return graph

    @staticmethod
    def compile_xgraph(xmodel_file, xgraph, target, fingerprint):
        if fingerprint is not None:
            cmd = {
                "inspector": True,
                "fingerprint": [fingerprint],
            }
        elif target is not None:
            cmd = {
                "inspector": True,
                "target": [target],
            }

        compiled_graph = xcompiler.xcompiler(xgraph.graph, cmd)
        if GtxOption.gtx_inspect_debug.value:
            output_file = "_".join([xmodel_file, target]) + gtx_KEYS.XMODEL_SUFFIX
            compiled_graph.serialize(output_file)
            GtxScreenLogger().info(f"The compiled graph is generated.({output_file})")
        return compiled_graph
