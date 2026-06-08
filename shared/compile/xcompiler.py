try:
    import xcompiler
except ModuleNotFoundError:
    raise Exception("please install xcompiler package")
from shared.base import KEYS
from shared.utils import Option, ScreenLogger


class XCompiler(object):

    @staticmethod
    def compile(xmodel_file, target):
        output_file = "_".join([xmodel_file, target]) + KEYS.XMODEL_SUFFIX
        args = [
            "-i",
            f"{xmodel_file}_int{KEYS.XMODEL_SUFFIX}",
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
        graph = graph.Graph.deserialize(output_file)
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
        if Option.inspect_debug.value:
            output_file = "_".join([xmodel_file, target]) + KEYS.XMODEL_SUFFIX
            compiled_graph.serialize(output_file)
            ScreenLogger().info(f"The compiled graph is generated.({output_file})")
        return compiled_graph
