class CompilerFactory(object):
    @staticmethod
    def get_compiler(compile_backend):
        if compile_backend == "xmodel":
            from .gtx_compiler import GtxCompiler

            return GtxCompiler()
        elif compile_backend == "xcompiler":
            from .xcompiler import XCompiler

            return XCompiler()
        else:
            raise NotImplementedError("other compiler is not implemented except gtx")
