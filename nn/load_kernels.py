# import imp
import os
import sys
import torch
try:
    from torch.utils.cpp_extension import load, _import_module_from_library
except ImportError:
    load = None
    _import_module_from_library = None
from gtx_shared.utils import create_work_dir, GtxScreenLogger, QError, QWarning
from gtx_utils.torch_utils import CmpFlag, compare_torch_version

_cur_dir = os.path.dirname(os.path.realpath(__file__))
_aot = False

for name in os.listdir(_cur_dir):
    if name.split(".")[-1] == "so":
        _aot = True
        break

new_kernel = False

if _aot:
    try:
        if not new_kernel:
            from nn import _kernels
            gtx_kernels = None
        else:
            file_ext = ".so"
            gtx_kernel_lib = [
                _ for _ in os.listdir(_cur_dir) if _.endswith(file_ext)
            ][0]
            lib_abspath = os.path.join(_cur_dir, gtx_kernel_lib)
            torch.ops.load_library(lib_abspath)
    except ImportError as e:
        GtxScreenLogger().error2user(QError.IMPORT_KERNEL, f"{str(e)}")
        sys.exit(1)
    else:
        GtxScreenLogger().info("Loading SuperGate GTX kernels...")

else:
    if os.path.exists(os.path.join(_cur_dir, "kernel")):
        # from .kernel import NN_PATH
        pass
    else:
        NN_PATH = _cur_dir
    try:
        cwd = NN_PATH
        lib_path = os.path.join(cwd, "lib")
        create_work_dir(lib_path)
        # cpu_src_path = os.path.join(cwd, "../../../csrc/cpu")
        source_files = []
        # for name in os.listdir(cpu_src_path):
        #   if name.split(".")[-1] in ["cpp", "cc", "c"]:
        #     source_files.append(os.path.join(cpu_src_path, name))

        extra_include_paths = [
            # os.path.join(cwd, "../../../include/cpu"),
            # os.path.join(cwd, "include")
        ]

        with_cuda = False
        extra_cflags = ""
        # CPU-only mode: CUDA kernel loading removed
        # CPU source files can be added here if needed:
        # cpp_src_path = os.path.join(cwd, "src/cpu")
        # for name in os.listdir(cpp_src_path):
        #     if name.split(".")[-1] in ["cpp", "cc", "c"]:
        #         source_files.append(os.path.join(cpp_src_path, name))

        is_python_module = False if new_kernel else True
        if source_files:
            gtx_kernels = load(
                name="gtx_kernels",
                sources=source_files,
                verbose=False,
                build_directory=lib_path,
                extra_cflags=[extra_cflags],
                extra_include_paths=extra_include_paths,
                with_cuda=with_cuda,
                is_python_module=is_python_module,
            )
        else:
            gtx_kernels = None

    except ImportError as e:
        GtxScreenLogger().error2user(QError.IMPORT_KERNEL, f"{str(e)}")
        sys.exit(1)
    else:
        GtxScreenLogger().info(f"Loading SuperGate GTX kernels...")
