import imp
import os
import sys
import torch
from torch.utils.cpp_extension import load, _import_module_from_library
from gtx_shared.utils import create_work_dir, GtxScreenLogger, QError, QWarning
from utils.torch_utils import CmpFlag, compare_torch_version

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
        GtxScreenLogger().info(f"Loading SuperGate GTX kernels...")

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
        exit()
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
        if "CUDA_HOME" in os.environ:
            GtxScreenLogger().check2user(
                QError.TORCH_VERSION,
                f"CUDA_HOME is set in environment, \
but pytorch installed is CPU version. \
Please install CUDA version pytorch.",
                torch.cuda.is_available(),
            )
            cuda_src_path = os.path.join(cwd, "../../../csrc/cuda")
            for name in os.listdir(cuda_src_path):
                if name.split(".")[-1] in ["cu", "cpp", "cc", "c"]:
                    source_files.append(os.path.join(cuda_src_path, name))

            cpp_src_path = os.path.join(cwd, "src/cuda")
            for name in os.listdir(cpp_src_path):
                if name.split(".")[-1] in ["cpp", "cc", "c"]:
                    source_files.append(os.path.join(cpp_src_path, name))

            extra_include_paths.append(os.path.join(cwd, "../../../include/cuda"))
            with_cuda = None
        else:
            print(
                "CUDA is not available, or CUDA_HOME not found in the environment "
                "so building CPU Only support."
            )
            # 추후 이부분 넣어야 할 듯?
            # cpp_src_path = os.path.join(cwd, "src/cpu")
            # for name in os.listdir(cpp_src_path):
            #     if name.split(".")[-1] in ["cpp", "cc", "c"]:
            #         pass
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
