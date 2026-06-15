# import imp
import os
import sys
import torch
try:
    from torch.utils.cpp_extension import load, _import_module_from_library
except ImportError:
    load = None
    _import_module_from_library = None
from shared.utils import create_work_dir, ScreenLogger, QError, QWarning
from utils.torch_utils import CmpFlag, compare_torch_version

_cur_dir = os.path.dirname(os.path.realpath(__file__))
_aot = False

# 단일 .so(Nuitka --module) 배포 시 nn/ 가 디스크 디렉토리가 아니므로 listdir 가 실패한다.
# 그 경우 AOT 커널/소스 빌드는 불필요(CPU 경로) → 빈 목록으로 간주하고 kernels=None.
try:
    _entries = os.listdir(_cur_dir)
except (FileNotFoundError, NotADirectoryError):
    _entries = None

# AOT 커널은 _kernels 확장(.so) 로만 판별. Cython 소스보호 배포에선 nn/ 의 모든 모듈이
# .so 라 "임의의 .so" 판별은 오탐(=실재하지 않는 _kernels import 시도) → _kernels 전용으로 좁힌다.
for name in (_entries or []):
    if name.startswith("_kernels") and name.split(".")[-1] == "so":
        _aot = True
        break

new_kernel = False
kernels = None

if _entries is None:
    # nn/ 디렉토리 부재(임베드 .so) — 커널 로드/빌드 생략.
    ScreenLogger().info("Loading SuperGate  kernels... (embedded, none)")
elif _aot:
    try:
        if not new_kernel:
            from nn import _kernels
            kernels = None
        else:
            file_ext = ".so"
            kernel_lib = [
                _ for _ in os.listdir(_cur_dir) if _.endswith(file_ext)
            ][0]
            lib_abspath = os.path.join(_cur_dir, kernel_lib)
            torch.ops.load_library(lib_abspath)
    except ImportError as e:
        ScreenLogger().error2user(QError.IMPORT_KERNEL, f"{str(e)}")
        sys.exit(1)
    else:
        ScreenLogger().info("Loading SuperGate  kernels...")

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
            kernels = load(
                name="kernels",
                sources=source_files,
                verbose=False,
                build_directory=lib_path,
                extra_cflags=[extra_cflags],
                extra_include_paths=extra_include_paths,
                with_cuda=with_cuda,
                is_python_module=is_python_module,
            )
        else:
            kernels = None

    except ImportError as e:
        ScreenLogger().error2user(QError.IMPORT_KERNEL, f"{str(e)}")
        sys.exit(1)
    else:
        ScreenLogger().info(f"Loading SuperGate  kernels...")
