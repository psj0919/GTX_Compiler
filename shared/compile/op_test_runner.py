# Copyright (C) Supergate - All Rights Reserved
#  Compiler - Per-Operator ELF Build & Simulator Runner
#
# 연산자별 독립 ELF를 tempfile로 빌드하여 ISS에서 실행/검증하는 모듈.
# compile() 이후 모든 연산자를 하나의 ELF로 합치는 기능도 제공.

import os
import pty
import select
import struct
import subprocess
import tempfile
import time
import shutil
import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from shared.compile.memory_planner import (
    DDR_INPUT_BASE,
    DDR_OUTPUT_BASE,
    DDR_WEIGHT_BASE,
    DDR_TEMP_BASE,
)

logger = logging.getLogger(__name__)

# 프로젝트 루트 (nn/include, nn/src 등이 있는 디렉토리)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 기본 시뮬레이터 경로
_DEFAULT_SIM_PATH = os.path.join(_PROJECT_ROOT, "simulator", "ISS")

# RISC-V 크로스 컴파일러
_CC = "riscv64-unknown-elf-gcc"


def _make_linker_ld() -> str:
    """최소 linker.ld 내용 반환."""
    return """\
OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY {
    IMEM (rx)  : ORIGIN = 0x00000000, LENGTH = 1M
    DMEM (rw)  : ORIGIN = 0x00100000, LENGTH = 1M
    L2SPM (rw) : ORIGIN = 0x10000000, LENGTH = 16M
    DDR (rw)   : ORIGIN = 0x80000000, LENGTH = 2048M
}
SECTIONS {
    .text : { KEEP(*(.text.init)) *(.text*) } > IMEM
    .rodata : { *(.rodata*) } > IMEM
    .data : { *(.data*) *(.sdata*) } > DMEM
    .bss (NOLOAD) : {
        __bss_start = .; *(.bss*) *(.sbss*) *(COMMON) __bss_end = .;
    } > DMEM
    .stack (NOLOAD) : {
        . = ALIGN(16); __stack_bottom = .;
        . += 0x10000; __stack_top = .;
    } > DMEM
    __global_pointer$ = ADDR(.data) + 0x800;
}
"""


def _make_crt0_S() -> str:
    """최소 crt0.S 내용 반환."""
    return r"""
    .section .text.init, "ax", @progbits
    .global _start
    .type   _start, @function
_start:
    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop
    la      sp, __stack_top
    andi    sp, sp, -16
    la      t0, __bss_start
    la      t1, __bss_end
    bgeu    t0, t1, .Lbss_done
.Lbss_loop:
    sd      zero, 0(t0)
    addi    t0, t0, 8
    bltu    t0, t1, .Lbss_loop
.Lbss_done:
    li      a0, 0
    li      a1, 0
    call    main
.Lhalt_loop:
    j       .Lhalt_loop
    .size   _start, . - _start
"""


def _get_intrinsic_sources() -> List[str]:
    """intrinsic C 소스 파일 경로 목록 반환."""
    src_dir = os.path.join(_PROJECT_ROOT, "nn", "src", "")
    return [
        os.path.join(src_dir, f)
        for f in [
            "intrin_level1.c",
            "intrin_level2.c",
            "intrin_level3.c",
            "utils.c",
        ]
        if os.path.isfile(os.path.join(src_dir, f))
    ]


def _get_include_dirs() -> List[str]:
    """컴파일에 필요한 include 경로 목록."""
    nn_inc = os.path.join(_PROJECT_ROOT, "nn", "include")
    inc = os.path.join(_PROJECT_ROOT, "nn", "include", "")
    return [nn_inc, inc]


def _compile_c_to_elf(
    c_path: str,
    elf_path: str,
    work_dir: str,
    extra_include_dirs: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    C 소스를 RISC-V ELF로 크로스 컴파일.

    Args:
        c_path: 소스 .c 파일 경로
        elf_path: 출력 .elf 경로
        work_dir: crt0.S, linker.ld가 있는 작업 디렉토리
        extra_include_dirs: 추가 include 경로

    Returns:
        (success, message)
    """
    crt0_path = os.path.join(work_dir, "crt0.S")
    linker_path = os.path.join(work_dir, "linker.ld")

    cmd = [_CC, "-O2", "-march=rv64gc", "-mabi=lp64d"]

    # Include 경로
    for d in _get_include_dirs():
        cmd.extend(["-I", d])
    if extra_include_dirs:
        for d in extra_include_dirs:
            cmd.extend(["-I", d])

    # 출력
    cmd.extend(["-o", elf_path])

    # 소스 파일: crt0.S + 모듈 C + intrinsic 소스
    cmd.append(crt0_path)
    cmd.append(c_path)
    cmd.extend(_get_intrinsic_sources())

    # 링커 플래그
    cmd.extend(["-T", linker_path, "-nostdlib", "-static"])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, cwd=work_dir
        )
        if result.returncode != 0:
            return False, f"Compilation failed:\n{result.stderr}"
        return True, "OK"
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out (60s)"
    except FileNotFoundError:
        return False, f"Cross compiler not found: {_CC}"


def _run_simulator(
    elf_path: str,
    sim_path: str = _DEFAULT_SIM_PATH,
    load_files: Optional[List[Tuple[str, int]]] = None,
    dump_addr: Optional[int] = None,
    dump_size: Optional[int] = None,
    dump_file: Optional[str] = None,
    monitoring: bool = False,
    timeout: int = 120,
    log_level: int = 0,
) -> Tuple[int, str]:
    """
    ISS 시뮬레이터 실행 (PTY 기반).

    SystemC 시뮬레이터는 stdout이 PIPE나 파일로 리다이렉트되면 hang한다.
    이를 해결하기 위해 pty.openpty()로 pseudo-terminal을 생성하여
    시뮬레이터의 stdout/stderr를 연결한다. 시뮬레이터는 isatty()=true로 판단하고,
    master fd에서 출력을 읽어 캡처한다.

    Args:
        elf_path: 실행할 ELF 경로
        sim_path: ISS 경로
        load_files: [(filepath, address), ...] 메모리 로드 목록
        dump_addr: 메모리 덤프 시작 주소
        dump_size: 메모리 덤프 크기 (바이트)
        dump_file: 메모리 덤프 출력 파일명
        monitoring: -M 모드 (빠르지만 데이터 corrupt 가능)
        timeout: 실행 제한 시간 (초)
        log_level: 로그 레벨 (0~3)

    Returns:
        (return_code, combined_stdout_stderr)
    """
    if not os.path.isfile(sim_path):
        return -1, f"Simulator not found: {sim_path}"

    cmd = [sim_path, "-I", elf_path]

    if monitoring:
        cmd.append("-M")

    if log_level > 0:
        cmd.extend(["-l", str(log_level)])

    # 메모리 로드
    # 주의: ISS의 -S와 -L을 함께 사용하면 stoi 에러 발생 (simulator bug).
    # -L만 사용하면 기본 주소 0x100000에 로드된다.
    # DDR 주소(0x80000000 등)에 로드할 수 없으므로,
    # -M (monitoring) 모드에서는 데이터 로드 없이 코드 무결성만 검증한다.
    if load_files and not monitoring:
        # 비-monitoring 모드에서만 데이터 로드 시도 (현재 -S 버그로 인해 사실상 사용 불가)
        for filepath, addr in load_files:
            # -S와 -L 함께 사용 시 stoi crash — -L만 사용 (기본 주소 0x100000)
            cmd.extend(["-L", filepath])

    # 메모리 덤프
    # ISS -B/-E는 hex 형식으로 파싱됨 (strtoul base 16).
    # 출력 형식은 32바이트 단위 역순 hex 텍스트.
    if dump_file:
        cmd.extend(["-T", dump_file])
        if dump_addr is not None:
            cmd.extend(["-B", f"{dump_addr:X}"])
        if dump_size is not None:
            cmd.extend(["-E", f"{dump_size:X}"])

    cmd.extend(["-W", "0"])

    work_dir = os.path.dirname(elf_path)

    # PTY 기반 실행: pseudo-terminal을 생성하여 시뮬레이터가 TTY에 연결된
    # 것으로 인식하게 만든다. 이렇게 하면 SystemC의 isatty() 체크를 통과한다.
    master_fd, slave_fd = pty.openpty()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=slave_fd,
            stderr=slave_fd,
            stdin=subprocess.DEVNULL,
            cwd=work_dir,
        )
        # slave는 자식 프로세스에 전달되었으므로 부모에서 닫는다.
        os.close(slave_fd)
        slave_fd = -1

        # master fd에서 출력을 비동기로 읽는다.
        output_chunks: list[bytes] = []
        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # 타임아웃 — 프로세스 강제 종료
                proc.kill()
                proc.wait()
                output = b"".join(output_chunks).decode("utf-8", errors="replace")
                return -1, f"Simulator timed out after {timeout}s\n{output}"

            # select로 master fd에서 읽을 데이터가 있는지 확인
            ready, _, _ = select.select([master_fd], [], [], min(remaining, 0.5))

            if ready:
                try:
                    chunk = os.read(master_fd, 65536)
                    if chunk:
                        output_chunks.append(chunk)
                    else:
                        # EOF — 프로세스가 종료됨
                        break
                except OSError:
                    # slave 쪽이 닫히면 read에서 EIO 발생 가능
                    break

            # 프로세스가 이미 종료되었는지 확인
            retcode = proc.poll()
            if retcode is not None:
                # 프로세스 종료 후 남은 출력 읽기
                while True:
                    rd, _, _ = select.select([master_fd], [], [], 0.1)
                    if not rd:
                        break
                    try:
                        chunk = os.read(master_fd, 65536)
                        if chunk:
                            output_chunks.append(chunk)
                        else:
                            break
                    except OSError:
                        break
                break

        # 프로세스가 아직 실행 중이면 종료 대기
        if proc.poll() is None:
            proc.wait(timeout=5)
        retcode = proc.returncode

        output = b"".join(output_chunks).decode("utf-8", errors="replace")
        return retcode, output

    except Exception as e:
        return -1, f"Simulator execution error: {e}"
    finally:
        if slave_fd >= 0:
            try:
                os.close(slave_fd)
            except OSError:
                pass
        try:
            os.close(master_fd)
        except OSError:
            pass


def _save_fp16_binary(data: np.ndarray, filepath: str):
    """numpy array를 fp16 바이너리로 저장."""
    fp16 = data.astype(np.float16).flatten()
    fp16.tofile(filepath)


def _load_fp16_from_hex_dump(filepath: str, num_elements: int) -> Optional[np.ndarray]:
    """
    ISS의 hex 덤프 파일을 fp16 배열로 파싱.

    형식: 각 줄이 "ADDR: HEXDATA" 또는 순수 HEXDATA.
    ISS는 32바이트 단위로 바이트 순서를 역순으로 덤프하므로,
    파싱 후 32바이트 블록 단위로 reverse해야 올바른 데이터를 얻음.
    """
    if not os.path.isfile(filepath):
        return None

    raw_bytes = bytearray()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # "ADDR: DATA" 형식에서 DATA 부분만 추출
            if ":" in line:
                line = line.split(":", 1)[1].strip()
            # hex 문자열을 바이트로 변환
            line = line.replace(" ", "")
            try:
                raw_bytes.extend(bytes.fromhex(line))
            except ValueError:
                continue

    if len(raw_bytes) < num_elements * 2:
        logger.warning(
            f"Hex dump has {len(raw_bytes)} bytes, need {num_elements * 2} for {num_elements} fp16 elements"
        )
        if len(raw_bytes) == 0:
            return None

    # ISS는 32바이트 단위로 바이트 순서를 역순으로 덤프함
    # 32바이트 블록 단위로 reverse하여 올바른 순서로 복원
    corrected = bytearray()
    chunk_size = 32
    for i in range(0, len(raw_bytes), chunk_size):
        chunk = raw_bytes[i : i + chunk_size]
        corrected.extend(reversed(chunk))

    arr = np.frombuffer(bytes(corrected[: num_elements * 2]), dtype=np.float16)
    return arr


def _load_fp16_from_binary_dump(
    filepath: str, num_elements: int
) -> Optional[np.ndarray]:
    """바이너리 덤프 파일을 fp16 배열로 읽기."""
    if not os.path.isfile(filepath):
        return None
    data = np.fromfile(filepath, dtype=np.float16, count=num_elements)
    return data if len(data) == num_elements else None


class OpTestRunner:
    """
    연산자별 독립 ELF 빌드 & ISS 실행 관리자.

    사용법:
        codegen = CCodeGenerator(graph, output_dir=..., model_name=...)
        files = codegen.generate()

        runner = OpTestRunner(codegen, sim_path="simulator/ISS")

        # 연산자별 개별 빌드 & 실행
        for mod_id in runner.module_ids:
            result = runner.build_and_run_op(mod_id)

        # 전체 모델 단일 ELF 빌드
        elf_path = runner.compile_all(output_dir="output_mini")
    """

    def __init__(
        self,
        codegen,  # CCodeGenerator 인스턴스 (generate() 호출 후)
        sim_path: str = _DEFAULT_SIM_PATH,
    ):
        self._codegen = codegen
        self._sim_path = sim_path

    @property
    def module_ids(self) -> List[int]:
        """사용 가능한 모듈 ID 목록 (코드가 있는 것만)."""
        return sorted(
            mid
            for mid, entry in self._codegen._module_map.items()
            if entry["func_code"] is not None
        )

    @property
    def all_module_ids(self) -> List[int]:
        """모든 모듈 ID (no-op 포함)."""
        return sorted(self._codegen._module_map.keys())

    def get_module_info(self, mod_id: int) -> Optional[Dict[str, Any]]:
        """모듈 메타데이터 반환."""
        return self._codegen._module_map.get(mod_id)

    # ================================================================
    # 연산자별 개별 ELF 빌드 & 실행
    # ================================================================

    def build_op_elf(
        self, mod_id: int, tmp_dir: Optional[str] = None
    ) -> Tuple[bool, str, str]:
        """
        단일 연산자의 standalone ELF를 tempfile 디렉토리에 빌드.

        Args:
            mod_id: 모듈 ID
            tmp_dir: 지정하면 해당 디렉토리 사용, None이면 tempfile 생성

        Returns:
            (success, elf_path_or_error_msg, work_dir)
            work_dir는 caller가 cleanup 책임 (tmp_dir가 None인 경우)
        """
        entry = self._codegen._module_map.get(mod_id)
        if entry is None:
            return False, f"Module {mod_id} not found", ""
        if entry["func_code"] is None:
            return False, f"Module {mod_id} is a no-op (no code)", ""

        # 작업 디렉토리 결정
        if tmp_dir is None:
            work_dir = tempfile.mkdtemp(prefix=f"op_{mod_id}_")
        else:
            work_dir = tmp_dir
            os.makedirs(work_dir, exist_ok=True)

        # 필요한 파일 생성
        # 1) linker.ld
        with open(os.path.join(work_dir, "linker.ld"), "w") as f:
            f.write(_make_linker_ld())

        # 2) crt0.S
        with open(os.path.join(work_dir, "crt0.S"), "w") as f:
            f.write(_make_crt0_S())

        # 3) 헤더/가중치 파일 복사 (codegen output_dir에서)
        src_dir = self._codegen.output_dir
        for hfile in [f"{self._codegen.model_name}.h", "weight_map.h"]:
            src = os.path.join(src_dir, hfile)
            if os.path.isfile(src):
                shutil.copy2(src, work_dir)

        # 4) 모듈 C 소스 생성 (per-module standalone 형태)
        c_path = os.path.join(work_dir, f"module_{mod_id}.c")
        c_code = self._make_standalone_c(mod_id, entry)
        with open(c_path, "w") as f:
            f.write(c_code)

        # 5) 컴파일
        elf_path = os.path.join(work_dir, f"module_{mod_id}.elf")
        ok, msg = _compile_c_to_elf(
            c_path,
            elf_path,
            work_dir,
            extra_include_dirs=[src_dir],
        )
        if not ok:
            return False, msg, work_dir

        return True, elf_path, work_dir

    def run_op_on_simulator(
        self,
        mod_id: int,
        elf_path: str,
        input_data: Optional[np.ndarray] = None,
        weights_bin_path: Optional[str] = None,
        output_num_elements: Optional[int] = None,
        monitoring: bool = False,
        timeout: int = 120,
        log_level: int = 0,
    ) -> Dict[str, Any]:
        """
        시뮬레이터에서 단일 연산자 ELF 실행.

        Args:
            mod_id: 모듈 ID
            elf_path: 컴파일된 ELF 경로
            input_data: 입력 데이터 (numpy array, fp16 변환됨)
            weights_bin_path: 가중치 바이너리 경로 (없으면 codegen output_dir에서 가져옴)
            output_num_elements: 출력 원소 수 (메모리 덤프 크기 결정)
            monitoring: -M 모드
            timeout: 제한 시간
            log_level: 0~3

        Returns:
            {
                "return_code": int,
                "output": str,  # 시뮬레이터 stdout+stderr
                "output_data": Optional[np.ndarray],  # fp16 출력 결과
                "success": bool,
            }
        """
        work_dir = os.path.dirname(elf_path)
        load_files = []

        # 입력 데이터 저장 & 로드 설정
        if input_data is not None:
            input_bin = os.path.join(work_dir, "input.bin")
            _save_fp16_binary(input_data, input_bin)
            load_files.append((input_bin, DDR_INPUT_BASE))

        # 가중치 로드
        if weights_bin_path is None:
            weights_bin_path = os.path.join(self._codegen.output_dir, "weights.bin")
        if os.path.isfile(weights_bin_path):
            load_files.append((weights_bin_path, DDR_WEIGHT_BASE))

        # 출력 덤프 설정
        dump_file = None
        dump_addr = None
        dump_size = None
        if output_num_elements is not None and output_num_elements > 0:
            dump_file = os.path.join(work_dir, "output_dump.hex")
            dump_addr = DDR_OUTPUT_BASE
            dump_size = output_num_elements * 2  # fp16 = 2 bytes

        # 실행
        retcode, sim_output = _run_simulator(
            elf_path=elf_path,
            sim_path=self._sim_path,
            load_files=load_files if load_files else None,
            dump_addr=dump_addr,
            dump_size=dump_size,
            dump_file=dump_file,
            monitoring=monitoring,
            timeout=timeout,
            log_level=log_level,
        )

        result = {
            "return_code": retcode,
            "output": sim_output,
            "output_data": None,
            "success": retcode == 0,
        }

        # 출력 파싱
        if dump_file and os.path.isfile(dump_file):
            out_data = _load_fp16_from_hex_dump(dump_file, output_num_elements)
            if out_data is None:
                out_data = _load_fp16_from_binary_dump(dump_file, output_num_elements)
            result["output_data"] = out_data

        return result

    def build_and_run_op(
        self,
        mod_id: int,
        input_data: Optional[np.ndarray] = None,
        output_num_elements: Optional[int] = None,
        monitoring: bool = True,
        timeout: int = 120,
        log_level: int = 0,
        cleanup: bool = True,
    ) -> Dict[str, Any]:
        """
        단일 연산자: tempfile로 빌드 → 시뮬레이터 실행 → 결과 반환.

        편의 메서드: build_op_elf + run_op_on_simulator를 한 번에 수행.
        """
        ok, elf_or_msg, work_dir = self.build_op_elf(mod_id)
        if not ok:
            return {
                "mod_id": mod_id,
                "success": False,
                "error": elf_or_msg,
                "build_ok": False,
            }

        result = self.run_op_on_simulator(
            mod_id=mod_id,
            elf_path=elf_or_msg,
            input_data=input_data,
            output_num_elements=output_num_elements,
            monitoring=monitoring,
            timeout=timeout,
            log_level=log_level,
        )
        result["mod_id"] = mod_id
        result["build_ok"] = True
        result["elf_path"] = elf_or_msg
        result["work_dir"] = work_dir

        if cleanup and work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

        return result

    # ================================================================
    # 전체 모델 단일 ELF 빌드
    # ================================================================

    def compile_all(self, output_dir: Optional[str] = None) -> Tuple[bool, str]:
        """
        CCodeGenerator.generate() 이후 모든 연산자를 하나의 ELF로 빌드.

        codegen의 output_dir에 이미 생성된 .c, .h, weight_map.h,
        crt0.S, linker.ld, Makefile이 있다고 가정.
        없으면 자동 생성.

        Args:
            output_dir: 출력 디렉토리 (None이면 codegen의 output_dir 사용)

        Returns:
            (success, elf_path_or_error_msg)
        """
        if output_dir is None:
            output_dir = self._codegen.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model_name = self._codegen.model_name

        # 빌드 파일이 없으면 생성
        linker_path = os.path.join(output_dir, "linker.ld")
        if not os.path.isfile(linker_path):
            with open(linker_path, "w") as f:
                f.write(_make_linker_ld())

        crt0_path = os.path.join(output_dir, "crt0.S")
        if not os.path.isfile(crt0_path):
            with open(crt0_path, "w") as f:
                f.write(_make_crt0_S())

        # 소스 파일 확인
        c_path = os.path.join(output_dir, f"{model_name}.c")
        if not os.path.isfile(c_path):
            return (
                False,
                f"Source file not found: {c_path} (run CCodeGenerator.generate() first)",
            )

        elf_path = os.path.join(output_dir, f"{model_name}.elf")
        ok, msg = _compile_c_to_elf(
            c_path,
            elf_path,
            output_dir,
            extra_include_dirs=[output_dir],
        )
        if not ok:
            return False, msg

        return True, elf_path

    def run_full_model(
        self,
        elf_path: str,
        input_data: Optional[np.ndarray] = None,
        output_num_elements: Optional[int] = None,
        monitoring: bool = False,
        timeout: int = 120,
        log_level: int = 0,
    ) -> Dict[str, Any]:
        """
        전체 모델 ELF를 시뮬레이터에서 실행.
        """
        work_dir = os.path.dirname(elf_path)
        load_files = []

        if input_data is not None:
            input_bin = os.path.join(work_dir, "input.bin")
            _save_fp16_binary(input_data, input_bin)
            load_files.append((input_bin, DDR_INPUT_BASE))

        weights_bin = os.path.join(self._codegen.output_dir, "weights.bin")
        if os.path.isfile(weights_bin):
            load_files.append((weights_bin, DDR_WEIGHT_BASE))

        dump_file = None
        dump_addr = None
        dump_size = None
        if output_num_elements is not None and output_num_elements > 0:
            dump_file = os.path.join(work_dir, "output_dump.hex")
            # 전체 모델의 최종 출력 주소 추적
            dump_addr = self._get_final_output_ddr_addr()
            dump_size = output_num_elements * 2

        retcode, sim_output = _run_simulator(
            elf_path=elf_path,
            sim_path=self._sim_path,
            load_files=load_files if load_files else None,
            dump_addr=dump_addr,
            dump_size=dump_size,
            dump_file=dump_file,
            monitoring=monitoring,
            timeout=timeout,
            log_level=log_level,
        )

        result = {
            "return_code": retcode,
            "output": sim_output,
            "output_data": None,
            "success": retcode == 0,
        }

        if dump_file and os.path.isfile(dump_file):
            out_data = _load_fp16_from_hex_dump(dump_file, output_num_elements)
            if out_data is None:
                out_data = _load_fp16_from_binary_dump(dump_file, output_num_elements)
            result["output_data"] = out_data

        return result

    # ================================================================
    # 전체 연산자 순회 실행
    # ================================================================

    def build_and_run_all_ops(
        self,
        monitoring: bool = True,
        timeout: int = 120,
        log_level: int = 0,
        cleanup: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        모든 연산자를 개별적으로 빌드 & 시뮬레이터 실행.

        Returns:
            모듈별 결과 리스트
        """
        results = []
        for mod_id in self.module_ids:
            logger.info(f"Building and running module {mod_id}...")
            result = self.build_and_run_op(
                mod_id=mod_id,
                monitoring=monitoring,
                timeout=timeout,
                log_level=log_level,
                cleanup=cleanup,
            )
            results.append(result)
            status = "OK" if result.get("success") else "FAIL"
            logger.info(f"  Module {mod_id}: {status}")
        return results

    # ================================================================
    # 내부 헬퍼
    # ================================================================

    def _make_standalone_c(self, mod_id: int, entry: Dict) -> str:
        """단일 연산자용 standalone C 소스 생성."""
        from shared.base.key_names import OP

        func_code = entry["func_code"]
        func_name = entry["func_name"]
        node = entry["node"]

        lines = []
        lines.append("// Auto-generated by OpTestRunner — standalone operator test")
        lines.append(f"// Module: {mod_id}, Node: {node.name}, Op: {node.op.type}")
        lines.append("")
        lines.append(f'#include "{self._codegen.model_name}.h"')
        lines.append('#include "weight_map.h"')
        lines.append("")

        if func_code:
            lines.append(func_code)
            lines.append("")

        # main(): split → op(DDR_INPUT_BASE, DDR_OUTPUT_BASE) → join → halt
        lines.append("int main() {")
        lines.append("    __wrspr(NEST_SELECT, 0, 0, 0xFFFFFFFFFFFFFFFF);")
        lines.append("    __split();")

        if func_name:
            op_type = node.op.type
            if op_type in (OP.ADD, OP.MULTIPLY):
                lines.append(
                    f"    {func_name}(DDR_INPUT_BASE, DDR_INPUT_BASE, DDR_OUTPUT_BASE);"
                )
            else:
                lines.append(f"    {func_name}(DDR_INPUT_BASE, DDR_OUTPUT_BASE);")
        else:
            lines.append(f"    // module_{mod_id}: {node.name} — no-op")

        lines.append("    __join();")
        lines.append("    __halt();")
        lines.append("    return 0;")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _get_final_output_ddr_addr(self) -> int:
        """전체 모델의 최종 출력 DDR 주소를 추적."""
        # 마지막 모듈의 출력 DDR 주소 찾기
        last_mod_id = max(self._codegen._module_map.keys())
        last_node = self._codegen._module_map[last_mod_id]["node"]

        if last_node.name in self._codegen._ddr_buffers:
            return self._codegen._ddr_buffers[last_node.name]

        # Fallback: no-op 모듈이면 입력 추적
        for mid in sorted(self._codegen._module_map.keys(), reverse=True):
            node = self._codegen._module_map[mid]["node"]
            if node.name in self._codegen._ddr_buffers:
                return self._codegen._ddr_buffers[node.name]

        return DDR_OUTPUT_BASE
