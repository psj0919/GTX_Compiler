"""
GTX Compiler 수치 검증 테스트.

미니 모델을 사용하여:
  1) PyTorch fp16 추론 참조값 계산
  2) C 코드 생성 + 빌드 + 시뮬레이터 무결성 검증
  3) ELF 내 weights/input 임베딩 검증
  4) 생성된 C 코드의 연산 시퀀스 정확성 검증

시뮬레이터 제약사항:
  - 비-monitoring 모드: DMA/credit 동기화 문제로 hang (시뮬레이터 버그)
  - -M (monitoring) 모드: 코드 무결성만 검증, DMA 실행 안 됨
  - 따라서 현재는 output 수치 비교가 불가능 — 시뮬레이터 수정 후 활성화 예정
"""

import os
import sys
import shutil
import subprocess
import tempfile
import pty
import select
import time

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gtx_shared.compile.c_codegen import CCodeGenerator
from gtx_shared.compile.memory_planner import (
    DDR_INPUT_BASE,
    DDR_OUTPUT_BASE,
    DDR_WEIGHT_BASE,
    DDR_TEMP_BASE,
)
from compile_to_c import (
    generate_makefile,
    generate_linker_script,
    generate_startup_code,
    generate_data_embed,
)

# ============================================================
# 고정 시드 & 미니 모델 정의
# ============================================================
SEED = 42
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_PATH = os.path.join(PROJECT_ROOT, "simulator", "GTX_ISS")

# 미니 모델 구조:
# Input[1,2,4,4] → Conv2d(k3,p1) → BN → ReLU → MaxPool(2) → Flatten → Dense(8→4) → Softmax
MINI_INPUT_SHAPE = (1, 2, 4, 4)
MINI_OUTPUT_SHAPE = (1, 4)


class MiniModel(nn.Module):
    """미니 모델 PyTorch 구현 (fp16 참조값 계산용)."""

    def __init__(self, conv_w, bn_gamma, bn_beta, bn_mean, bn_var, fc_w, fc_b):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(8, 4)

        # 가중치 설정
        with torch.no_grad():
            self.conv.weight.copy_(torch.from_numpy(conv_w))
            self.bn.weight.copy_(torch.from_numpy(bn_gamma))
            self.bn.bias.copy_(torch.from_numpy(bn_beta))
            self.bn.running_mean.copy_(torch.from_numpy(bn_mean))
            self.bn.running_var.copy_(torch.from_numpy(bn_var))
            self.fc.weight.copy_(torch.from_numpy(fc_w))
            self.fc.bias.copy_(torch.from_numpy(fc_b))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.softmax(x, dim=-1)
        return x


def _make_weights(seed=SEED):
    """고정 시드로 미니 모델 가중치 생성."""
    rng = np.random.RandomState(seed)
    conv_w = rng.randn(2, 2, 3, 3).astype(np.float32)
    bn_gamma = np.ones(2, dtype=np.float32)
    bn_beta = np.zeros(2, dtype=np.float32)
    bn_mean = np.array([0.1, -0.2], dtype=np.float32)
    bn_var = np.array([1.0, 1.0], dtype=np.float32)
    fc_w = rng.randn(4, 8).astype(np.float32)
    fc_b = rng.randn(4).astype(np.float32)
    return conv_w, bn_gamma, bn_beta, bn_mean, bn_var, fc_w, fc_b


def _make_input(seed=SEED):
    """고정 시드로 입력 텐서 생성."""
    rng = np.random.RandomState(seed + 1)  # 가중치와 다른 시드
    return rng.randn(*MINI_INPUT_SHAPE).astype(np.float32)


def _create_mini_graph(seed=SEED):
    """고정 시드로 미니 GTX Graph 생성."""
    from gtx_shared.base.key_names import GTX_OP

    conv_w, bn_gamma, bn_beta, bn_mean, bn_var, fc_w, fc_b = _make_weights(seed)

    class MockOp:
        def __init__(self, op_type, params_dict, attrs_dict):
            self.type = op_type
            self._params = {}
            self._attrs = attrs_dict
            self.attrs = attrs_dict
            for pn, pd in params_dict.items():
                self._params[pn] = MockTensor(pn, list(pd.shape), pd)

        @property
        def params(self):
            return self._params

        def get_attr(self, name):
            return self._attrs.get(name)

        class ParamName:
            WEIGHTS = "weights"
            BIAS = "bias"
            GAMMA = "gamma"
            BETA = "beta"
            MEAN = "mean"
            VAR = "var"

    class MockTensor:
        def __init__(self, name, shape, data=None):
            self.name = name
            self.shape = shape
            self.data = data
            self.ndim = len(shape) if shape else 0
            self.uses = []

    class MockNode:
        def __init__(self, name, op_type, in_shape, out_shape, params, attrs):
            self.name = name
            self.op = MockOp(op_type, params, attrs)
            self.in_tensors = [MockTensor(f"{name}_in", in_shape)] if in_shape else []
            self.out_tensors = [MockTensor(f"{name}_out", out_shape)]
            self._in_nodes = set()
            self._out_nodes = set()

        @property
        def in_nodes(self):
            return self._in_nodes

        @property
        def out_nodes(self):
            return self._out_nodes

    class MockGraph:
        def __init__(self, name):
            self.name = name
            self._nodes = []

        @property
        def nodes(self):
            return self._nodes

        def add_node(self, node):
            self._nodes.append(node)

    layers = [
        ("input_0", GTX_OP.INPUT, None, [1, 2, 4, 4], {}, {}),
        (
            "module_1_conv",
            GTX_OP.CONV2D,
            [1, 2, 4, 4],
            [1, 2, 4, 4],
            {"weights": conv_w},
            {
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": [1, 1],
                "dilation": [1, 1],
                "groups": 1,
                "bias": False,
            },
        ),
        (
            "module_2_bn",
            GTX_OP.BATCH_NORM,
            [1, 2, 4, 4],
            [1, 2, 4, 4],
            {
                "gamma": bn_gamma,
                "beta": bn_beta,
                "mean": bn_mean,
                "var": bn_var,
            },
            {"eps": 1e-5},
        ),
        ("module_3_relu", GTX_OP.RELU, [1, 2, 4, 4], [1, 2, 4, 4], {}, {}),
        (
            "module_4_pool",
            GTX_OP.MAX_POOL,
            [1, 2, 4, 4],
            [1, 2, 2, 2],
            {},
            {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0]},
        ),
        ("module_5_flat", GTX_OP.FLATTEN, [1, 2, 2, 2], [1, 8], {}, {}),
        (
            "module_6_fc",
            GTX_OP.DENSE,
            [1, 8],
            [1, 4],
            {"weights": fc_w, "bias": fc_b},
            {"in_features": 8, "out_features": 4, "bias": True},
        ),
        ("module_7_sm", GTX_OP.SOFTMAX, [1, 4], [1, 4], {}, {}),
    ]

    g = MockGraph("MiniTest")
    prev = None
    for name, op_type, in_shape, out_shape, params, attrs in layers:
        node = MockNode(name, op_type, in_shape, out_shape, params, attrs)
        if prev:
            node._in_nodes.add(prev)
            prev._out_nodes.add(node)
        g.add_node(node)
        prev = node
    return g


def _run_simulator_pty(cmd, cwd, timeout=60):
    """PTY 기반 시뮬레이터 실행. (return_code, output_str)."""
    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=slave_fd,
            stderr=slave_fd,
            stdin=subprocess.DEVNULL,
            cwd=cwd,
        )
        os.close(slave_fd)
        slave_fd = -1

        output_chunks = []
        deadline = time.monotonic() + timeout

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                proc.kill()
                proc.wait()
                return -1, "TIMEOUT"

            ready, _, _ = select.select([master_fd], [], [], min(remaining, 0.5))
            if ready:
                try:
                    chunk = os.read(master_fd, 65536)
                    if chunk:
                        output_chunks.append(chunk)
                    else:
                        break
                except OSError:
                    break

            retcode = proc.poll()
            if retcode is not None:
                # Drain remaining output
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
                output = b"".join(output_chunks).decode("utf-8", errors="replace")
                return retcode, output

        # Process may have ended by the time we exit the loop
        retcode = proc.wait()
        output = b"".join(output_chunks).decode("utf-8", errors="replace")
        return retcode, output
    finally:
        if slave_fd >= 0:
            os.close(slave_fd)
        os.close(master_fd)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="session")
def mini_build_dir():
    """세션 전체에서 공유되는 미니 모델 빌드 디렉토리."""
    build_dir = os.path.join(PROJECT_ROOT, "output_test")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    # 1. GTX Graph 생성
    graph = _create_mini_graph(SEED)

    # 2. C 코드 생성
    codegen = CCodeGenerator(graph, output_dir=build_dir, model_name="mini")
    codegen.generate()

    # 3. 빌드 파일 생성
    generate_makefile(build_dir, "mini")
    generate_linker_script(build_dir)
    generate_startup_code(build_dir)

    # 4. 입력 데이터 생성 + ELF 임베딩
    input_data = _make_input(SEED)
    input_data.astype(np.float16).tofile(os.path.join(build_dir, "input.bin"))
    generate_data_embed(build_dir, input_bin="input.bin")

    # 5. 크로스 컴파일
    result = subprocess.run(["make", "-C", build_dir], capture_output=True, text=True)
    assert result.returncode == 0, f"Build failed:\n{result.stderr}"

    yield build_dir

    # Cleanup (optional — keep for debugging)
    # shutil.rmtree(build_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def pytorch_reference():
    """PyTorch fp16 추론 참조값."""
    weights = _make_weights(SEED)
    conv_w, bn_gamma, bn_beta, bn_mean, bn_var, fc_w, fc_b = weights

    model = MiniModel(conv_w, bn_gamma, bn_beta, bn_mean, bn_var, fc_w, fc_b)
    model.eval()
    model.half()

    input_data = _make_input(SEED)
    input_tensor = torch.from_numpy(input_data).half()

    with torch.no_grad():
        output = model(input_tensor)

    # 각 레이어별 중간값도 계산
    with torch.no_grad():
        x = input_tensor
        conv_out = model.conv(x)
        bn_out = model.bn(conv_out)
        relu_out = model.relu(bn_out)
        pool_out = model.pool(relu_out)
        flat_out = pool_out.flatten(1)
        fc_out = model.fc(flat_out)
        softmax_out = F.softmax(fc_out, dim=-1)

    return {
        "input": input_tensor.numpy(),
        "conv_out": conv_out.numpy(),
        "bn_out": bn_out.numpy(),
        "relu_out": relu_out.numpy(),
        "pool_out": pool_out.numpy(),
        "flat_out": flat_out.numpy(),
        "fc_out": fc_out.numpy(),
        "softmax_out": softmax_out.numpy(),
        "final_output": output.numpy(),
    }


# ============================================================
# Tests
# ============================================================


class TestCodeGeneration:
    """C 코드 생성 정확성 테스트."""

    def test_files_generated(self, mini_build_dir):
        """필수 파일들이 모두 생성되었는지 확인."""
        required_files = [
            "mini.h",
            "mini.c",
            "weights.h",
            "weight_map.h",
            "weights.bin",
            "Makefile",
            "linker.ld",
            "crt0.S",
            "data_embed.S",
        ]
        for f in required_files:
            path = os.path.join(mini_build_dir, f)
            assert os.path.isfile(path), f"Missing file: {f}"
            assert os.path.getsize(path) > 0, f"Empty file: {f}"

    def test_elf_built(self, mini_build_dir):
        """ELF 파일이 정상 빌드되었는지 확인."""
        elf_path = os.path.join(mini_build_dir, "mini.elf")
        assert os.path.isfile(elf_path), "mini.elf not found"
        assert os.path.getsize(elf_path) > 1000, "mini.elf too small"

    def test_header_defines(self, mini_build_dir):
        """헤더 파일에 필수 #define이 포함되어 있는지 확인."""
        header_path = os.path.join(mini_build_dir, "mini.h")
        with open(header_path) as f:
            content = f.read()

        required_defines = [
            "#define DDR_INPUT_BASE",
            "#define DDR_OUTPUT_BASE",
            "#define DDR_WEIGHT_BASE",
            "#define NEST_ID",
            "#define SPU_ID",
            "#define CREDIT_SPU",
            "#define CREDIT_NEST",
            "#define BANK_A",
            "#define BANK_B",
            "#define BANK_C",
            "#define BANK_R",
        ]
        for d in required_defines:
            assert d in content, f"Missing define: {d}"

    def test_nest_spu_configurable(self, mini_build_dir):
        """NEST_ID/SPU_ID가 설정 가능하고 기본값이 0인지 확인."""
        header_path = os.path.join(mini_build_dir, "mini.h")
        with open(header_path) as f:
            content = f.read()

        assert "#define NEST_ID     0" in content
        assert "#define SPU_ID      0" in content
        assert "#define CREDIT_SPU  0x1" in content
        assert "#define CREDIT_NEST 0x1" in content

    def test_c_code_uses_macros(self, mini_build_dir):
        """생성된 C 코드가 하드코딩된 0 대신 매크로를 사용하는지 확인."""
        c_path = os.path.join(mini_build_dir, "mini.c")
        with open(c_path) as f:
            content = f.read()

        # 매크로 사용 확인
        assert "__start_plan(NEST_ID)" in content
        assert "__end_plan(NEST_ID)" in content
        assert "__start_thread(SPU_ID)" in content
        assert "__end_thread(SPU_ID)" in content
        assert "__wrspr(NEST_SELECT, 0, NEST_ID," in content

        # 하드코딩된 0이 없는지 확인
        assert "__start_plan(0)" not in content
        assert "__end_plan(0)" not in content
        assert "__start_thread(0)" not in content
        assert "__end_thread(0)" not in content

    def test_inference_function_exists(self, mini_build_dir):
        """추론 함수가 올바르게 생성되었는지 확인."""
        c_path = os.path.join(mini_build_dir, "mini.c")
        with open(c_path) as f:
            content = f.read()

        assert "void mini_inference(" in content
        assert "int main()" in content
        assert "__split()" in content
        assert "__join()" in content
        assert "__halt()" in content

    def test_all_layers_present(self, mini_build_dir):
        """모든 레이어가 C 코드에 포함되어 있는지 확인."""
        c_path = os.path.join(mini_build_dir, "mini.c")
        with open(c_path) as f:
            content = f.read()

        expected_layers = [
            "module_1_conv2d",
            "module_2_batchnorm",
            "module_3_relu",
            "module_4_maxpool",
            "module_6_dense",
            "module_7_softmax",
        ]
        for layer in expected_layers:
            assert layer in content, f"Missing layer function: {layer}"

    def test_custom_nest_spu_ids(self):
        """nest_id=2, spu_id=1 설정 시 올바른 매크로 생성 확인."""
        graph = _create_mini_graph(SEED)
        with tempfile.TemporaryDirectory() as tmpdir:
            codegen = CCodeGenerator(
                graph,
                output_dir=tmpdir,
                model_name="test",
                nest_id=2,
                spu_id=1,
            )
            codegen.generate()

            with open(os.path.join(tmpdir, "test.h")) as f:
                header = f.read()

            assert "#define NEST_ID     2" in header
            assert "#define SPU_ID      1" in header
            assert "#define CREDIT_SPU  0x2" in header  # 1 << 1
            assert "#define CREDIT_NEST 0x4" in header  # 1 << 2


class TestELFEmbedding:
    """ELF 파일에 weights/input 데이터가 올바르게 임베딩되었는지 검증."""

    def test_elf_sections_present(self, mini_build_dir):
        """ELF에 .ddr_weights, .ddr_input 섹션이 존재하는지 확인."""
        elf_path = os.path.join(mini_build_dir, "mini.elf")
        result = subprocess.run(
            ["riscv64-unknown-elf-objdump", "-h", elf_path],
            capture_output=True,
            text=True,
        )
        assert ".ddr_weights" in result.stdout, ".ddr_weights section missing"
        assert ".ddr_input" in result.stdout, ".ddr_input section missing"

    def test_weights_at_correct_address(self, mini_build_dir):
        """weights 섹션이 DDR_WEIGHT_BASE(0xA0000000)에 배치되었는지 확인."""
        elf_path = os.path.join(mini_build_dir, "mini.elf")
        result = subprocess.run(
            ["riscv64-unknown-elf-objdump", "-h", elf_path],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.split("\n"):
            if ".ddr_weights" in line:
                # VMA is the 4th field
                parts = line.split()
                vma = int(parts[3], 16)
                assert vma == DDR_WEIGHT_BASE, (
                    f"Weights at 0x{vma:X}, expected 0x{DDR_WEIGHT_BASE:X}"
                )
                break
        else:
            pytest.fail(".ddr_weights section not found in objdump output")

    def test_input_at_correct_address(self, mini_build_dir):
        """input 섹션이 DDR_INPUT_BASE(0x80000000)에 배치되었는지 확인."""
        elf_path = os.path.join(mini_build_dir, "mini.elf")
        result = subprocess.run(
            ["riscv64-unknown-elf-objdump", "-h", elf_path],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.split("\n"):
            if ".ddr_input" in line:
                parts = line.split()
                vma = int(parts[3], 16)
                assert vma == DDR_INPUT_BASE, (
                    f"Input at 0x{vma:X}, expected 0x{DDR_INPUT_BASE:X}"
                )
                break
        else:
            pytest.fail(".ddr_input section not found in objdump output")

    def test_weights_size_matches(self, mini_build_dir):
        """ELF의 weights 섹션 크기가 weights.bin과 일치하는지 확인."""
        weights_bin = os.path.join(mini_build_dir, "weights.bin")
        expected_size = os.path.getsize(weights_bin)

        elf_path = os.path.join(mini_build_dir, "mini.elf")
        result = subprocess.run(
            ["riscv64-unknown-elf-objdump", "-h", elf_path],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.split("\n"):
            if ".ddr_weights" in line:
                parts = line.split()
                size = int(parts[2], 16)
                assert size == expected_size, (
                    f"Section size {size} != weights.bin size {expected_size}"
                )
                break

    @pytest.mark.skipif(not os.path.isfile(SIM_PATH), reason="Simulator not found")
    def test_weights_readable_from_simulator(self, mini_build_dir):
        """시뮬레이터가 ELF에서 weights를 올바르게 로드하는지 메모리 덤프로 확인."""
        elf_path = os.path.join(mini_build_dir, "mini.elf")
        dump_file = os.path.join(mini_build_dir, "weight_verify.hex")

        if os.path.exists(dump_file):
            os.remove(dump_file)

        # weights.bin 크기
        weights_bin = os.path.join(mini_build_dir, "weights.bin")
        w_size = os.path.getsize(weights_bin)

        cmd = [
            SIM_PATH,
            "-I",
            elf_path,
            "-M",
            "-T",
            dump_file,
            "-B",
            f"{DDR_WEIGHT_BASE:X}",
            "-E",
            f"{w_size:X}",
            "-W",
            "0",
        ]

        retcode, output = _run_simulator_pty(cmd, mini_build_dir, timeout=60)

        assert os.path.isfile(dump_file), "Dump file not created"
        assert os.path.getsize(dump_file) > 0, "Dump file is empty"

        # 덤프 데이터 파싱 (32바이트 역순)
        with open(dump_file) as f:
            hex_data = f.read().strip()
        dump_bytes = bytearray(bytes.fromhex(hex_data))
        corrected = bytearray()
        for i in range(0, len(dump_bytes), 32):
            chunk = dump_bytes[i : i + 32]
            corrected.extend(reversed(chunk))

        # weights.bin과 비교
        expected = np.fromfile(weights_bin, dtype=np.uint8)
        actual = np.frombuffer(bytes(corrected[: len(expected)]), dtype=np.uint8)
        np.testing.assert_array_equal(
            actual, expected, err_msg="Weight data in ELF doesn't match weights.bin"
        )


class TestSimulatorIntegrity:
    """시뮬레이터 코드 무결성 검증 (-M 모드)."""

    @pytest.mark.skipif(not os.path.isfile(SIM_PATH), reason="Simulator not found")
    def test_monitoring_mode_completes(self, mini_build_dir):
        """-M 모드에서 시뮬레이터가 정상 종료하는지 확인."""
        elf_path = os.path.join(mini_build_dir, "mini.elf")
        cmd = [SIM_PATH, "-I", elf_path, "-M", "-W", "0"]

        retcode, output = _run_simulator_pty(cmd, mini_build_dir, timeout=60)

        # return code가 0 또는 None(정상 종료)
        assert retcode is not None and retcode != -1, (
            f"Simulator timeout or error: retcode={retcode}"
        )

    @pytest.mark.skipif(not os.path.isfile(SIM_PATH), reason="Simulator not found")
    def test_cycle_count_reasonable(self, mini_build_dir):
        """실행 사이클 수가 합리적인 범위인지 확인."""
        elf_path = os.path.join(mini_build_dir, "mini.elf")
        cmd = [SIM_PATH, "-I", elf_path, "-M", "-l", "3", "-W", "0"]

        retcode, output = _run_simulator_pty(cmd, mini_build_dir, timeout=60)

        # 사이클 수 추출
        import re

        match = re.search(r"Total CPU cycles\s*:\s*(\d+)", output)
        assert match is not None, "Could not find cycle count in output"

        cycles = int(match.group(1))
        assert 100 < cycles < 100000, (
            f"Unexpected cycle count: {cycles} (expected 100-100000)"
        )

    @pytest.mark.skipif(not os.path.isfile(SIM_PATH), reason="Simulator not found")
    def test_no_fatal_errors(self, mini_build_dir):
        """시뮬레이터 출력에 치명적 에러가 없는지 확인."""
        elf_path = os.path.join(mini_build_dir, "mini.elf")
        cmd = [SIM_PATH, "-I", elf_path, "-M", "-W", "0"]

        retcode, output = _run_simulator_pty(cmd, mini_build_dir, timeout=60)

        # "COMMAND not finished (can't escape wjoin)" 은 -M 모드에서 정상적으로 발생
        # 진짜 치명적 에러만 체크
        fatal_patterns = [
            "Segmentation fault",
            "SIGSEGV",
            "Aborted",
            "core dumped",
            "illegal instruction",
        ]
        for pattern in fatal_patterns:
            assert pattern.lower() not in output.lower(), (
                f"Fatal error detected: {pattern}"
            )


class TestPyTorchReference:
    """PyTorch fp16 참조값 검증."""

    def test_output_shape(self, pytorch_reference):
        """출력 shape이 올바른지 확인."""
        output = pytorch_reference["final_output"]
        assert output.shape == MINI_OUTPUT_SHAPE, (
            f"Output shape {output.shape} != expected {MINI_OUTPUT_SHAPE}"
        )

    def test_softmax_sums_to_one(self, pytorch_reference):
        """Softmax 출력이 1에 합산되는지 확인."""
        output = pytorch_reference["final_output"]
        total = output.sum()
        np.testing.assert_allclose(
            float(total),
            1.0,
            atol=0.01,
            err_msg="Softmax output doesn't sum to ~1.0",
        )

    def test_all_probabilities_positive(self, pytorch_reference):
        """모든 확률값이 양수인지 확인."""
        output = pytorch_reference["final_output"]
        assert np.all(output >= 0), "Negative probability in softmax output"

    def test_intermediate_shapes(self, pytorch_reference):
        """중간 레이어 shape이 올바른지 확인."""
        expected = {
            "conv_out": (1, 2, 4, 4),
            "bn_out": (1, 2, 4, 4),
            "relu_out": (1, 2, 4, 4),
            "pool_out": (1, 2, 2, 2),
            "flat_out": (1, 8),
            "fc_out": (1, 4),
            "softmax_out": (1, 4),
        }
        for name, shape in expected.items():
            actual = pytorch_reference[name].shape
            assert actual == shape, f"{name}: shape {actual} != expected {shape}"

    def test_relu_non_negative(self, pytorch_reference):
        """ReLU 출력이 모두 0 이상인지 확인."""
        relu_out = pytorch_reference["relu_out"]
        assert np.all(relu_out >= 0), "ReLU output has negative values"

    def test_reference_values_logged(self, pytorch_reference):
        """참조값 기록 (향후 시뮬레이터 비교용)."""
        output = pytorch_reference["final_output"]
        # 이 값들은 시뮬레이터가 수정되면 비교에 사용됨
        print(f"\n  PyTorch fp16 reference output: {output.flatten()}")
        print(f"  Sum: {output.sum():.4f}")
        print(f"  Argmax: {output.argmax()}")


class TestPerModuleBuilds:
    """개별 모듈 빌드 검증."""

    def test_module_files_generated(self, mini_build_dir):
        """개별 모듈 C 파일들이 생성되었는지 확인."""
        modules_dir = os.path.join(mini_build_dir, "modules")
        assert os.path.isdir(modules_dir), "modules/ directory not found"

        # 최소한 conv, bn, relu, pool, dense, softmax 모듈이 있어야 함
        c_files = [f for f in os.listdir(modules_dir) if f.endswith(".c")]
        assert len(c_files) >= 6, f"Only {len(c_files)} module files (expected ≥6)"

    def test_module_code_uses_macros(self, mini_build_dir):
        """개별 모듈 C 파일에서도 매크로가 사용되는지 확인."""
        modules_dir = os.path.join(mini_build_dir, "modules")
        for fname in os.listdir(modules_dir):
            if not fname.endswith(".c"):
                continue
            with open(os.path.join(modules_dir, fname)) as f:
                content = f.read()

            if "__start_plan" in content:
                assert "__start_plan(NEST_ID)" in content, (
                    f"{fname}: uses __start_plan but not with NEST_ID"
                )
            if "__wrspr" in content:
                assert "NEST_ID" in content, (
                    f"{fname}: uses __wrspr but NEST_ID not referenced"
                )
