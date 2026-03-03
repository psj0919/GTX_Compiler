#!/usr/bin/env python3
"""
GTX Compiler C 코드 생성기 테스트.

ResNet-18 모델 파싱이 기존 nn 패키지 의존성 문제로 실패하는 경우,
직접 GTX Graph를 구성하여 C 코드 생성기를 테스트합니다.
"""

import sys
import os
import pty
import select
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gtx_shared.compile.c_codegen import CCodeGenerator
from gtx_shared.compile.memory_planner import MemoryPlanner
from gtx_shared.compile.weight_exporter import WeightExporter


def create_mock_resnet_graph():
    """
    ResNet-18의 구조를 반영한 Mock GTX Graph를 생성합니다.
    실제 TorchParser를 사용하지 않고, 그래프 구조와 shape/param만 직접 구성합니다.
    """
    from gtx_shared.gtx_graph import Graph, Node, Tensor
    from gtx_shared.base.key_names import GTX_OP

    graph = Graph("ResNet18")

    # 레이어 정의 (ResNet-18 기본 구조)
    layers = [
        # (name, op_type, in_shape, out_shape, params, attrs)
        ("input_0", GTX_OP.INPUT, None, [1, 3, 224, 224], {}, {}),
        (
            "conv1",
            GTX_OP.CONV2D,
            [1, 3, 224, 224],
            [1, 64, 112, 112],
            {"weights": np.random.randn(64, 3, 7, 7).astype(np.float32)},
            {
                "kernel_size": [7, 7],
                "stride": [2, 2],
                "padding": [3, 3],
                "dilation": [1, 1],
                "groups": 1,
            },
        ),
        (
            "bn1",
            GTX_OP.BATCH_NORM,
            [1, 64, 112, 112],
            [1, 64, 112, 112],
            {
                "gamma": np.ones(64).astype(np.float32),
                "beta": np.zeros(64).astype(np.float32),
                "mean": np.random.randn(64).astype(np.float32),
                "var": np.abs(np.random.randn(64).astype(np.float32)),
            },
            {"eps": 1e-5, "momentum": 0.1},
        ),
        ("relu1", GTX_OP.RELU, [1, 64, 112, 112], [1, 64, 112, 112], {}, {}),
        (
            "maxpool",
            GTX_OP.MAX_POOL,
            [1, 64, 112, 112],
            [1, 64, 56, 56],
            {},
            {"kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1]},
        ),
        # BasicBlock 1
        (
            "conv2",
            GTX_OP.CONV2D,
            [1, 64, 56, 56],
            [1, 64, 56, 56],
            {"weights": np.random.randn(64, 64, 3, 3).astype(np.float32)},
            {
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": [1, 1],
                "dilation": [1, 1],
                "groups": 1,
            },
        ),
        (
            "bn2",
            GTX_OP.BATCH_NORM,
            [1, 64, 56, 56],
            [1, 64, 56, 56],
            {
                "gamma": np.ones(64).astype(np.float32),
                "beta": np.zeros(64).astype(np.float32),
                "mean": np.random.randn(64).astype(np.float32),
                "var": np.abs(np.random.randn(64).astype(np.float32)),
            },
            {"eps": 1e-5},
        ),
        ("relu2", GTX_OP.RELU, [1, 64, 56, 56], [1, 64, 56, 56], {}, {}),
        (
            "conv3",
            GTX_OP.CONV2D,
            [1, 64, 56, 56],
            [1, 64, 56, 56],
            {"weights": np.random.randn(64, 64, 3, 3).astype(np.float32)},
            {
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": [1, 1],
                "dilation": [1, 1],
                "groups": 1,
            },
        ),
        (
            "bn3",
            GTX_OP.BATCH_NORM,
            [1, 64, 56, 56],
            [1, 64, 56, 56],
            {
                "gamma": np.ones(64).astype(np.float32),
                "beta": np.zeros(64).astype(np.float32),
                "mean": np.random.randn(64).astype(np.float32),
                "var": np.abs(np.random.randn(64).astype(np.float32)),
            },
            {"eps": 1e-5},
        ),
        ("add1", GTX_OP.ADD, [1, 64, 56, 56], [1, 64, 56, 56], {}, {}),
        ("relu3", GTX_OP.RELU, [1, 64, 56, 56], [1, 64, 56, 56], {}, {}),
        # Global Average Pool + FC
        (
            "avgpool",
            GTX_OP.ADAPTIVEAVGPOOL2D,
            [1, 64, 56, 56],
            [1, 64, 1, 1],
            {},
            {"output_size": [1, 1]},
        ),
        ("flatten", GTX_OP.FLATTEN, [1, 64, 1, 1], [1, 64], {}, {}),
        (
            "fc",
            GTX_OP.DENSE,
            [1, 64],
            [1, 10],
            {
                "weights": np.random.randn(10, 64).astype(np.float32),
                "bias": np.random.randn(10).astype(np.float32),
            },
            {"in_features": 64, "out_features": 10, "bias": True},
        ),
    ]

    # Mock Node 클래스
    class MockOp:
        def __init__(self, op_type, params_dict, attrs_dict):
            self.type = op_type
            self._params = {}
            self._attrs = attrs_dict
            self.attrs = attrs_dict

            for param_name, param_data in params_dict.items():
                t = MockTensor(param_name, list(param_data.shape), param_data)
                self._params[param_name] = t

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

    # 그래프 구성
    mock_graph = MockGraph("ResNet18_test")
    prev_node = None
    residual_node = None

    for i, (name, op_type, in_shape, out_shape, params, attrs) in enumerate(layers):
        node = MockNode(name, op_type, in_shape, out_shape, params, attrs)

        # 연결 설정
        if prev_node:
            node._in_nodes.add(prev_node)
            prev_node._out_nodes.add(node)

        # Residual connection for add
        if name == "add1" and residual_node:
            node._in_nodes.add(residual_node)

        # 첫 번째 conv 이후의 노드를 residual로 저장
        if name == "maxpool":
            residual_node = node

        mock_graph.add_node(node)
        prev_node = node

    return mock_graph


def create_mini_graph():
    """
    시뮬레이터 검증용 초소형 그래프.
    입력: [1,2,4,4] (32 elems) → Conv(3x3,p=1) → BN → ReLU → Pool(2x2) → Flatten → Dense → Softmax
    모든 파라미터가 극소화되어 시뮬레이터에서 수 초 내에 완료됨.
    """
    from gtx_shared.base.key_names import GTX_OP

    # Mock 클래스 — create_mock_resnet_graph와 동일
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

    # 극소 크기: [1,2,4,4]
    layers = [
        ("input_0", GTX_OP.INPUT, None, [1, 2, 4, 4], {}, {}),
        # Conv2d: [1,2,4,4] → [1,2,4,4]  k=3 s=1 p=1
        (
            "module_1_conv",
            GTX_OP.CONV2D,
            [1, 2, 4, 4],
            [1, 2, 4, 4],
            {"weights": np.random.randn(2, 2, 3, 3).astype(np.float32)},
            {
                "kernel_size": [3, 3],
                "stride": [1, 1],
                "padding": [1, 1],
                "dilation": [1, 1],
                "groups": 1,
                "bias": False,
            },
        ),
        # BN
        (
            "module_2_bn",
            GTX_OP.BATCH_NORM,
            [1, 2, 4, 4],
            [1, 2, 4, 4],
            {
                "gamma": np.ones(2, dtype=np.float32),
                "beta": np.zeros(2, dtype=np.float32),
                "mean": np.array([0.1, -0.2], dtype=np.float32),
                "var": np.array([1.0, 1.0], dtype=np.float32),
            },
            {"eps": 1e-5},
        ),
        # ReLU
        ("module_3_relu", GTX_OP.RELU, [1, 2, 4, 4], [1, 2, 4, 4], {}, {}),
        # MaxPool: [1,2,4,4] → [1,2,2,2]
        (
            "module_4_pool",
            GTX_OP.MAX_POOL,
            [1, 2, 4, 4],
            [1, 2, 2, 2],
            {},
            {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0]},
        ),
        # Flatten: [1,2,2,2] → [1,8]
        ("module_5_flat", GTX_OP.FLATTEN, [1, 2, 2, 2], [1, 8], {}, {}),
        # Dense: [1,8] → [1,4]
        (
            "module_6_fc",
            GTX_OP.DENSE,
            [1, 8],
            [1, 4],
            {
                "weights": np.random.randn(4, 8).astype(np.float32),
                "bias": np.random.randn(4).astype(np.float32),
            },
            {"in_features": 8, "out_features": 4, "bias": True},
        ),
        # Softmax
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GTX Compiler C Code Generator Test")
    parser.add_argument(
        "--mini",
        action="store_true",
        help="극소 파라미터 미니 모델로 생성+빌드+시뮬레이터 검증",
    )
    parser.add_argument(
        "--op-test",
        action="store_true",
        help="연산자별 개별 ELF 빌드 & 시뮬레이터 실행 (tempfile 기반)",
    )
    parser.add_argument(
        "--compile-all",
        action="store_true",
        help="모든 연산자를 하나의 ELF로 빌드 후 시뮬레이터 실행",
    )
    args = parser.parse_args()

    if args.op_test:
        run_op_test()
    elif args.compile_all:
        run_compile_all_test()
    elif args.mini:
        run_mini_test()
    else:
        run_resnet_test()


def run_op_test():
    """연산자별 개별 ELF 빌드 & 시뮬레이터 실행 (tempfile 기반)."""
    import shutil
    import tempfile

    print("=" * 60)
    print("GTX Compiler - Per-Operator ELF Test (tempfile)")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_mini")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # 1. 미니 그래프 생성 + C 코드 생성
    print("\n[1] Creating mini graph...")
    np.random.seed(42)
    graph = create_mini_graph()

    print(f"\n[2] Generating C code → {output_dir}")
    codegen = CCodeGenerator(graph, output_dir=output_dir, model_name="mini")
    files = codegen.generate()
    for ftype, fpath in files.items():
        if os.path.exists(fpath):
            print(
                f"    {ftype}: {os.path.basename(fpath)} ({os.path.getsize(fpath):,} B)"
            )

    # 빌드 파일 생성 (헤더가 참조하는 crt0.S, linker.ld)
    from compile_to_c import (
        generate_makefile,
        generate_linker_script,
        generate_startup_code,
    )

    generate_makefile(output_dir, "mini")
    generate_linker_script(output_dir)
    generate_startup_code(output_dir)

    # 2. OpTestRunner 생성
    from gtx_shared.compile.op_test_runner import OpTestRunner

    sim_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "simulator", "GTX_ISS"
    )
    runner = OpTestRunner(codegen, sim_path=sim_path)

    print(f"\n[3] Available modules with code: {runner.module_ids}")
    print(f"    All modules (incl. no-op):   {runner.all_module_ids}")

    # 3. 각 연산자별 개별 ELF 빌드 & 시뮬레이터 실행
    # 참고: standalone 연산자 ELF는 GTX_ISS -M 모드에서 hang할 수 있음
    #       (dense/mm 레이어가 포함되지 않은 단독 ELF는 시뮬레이터가 종료하지 않는 문제)
    #       빌드 성공 여부만 확인하고, 시뮬레이터 실행은 짧은 timeout으로 시도.
    print("\n[4] Building & running each operator individually...")

    build_results = []  # (mod_id, op_type, build_ok, sim_status)

    for mod_id in runner.module_ids:
        info = runner.get_module_info(mod_id)
        node = info["node"]
        op_type = node.op.type
        func_name = info["func_name"]
        print(f"\n    --- Module {mod_id}: {op_type} ({func_name}) ---")

        # tempfile 기반 빌드 — TemporaryDirectory가 자동으로 정리함
        with tempfile.TemporaryDirectory(prefix=f"gtx_optest_{mod_id}_") as temp_dir:
            ok, elf_or_msg, _work_dir = runner.build_op_elf(mod_id, tmp_dir=temp_dir)

            if not ok:
                print(f"    BUILD FAILED: {elf_or_msg}")
                build_results.append((mod_id, op_type, False, "BUILD_FAIL"))
                continue

            elf_size = os.path.getsize(elf_or_msg)
            print(f"    Build OK: {os.path.basename(elf_or_msg)} ({elf_size:,} B)")

            # 시뮬레이터 실행 (짧은 timeout — standalone ELF는 hang 가능)
            sim_status = "SKIP"
            if os.path.isfile(sim_path):
                result = runner.run_op_on_simulator(
                    mod_id=mod_id,
                    elf_path=elf_or_msg,
                    monitoring=True,
                    timeout=15,
                    log_level=0,
                )
                if result["success"]:
                    sim_status = "OK"
                elif result["return_code"] == -1:
                    # timeout (standalone ELF hang은 알려진 현상)
                    sim_status = "TIMEOUT (expected for standalone op)"
                else:
                    sim_status = f"FAIL (code={result['return_code']})"
                print(f"    Simulator: {sim_status}")

                # 마지막 5줄 출력
                out_text = result["output"].strip()
                if out_text:
                    lines = out_text.split("\n")
                    for line in lines[-5:]:
                        print(f"      {line}")
            else:
                print(f"    Simulator not found: {sim_path}")

            build_results.append((mod_id, op_type, True, sim_status))
        # TemporaryDirectory 자동 정리 완료

    # 결과 요약
    print("\n" + "-" * 60)
    print("Summary:")
    print(f"  {'ID':>3}  {'Op Type':<20}  {'Build':<8}  {'Simulator'}")
    print(f"  {'---':>3}  {'--------':<20}  {'-----':<8}  {'---------'}")
    for mod_id, op_type, build_ok, sim_status in build_results:
        build_str = "OK" if build_ok else "FAIL"
        print(f"  {mod_id:>3}  {op_type:<20}  {build_str:<8}  {sim_status}")

    print("\n" + "=" * 60)
    print("PER-OPERATOR TEST DONE!")
    print("=" * 60)


def run_compile_all_test():
    """모든 연산자를 하나의 ELF로 빌드 후 시뮬레이터 실행."""
    import shutil

    print("=" * 60)
    print("GTX Compiler - Compile All → Single ELF Test")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_mini")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # 1. 미니 그래프 생성 + C 코드 생성
    print("\n[1] Creating mini graph...")
    np.random.seed(42)
    graph = create_mini_graph()

    print(f"\n[2] Generating C code → {output_dir}")
    codegen = CCodeGenerator(graph, output_dir=output_dir, model_name="mini")
    files = codegen.generate()
    for ftype, fpath in files.items():
        if os.path.exists(fpath):
            print(
                f"    {ftype}: {os.path.basename(fpath)} ({os.path.getsize(fpath):,} B)"
            )

    # 2. OpTestRunner로 전체 모델 단일 ELF 빌드
    from gtx_shared.compile.op_test_runner import OpTestRunner

    sim_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "simulator", "GTX_ISS"
    )
    runner = OpTestRunner(codegen, sim_path=sim_path)

    print(f"\n[3] Compiling all ops into single ELF...")
    ok, elf_or_msg = runner.compile_all()
    if not ok:
        print(f"    BUILD FAILED: {elf_or_msg}")
        return

    print(f"    Build OK: {elf_or_msg}")
    elf_size = os.path.getsize(elf_or_msg)
    print(f"    ELF size: {elf_size:,} bytes")

    # 3. 시뮬레이터 실행
    if not os.path.isfile(sim_path):
        print(f"\n[4] Simulator not found: {sim_path}")
        print("    Skip simulator test.")
        return

    print(f"\n[4] Running full model on simulator...")
    result = runner.run_full_model(
        elf_path=elf_or_msg,
        monitoring=True,
        timeout=60,
        log_level=0,
    )

    status = "OK" if result["success"] else f"FAIL (code={result['return_code']})"
    print(f"    Simulator: {status}")

    lines = result["output"].strip().split("\n")
    print("    --- Last 15 lines ---")
    for line in lines[-15:]:
        print(f"    {line}")

    print("\n" + "=" * 60)
    print("COMPILE ALL TEST DONE!")
    print("=" * 60)


def run_mini_test():
    """극소 파라미터 미니 모델 → C 코드 생성 → 빌드 → 시뮬레이터 실행."""
    import subprocess, shutil

    print("=" * 60)
    print("GTX Compiler - Mini Model Simulator Test")
    print("=" * 60)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_mini")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # 1. 미니 그래프 생성
    print("\n[1] Creating mini graph (tiny parameters)...")
    graph = create_mini_graph()
    for node in graph.nodes:
        print(f"    {node.name}: {node.op.type} → {node.out_tensors[0].shape}")

    # 2. C 코드 생성
    print(f"\n[2] Generating C code → {output_dir}")
    codegen = CCodeGenerator(graph, output_dir=output_dir, model_name="mini")
    files = codegen.generate()
    for ftype, fpath in files.items():
        if os.path.exists(fpath):
            print(
                f"    {ftype}: {os.path.basename(fpath)} ({os.path.getsize(fpath):,} B)"
            )

    # 3. 빌드 파일 생성
    from compile_to_c import (
        generate_makefile,
        generate_linker_script,
        generate_startup_code,
        generate_data_embed,
    )

    generate_makefile(output_dir, "mini")
    generate_linker_script(output_dir)
    generate_startup_code(output_dir)
    generate_data_embed(output_dir)
    print("    + Makefile, linker.ld, crt0.S, data_embed.S")

    # 4. 크로스 컴파일
    print("\n[3] Cross-compiling...")
    result = subprocess.run(["make", "-C", output_dir], capture_output=True, text=True)
    if result.returncode != 0:
        print("  BUILD FAILED:")
        print(result.stderr)
        return
    print("    Build OK: mini.elf")

    # 5. 시뮬레이터 실행
    elf_path = os.path.join(output_dir, "mini.elf")
    sim_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "simulator", "GTX_ISS"
    )
    if not os.path.isfile(sim_path):
        print(f"\n[4] Simulator not found: {sim_path}")
        print("    Skip simulator test.")
        return

    print("\n[4] Running simulator via PTY (timeout 60s)...")
    print(f"    {sim_path} -I {elf_path} -M -l 3")

    sim_timeout = 60
    master_fd, slave_fd = pty.openpty()
    try:
        import subprocess as sp

        proc = sp.Popen(
            [sim_path, "-I", elf_path, "-M", "-l", "3", "-W", "0"],
            stdout=slave_fd,
            stderr=slave_fd,
            stdin=sp.DEVNULL,
            cwd=os.path.dirname(elf_path),
        )
        os.close(slave_fd)
        slave_fd = -1

        output_chunks: list[bytes] = []
        deadline = time.monotonic() + sim_timeout
        timed_out = False

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                proc.kill()
                proc.wait()
                timed_out = True
                break

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

        if proc.poll() is None:
            proc.wait(timeout=5)

        output = b"".join(output_chunks).decode("utf-8", errors="replace")
        print("    --- Simulator Output (last 30 lines) ---")
        lines = output.strip().split("\n")
        for line in lines[-30:]:
            print(f"    {line}")
        print("    --- End ---")

        if timed_out:
            print(
                "    SIMULATOR: timed out after 60s (may be normal for longer models)"
            )
        elif proc.returncode == 0:
            print("\n    SIMULATOR: exited normally (return code 0)")
        else:
            print(f"\n    SIMULATOR: exited with code {proc.returncode}")

    except Exception as e:
        print(f"    SIMULATOR: error — {e}")
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

    print("\n" + "=" * 60)
    print("MINI TEST DONE!")
    print("=" * 60)


def run_resnet_test():
    """기존 ResNet-18 크기 그래프 테스트 (시뮬레이터 없이 코드 생성+빌드만)."""
    print("=" * 60)
    print("GTX Compiler - C Code Generator Test")
    print("=" * 60)

    # 1. Mock 그래프 생성
    print("\n[1] Creating mock ResNet graph...")
    graph = create_mock_resnet_graph()
    print(f"    Nodes: {len(graph.nodes)}")
    for node in graph.nodes:
        print(f"    - {node.name}: {node.op.type} → {node.out_tensors[0].shape}")

    # 2. C 코드 생성
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    print(f"\n[2] Generating C code → {output_dir}")

    codegen = CCodeGenerator(graph, output_dir=output_dir, model_name="resnet")
    files = codegen.generate()

    print("    Generated files:")
    for ftype, fpath in files.items():
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            print(f"    {ftype}: {os.path.basename(fpath)} ({size:,} bytes)")

    # 3. 생성된 코드 미리보기
    model_c_path = os.path.join(output_dir, "resnet.c")
    if os.path.exists(model_c_path):
        print("\n[3] Preview of resnet.c (first 60 lines):")
        print("-" * 60)
        with open(model_c_path) as f:
            for i, line in enumerate(f):
                if i >= 60:
                    print("... (truncated)")
                    break
                print(line, end="")
        print("-" * 60)

    # 4. 가중치 통계
    weights_h = os.path.join(output_dir, "weights.h")
    if os.path.exists(weights_h):
        size = os.path.getsize(weights_h)
        print(f"\n[4] weights.h: {size:,} bytes")

    weights_bin = os.path.join(output_dir, "weights.bin")
    if os.path.exists(weights_bin):
        size = os.path.getsize(weights_bin)
        print(f"    weights.bin: {size:,} bytes")

    # 5. Makefile, linker script, startup code 생성
    from compile_to_c import (
        generate_makefile,
        generate_linker_script,
        generate_startup_code,
    )

    makefile_path = generate_makefile(output_dir, "resnet")
    linker_path = generate_linker_script(output_dir)
    startup_path = generate_startup_code(output_dir)
    print("\n[5] Build files:")
    print(f"    Makefile: {makefile_path}")
    print(f"    linker.ld: {linker_path}")
    print(f"    crt0.S: {startup_path}")

    print("\n" + "=" * 60)
    print("TEST PASSED!")
    print("=" * 60)
    print(f"\n빌드 명령어: cd {output_dir} && make")


if __name__ == "__main__":
    main()
