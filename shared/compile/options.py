#
# Copyright 2025 Supergate.cc, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""codegen 동작 옵션.

**환경변수를 읽지 않는다.** 옵션은 `main()` 의 argparse 에서 만들어져
`compile_model` → `generate_ggml_code` → `VispCodeGenerator` 로 명시적으로 흘러간다.

왜 env 가 아니라 인자인가:
- 생성 결과가 **호출부에 다 드러난다.** 셸 환경에 따라 다른 코드가 나오지 않는다.
- 라이브러리로 임포트해 쓰는 쪽(테스트·하네스)이 전역 상태 없이 그대로 호출할 수 있다.
- 재현이 명령줄 한 줄로 끝난다 — 무슨 env 가 걸려 있었는지 되짚을 필요가 없다.
"""

from dataclasses import dataclass, field
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class CodegenOptions:
    """생성 동작을 바꾸는 옵션 묶음. 기본값 = 검증된 동작(100/100 통과 설정)."""

    # --- 최적화 ---------------------------------------------------------
    const_fold: bool = False
    """입력-무관 subgraph 를 numpy 로 접어 GGUF 에 baking.

    **기본 off.** skip 한 노드의 출력이 baked 에 안 들어가는 경우가 있어, 하류가
    그 텐서를 못 찾고 그래프 입력(= 이미지)으로 폴백한다(pvt: 410 노드 skip / 16 텐서만
    bake → residual add 가 can_repeat 로 abort). 정합성이 보장되면 기본값을 되돌린다.
    """

    dce: bool = True
    """출력에서 역방향 도달 못 하는 `tensor <var> = …;` 선언 제거.

    끄면 진단용으로 전체 emit 을 볼 수 있다. 켠 채로 두는 게 정상 — 죽은 코드도
    **실행은 되므로** 거기서 assert 가 나면 살아있는 모델이 죽는다.
    """

    skip_opts: FrozenSet[str] = frozenset()
    """건너뛸 dev-graph 최적화 패스 이름. `{"all"}` 이면 전부 건너뛴다.

    어떤 패스가 그래프를 망가뜨렸는지 이분 탐색할 때 쓴다.
    """

    # --- 출력 형태 ------------------------------------------------------
    dense_out: Optional[bool] = None
    """NMS-free(v10) 후처리 계열에서 dense 예측까지만 출력.

    `None` = 그래프에 top-k/gather 류가 있으면 자동 판단(기존 동작).
    """

    # --- 진단 -----------------------------------------------------------
    trace_shapes: bool = False
    """텐서마다 런타임 ne 를 stderr 로 찍는 줄을 끼운다.

    shape assert 는 **그래프 구축 중**에 터지므로 완성된 그래프를 나중에 훑을 수 없다.
    """

    debug_taps: Optional[str] = None
    """중간 텐서 탭. `"all"` 또는 `"12,45"` 처럼 노드 인덱스 목록."""

    extra_outputs: tuple = field(default_factory=tuple)
    """추가 graph output 으로 등록할 중간 텐서 변수명들(값 덤프용)."""

    # ------------------------------------------------------------------
    @property
    def tap_set(self):
        """`debug_taps` 를 노드 인덱스 집합으로. `"all"`/미지정이면 None."""
        if not self.debug_taps or self.debug_taps == "all":
            return None
        return {int(i) for i in self.debug_taps.split(",") if i.strip().isdigit()}

    def skips(self, pass_name: str) -> bool:
        return "all" in self.skip_opts or pass_name in self.skip_opts


DEFAULT = CodegenOptions()


def add_cli_arguments(ap):
    """`argparse.ArgumentParser` 에 codegen 옵션 플래그를 등록한다."""
    g = ap.add_argument_group("codegen 옵션")
    g.add_argument("--const-fold", action="store_true",
                   help="입력-무관 subgraph 를 접어 GGUF 에 baking (기본 off — 정합성 미보장)")
    g.add_argument("--no-dce", action="store_true",
                   help="죽은 선언 제거를 끈다 (진단용)")
    g.add_argument("--skip-opts", default="",
                   help="건너뛸 dev-graph 최적화 패스(쉼표 구분). 'all' 이면 전부")
    g.add_argument("--dense-out", choices=("auto", "on", "off"), default="auto",
                   help="NMS-free 계열에서 dense 예측까지만 출력 (기본 auto)")
    g.add_argument("--trace-shapes", action="store_true",
                   help="텐서마다 런타임 ne 를 stderr 로 찍는다 (진단용)")
    g.add_argument("--debug-taps", default=None,
                   help="중간 텐서 탭: 'all' 또는 '12,45' (진단용)")
    g.add_argument("--extra-outputs", default="",
                   help="추가 graph output 으로 뽑을 변수명(쉼표 구분, 진단용)")
    return ap


def from_args(args) -> CodegenOptions:
    """argparse 결과 → `CodegenOptions`."""
    dense = {"auto": None, "on": True, "off": False}[getattr(args, "dense_out", "auto")]
    return CodegenOptions(
        const_fold=bool(getattr(args, "const_fold", False)),
        dce=not bool(getattr(args, "no_dce", False)),
        skip_opts=frozenset(s.strip() for s in (getattr(args, "skip_opts", "") or "").split(",")
                            if s.strip()),
        dense_out=dense,
        trace_shapes=bool(getattr(args, "trace_shapes", False)),
        debug_taps=getattr(args, "debug_taps", None),
        extra_outputs=tuple(s.strip() for s in (getattr(args, "extra_outputs", "") or "").split(",")
                            if s.strip()),
    )
