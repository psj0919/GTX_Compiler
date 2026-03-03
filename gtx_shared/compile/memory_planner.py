# Copyright (C) Supergate - All Rights Reserved
# GTX Compiler - SPM Memory Planner
# L1SPM 메모리 할당 및 DMA 스케줄링을 담당하는 모듈

"""
GTX 하드웨어 메모리 계층:
  DDR (External) ↔ L2SPM (Shared) ↔ L1SPM (4 banks: A, B, C, R)

L1SPM Bank 기본 주소 (from intrin_level2.h __init_spm_addr):
  A: 0x00000  (입력 데이터)
  B: 0x20000  (가중치)
  C: 0x30000  (출력 데이터)
  R: 0x50000  (임시/중간 결과)

각 bank 주소는 24비트 (최대 16MB 주소 공간)
fp16 기준: 1개 원소 = 2바이트
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# L1SPM Bank 기본 설정
L1SPM_BANK_A_BASE = 0x00000  # 입력
L1SPM_BANK_B_BASE = 0x20000  # 가중치
L1SPM_BANK_C_BASE = 0x30000  # 출력
L1SPM_BANK_R_BASE = 0x50000  # 임시

# Bank 크기 (기본값, 바이트 단위)
L1SPM_BANK_A_SIZE = 0x20000  # 128KB
L1SPM_BANK_B_SIZE = 0x10000  # 64KB
L1SPM_BANK_C_SIZE = 0x20000  # 128KB
L1SPM_BANK_R_SIZE = 0x10000  # 64KB

# L2 SPM 설정 (NEST 공유 메모리)
L2SPM_BASE = 0x10000000  # L2 SPM 시작 주소
L2SPM_SIZE = 0x01000000  # 16MB

# L2 SPM 영역 분할 (오프셋 기준)
L2_INPUT_OFFSET = 0x000000  # 입력 데이터 영역
L2_WEIGHT_OFFSET = 0x400000  # 가중치 영역
L2_OUTPUT_OFFSET = 0x800000  # 출력/결과 영역

# DDR 기본 주소
DDR_INPUT_BASE = 0x80000000
DDR_OUTPUT_BASE = 0x90000000
DDR_WEIGHT_BASE = 0xA0000000
DDR_TEMP_BASE = 0xB0000000

# 데이터 타입 크기
DTYPE_SIZE = {
    "fp16": 2,
    "fp32": 4,
    "int8": 1,
    "int16": 2,
}


@dataclass
class MemoryRegion:
    """메모리 영역 정보"""

    name: str
    base_addr: int
    size: int  # 바이트 단위
    dtype: str = "fp16"

    @property
    def end_addr(self):
        return self.base_addr + self.size

    @property
    def num_elements(self):
        return self.size // DTYPE_SIZE.get(self.dtype, 2)


@dataclass
class TileConfig:
    """타일링 설정"""

    tile_h: int  # 타일 높이
    tile_w: int  # 타일 너비
    tile_c: int  # 타일 채널 수
    num_tiles_h: int  # H 방향 타일 수
    num_tiles_w: int  # W 방향 타일 수
    num_tiles_c: int  # C 방향 타일 수


@dataclass
class DMATransfer:
    """DMA 전송 정보"""

    src_addr: str  # C 코드에서의 주소 표현 (변수명 또는 상수)
    dst_addr: str
    length: int  # 바이트 단위 전송 길이
    height: int  # 2D 전송 높이
    read_stride: int  # 소스 stride
    write_stride: int  # 목적지 stride
    direction: str  # 'load' (DDR→L1) 또는 'store' (L1→DDR)
    use_credit: bool = True


class MemoryPlanner:
    """
    GTX L1SPM 메모리 할당 및 타일링 계획을 수립하는 클래스.

    각 레이어별로:
    1. 필요한 입력/출력/가중치 크기를 계산
    2. L1SPM에 맞는 타일 크기를 결정
    3. DMA 전송 스케줄을 생성
    """

    def __init__(self, dtype="fp16"):
        self.dtype = dtype
        self.elem_size = DTYPE_SIZE.get(dtype, 2)

        # DDR 메모리 할당 추적
        self._ddr_weight_offset = 0
        self._ddr_temp_offset = 0

        # 레이어별 DDR 주소 매핑
        self.layer_input_addrs: Dict[str, int] = {}
        self.layer_output_addrs: Dict[str, int] = {}
        self.weight_addrs: Dict[str, int] = {}

    def calc_tensor_size(self, shape: List[int]) -> int:
        """텐서 크기를 바이트 단위로 계산"""
        num_elems = 1
        for s in shape:
            num_elems *= s
        return num_elems * self.elem_size

    def alloc_ddr_weight(self, name: str, size: int) -> int:
        """DDR에 가중치 영역 할당"""
        addr = DDR_WEIGHT_BASE + self._ddr_weight_offset
        self.weight_addrs[name] = addr
        self._ddr_weight_offset += size
        # 64바이트 정렬
        self._ddr_weight_offset = (self._ddr_weight_offset + 63) & ~63
        return addr

    def alloc_ddr_temp(self, name: str, size: int) -> int:
        """DDR에 임시 버퍼 할당"""
        addr = DDR_TEMP_BASE + self._ddr_temp_offset
        self._ddr_temp_offset += size
        self._ddr_temp_offset = (self._ddr_temp_offset + 63) & ~63
        return addr

    def plan_conv2d_tiling(
        self,
        in_shape: List[int],  # [N, C, H, W]
        weight_shape: List[int],  # [OC, IC, KH, KW]
        stride: List[int],
        padding: List[int],
    ) -> TileConfig:
        """
        Conv2d 타일링 계획.
        L1SPM 크기 제한에 맞춰 입력/가중치/출력 타일 크기를 결정.
        """
        _, ic, ih, iw = in_shape
        oc, _, kh, kw = weight_shape
        sh, sw = stride[0], stride[1]
        ph, pw = padding[0], padding[1]

        oh = (ih + 2 * ph - kh) // sh + 1
        ow = (iw + 2 * pw - kw) // sw + 1

        # 가중치 크기: OC * IC * KH * KW * elem_size
        weight_size_per_oc = ic * kh * kw * self.elem_size

        # Bank B에 맞는 output channel 타일 크기 결정
        max_oc_per_tile = L1SPM_BANK_B_SIZE // weight_size_per_oc
        max_oc_per_tile = max(1, min(max_oc_per_tile, oc))

        # 입력 타일: Bank A에 맞는 spatial 타일 크기
        # 입력 타일은 패딩 포함 + 커널 오버랩 고려
        input_row_size = iw * ic * self.elem_size
        max_tile_h = L1SPM_BANK_A_SIZE // input_row_size
        max_tile_h = max(kh, min(max_tile_h, ih + 2 * ph))

        # 출력 타일 높이
        out_tile_h = (max_tile_h - kh) // sh + 1
        out_tile_h = max(1, out_tile_h)

        # 타일 수 계산
        num_tiles_h = math.ceil(oh / out_tile_h)
        num_tiles_c = math.ceil(oc / max_oc_per_tile)

        return TileConfig(
            tile_h=max_tile_h,
            tile_w=iw + 2 * pw,  # 전체 폭 사용
            tile_c=max_oc_per_tile,
            num_tiles_h=num_tiles_h,
            num_tiles_w=1,
            num_tiles_c=num_tiles_c,
        )

    def plan_elementwise_tiling(self, shape: List[int]) -> TileConfig:
        """
        Elementwise 연산 (relu, add, etc.) 타일링 계획.
        가능하면 전체를 한번에 처리, 아니면 채널 단위로 타일링.
        """
        total_size = self.calc_tensor_size(shape)

        if total_size <= L1SPM_BANK_A_SIZE:
            # 전체를 한번에 처리 가능
            return TileConfig(
                tile_h=shape[2] if len(shape) > 2 else 1,
                tile_w=shape[3] if len(shape) > 3 else shape[-1],
                tile_c=shape[1] if len(shape) > 1 else 1,
                num_tiles_h=1,
                num_tiles_w=1,
                num_tiles_c=1,
            )

        # 채널 단위 타일링
        if len(shape) >= 4:
            c = shape[1]
            per_channel_size = self.calc_tensor_size([1, 1, shape[2], shape[3]])
            channels_per_tile = L1SPM_BANK_A_SIZE // per_channel_size
            channels_per_tile = max(1, min(channels_per_tile, c))

            return TileConfig(
                tile_h=shape[2],
                tile_w=shape[3],
                tile_c=channels_per_tile,
                num_tiles_h=1,
                num_tiles_w=1,
                num_tiles_c=math.ceil(c / channels_per_tile),
            )

        # Fallback: 단순 분할
        total_elems = 1
        for s in shape:
            total_elems *= s
        elems_per_tile = L1SPM_BANK_A_SIZE // self.elem_size
        num_tiles = math.ceil(total_elems / elems_per_tile)

        return TileConfig(
            tile_h=1,
            tile_w=min(total_elems, elems_per_tile),
            tile_c=1,
            num_tiles_h=1,
            num_tiles_w=num_tiles,
            num_tiles_c=1,
        )

    def get_spm_bank_addrs(self) -> Dict[str, int]:
        """기본 SPM bank 주소 반환"""
        return {
            "A": L1SPM_BANK_A_BASE,
            "B": L1SPM_BANK_B_BASE,
            "C": L1SPM_BANK_C_BASE,
            "R": L1SPM_BANK_R_BASE,
        }
