import numpy as np
import torch
import struct


class bfloat16:
    """
    NumPy와 호환되는 bfloat16 구현
    .view() 메서드 지원 및 배열 연산 최적화
    """
    
    def __init__(self, value=0.0):
        if isinstance(value, (int, float)):
            self.bits = self._float32_to_bfloat16(float(value))
        elif isinstance(value, bfloat16):
            self.bits = value.bits
        elif isinstance(value, np.ndarray):
            # NumPy 배열인 경우 각 원소를 변환
            self.bits = np.array([self._float32_to_bfloat16(float(x)) for x in value.flat]).reshape(value.shape)
        else:
            self.bits = int(value)
    
    def _float32_to_bfloat16(self, f32_value):
        """float32를 bfloat16으로 변환"""
        if f32_value == 0.0:
            return 0
        # float32를 바이트로 변환하고 상위 16비트만 사용
        f32_bytes = struct.pack('>f', f32_value)
        bf16_bytes = f32_bytes[:2]
        return struct.unpack('>H', bf16_bytes)[0]
    
    def _bfloat16_to_float32(self, bf16_bits):
        """bfloat16을 float32로 변환"""
        if bf16_bits == 0:
            return 0.0
        # 16비트에 0을 16비트 추가하여 float32 생성
        f32_bytes = struct.pack('>H', bf16_bits) + b'\x00\x00'
        return struct.unpack('>f', f32_bytes)[0]
    
    def to_float(self):
        """float32로 변환"""
        if isinstance(self.bits, np.ndarray):
            return np.array([self._bfloat16_to_float32(x) for x in self.bits.flat]).reshape(self.bits.shape)
        else:
            return self._bfloat16_to_float32(self.bits)
    
    def view(self, dtype):
        """PyTorch의 .view() 메서드와 유사한 기능"""
        if dtype == np.int16:
            if isinstance(self.bits, np.ndarray):
                return self.bits.astype(np.int16)
            else:
                return np.int16(self.bits)
        else:
            raise NotImplementedError(f"view({dtype}) not implemented")
    
    def astype(self, dtype):
        """NumPy의 astype과 유사한 기능"""
        if dtype == np.float32:
            return self.to_float()
        else:
            raise NotImplementedError(f"astype({dtype}) not implemented")
    
    def __float__(self):
        return self.to_float()
    
    @property
    def shape(self):
        if isinstance(self.bits, np.ndarray):
            return self.bits.shape
        else:
            return ()

import numpy as np
import torch
import struct


class bfloat16:
    """
    NumPy와 호환되는 bfloat16 구현
    .view() 메서드 지원 및 배열 연산 최적화
    """
    
    def __init__(self, value=0.0):
        if isinstance(value, (int, float)):
            self.bits = self._float32_to_bfloat16(float(value))
        elif isinstance(value, bfloat16):
            self.bits = value.bits
        elif isinstance(value, np.ndarray):
            # NumPy 배열인 경우 각 원소를 변환
            self.bits = np.array([self._float32_to_bfloat16(float(x)) for x in value.flat]).reshape(value.shape)
        else:
            self.bits = int(value)
    
    def _float32_to_bfloat16(self, f32_value):
        """float32를 bfloat16으로 변환"""
        if f32_value == 0.0:
            return 0
        # float32를 바이트로 변환하고 상위 16비트만 사용
        f32_bytes = struct.pack('>f', f32_value)
        bf16_bytes = f32_bytes[:2]
        return struct.unpack('>H', bf16_bytes)[0]
    
    def _bfloat16_to_float32(self, bf16_bits):
        """bfloat16을 float32로 변환"""
        if bf16_bits == 0:
            return 0.0
        # 16비트에 0을 16비트 추가하여 float32 생성
        f32_bytes = struct.pack('>H', bf16_bits) + b'\x00\x00'
        return struct.unpack('>f', f32_bytes)[0]
    
    def to_float(self):
        """float32로 변환"""
        if isinstance(self.bits, np.ndarray):
            return np.array([self._bfloat16_to_float32(x) for x in self.bits.flat]).reshape(self.bits.shape)
        else:
            return self._bfloat16_to_float32(self.bits)
    
    def view(self, dtype):
        """PyTorch의 .view() 메서드와 유사한 기능"""
        if dtype == np.int16:
            if isinstance(self.bits, np.ndarray):
                return self.bits.astype(np.int16)
            else:
                return np.int16(self.bits)
        else:
            raise NotImplementedError(f"view({dtype}) not implemented")
    
    def astype(self, dtype):
        """NumPy의 astype과 유사한 기능"""
        if dtype == np.float32:
            return self.to_float()
        else:
            raise NotImplementedError(f"astype({dtype}) not implemented")
    
    def __float__(self):
        return self.to_float()
    
    @property
    def shape(self):
        if isinstance(self.bits, np.ndarray):
            return self.bits.shape
        else:
            return ()


def numpy_bfloat16_array(arr):
    """NumPy 배열을 bfloat16 배열로 변환하는 최적화된 함수"""
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    
    # 각 원소를 bfloat16으로 변환
    flat_arr = arr.flatten()
    bf16_bits = np.zeros(len(flat_arr), dtype=np.uint16)
    
    for i, val in enumerate(flat_arr):
        if val == 0.0:
            bf16_bits[i] = 0
        else:
            f32_bytes = struct.pack('>f', float(val))
            bf16_bits[i] = struct.unpack('>H', f32_bytes[:2])[0]
    
    result = bfloat16(0)
    result.bits = bf16_bits.reshape(arr.shape)
    return result


def bfloat16_to_numpy(bf16_obj):
    """bfloat16 객체를 NumPy float32 배열로 변환"""
    if isinstance(bf16_obj.bits, np.ndarray):
        flat_bits = bf16_obj.bits.flatten()
        float_vals = np.zeros(len(flat_bits), dtype=np.float32)
        
        for i, bits in enumerate(flat_bits):
            if bits == 0:
                float_vals[i] = 0.0
            else:
                f32_bytes = struct.pack('>H', bits) + b'\x00\x00'
                float_vals[i] = struct.unpack('>f', f32_bytes)[0]
        
        return float_vals.reshape(bf16_obj.bits.shape)
    else:
        if bf16_obj.bits == 0:
            return 0.0
        f32_bytes = struct.pack('>H', bf16_obj.bits) + b'\x00\x00'
        return struct.unpack('>f', f32_bytes)[0]


def int16_to_bfloat16(int16_arr):
    """int16 배열을 bfloat16으로 변환 (비트 재해석)"""
    result = bfloat16(0)
    result.bits = int16_arr.astype(np.uint16)
    return result

