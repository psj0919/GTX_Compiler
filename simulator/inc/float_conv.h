
#ifndef FLOAT_CONV_H_
#define FLOAT_CONV_H_

#include <cstdint>
#include <cmath>
#include <cstring>
//FP8 -> float32
inline float fp16_to_32(uint16_t h);

inline int8_t fp16_to_i8(uint16_t h) {
    float f = fp16_to_32(h);

    if (std::isnan(f) || std::isinf(f)) return 0; // 
    if (f > 127.0f) return 127;
    if (f < -128.0f) return -128;

    return static_cast<int8_t>(std::round(f));  // 
}

inline uint8_t fp16_to_8(uint16_t h) {
      uint32_t h_sign = (h >> 15) & 0x1;
        uint32_t h_exp  = (h >> 10) & 0x1F;   // 5 bits
        uint32_t h_frac =  h        & 0x03FF; // 10 bits

        uint8_t  sign8  = static_cast<uint8_t>(h_sign << 7);
        uint8_t  exp8, frac8;

        // NaN/Inf
        if (h_exp == 0x1F) {
            if (h_frac) { // NaN
                return sign8 | 0xF8 | 0x01;   // qNaN 최소 분수 유지(관례)
            } else {       // Inf
                return sign8 | 0xF8;
            }
        }

        // 유효 가수 및 FP16 unbiased exponent
        int e16 = (h_exp == 0) ? (-14) : (int(h_exp) - 15);
        // FP8 목표 exponent (biased)
        int new_e = e16 + 7; // bias16=15, bias8=7 → e16 + (7)

        // FP16 가수(정규는 1.xxx = 11비트, 서브노멀은 0.xxx)
        uint32_t sig = (h_exp == 0) ? h_frac : (0x400u | h_frac); // 0x400 = 1<<10 (hidden 1)

        // 정규 범위: 1..14
        if (new_e >= 1 && new_e <= 14) {
            // 11비트(sig)를 1+3비트(=4비트)로 반올림 → 그중 낮은 3비트가 저장 분수
            const uint32_t out_bits = 4;               // 1+frac(3)
            const uint32_t shift = 11 - out_bits;      // 7
            uint32_t main = sig >> shift;              // 4비트 후보 (1.xxx)
            uint32_t round_bit = (sig >> (shift - 1)) & 1u;
            uint32_t sticky    = (shift >= 2) ? (sig & ((1u << (shift - 1)) - 1u)) : 0u;

            // RNE
            if (round_bit && (sticky || (main & 1u))) {
                ++main;
            }

            // 반올림으로 1.111 -> 10.000 이 되면 exponent 승격
            if (main == (1u << out_bits)) { // 16
                ++new_e;
                main = (1u << (out_bits - 1)); // 1.000
                if (new_e >= 0xF) { // overflow to Inf
                    return sign8 | 0xF8;
                }
            }

            exp8  = static_cast<uint8_t>(new_e & 0xF);
            frac8 = static_cast<uint8_t>(main & 0x7u); // 하위 3비트 저장
            return sign8 | (exp8 << 3) | frac8;
        }

        // 언더플로우 측: FP8 서브노멀( exp=0, 값 = frac * 2^{(1-bias)-frac_bits} = frac * 2^{-9} )
        if (new_e <= 0) {
            // total_shift = (11 - frac_bits) + (1 - new_e)
            //  = 7 + (1 - new_e) = 8 - new_e
            uint32_t total_shift = 8u - static_cast<uint32_t>(new_e); // new_e <= 0 → 8.. 커짐

            // 너무 작으면 0
            if (total_shift >= 32u) return sign8;

            uint32_t frac = sig >> total_shift;                 // 3비트 후보
            if (total_shift == 0u) { /* 이 경우는 논리상 오지 않음(new_e<=0) */ }

            // RNE for subnormals: round_bit = bit just below, sticky = remaining low bits
            uint32_t rb_pos = total_shift - 1u;
            uint32_t round_bit = (sig >> rb_pos) & 1u;
            uint32_t sticky = (rb_pos > 0u) ? (sig & ((1u << rb_pos) - 1u)) : 0u;

            if (round_bit && (sticky || (frac & 1u))) {
                ++frac;
                // 서브노멀에서 반올림으로 정규화될 수 있음: 0b1000 (=1.000) → exp=1, frac=0
                if (frac == 0x8u) {
                    // exp=1, frac=0
                    return sign8 | (1u << 3);
                }
            }

        frac8 = static_cast<uint8_t>(frac & 0x7u);
        return sign8 | frac8; // exp=0
        }

    // 오버플로우: 정규쪽에서 이미 처리하지만, 방어적으로
    return sign8 | 0xF8; // Inf
}
inline float int8_to_32(int8_t x) {
    return static_cast<float>(x);
}
inline float fp8_to_32(uint8_t h) {
    uint8_t h_sign = (h & 0x80) >> 7;
    uint8_t h_exp  = (h & 0x78) >> 3;
    uint8_t h_frac = (h & 0x07);

    uint32_t f_sign = h_sign << 31;
    uint32_t f_exp, f_frac;

    if (h_exp == 0) {
        if (h_frac == 0) {
            // zero
            f_exp = 0;
            f_frac = 0;
        } else {
            // subnormal
            float f = h_frac / 8.0f;  // 2^3
            f *= std::pow(2, -6);     // 2^(1 - bias) = 2^-6
            if (h_sign) f = -f;
            return f;
        }
    } else if (h_exp == 0xF) {
        // Inf or NaN
        f_exp = 0xFF << 23;
        f_frac = h_frac << 20;
    } else {
        // normalized
        f_exp = (h_exp + (127 - 7)) << 23;
        f_frac = h_frac << 20;
    }

    uint32_t f_bits = f_sign | f_exp | f_frac;
    float result;
    std::memcpy(&result, &f_bits, sizeof(float));
    return result;
}
// FP16 → float32
inline float fp16_to_32(uint16_t h) {
    uint16_t h_sign = (h & 0x8000) >> 15;
    uint16_t h_exp  = (h & 0x7C00) >> 10;
    uint16_t h_frac = (h & 0x03FF);

    uint32_t f_sign = h_sign << 31;
    uint32_t f_exp, f_frac;

    if (h_exp == 0) {
        if (h_frac == 0) {
            // zero
            f_exp = 0;
            f_frac = 0;
        } else {
            // subnormal
            float f = h_frac / 1024.0f;
            f *= std::pow(2, -14);
            if (h_sign) f = -f;
            return f;
        }
    } else if (h_exp == 0x1F) {
        // Inf or NaN
        f_exp = 0xFF << 23;
        f_frac = h_frac << 13;
    } else {
        // normalized
        f_exp = (h_exp + (127 - 15)) << 23;
        f_frac = h_frac << 13;
    }

    uint32_t f_bits = f_sign | f_exp | f_frac;
    float result;
    std::memcpy(&result, &f_bits, sizeof(float));
    return result;
}



// float32 → FP16
inline uint16_t fp32_to_16(float f) {
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(float));

    uint16_t h_sign = (f_bits >> 16) & 0x8000;
    int32_t  f_exp  = ((f_bits >> 23) & 0xFF) - 127 + 15;
    uint32_t f_frac = f_bits & 0x007FFFFF;

    if (f_exp <= 0) {
        if (f_exp < -10) {
            // FP16 서브노멀로도 표현 불가 → 부호만 남은 0
            return h_sign;
        }

        // FP16 서브노멀 생성 (RNE)
        uint32_t mant = 0x00800000u | f_frac; // 1.xxx (24비트)
        uint32_t shift = 1u - static_cast<uint32_t>(f_exp); // 1..10
        uint32_t total_shift = 13u + shift;                 // 14..23

        // 결과 가수(10비트 후보)
        uint16_t h_frac = static_cast<uint16_t>(mant >> total_shift);

        // RNE: round bit & sticky bit
        uint32_t round_bit = (mant >> (total_shift - 1)) & 1u;
        uint32_t sticky_mask = (1u << (total_shift - 1)) - 1u;
        uint32_t sticky = (mant & sticky_mask) ? 1u : 0u;

        if (round_bit && (sticky || (h_frac & 1))) {
            ++h_frac;
            // 서브노멀에서 올림으로 정상화될 수 있음: 0x400(11비트) → exp=1, frac=0
            if (h_frac & 0x0400u) {
                return h_sign | 0x0400u; // exp=1(=0x0400), frac=0
            }
        }

        return h_sign | h_frac; // exp=0 (서브노멀), frac=10비트
    } else if (f_exp >= 0x1F) {
        // overflow or NaN
        return h_sign | 0x7C00;
    }

    uint16_t h_exp  = (uint16_t)(f_exp << 10);
     // -----------------------
    // RNE (Round to Nearest Even)
    // -----------------------
    uint32_t round_mask = 0x1FFF;        // 
    uint32_t round_bits = f_frac & round_mask;
    uint32_t add = 0;

    if (round_bits > 0x1000) {
        add = 1;
    } else if (round_bits == 0x1000) {
        if ((f_frac >> 13) & 1) { // tie-breaking: if LSB is 1, round up
            add = 1;
        }
    }
 
    uint16_t h_frac = (uint16_t)((f_frac >> 13) + add);

    
    if (h_frac & 0x0400) {
        h_frac = 0;      // mantissa overflow
        h_exp += 0x0400; // exponent + 1
        if (h_exp >= 0x7C00) { // exponent overflow
            h_exp = 0x7C00;
        }
    }

    return h_sign | h_exp | h_frac;
}

inline uint16_t int32_to_f16(int32_t x) {
    float f = static_cast<float>(x); // 안전하게 변환 가능
    return fp32_to_16(f);
}

inline uint16_t int8_to_f16(int8_t x) {
    float f = static_cast<float>(x); // 1. int8 → float32
    return fp32_to_16(f);          // 2. float32 → fp16
}
#endif