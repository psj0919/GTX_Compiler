#ifndef F_EXTENSION__H
#define F_EXTENSION__H

#include <cmath>
#include <cfenv>
#include "systemc"

#include "extension_base.h"
#include "Registers.h"
#include "Instrcycle_info.h"

namespace riscv_tlm {

    typedef enum{
        OP_F_FMADD_S,
        OP_F_FMSUB_S,
        OP_F_FNMSUB_S,
        OP_F_FNMADD_S,
        OP_F_FADD_S,
        OP_F_FSUB_S,
        OP_F_FMUL_S,
        OP_F_FDIV_S,
        OP_F_FSQRT_S,
        OP_F_FSGNJ_S,
        OP_F_FSGNJN_S,
        OP_F_FSGNJX_S,
        OP_F_FMIN_S,
        OP_F_FMAX_S,
        OP_F_FCVT_W_S,
        OP_F_FCVT_WU_S,
        OP_F_FMV_X_W,
        OP_F_FEQ_S,
        OP_F_FLT_S,
        OP_F_FLE_S,
        OP_F_FCLASS_S,
        OP_F_FCVT_S_W,
        OP_F_FCVT_S_WU,
        OP_F_FMV_W_X,
        OP_F_FMADD_D,
        OP_F_FMSUB_D,
        OP_F_FNMSUB_D,
        OP_F_FNMADD_D,
        OP_F_FADD_D,
        OP_F_FSUB_D,
        OP_F_FMUL_D,
        OP_F_FDIV_D,
        OP_F_FSQRT_D,
        OP_F_FSGNJ_D,
        OP_F_FSGNJN_D,
        OP_F_FSGNJX_D,
        OP_F_FMIN_D,
        OP_F_FMAX_D,
        OP_F_FCVT_S_D,
        OP_F_FCVT_D_S,
        OP_F_FEQ_D,
        OP_F_FLT_D,
        OP_F_FLE_D,
        OP_F_FCLASS_D,
        OP_F_FCVT_W_D,
        OP_F_FCVT_WU_D,
        OP_F_FCVT_D_W,
        OP_F_FCVT_D_WU,
        OP_F_FLW,
        OP_F_FSW,
        OP_F_FLD,
        OP_F_FSD,
        OP_F_FCVT_L_S,
        OP_F_FCVT_LU_S,
        OP_F_FCVT_S_L,
        OP_F_FCVT_S_LU,
        OP_F_FCVT_L_D,
        OP_F_FCVT_LU_D,
        OP_F_FMV_X_D,
        OP_F_FCVT_D_L,
        OP_F_FCVT_D_LU,
        OP_F_FMV_D_X,
        OP_F_ERROR


    } op_F_Codes;

    typedef enum {
        //.s, .d
        FMADD  = 0b1000011,
        FMSUB  = 0b1000111,
        FNMSUB = 0b1001011,
        FNMADD = 0b1001111,
        F_D = 0b01,
        F_S = 0b00,

        F_OP   = 0b1010011,
        FADD   = 0b00000,
        FSUB   = 0b00001,
        FMUL   = 0b00010,
        FDIV   = 0b00011,
        FSQRT  = 0b01011,
        FSGN  = 0b00100,
        FSGN_JX  = 0b010,
        FSGN_JN  = 0b001,
        FSGN_J   = 0b000,

        F_CSR  = 0b1110011,        

        FM    = 0b00101,
        FMIN  = 0b000,
        FMAX  = 0b001,
        FCVT__F  = 0b11000,
        FCVT_W  = 0b00000,
        FCVT_WU = 0b00001,
        FMV_X = 0b11100,
        FMV_X_F = 0b000,
        FMV_X_F2 = 0b00000,

        FCOMP = 0b10100,
        FEQ   = 0b010,
        FLT   = 0b001,
        FLE   = 0b000,
        FCLASS = 0b11100,
        FCLASS_F = 0b001,
        FCLASS_F2 = 0b00000,

        FCVT_F = 0b11010,

        FCVT_F_F = 0b01000,

        FMV_W_X = 0b11110,
        FMV_W_X_F = 0b000,
        FMV_W_X_F2 = 0b00000,

        FCVT_L_S = 0b11000,
        FCVT_S_L = 0b11010,
        FCVT_L = 0b00010,
        FCVT_U = 0b00011,

        FLWD = 0b0000111,
        FLW_F = 0b010,
        FSWD = 0b0100111,
        FSW_F = 0b010,
        FLD_F = 0b011,
        FSD_F = 0b011
        

    } F_Codes;



    template<typename T>
    class F_extension : public extension_base<T> {
    public:
        using extension_base<T>::extension_base;

        using signed_T = typename std::make_signed<T>::type;
        using unsigned_T = typename std::make_unsigned<T>::type;
        union BitsToDouble
        {
            uint64_t u64;
            double d;
            int64_t i;
        };

        union BitsToFloat
        {
            uint32_t u32;
            float f;
            int32_t i;
        };

        int classify_fp32(uint32_t input) {
            // rd Bit
            // 0 // rs1 is -infinity
            // 1 // rs1 is a negative normal number.
            // 2 // rs1 is a negative subnormal number.
            // 3 // rs1 is −0.
            // 4 // rs1 is +0.
            // 5 // rs1 is a positive subnormal number.
            // 6 // rs1 is a positive normal number.
            // 7 // rs1 is +infinity
            // 8 // rs1 is a signaling NaN.
            // 9 // rs1 is a quiet NaN.

            uint32_t bits;
            std::memcpy(&bits, &input, sizeof(bits));  // safe bit reinterpretation

            uint32_t sign     = (bits >> 31) & 0x1;
            uint32_t exponent = (bits >> 23) & 0xFF;
            uint32_t fraction = bits & 0x7FFFFF;
            int bit_shift;

            if (exponent == 0xFF) {
                if (fraction == 0) {
                    bit_shift = sign ? 0 : 7; // -inf / +inf
                } else {
                    bool is_qnan = (fraction >> 22) & 0x1; // MSB of fraction
                    bit_shift = is_qnan ? 9 : 8;
                }
            }
            else if (exponent == 0x00) {
                if (fraction == 0) {
                    bit_shift = sign ? 3 : 4; // -0 / +0
                } else {
                    bit_shift = sign ? 2 : 5; // subnormal
                }
            }
            else{
                // normal
                bit_shift = sign ? 1 : 6;
            }

            return (1 << bit_shift);
        }

        int classify_fp64(uint64_t input) {
            uint64_t bits;
            std::memcpy(&bits, &input, sizeof(bits)); // bit-level access

            uint64_t sign     = (bits >> 63) & 0x1;
            uint64_t exponent = (bits >> 52) & 0x7FF;
            uint64_t fraction = bits & 0xFFFFFFFFFFFFF;
            int bit_shift;

            if (exponent == 0x7FF) {
                if (fraction == 0) {
                    bit_shift = sign ? 0 : 7; // -inf : +inf
                } else {
                    bool is_qnan = (fraction >> 51) & 0x1; // MSB of fraction
                    bit_shift = is_qnan ? 9 : 8;
                }
            }
            else if (exponent == 0x000) {
                if (fraction == 0) {
                    bit_shift = sign ? 3 : 4; // -0 : +0
                } else {
                    bit_shift = sign ? 2 : 5; // subnormal
                }
            }
            else{
                // normal
                bit_shift = sign ? 1 : 6;
            }

            return (1 << bit_shift);

        }

        inline std::uint32_t get_funct5() const {                                                                                                                                                 
            return this->m_instr.range(31, 27);
        }
        inline std::uint32_t get_rs3() const {
            return this->m_instr.range(31, 27);
        }
        inline std::uint32_t get_fmt() const {
            return this->m_instr.range(26,25);
        }
        inline unsigned_T opcode() const  {
            return this->m_instr.range(6,0);
        }
        inline unsigned int get_rm() const  {
            return this->m_instr.range(14,12);
        }
        inline unsigned int get_imm12() const  {
            return this->m_instr.range(31,20);
        }
        inline unsigned int get_imm12_apart() const  {
            return this->m_instr.range(31,25);
        }
        inline unsigned int get_imm7() const  {
            return this->m_instr.range(31,25);
        }
        inline unsigned int get_imm5() const  {
            return this->m_instr.range(11,7);
        }
        [[nodiscard]] op_F_Codes decode() const{
                switch(opcode()){
                    case FMADD:
                        switch(get_fmt()){
                            case F_S:
                                return OP_F_FMADD_S;
                                break;
                            case F_D:
                                return OP_F_FMADD_D;
                                break;
                            default:
                                return OP_F_ERROR;
                                break; 
                        }
                        break;
                    case FMSUB:
                        switch(get_fmt()){
                            case F_S:
                                return OP_F_FMSUB_S;
                                break;
                            case F_D:
                                return OP_F_FMSUB_D;
                                break;
                            default:
                                return OP_F_ERROR;
                                break; 
                        }
                        break;
                    case FNMSUB:
                        switch(get_fmt()){
                            case F_S:
                                return OP_F_FNMSUB_S;
                                break;
                            case F_D:
                                return OP_F_FNMSUB_D;
                                break;
                            default:
                                return OP_F_ERROR;
                                break; 
                        }
                        break;
                    case FNMADD:
                        switch(get_fmt()){
                            case F_S:
                                return OP_F_FNMADD_S;
                                break;
                            case F_D:
                                return OP_F_FNMADD_D;
                                break;
                            default:
                                return OP_F_ERROR;
                                break; 
                        }
                        break;
                    case F_OP:
                        switch(get_funct5()){
                            case FADD:
                                switch(get_fmt()){
                                    case F_S:
                                        return OP_F_FADD_S;
                                        break;
                                    case F_D:
                                        return OP_F_FADD_D;
                                        break;                            
                                    default:
                                        return OP_F_ERROR;
                                        break; 
                                }
                                break;
                            case FSUB:
                                switch(get_fmt()){
                                    case F_S:
                                        return OP_F_FSUB_S;
                                        break;
                                    case F_D:
                                        return OP_F_FSUB_D;
                                        break;                            
                                    default:
                                        return OP_F_ERROR;
                                        break; 
                                }
                                break;
                            case FMUL:
                                switch(get_fmt()){
                                    case F_S:
                                        return OP_F_FMUL_S;
                                        break;
                                    case F_D:
                                        return OP_F_FMUL_D;
                                        break;                            
                                    default:
                                        return OP_F_ERROR;
                                        break; 
                                }
                                break;
                            case FDIV:
                                switch(get_fmt()){
                                    case F_S:
                                        return OP_F_FDIV_S;
                                        break;
                                    case F_D:
                                        return OP_F_FDIV_D;
                                        break;                            
                                    default:
                                        return OP_F_ERROR;
                                        break; 
                                }
                                break;
                            case FSQRT:
                                if((this->get_rs2() == 0b00000)){
                                    switch(get_fmt()){
                                    case F_S:
                                        return OP_F_FSQRT_S;
                                        break;
                                    case F_D:
                                        return OP_F_FSQRT_D;
                                        break;                            
                                    default:
                                        return OP_F_ERROR;
                                        break; 
                                }
                                }else{
                                    return OP_F_ERROR;
                                }
                                break;
                            case FSGN:
                                switch(this->get_funct3()){
                                    case FSGN_J:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FSGNJ_S;
                                                break;
                                            case F_D:
                                                return OP_F_FSGNJ_D;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    case FSGN_JN:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FSGNJN_S;
                                                break;
                                            case F_D:
                                                return OP_F_FSGNJN_D;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    case FSGN_JX:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FSGNJX_S;
                                                break;
                                            case F_D:
                                                return OP_F_FSGNJX_D;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    default:
                                        return OP_F_ERROR;
                                        break;
                                }
                                break;
                            case FM:
                                switch(this->get_funct3()){
                                    case FMIN:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FMIN_S;
                                                break;
                                            case F_D:
                                                return OP_F_FMIN_D;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    case FMAX:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FMAX_S;
                                                break;
                                            case F_D:
                                                return OP_F_FMAX_D;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    default:
                                        return OP_F_ERROR;
                                        break;
                                }
                                break;
                            case FCVT__F:
                                switch(this->get_rs2()){
                                    case FCVT_W:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FCVT_W_S;
                                                break;
                                            case F_D:
                                                return OP_F_FCVT_W_D;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    case FCVT_WU:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FCVT_WU_S;
                                                break;
                                            case F_D:
                                                return OP_F_FCVT_WU_D;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    case FCVT_L:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FCVT_L_S;
                                                break;
                                            case F_D:
                                                return OP_F_FCVT_L_D;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }

                                        break;
                                    case FCVT_U:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FCVT_LU_S;
                                                break;
                                            case F_D:
                                                return OP_F_FCVT_LU_D;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    default:
                                        return OP_F_ERROR;
                                        break;
                                }
                                break;
                            case FMV_X:
                                if((this->get_funct3() == FMV_X_F)&& (this->get_rs2() == FMV_X_F2)){
                                    switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FMV_X_W;
                                                break;
                                            case F_D:
                                                return OP_F_FMV_X_D;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                    }
                                    return OP_F_FMV_X_W;
                                } else if((this->get_funct3() == FCLASS_F)&& (this->get_rs2() == FCLASS_F2)){
                                    switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FCLASS_S;
                                                break;
                                            case F_D:
                                                return OP_F_FCLASS_D;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                    }
                                } else {
                                    return OP_F_ERROR;
                                }
                                break;
                            case FCOMP:
                                switch(this->get_funct3()){
                                    case FEQ:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FEQ_S;
                                                break;
                                            case F_D:
                                                return OP_F_FEQ_D;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    case FLT:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FLT_S;
                                                break;
                                            case F_D:
                                                return OP_F_FLT_D;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    case FLE:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FLE_S;
                                                break;
                                            case F_D:
                                                return OP_F_FLE_D;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    default:
                                        return OP_F_ERROR;
                                        break;
                                }
                                break;
                            case FCVT_F_F:
                                if((this->get_rs2() == 0b00001)&&(this->get_fmt() == 00)){
                                    return OP_F_FCVT_S_D;
                                } else if((this->get_rs2() == 0b00000)&&(this->get_fmt() == 01)){
                                    return OP_F_FCVT_D_S;
                                } else{
                                    return OP_F_ERROR;
                                }
                                break;
                                
                            case FCVT_F:
                                switch(this->get_rs2()){
                                    case FCVT_W:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FCVT_S_W;
                                                break;
                                            case F_D:
                                                return OP_F_FCVT_D_W;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    case FCVT_WU:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FCVT_S_WU;
                                                break;
                                            case F_D:
                                                return OP_F_FCVT_D_WU;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    case FCVT_L:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FCVT_S_L;
                                                break;
                                            case F_D:
                                                return OP_F_FCVT_D_L;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    case FCVT_U:
                                        switch(get_fmt()){
                                            case F_S:
                                                return OP_F_FCVT_S_LU;
                                                break;
                                            case F_D:
                                                return OP_F_FCVT_D_LU;
                                                break;                            
                                            default:
                                                return OP_F_ERROR;
                                                break; 
                                        }
                                        break;
                                    default:
                                        return OP_F_ERROR;
                                        break;
                                }
                                break;
                            case FMV_W_X:
                                if((this->get_funct3()== FMV_W_X_F)&& (this->get_rs2() == FMV_W_X_F2) && (this->get_fmt() == 0b00)){
                                    return OP_F_FMV_W_X;
                                } else if((this->get_funct3()== FMV_W_X_F)&& (this->get_rs2() == FMV_W_X_F2) && (this->get_fmt() == 0b01)){
                                    return OP_F_FMV_D_X;
                                } else{return OP_F_ERROR;}
                                break;                                                       
                            default:
                                return OP_F_ERROR;
                                break; 
                        }
                        break;
                    case FLWD:
                        switch(this->get_funct3()){
                            case FLW_F:
                                return OP_F_FLW;
                                break;
                            case FLD_F:
                                return OP_F_FLD;
                                break;
                            default:
                                return OP_F_ERROR;
                                break;
                        }
                        break;
                    case FSWD:
                        switch(this->get_funct3()){
                            case FSW_F:
                                return OP_F_FSW;
                                break;
                            case FSD_F:
                                return OP_F_FSD;
                                break;
                            default:
                                return OP_F_ERROR;
                                break;
                        }
                        break;
                    default:
                        return OP_F_ERROR;
                        break;

                }
                return OP_F_ERROR;
        }

        void Exec_F_FMADD_S(){
            unsigned int rs3,rs2,rs1,rm ,rd;
            BitsToFloat fd, fs1, fs2, fs3;
            int fflags = 0;
            rs3 = this->get_rs3();
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this ->get_rm();//rounding
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            fs3.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs3));
            
            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            fd.f = fs1.f * fs2.f + fs3.f;

            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.f = std::numeric_limits<float>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX

            this -> regs-> setCSR(CSR_ADDR_FFLAGS, fflags);

            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FMADD_S_CYC);
        }  
        void EXEC_F_FMSUB_S(){
            unsigned int rs3,rs2,rs1,rm ,rd;
            BitsToFloat fd, fs1, fs2, fs3;
            int fflags = 0;
            rs3 = this->get_rs3();
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this ->get_rm();//rounding
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            fs3.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs3));
            
            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            fd.f = fs1.f * fs2.f - fs3.f;

            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.f = std::numeric_limits<float>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this -> regs-> setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FMSUB_S_CYC);
        }
        void EXEC_F_FNMADD_S(){
            unsigned int rs3,rs2,rs1,rm,rd;
            BitsToFloat fd, fs1, fs2, fs3;
            int fflags = 0;
            rs3 = this->get_rs3();
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this ->get_rm();//rounding
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            fs3.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs3));
            
            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            fd.f = -fs1.f * fs2.f - fs3.f;

            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.f = std::numeric_limits<float>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this -> regs-> setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FNMADD_S_CYC);
        };
        void EXEC_F_FNMSUB_S(){
            unsigned int rs3,rs2,rs1,rm,rd;
            BitsToFloat fd, fs1, fs2, fs3;
            int fflags = 0;
            rs3 = this->get_rs3();
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this ->get_rm();//rounding
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            fs3.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs3));
            
            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            fd.f = -fs1.f * fs2.f + fs3.f;
            
            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.f = std::numeric_limits<float>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this -> regs-> setCSR(CSR_ADDR_FFLAGS, fflags);

            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FNMSUB_S_CYC);
        };
        void EXEC_F_FADD_S(){
            unsigned int rs2,rs1,rm,rd;
            BitsToFloat fd, fs1, fs2;
            int fflags = 0;

            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this ->get_rm();//rounding
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            
            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            fd.f = fs1.f + fs2.f;
            
            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.f = std::numeric_limits<float>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this -> regs-> setCSR(CSR_ADDR_FFLAGS, fflags);

            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FADD_S_CYC);
        };
        void EXEC_F_FSUB_S(){
            unsigned int rs2,rs1,rm,rd;
            BitsToFloat fd, fs1, fs2;
            int fflags = 0;

            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this ->get_rm();//rounding
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            
            feclearexcept(FE_ALL_EXCEPT);

            fd.f = fs1.f - fs2.f;
            
            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.f = std::numeric_limits<float>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX

            this -> regs-> setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FSUB_S_CYC);
        };
        void EXEC_F_FMUL_S(){
            unsigned int rs2,rs1,rm,rd;
            BitsToFloat fd, fs1, fs2;
            int fflags = 0;

            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this ->get_rm();//rounding
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            
            feclearexcept(FE_ALL_EXCEPT);

            fd.f = fs1.f * fs2.f;

            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.f = std::numeric_limits<float>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this -> regs-> setCSR(CSR_ADDR_FFLAGS, fflags);

            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FMUL_S_CYC);
        };
        void EXEC_F_FDIV_S(){
            unsigned int rs2,rs1,rm,rd;
            BitsToFloat fd, fs1, fs2;
            int fflags = 0;

            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this ->get_rm();//rounding
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            
            feclearexcept(FE_ALL_EXCEPT);
            
            fd.f = fs1.f / fs2.f;

            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.f = std::numeric_limits<float>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this -> regs-> setCSR(CSR_ADDR_FFLAGS, fflags);

            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FDIV_S_CYC);
        };
        void EXEC_F_FSQRT_S(){
            unsigned int rs1,rm,rd;
            BitsToFloat fd, fs1;
            int fflags = 0;

            rs1 = this->get_rs1();
            rm = this ->get_rm();//rounding
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            
            feclearexcept(FE_ALL_EXCEPT);
            
            fd.f = sqrt(fs1.f);

            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.f = std::numeric_limits<float>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this -> regs-> setCSR(CSR_ADDR_FFLAGS, fflags);

            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FSQRT_S_CYC);
        };
        void EXEC_F_FSGNJ_S(){
            unsigned int rs2,rs1,rd;
            BitsToFloat fd, fs1, fs2;
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd  = this->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            fd.u32 = (fs2.u32 & 0x80000000) | (fs1.u32 & 0x7FFFFFFF);
            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FSGNJ_S_CYC);
        };
        void EXEC_F_FSGNJN_S(){
            unsigned int rs2,rs1,rd;
            BitsToFloat fd, fs1, fs2;
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            fd.u32 = (~(fs2.u32) & 0x80000000) | fs1.u32 & 0x7FFFFFFF;
            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FSGNJN_S_CYC);
        };
        void EXEC_F_FSGNJX_S(){
            unsigned int rs2,rs1,rd;
            BitsToFloat fd, fs1, fs2;
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            fd.u32 = ((fs2.u32 & 0x80000000) ^ (fs1.u32 & 0x80000000)) | fs1.u32 & 0x7FFFFFFF;
            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FSGNJX_S_CYC);
        };
        void EXEC_F_FMIN_S(){
            unsigned int rs2, rs1, rd;
            BitsToFloat fs1, fs2, fd;
            
            int fflags = 0;
            bool fs1_qnan, fs2_qnan;
            bool fs1_snan, fs2_snan;
            
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            
            fs1_qnan = (classify_fp32(fs1.u32) == (1<<9));
            fs2_qnan = (classify_fp32(fs2.u32) == (1<<9));
            fs1_snan = (classify_fp32(fs1.u32) == (1<<8));
            fs2_snan = (classify_fp32(fs2.u32) == (1<<8));

            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            if ( fs1.f < fs2.f ) {
                fd.f = fs1.f;
            }
            else if (fs1.f == fs2.f) {
                if(classify_fp32(fs1.u32) == (1<<3))    fd.f = fs1.f;
                else    fd.f = fs2.f;
            }
            else {
                fd.f = fs2.f;
            }
            
            if ((fs1_qnan && fs2_qnan) || (fs1_snan || fs2_snan))  { // NV
                fflags |= (1 << 4); 
                fd.f = std::numeric_limits<float>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            
            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FMIN_S_CYC);
        };
        void EXEC_F_FMAX_S(){
            unsigned int rs2, rs1, rd;
            BitsToFloat fs2, fs1, fd;
            
            int fflags = 0;
            bool fs1_qnan, fs2_qnan;
            bool fs1_snan, fs2_snan;
            
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            
            fs1_qnan = (classify_fp32(fs1.u32) == (1<<9));
            fs2_qnan = (classify_fp32(fs2.u32) == (1<<9));
            fs1_snan = (classify_fp32(fs1.u32) == (1<<8));
            fs2_snan = (classify_fp32(fs2.u32) == (1<<8));

            feclearexcept(FE_ALL_EXCEPT); // reset exception flags
            
            if ( fs1.f > fs2.f ){
                fd.f = fs1.f;
            }
            else if (fs1.f == fs2.f) {
                if(classify_fp32(fs1.u32) == (1<<4)) {
                    fd.f = fs1.f;
                }
                else {
                    fd.f = fs2.f;
                }  
            }
            else {
                fd.f = fs2.f;
            }

            if ((fs1_qnan && fs2_qnan) || (fs1_snan || fs2_snan))  { // NV
                fflags |= (1 << 4); 
                fd.f = std::numeric_limits<float>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX


            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            
            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FMAX_S_CYC);
        };
        void EXEC_F_FCVT_W_S(){     // FIXME Rounding
            unsigned int rs1,rd;
            int32_t xd;
            BitsToFloat fs1;
            int32_t min = 0x80000000;
            int32_t max = 0x7FFFFFFF;
            int classify;
            int fflags = 0;
            
            rs1 = this->get_rs1();
            rd = this->get_rd();
            fs1.u32=this->regs->getfValue(rs1);
            
            // Classify
            // 0 bit // rs1 is -infinity
            // 1 bit // rs1 is a negative normal number.
            // 2 bit // rs1 is a negative subnormal number.
            // 3 bit // rs1 is −0.
            // 4 bit // rs1 is +0.
            // 5 bit // rs1 is a positive subnormal number.
            // 6 bit // rs1 is a positive normal number.
            // 7 bit // rs1 is +infinity
            // 8 bit // rs1 is a signaling NaN.
            // 9 bit // rs1 is a quiet NaN.
            classify = classify_fp32(fs1.u32);
            
            if (fs1.f < min) {      // min, -Inf
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.f;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd = min;
            }
            else if ((fs1.f > max) | (classify == (1<<9)) | (classify == (1<<8))) {     // max, +Inf, NaN
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.f;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd = max;
            }
            else {
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.f;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
            }

            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setValue(rd,(xd));
            this->perf->cycleInc(FCVT_W_S_CYC);
        };
        void EXEC_F_FCVT_WU_S(){    // FIXME Rounding
            unsigned int rs1,rd;
            BitsToFloat xd;
            BitsToFloat fs1;
            
            uint32_t min = 0x00000000;
            uint32_t max = 0xFFFFFFFF;
            int classify;
            int fflags = 0;
            
            rs1 = this->get_rs1();
            rd = this->get_rd();
            fs1.u32=this->regs->getfValue(rs1);
            
            // Classify
            // 0 bit // rs1 is -infinity
            // 1 bit // rs1 is a negative normal number.
            // 2 bit // rs1 is a negative subnormal number.
            // 3 bit // rs1 is −0.
            // 4 bit // rs1 is +0.
            // 5 bit // rs1 is a positive subnormal number.
            // 6 bit // rs1 is a positive normal number.
            // 7 bit // rs1 is +infinity
            // 8 bit // rs1 is a signaling NaN.
            // 9 bit // rs1 is a quiet NaN.

            classify = classify_fp32(fs1.u32);
            
            if ((((classify == (1<<1)) | (classify == (1<<2))) & (fs1.f < min)) | (classify == (1<<0))) {  // min, -Inf
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd.u32 = fs1.f;
                
                if (fs1.f <= -1.0)              fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd.u32 = min;
            }
            else if ((fs1.f > max) | (classify == (1<<9)) | (classify == (1<<8))) {     // max, +Inf, NaN
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd.u32 = fs1.f;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd.u32 = max;
            }
            else {
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd.u32 = fs1.f;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
            }
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setValue(rd,(xd.i));
            this->perf->cycleInc(FCVT_WU_S_CYC);
        };

        void EXEC_F_FMV_X_W(){
            unsigned int rs1, rd;
            BitsToFloat fs1;
            BitsToDouble fd;
            rs1 = this->get_rs1();
            rd = this->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this->regs->getfValue(rs1));
            fs1.i = static_cast<std::int32_t>(fs1.u32);
            fd.u64 = static_cast<std::int64_t>(fs1.i);
            this->regs->setValue(rd, (fd.u64));
            this->perf->cycleInc(FMV_X_W_CYC);
        };
        void EXEC_F_FEQ_S(){
            unsigned int rs2,rs1,rd;
            BitsToFloat fs1, fs2;
            unsigned int xd;
            int fflags = 0;


            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2)); 

            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            xd = (fs1.f == fs2.f);

            if (fetestexcept(FE_INVALID)) {  // NV
                fflags |= (1 << 4); 
               
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setValue(rd, (xd));
            this->perf->cycleInc(FEQ_S_CYC);
        };
        void EXEC_F_FLT_S(){
            unsigned int rs2,rs1,rd;
            BitsToFloat fs1, fs2;
            unsigned int xd;
            int fflags = 0;


            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            xd = (fs1.f < fs2.f);
            
            if (fetestexcept(FE_INVALID)) {  // NV
                fflags |= (1 << 4); 
               
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);


            this->regs->setValue(rd, (xd));
            this->perf->cycleInc(FLT_S_CYC);
        };
        void EXEC_F_FLE_S(){
            unsigned int rs2,rs1,rd;
            BitsToFloat fs1, fs2;
            unsigned int xd;
            int fflags = 0;


            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            fs2.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs2));
            
            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            xd = (fs1.f <= fs2.f);
            
            if (fetestexcept(FE_INVALID)) {  // NV
                fflags |= (1 << 4); 
               
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);


            this->regs->setValue(rd, (xd));
            this->perf->cycleInc(FLE_S_CYC);
        };
        void EXEC_F_FCLASS_S(){
            unsigned int rs1,rm,rd;
            BitsToFloat fd, fs1;
            int xd;
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getfValue(rs1));
            xd = classify_fp32(fs1.u32);
            this->regs->setValue(rd, (xd));
            this->perf->cycleInc(FCLASS_S_CYC);
        };
        void EXEC_F_FCVT_S_W(){
            unsigned int rs1,rd;
            int32_t i32;
            BitsToFloat fs1;

            rs1 = this->get_rs1();
            rd = this->get_rd();

            i32=this->regs->getValue(rs1);
            fs1.f=i32;

            this->regs->setfValue(rd,(fs1.u32));
            this->perf->cycleInc(FCVT_S_W_CYC);
        };
        void EXEC_F_FCVT_S_WU(){
            unsigned int rs1,rd;
            BitsToFloat fs1;

            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.f = static_cast<std::uint32_t>(this -> regs-> getValue(rs1));

            this->regs->setfValue(rd,(fs1.u32));
            this->perf->cycleInc(FCVT_S_WU_CYC);
        };
        void EXEC_F_FMV_W_X(){
            unsigned int rs1,rd;
            BitsToFloat fd, fs1;
            uint32_t xs1;
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.u32 = static_cast<std::uint32_t>(this -> regs-> getValue(rs1));
            fd.f = fs1.f;
            this->regs->setfValue(rd, (fd.u32));
            this->perf->cycleInc(FMV_W_X_CYC);
        };
        void EXEC_F_FMADD_D(){
            unsigned int rs3,rs2,rs1,rm ,rd;
            BitsToDouble fd, fs1, fs2, fs3;

            unsigned int fflags = 0;

            rs3 = this->get_rs3();
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this ->get_rm();//rounding
            rd = this ->get_rd();
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));
            fs3.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs3));
            
            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            fd.d = fs1.d * fs2.d + fs3.d;

            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.d = std::numeric_limits<double>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);

            this->regs->setfValue(rd, (fd.u64));
            this->perf->cycleInc(FMADD_D_CYC);
        };
        void EXEC_F_FMSUB_D(){
            unsigned int rs3,rs2,rs1,rm ,rd;
            BitsToDouble fd, fs1, fs2, fs3;

            unsigned int fflags = 0;

            rs3 = this->get_rs3();
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this ->get_rm();//rounding
            rd = this ->get_rd();

            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));
            fs3.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs3));

            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            fd.d = fs1.d * fs2.d - fs3.d;

            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.d = std::numeric_limits<double>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);

            this->regs->setfValue(rd, (fd.u64));
            this->perf->cycleInc(FMSUB_D_CYC);
        };
        void EXEC_F_FNMSUB_D(){
            unsigned int rs3,rs2,rs1,rm,rd;
            BitsToDouble fd, fs1, fs2, fs3;

            unsigned int fflags = 0;

            rs3 = this->get_rs3();
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this ->get_rm();
            rd = this ->get_rd();
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));
            fs3.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs3));

            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            fd.d = -fs1.d * fs2.d + fs3.d;

            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.d = std::numeric_limits<double>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setfValue(rd, (fd.u64));
            this->perf->cycleInc(FNMSUB_D_CYC);
        };
        void EXEC_F_FNMADD_D(){
            unsigned int rs3,rs2,rs1,rm,rd;
            BitsToDouble fd, fs1, fs2, fs3;

            unsigned int fflags = 0;

            rs3 = this->get_rs3();
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this ->get_rm();//rounding
            rd = this ->get_rd();
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));
            fs3.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs3));

            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            fd.d = -fs1.d * fs2.d - fs3.d;

            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.d = std::numeric_limits<double>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);

            this->regs->setfValue(rd, (fd.u64));
            this->perf->cycleInc(FNMADD_D_CYC);
        };
        void EXEC_F_FADD_D(){
            unsigned int rs2, rs1, rm, rd;
            BitsToDouble fd, fs1, fs2;
            
            unsigned int fflags = 0;

            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this->get_rm();    //rounding
            rd = this->get_rd();
            
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));
            
            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            fd.d = fs1.d + fs2.d;
            
            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.d = std::numeric_limits<double>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setfValue(rd, (fd.u64));
            this->perf->cycleInc(FADD_D_CYC);
        };
        void EXEC_F_FSUB_D(){
            unsigned int rs2, rs1, rm, rd;
            BitsToDouble fd, fs1, fs2;
            
            unsigned int fflags = 0;

            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this->get_rm();    //rounding
            rd = this->get_rd();
            
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));
            
            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            fd.d = fs1.d - fs2.d;
            
            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.d = std::numeric_limits<double>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setfValue(rd, (fd.u64));
            this->perf->cycleInc(FSUB_D_CYC);
        };
        void EXEC_F_FMUL_D(){
            unsigned int rs2, rs1, rm, rd;
            BitsToDouble fd, fs1, fs2;
            
            unsigned int fflags = 0;

            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this->get_rm();    //rounding
            rd = this->get_rd();
            
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));
            
            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            fd.d = fs1.d * fs2.d;
            
            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.d = std::numeric_limits<double>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setfValue(rd, (fd.u64));
            this->perf->cycleInc(FMUL_D_CYC);
        };
        void EXEC_F_FDIV_D(){
            unsigned int rs2, rs1, rm, rd;
            BitsToDouble fd, fs1, fs2;
            
            unsigned int fflags = 0;

            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rm = this->get_rm();    //rounding
            rd = this->get_rd();
            
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));
            
            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            fd.d = fs1.d / fs2.d;
            
            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.d = std::numeric_limits<double>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setfValue(rd, (fd.u64));
            this->perf->cycleInc(FDIV_D_CYC);
        };
        void EXEC_F_FSQRT_D(){
            unsigned int rs1, rm, rd;
            BitsToDouble fd, fs1;
            
            unsigned int fflags = 0;

            rs1 = this->get_rs1();
            rm = this->get_rm();    //rounding
            rd = this->get_rd();
            
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            
            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            fd.d = sqrt(fs1.d);
            
            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                fd.d = std::numeric_limits<double>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setfValue(rd, (fd.u64));
            this->perf->cycleInc(FSQRT_D_CYC);
        };
        void EXEC_F_FSGNJ_D(){
            unsigned int rs2,rs1,rd;
            BitsToDouble fd, fs1, fs2;
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));
            fd.u64 = fs2.u64 & 0x8000000000000000 | fs1.u64 & 0x7FFFFFFFFFFFFFFF;



            this->regs->setfValue(rd, (fd.u64));
            this->perf->cycleInc(FSGNJ_D_CYC);
        };
        void EXEC_F_FSGNJN_D(){
            unsigned int rs2,rs1,rd;
            BitsToDouble fd, fs1, fs2;
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));
            fd.u64 = (~fs2.u64) & 0x8000000000000000 | fs1.u64 & 0x7FFFFFFFFFFFFFFF;
            this->regs->setfValue(rd, (fd.u64));
            this->perf->cycleInc(FSGNJN_D_CYC);
        };
        void EXEC_F_FSGNJX_D(){
            unsigned int rs2,rs1,rd;
            BitsToDouble fd, fs1, fs2;
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));
            fd.u64 = ((fs2.u64 & 0x8000000000000000) ^ (fs1.u64 & 0x8000000000000000)) | fs1.u64 & 0x7FFFFFFFFFFFFFFF;
            this->regs->setfValue(rd, (fd.u64));
            this->perf->cycleInc(FSGNJX_D_CYC);
        };
        void EXEC_F_FMIN_D(){
            unsigned int rs2, rs1, rd;
            BitsToDouble fs1, fs2, fd;
            
            int fflags = 0;
            
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));
            
            bool fs1_qnan = (classify_fp64(fs1.u64) == (1<<9));
            bool fs2_qnan = (classify_fp64(fs2.u64) == (1<<9));
            bool fs1_snan = (classify_fp64(fs1.u64) == (1<<8));
            bool fs2_snan = (classify_fp64(fs2.u64) == (1<<8));

            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            if ( fs1.d < fs2.d ) {
                fd.d = fs1.d;
            }
            else if (fs1.d == fs2.d) {
                if(classify_fp64(fs1.u64) == (1<<3)) {
                    fd.d = fs1.d;
                }
                else {
                    fd.d = fs2.d;
                }
            }
            else {
                fd.d = fs2.d;
            }
            
            if ((fs1_qnan && fs2_qnan) || (fs1_snan || fs2_snan))  { // NV
                fflags |= (1 << 4); 
                fd.d = std::numeric_limits<double>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            
            this->regs->setfValue(rd, (fd.u64));
            this->perf->cycleInc(FMIN_D_CYC);
        };
        void EXEC_F_FMAX_D(){
            unsigned int rs2, rs1, rd;
            BitsToDouble fs1, fs2, fd;
            
            int fflags = 0;
            
            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));
            
            bool fs1_qnan = (classify_fp64(fs1.u64) == (1<<9));
            bool fs2_qnan = (classify_fp64(fs2.u64) == (1<<9));
            bool fs1_snan = (classify_fp64(fs1.u64) == (1<<8));
            bool fs2_snan = (classify_fp64(fs2.u64) == (1<<8));

            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            if ( fs1.d > fs2.d ){
                fd.d = fs1.d;
                
            }
            else if (fs1.d == fs2.d) {
                if(classify_fp64(fs1.u64) == (1<<4)) {
                    fd.d = fs1.d;
                
                }
                else {
                    fd.d = fs2.d;
                
                }  
            }
            else {
                fd.d = fs2.d;
            }
            
            if ((fs1_qnan && fs2_qnan) || (fs1_snan || fs2_snan))  { // NV
                fflags |= (1 << 4); 
                fd.d = std::numeric_limits<double>::quiet_NaN();
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            
            this->regs->setfValue(rd, (fd.u64));
            this->perf->cycleInc(FMAX_D_CYC);
        };
        void EXEC_F_FCVT_S_D(){     // FIXME
            unsigned int rs1, rd;
            BitsToFloat fs1;
            BitsToDouble fs2, d1;
            uint64_t NaN = 0x7ff8000000000000;

            rs1 = this->get_rs1();  //src
            rd = this->get_rd();    //dest

            fs2.u64=this -> regs-> getfValue(rs1);

            if(classify_fp64(fs2.u64) == (1<<9)) { //is qnan
                fs2.u64 = NaN;
                fs1.f = fs2.d;
            }
            else {
                fs1.f = fs2.d;
            }
 

            
            this->regs->setfValue(rd,(fs1.u32));
            this->perf->cycleInc(FCVT_S_D_CYC);
        };
        void EXEC_F_FCVT_D_S(){     // FIXME
            unsigned int rs1, rd;
            BitsToFloat fs1;
            BitsToDouble fs2;
            uint64_t NaN = 0x7ff8000000000000;

            rs1 = this->get_rs1();  //src
            rd = this->get_rd();    //dest

            fs1.u32=this -> regs-> getfValue(rs1);

            if(classify_fp64(fs2.u64) == (1<<9)) { //is qnan
                fs1.u32 = NaN;
                fs2.d=fs1.f;
            }
            else {
                fs2.d=fs1.f;
            }

            fs2.d=fs1.f;

            this->regs->setfValue(rd,(fs2.u64));
            this->perf->cycleInc(FCVT_D_S_CYC);
        };
        void EXEC_F_FEQ_D(){
            unsigned int rs2,rs1,rd;
            BitsToDouble fs1, fs2;

            unsigned int xd;
            unsigned int fflags = 0;

            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));

            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            xd = (fs1.d == fs2.d);
            
            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                xd = 0;
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);

            
            this->regs->setValue(rd, (xd));
            this->perf->cycleInc(FEQ_D_CYC);
        };
        void EXEC_F_FLT_D(){
            unsigned int rs2,rs1,rd;
            BitsToDouble fs1, fs2;

            unsigned int xd;
            unsigned int fflags = 0;

            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));

            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            xd = (fs1.d < fs2.d);
            
            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                xd = 0;
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);

            
            this->regs->setValue(rd, (xd));
            this->perf->cycleInc(FLT_D_CYC);
        };
        void EXEC_F_FLE_D(){
            unsigned int rs2,rs1,rd;
            BitsToDouble fs1, fs2;

            unsigned int xd;
            unsigned int fflags = 0;

            rs2 = this->get_rs2();
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            fs2.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs2));

            feclearexcept(FE_ALL_EXCEPT); // reset exception flags

            xd = (fs1.d <= fs2.d);
            
            if (fetestexcept(FE_INVALID)) {                     // NV
                fflags |= (1 << 4); 
                xd = 0;
            }
            if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3); // DZ
            if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2); // OF
            if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1); // UF
            if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0); // NX
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);

            
            this->regs->setValue(rd, (xd));
            this->perf->cycleInc(FLE_D_CYC);
        };
        void EXEC_F_FCLASS_D(){     // FIXME
            unsigned int rs1, rm, rd;
            BitsToDouble fd, fs1;
            unsigned int xd;

            rs1 = this->get_rs1();
            rd = this->get_rd();
            
            fs1.u64 = static_cast<std::uint64_t>(this -> regs-> getfValue(rs1));
            xd = classify_fp64(fs1.u64);

            
            this->regs->setValue(rd, (xd));
            this->perf->cycleInc(FCLASS_D_CYC);
        };
        void EXEC_F_FCVT_W_D(){     // FIXME 
            unsigned int rs1,rd;
            BitsToDouble fs1;
            
            int32_t xd;
            int32_t min = 0x80000000;
            int32_t max = 0x7FFFFFFF;
            
            unsigned int classify;
            unsigned int fflags = 0;
            
            rs1 = this->get_rs1();
            rd = this ->get_rd();
            fs1.u64 = this->regs->getfValue(rs1);
            
            classify = classify_fp64(fs1.u64);
            
            if ((((classify == (1<<1)) | (classify == (1<<2))) & (fs1.d < min)) | (classify == (1<<0))) {      // min, -Inf
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.d;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd = min;
            }
            else if ((fs1.d > max) | (classify == (1<<9)) | (classify == (1<<8))) {     // max, +Inf, NaN
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.d;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd = max;
            }
            else {
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.d;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
            }
    
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setValue(rd,(xd));
            this->perf->cycleInc(FCVT_W_D_CYC);
        };
        void EXEC_F_FCVT_WU_D(){    // FIXME
            unsigned int rs1,rd;
            BitsToDouble fs1;
            
            BitsToFloat xd;
            uint32_t min = 0x00000000;
            uint32_t max = 0xFFFFFFFF;
            
            unsigned int classify;
            unsigned int fflags = 0;
            
            rs1 = this->get_rs1();
            rd = this->get_rd();
            fs1.u64 = this->regs->getfValue(rs1);
            
            classify = classify_fp64(fs1.u64);
            
            if ((((classify == (1<<1)) | (classify == (1<<2))) & (fs1.d < min)) | (classify == (1<<0))) {      // min, -Inf
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd.u32 = fs1.d;
                if (fs1.d <= -1.0)   fflags |= (1 << 4);                // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd.u32 = min;
            }
            else if ((fs1.d > max) | (classify == (1<<9)) | (classify == (1<<8))) {     // max, +Inf, NaN
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd.u32 = fs1.d;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd.u32 = max;
            }
            else {
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd.u32 = fs1.d;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
            }
    
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setValue(rd,(xd.i));
            this->perf->cycleInc(FCVT_WU_D_CYC);
        };
        void EXEC_F_FCVT_D_W(){     // FIXME
            unsigned int rs1,rd;
            int32_t i32;
            BitsToDouble fs1;
            
            rs1 = this->get_rs1();
            rd = this ->get_rd();

            i32=this->regs->getValue(rs1);
            fs1.d=i32;

            this->regs->setfValue(rd,(fs1.u64));
            this->perf->cycleInc(FCVT_D_W_CYC);
        };
        void EXEC_F_FCVT_D_WU(){    // FIXME
            unsigned int rs1,rd;
            BitsToDouble fs1;
            uint32_t ui32;

            rs1 = this->get_rs1();
            rd = this-> get_rd();
            ui32 = static_cast<std::uint64_t>(this -> regs-> getValue(rs1));
            fs1.d = ui32;

            this->regs->setfValue(rd,(fs1.u64));
            this->perf->cycleInc(FCVT_D_WU_CYC);
        };           
        void EXEC_F_FLW(){
            unsigned int rd, rs1;
            signed_T imm;
            std::uint32_t udata32;
            std::uint64_t udata64, NaNBOX;
            unsigned_T mem_addr;

            NaNBOX = 0xffffffff;

            rs1 = this -> get_rs1();
            rd = this -> get_rd();
            imm = this -> get_imm12();
            if((imm&0x800) == 0x800){
                imm = imm |(0xFFFFFFFFFFFFF000);
            }
            mem_addr = this -> regs ->getValue(rs1) + imm;
            udata32 = static_cast<std::uint32_t>(this-> mem_intf->readDataMem(mem_addr,4));
            udata64 = NaNBOX <<32 | udata32;

            this->perf->dataMemoryRead();
            this->regs->setfValue(rd, udata64);
            this->perf->cycleInc(FLW_CYC);

            return;
        }
        void EXEC_F_FSW(){
            unsigned int rs1, rs2;
            signed_T imm;
            std::int32_t data;
            unsigned_T mem_addr;
            rs1 = this -> get_rs1();
            rs2 = this -> get_rs2();
            imm = ((this -> get_imm7()) << 5) | this -> get_rd();
            if((imm&0x800) == 0x800){
                imm = imm |(0xFFFFFFFFFFFFF000);
            }
            mem_addr = imm + this->regs->getValue(rs1);
            data = static_cast<std::int32_t>(this->regs->getfValue(rs2));
            
            this->perf->dataMemoryWrite();
            this->mem_intf->writeDataMem(mem_addr,data,4);
            this->perf->cycleInc(FSW_CYC);

            return;
        }
        void EXEC_F_FLD(){          // FIXME
            unsigned int rd, rs1;
            signed_T imm;
            std::uint64_t data;
            unsigned_T mem_addr;
            
            rs1 = this -> get_rs1();
            rd = this -> get_rd();
            imm = this -> get_imm12();
            if((imm&0x800) == 0x800){
                imm = imm |(0xFFFFFFFFFFFFF000);
            }
            mem_addr = imm + this -> regs ->getValue(rs1);

            data = static_cast<std::uint32_t>(this->mem_intf->readDataMem(mem_addr, 4));
            std::uint64_t aux = static_cast<std::uint32_t>(this->mem_intf->readDataMem(mem_addr + 4, 4));
            data |= aux << 32;

            this->perf->dataMemoryRead();
            this->regs->setfValue(rd, data);
            this->perf->cycleInc(FLD_CYC);

            return;
        };
        void EXEC_F_FSD(){         
            unsigned int rs1, rs2;
            signed_T imm;
            std::int64_t data32, data64;
            unsigned_T mem_addr32, mem_addr64;

            rs1 = this -> get_rs1();
            rs2 = this -> get_rs2();
            imm = ((this -> get_imm7()) << 5) | this -> get_rd();
            if((imm&0x800) == 0x800){
                imm = imm |(0xFFFFFFFFFFFFF000);
            }
            mem_addr32 = imm + this->regs->getValue(rs1);
            data32 = static_cast<std::int64_t>(this->regs->getfValue(rs2));

            this->perf->dataMemoryWrite();
            this->mem_intf->writeDataMem(mem_addr32,data32,4);

            mem_addr64 = 0x4+mem_addr32;
            data64 = data32>>32;

            this->perf->dataMemoryWrite();
            this->mem_intf->writeDataMem(mem_addr64,data64,4);
            this->perf->cycleInc(FSD_CYC);
            return;
        };
        void EXEC_F_FCVT_L_S(){     // FIXME
            unsigned int rs1,rd;
            BitsToFloat fs1;
            
            int64_t xd;
            int64_t min = 0x8000000000000000;
            int64_t max = 0x7FFFFFFFFFFFFFFF;
            
            unsigned int classify;
            unsigned int fflags = 0;
            
            rs1 = this->get_rs1();
            rd = this->get_rd();
            fs1.u32 = this->regs->getfValue(rs1);
            
            // Classify
            // 0 bit // rs1 is -infinity
            // 1 bit // rs1 is a negative normal number.
            // 2 bit // rs1 is a negative subnormal number.
            // 3 bit // rs1 is −0.
            // 4 bit // rs1 is +0.
            // 5 bit // rs1 is a positive subnormal number.
            // 6 bit // rs1 is a positive normal number.
            // 7 bit // rs1 is +infinity
            // 8 bit // rs1 is a signaling NaN.
            // 9 bit // rs1 is a quiet NaN.
            
            classify = classify_fp32(fs1.u32);
            
            if ((((classify == (1<<1)) | (classify == (1<<2))) & (fs1.f < min)) | (classify == (1<<0))) {     // min, -Inf
                feclearexcept(FE_ALL_EXCEPT);   // reset exception flags
                xd = fs1.f;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd = min;
            }
            else if ((fs1.f > max) | (classify == (1<<9)) | (classify == (1<<8))) {     // max, +Inf, NaN
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.f;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd = max;
            }
            else {
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.f;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
            }

            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setValue(rd,(xd));
        };
        void EXEC_F_FCVT_LU_S(){    // FIXME
            unsigned int rs1,rd;
            BitsToFloat fs1;
            
            uint64_t xd;
            uint64_t min = 0x0000000000000000;
            uint64_t max = 0xFFFFFFFFFFFFFFFF;
            
            unsigned int classify;
            unsigned int fflags = 0;
            
            rs1 = this->get_rs1();
            rd = this->get_rd();
            fs1.u32=this->regs->getfValue(rs1);
            
            // Classify
            // 0 bit // rs1 is -infinity
            // 1 bit // rs1 is a negative normal number.
            // 2 bit // rs1 is a negative subnormal number.
            // 3 bit // rs1 is −0.
            // 4 bit // rs1 is +0.
            // 5 bit // rs1 is a positive subnormal number.
            // 6 bit // rs1 is a positive normal number.
            // 7 bit // rs1 is +infinity
            // 8 bit // rs1 is a signaling NaN.
            // 9 bit // rs1 is a quiet NaN.

            classify = classify_fp32(fs1.u32);
            
            if ((((classify == (1<<1)) | (classify == (1<<2))) & (fs1.f < min)) | (classify == (1<<0))) {  // min, -Inf
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.f;
                
                if (fs1.f <= -1.0)              fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd = min;
            }
            else if ((fs1.f > max) | (classify == (1<<9)) | (classify == (1<<8))) {     // max, +Inf, NaN
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.f;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd = max;
            }
            else {
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.f;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
            }
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setValue(rd,(xd));
        };
        void EXEC_F_FCVT_S_L(){     // FIXME
            unsigned int rs1,rd;
            int64_t i64;
            BitsToFloat fs1;

            rs1 = this->get_rs1();
            rd = this->get_rd();

            i64 = this->regs->getValue(rs1);
            fs1.f = i64;


            this->regs->setfValue(rd,(fs1.u32));
        };
        void EXEC_F_FCVT_S_LU(){    // FIXME
            unsigned int rs1,rd;
            uint64_t ui64;
            BitsToFloat fs1;

            rs1 = this->get_rs1();
            rd = this->get_rd();

            ui64 = this->regs->getValue(rs1);
            fs1.f = ui64;


            this->regs->setfValue(rd,(fs1.u32));
        };
        void EXEC_F_FCVT_L_D(){     // FIXME
            unsigned int rs1,rd;
            BitsToDouble fs1;
            
            int64_t xd;
            int64_t min = 0x8000000000000000;
            int64_t max = 0x7FFFFFFFFFFFFFFF;
            
            unsigned int classify;
            unsigned int fflags = 0;
            
            rs1 = this->get_rs1();
            rd = this->get_rd();
            fs1.u64 = this->regs->getfValue(rs1);
            
            classify = classify_fp64(fs1.u64);
            
            if ((((classify == (1<<1)) | (classify == (1<<2))) & (fs1.d < min)) | (classify == (1<<0))) {      // min, -Inf
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.d;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd = min;
            }
            else if ((fs1.d > max) | (classify == (1<<9)) | (classify == (1<<8))) {     // max, +Inf, NaN
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.d;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd = max;
            }
            else {
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.d;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
            }
    
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setValue(rd,(xd));
        };
        void EXEC_F_FCVT_LU_D(){    // FIXME
            unsigned int rs1,rd;
            BitsToDouble fs1;
            
            uint64_t xd;
            uint64_t min = 0x0000000000000000;
            uint64_t max = 0xFFFFFFFFFFFFFFFF;
            
            unsigned int classify;
            unsigned int fflags = 0;
            
            rs1 = this->get_rs1();
            rd = this->get_rd();
            fs1.u64 = this->regs->getfValue(rs1);
            
            classify = classify_fp64(fs1.u64);
            
            if ((((classify == (1<<1)) | (classify == (1<<2))) & (fs1.d < min)) | (classify == (1<<0))) {      // min, -Inf
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.d;
                if (fs1.d <= -1.0)   fflags |= (1 << 4);                // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd = min;
            }
            else if ((fs1.d > max) | (classify == (1<<9)) | (classify == (1<<8))) {     // max, +Inf, NaN
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.d;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
                xd = max;
            }
            else {
                feclearexcept(FE_ALL_EXCEPT); // reset exception flags
                xd = fs1.d;
                if (fetestexcept(FE_INVALID))   fflags |= (1 << 4);      // NV
                if (fetestexcept(FE_DIVBYZERO)) fflags |= (1 << 3);      // DZ
                if (fetestexcept(FE_OVERFLOW))  fflags |= (1 << 2);      // OF
                if (fetestexcept(FE_UNDERFLOW)) fflags |= (1 << 1);      // UF
                if (fetestexcept(FE_INEXACT))   fflags |= (1 << 0);      // NX
            }
    
            this->regs->setCSR(CSR_ADDR_FFLAGS, fflags);
            this->regs->setValue(rd,(xd));
        };
        void EXEC_F_FMV_X_D(){      // FIXME
            unsigned int rs1, rd;
            BitsToDouble fs1, fd;
            
            rs1 = this->get_rs1();
            rd = this->get_rd();
            
            fs1.u64 = static_cast<std::uint64_t>(this->regs->getfValue(rs1));
            fd.i = static_cast<std::int64_t>(fs1.u64);
            
            this->regs->setValue(rd, (fd.i));
        };
        void EXEC_F_FCVT_D_L(){     // FIXME
            unsigned int rs1,rd;
            int64_t i64;
            BitsToDouble fs1;

            rs1 = this->get_rs1();
            rd = this->get_rd();
            i64 = this->regs->getValue(rs1);
            fs1.d = i64;


            this->regs->setfValue(rd,(fs1.u64));
        };
        void EXEC_F_FCVT_D_LU(){    // FIXME
            unsigned int rs1,rd;
            uint64_t ui64;
            BitsToDouble fs1;

            rd = this->get_rd();
            rs1 = this->get_rs1();

            ui64 = this->regs->getValue(rs1);
            fs1.d = ui64;


            this->regs->setfValue(rd,(fs1.u64));
        };
        void EXEC_F_FMV_D_X(){      // FIXME
            unsigned int rs1, rd;
            BitsToDouble fs1, fd;

            rs1 = this->get_rs1();
            rd = this->get_rd();
            
            fs1.u64 = static_cast<std::uint64_t>(this->regs->getValue(rs1));
            this->regs->setfValue(rd, (fs1.u64));

        };
        bool exec_instruction(Instruction &inst, op_F_Codes code){
            bool PC_not_affected = true;

            this -> setInstr(inst.getInstr());
            switch(code){
                case OP_F_FMADD_S:
                    Exec_F_FMADD_S();
                    break;
                case OP_F_FMSUB_S:
                    EXEC_F_FMSUB_S();
                    break;
                case OP_F_FNMADD_S:
                    EXEC_F_FNMADD_S();
                    break;
                case OP_F_FNMSUB_S:
                    EXEC_F_FNMSUB_S();
                    break;
                case OP_F_FADD_S:
                    EXEC_F_FADD_S();
                    break;
                case OP_F_FSUB_S:
                    EXEC_F_FSUB_S();
                    break;
                case OP_F_FMUL_S:
                    EXEC_F_FMUL_S();
                    break;
                case OP_F_FDIV_S:
                    EXEC_F_FDIV_S();
                    break;
                case OP_F_FSQRT_S:
                    EXEC_F_FSQRT_S();
                    break;
                case OP_F_FSGNJ_S:
                    EXEC_F_FSGNJ_S();
                    break;
                case OP_F_FSGNJN_S:
                    EXEC_F_FSGNJN_S();
                    break;
                case OP_F_FSGNJX_S:
                    EXEC_F_FSGNJX_S();
                    break;
                case OP_F_FMIN_S:
                    EXEC_F_FMIN_S();
                    break;
                case OP_F_FMAX_S:
                    EXEC_F_FMAX_S();
                    break;
                case OP_F_FCVT_W_S:
                    EXEC_F_FCVT_W_S();
                    break;
                case OP_F_FCVT_WU_S:
                    EXEC_F_FCVT_WU_S();
                    break;
                case OP_F_FMV_X_W:
                    EXEC_F_FMV_X_W();
                    break;
                case OP_F_FEQ_S:
                    EXEC_F_FEQ_S();
                    break;
                case OP_F_FLT_S:
                    EXEC_F_FLT_S();
                    break;
                case OP_F_FLE_S:
                    EXEC_F_FLE_S();
                    break;
                case OP_F_FCLASS_S:
                    EXEC_F_FCLASS_S();
                    break;
                case OP_F_FCVT_S_W:
                    EXEC_F_FCVT_S_W();
                    break;
                case OP_F_FCVT_S_WU:
                    EXEC_F_FCVT_S_WU();
                    break;
                case OP_F_FMV_W_X:
                    EXEC_F_FMV_W_X();
                    break;
                case OP_F_FMADD_D:
                    EXEC_F_FMADD_D();
                    break;
                case OP_F_FMSUB_D:
                    EXEC_F_FMSUB_D();
                    break;
                case OP_F_FNMSUB_D:
                    EXEC_F_FNMSUB_D();
                    break;
                case OP_F_FNMADD_D:
                    EXEC_F_FNMADD_D();
                    break;
                case OP_F_FADD_D:
                    EXEC_F_FADD_D();
                    break;
                case OP_F_FSUB_D:
                    EXEC_F_FSUB_D();
                    break;
                case OP_F_FMUL_D:
                    EXEC_F_FMUL_D();
                    break;
                case OP_F_FDIV_D:
                    EXEC_F_FDIV_D();
                    break;
                case OP_F_FSQRT_D:
                    EXEC_F_FSQRT_D();
                    break;
                case OP_F_FSGNJ_D:
                    EXEC_F_FSGNJ_D();
                    break;
                case OP_F_FSGNJN_D:
                    EXEC_F_FSGNJN_D();
                    break;
                case OP_F_FSGNJX_D:
                    EXEC_F_FSGNJX_D();
                    break;
                case OP_F_FMIN_D:
                    EXEC_F_FMIN_D();
                    break;
                case OP_F_FMAX_D:
                    EXEC_F_FMAX_D();
                    break;
                case OP_F_FCVT_S_D:
                    EXEC_F_FCVT_S_D();
                    break;
                case OP_F_FCVT_D_S:
                    EXEC_F_FCVT_D_S();
                    break;
                case OP_F_FEQ_D:
                    EXEC_F_FEQ_D();
                    break;
                case OP_F_FLT_D:
                    EXEC_F_FLT_D();
                    break;
                case OP_F_FLE_D:
                    EXEC_F_FLE_D();
                    break;
                case OP_F_FCLASS_D:
                    EXEC_F_FCLASS_D();
                    break;
                case OP_F_FCVT_W_D:
                    EXEC_F_FCVT_W_D();
                    break;
                case OP_F_FCVT_WU_D:
                    EXEC_F_FCVT_WU_D();
                    break;
                case OP_F_FCVT_D_W:
                    EXEC_F_FCVT_D_W();
                    break;
                case OP_F_FCVT_D_WU:
                    EXEC_F_FCVT_D_WU();
                    break;
                case OP_F_FLW:
                    EXEC_F_FLW();
                    break;
                case OP_F_FSW:
                    EXEC_F_FSW();
                    break;
                case OP_F_FLD:
                    EXEC_F_FLD();
                    break;
                case OP_F_FSD:
                    EXEC_F_FSD();
                    break;
                case OP_F_FCVT_L_S:
                    EXEC_F_FCVT_L_S();
                    break;
                case OP_F_FCVT_LU_S:
                    EXEC_F_FCVT_LU_S();
                    break;
                case OP_F_FCVT_S_L:
                    EXEC_F_FCVT_S_L();
                    break;
                case OP_F_FCVT_S_LU:
                    EXEC_F_FCVT_S_LU();
                    break;
                case OP_F_FCVT_L_D:
                    EXEC_F_FCVT_L_D();
                    break;
                case OP_F_FCVT_LU_D:
                    EXEC_F_FCVT_LU_D();
                    break;
                case OP_F_FMV_X_D:
                    EXEC_F_FMV_X_D();
                    break;
                case OP_F_FCVT_D_L:
                    EXEC_F_FCVT_D_L();
                    break;
                case OP_F_FCVT_D_LU:
                    EXEC_F_FCVT_D_LU();
                    break;
                case OP_F_FMV_D_X:
                    EXEC_F_FMV_D_X();
                    break;
                default:
                    return PC_not_affected;
                    break;
            }
            return PC_not_affected;
        }
    };
}

#endif
