

#ifndef GTX_EXTENSION__H
#define GTX_EXTENSION__H
#define NSU_NUM 1
#include <queue>
#include "systemc"

#include "extension_base.h"
#include "float_conv.h"
#include "NSU.h"
#include "SPU.h"
#include "Registers.h"
#include "Instrcycle_info.h"
namespace riscv_tlm {



    typedef enum {
//matrix Multiplication
        MM = 0b0001011,
        MM_S = 0b0001011,
        MM_O = 0b0001011,
        MM_V = 0b0001011,
        MM_T = 0b0001011,
        MMC = 0b0001011,
        MMC_S = 0b0001011,
        MMC_O = 0b0001011,
        MMC_V = 0b0001011,
        MMC_T = 0b0001011,
        //mm_func3
        MM_F = 0b010,
        MM_S_F = 0b000,
        MM_O_F = 0b001,
        MM_V_F = 0b011,
        MM_T_F = 0b111,
        MMC_F = 0b010,
        MMC_S_F = 0b000,
        MMC_O_F = 0b001,
        MMC_V_F = 0b011,
        MMC_T_F = 0b111,
        //mm func7
        MM_F7 = 0b0000000,
        MM_S_F7 = 0b0000000,
        MM_O_F7 = 0b0000000,
        MM_V_F7 = 0b0000000,
        MM_T_F7 = 0b0000000,
        MMC_F7 = 0b0000001,
        MMC_S_F7 = 0b0000001,
        MMC_O_F7 = 0b0000001,
        MMC_V_F7 = 0b0000001,
        MMC_T_F7 = 0b0000001,
        //convolution
        //CONV_NM = 0b0001011,
        //CONV_DW = 0b0001011,
        //CONV_DS = 0b0001011,
        IM2COL_N = 0b0001011,
        IM2COL_D = 0b0001011,
        //conv func3
        //CONV_NM_F = 0b000,
        //CONV_DW_F = 0b000,
        //CONV_DS_F = 0b000,
        IM2COL_N_F = 0b000,
        IM2COL_D_F = 0b000,
        //conv func7
        //CONV_NM_F7 = 0b0001000,
        //CONV_DW_F7 = 0b0001001,
        //CONV_DS_F7 = 0b0001010,
        IM2COL_N_F7 = 0b0001000,
        IM2COL_D_F7 = 0b0001001,
        //scalar calculation
        ADD_VS = 0b0001011,
        SUB_VS = 0b0001011,
        MUL_VS = 0b0001011,
        DIV_VS = 0b0001011,
        FMADD_VS = 0b0001011,
        MAX_VS = 0b0001011,
        MIN_VS = 0b0001011,
        ADD_IS = 0b0001011,
        SUB_IS = 0b0001011,
        MUL_IS = 0b0001011,
        DIV_IS = 0b0001011,
        FMADD_IS = 0b0001011,
        MAX_IS = 0b0001011,
        MIN_IS = 0b0001011,
        //scalar func3
        ADD_VS_F = 0b000,
        SUB_VS_F = 0b001,
        MUL_VS_F = 0b010,
        DIV_VS_F = 0b011,
        FMADD_VS_F = 0b000,
        MAX_VS_F = 0b000,
        MIN_VS_F = 0b001,
        ADD_IS_F = 0b100,
        SUB_IS_F = 0b101,
        MUL_IS_F = 0b110,
        DIV_IS_F = 0b111,
        FMADD_IS_F = 0b100,
        MAX_IS_F = 0b100,
        MIN_IS_F = 0b101,
        ADD_VS_F7 = 0b0010000,
        SUB_VS_F7 = 0b0010000,
        MUL_VS_F7 = 0b0010000,
        DIV_VS_F7 = 0b0010000,
        FMADD_VS_F7 = 0b0010001,
        MAX_VS_F7 = 0b0010011,
        MIN_VS_F7 = 0b0010011,
        ADD_IS_F7 = 0b0010000,
        SUB_IS_F7 = 0b0010000,
        MUL_IS_F7 = 0b0010000,
        DIV_IS_F7 = 0b0010000,
        FMADD_IS_F7 = 0b0010001,
        MAX_IS_F7 = 0b0010011,
        MIN_IS_F7 = 0b0010011,
        //vector calculation
        ADD_VV = 0b0001011,
        SUB_VV = 0b0001011,
        MUL_VV = 0b0001011,
        DIV_VV = 0b0001011,
        DOT_VVS = 0b0001011,
        FMADD_VVV = 0b0001011,
        SUM_VS  = 0b0001011,
        SQRT_V = 0b0001011,
        EXP_V = 0b0001011,
        LN_V = 0b0001011,
        ABS_V = 0b0001011,
        NEG_V = 0b0001011,
        SIGN_V = 0b0001011,
        STEP_V = 0b0001011,
        CEIL_V = 0b0001011,
        TRUNC_V = 0b0001011,
        FLOOR_V = 0b0001011,
        RNE_V = 0b0001011,
        CLAMP_MIN_V = 0b0001011,
        CLAMP_MAX_V = 0b0001011,
        ACCUM_V = 0b0001011,
        ARANGE_V = 0b0001011,
        ADD_II = 0b0001011,
        SUB_II = 0b0001011,
        MUL_II = 0b0001011,
        DIV_II = 0b0001011,
        DOT_IIS = 0b0001011,
        FMADD_III = 0b0001011,
        SUM_IS  = 0b0001011,
        SQRT_I = 0b0001011,
        EXP_I = 0b0001011,
        LN_I = 0b0001011,
        ABS_I = 0b0001011,
        NEG_I  = 0b0001011,
        SIGN_I = 0b0001011,
        STEP_I = 0b0001011,
        CEIL_I = 0b0001011,
        TRUNC_I = 0b0001011,
        FLOOR_I = 0b0001011,
        RNE_I = 0b0001011,
        AND_II = 0b0001011,
        OR_II = 0b0001011,
        NOT_I = 0b0001011,
        SHIFT_I = 0b0001011,
        //vector func3
        ADD_VV_F = 0b000,
        SUB_VV_F = 0b001,
        MUL_VV_F = 0b010,
        DIV_VV_F = 0b011,
        DOT_VVS_F = 0b000,
        FMADD_VVV_F = 0b000,
        SUM_VS_F  = 0b001,
        SQRT_V_F = 0b000,
        EXP_V_F = 0b001,
        LN_V_F = 0b010,
        ABS_V_F = 0b000,
        NEG_V_F = 0b001,
        SIGN_V_F = 0b010,
        STEP_V_F = 0b011,
        CEIL_V_F = 0b000,
        TRUNC_V_F = 0b001,
        FLOOR_V_F = 0b010,
        RNE_V_F = 0b011,
        CLAMP_MIN_F = 0b000,
        CLAMP_MAX_F = 0b001,
        ACCUM_V_F = 0b010,
        ARANGE_V_F = 0b011,
        ADD_II_F = 0b100,
        SUB_II_F = 0b101,
        MUL_II_F = 0b110,
        DIV_II_F = 0b111,
        DOT_IIS_F = 0b100,
        FMADD_III_F = 0b100,
        SUM_IS_F = 0b101,
        SQRT_I_F = 0b100,
        EXP_I_F = 0b101,
        LN_I_F = 0b110,
        ABS_I_F = 0b100,
        NEG_I_F = 0b101,
        SIGN_I_F = 0b110,
        STEP_I_F = 0b111,
        CEIL_I_F = 0b100,
        TRUNC_I_F = 0b101,
        FLOOR_I_F = 0b110,
        RNE_I_F = 0b111,
        AND_II_F = 0b100,
        OR_II_F = 0b101,
        NOT_I_F = 0b110,
        SHIFT_I_F = 0b111,
        //vector func7
        ADD_VV_F7 = 0b0011000,
        SUB_VV_F7 = 0b0011000,
        MUL_VV_F7 = 0b0011000,
        DIV_VV_F7 = 0b0011000,
        DOT_VVS_F7 = 0b0011010,
        FMADD_VVV_F7 = 0b0011001,
        SUM_VS_F7  = 0b0011010,
        SQRT_V_F7 = 0b0011100,
        EXP_V_F7  = 0b0011100,
        LN_V_F7 = 0b0011100,
        ABS_V_F7 = 0b0011101,
        NEG_V_F7 = 0b0011101,
        SIGN_V_F7 = 0b0011101,
        STEP_V_F7 = 0b0011101,
        CEIL_V_F7 = 0b0011110,
        TRUNC_V_F7 = 0b0011110,
        FLOOR_V_F7 = 0b0011110,
        RNE_V_F7 = 0b0011110,
        CLAMP_MIN_F7 = 0b0011111,
        CLAMP_MAX_F7 = 0b0011111,
        ACCUM_V_F7 = 0b0011111,
        ARANGE_V_F7 = 0b0011111,
        ADD_II_F7 = 0b0011000,
        SUB_II_F7 = 0b0011000,
        MUL_II_F7 = 0b0011000,
        DIV_II_F7 = 0b0011000,
        DOT_IIS_F7 = 0b0011010,
        FMADD_III_F7 = 0b0011001,
        SUM_IS_F7  = 0b0011010,
        SQRT_I_F7 = 0b0011100,
        EXP_I_F7 = 0b0011100,
        LN_I_F7 = 0b0011100,
        ABS_I_F7 = 0b0011101,
        NEG_I_F7 = 0b0011101,
        SIGN_I_F7 = 0b0011101,
        STEP_I_F7 = 0b0011101,
        CEIL_I_F7 = 0b0011110,
        TRUNC_I_F7 = 0b0011110,
        FLOOR_I_F7 = 0b0011110,
        RNE_I_F7 = 0b0011110,
        AND_I_F7 = 0b0011111,
        OR_I_F7 = 0b0011111,
        NOT_I_F7 = 0b0011111,
        SHIFT_I_F7 = 0b0011111,
        //Format conversion
        SCVT_QH = 0b0001011,
        SCVT_HQ = 0b0001011,
        SCVT_IH = 0b0001011,
        SCVT_HI = 0b0001011,
        SCVT_HN = 0b0001011,
        FCVT_SH = 0b0001011,
        FCVT_HS = 0b0001011,
        FCVT_DH = 0b0001011,
        FCVT_HD = 0b0001011,
        //Format func3
        SCVT_QH_F = 0b000,
        SCVT_HQ_F = 0b001,
        SCVT_IH_F = 0b000,
        SCVT_HI_F = 0b001,
        SCVT_HN_F = 0b001,
        FCVT_SH_F = 0b000,
        FCVT_HS_F = 0b001,
        FCVT_DH_F = 0b000,
        FCVT_HD_F = 0b001,
        //Format func7
        SCVT_QH_F7 = 0b0100000,
        SCVT_HQ_F7 = 0b0100000,
        SCVT_IH_F7 = 0b0100001,
        SCVT_HI_F7 = 0b0100001,
        SCVT_HN_F7 = 0b0100010,
        FCVT_SH_F7 = 0b0100100,
        FCVT_HS_F7 = 0b0100100,
        FCVT_DH_F7 = 0b0100101,
        FCVT_HD_F7 = 0b0100101,
        //Activation
        //RELU = 0b0001011,
        //RELU6 = 0b0001011,
        //LRELU = 0b0001011,
        PRELU = 0b0001011,
        GELU = 0b0001011,
        TANH = 0b0001011,
        SIGM = 0b0001011,
        LRELU_IMM = 0b0001011,
        PRELU_IMM = 0b0001011,
        GELU_IMM = 0b0001011,
        TANH_IMM = 0b0001011,
        SIGM_IMM = 0b0001011,
        //Activation func3
        //RELU_F = 0b000,
        //RELU6_F = 0b001,
        //LRELU_F = 0b010,
        PRELU_F = 0b011,
        GELU_F = 0b000,
        TANH_F = 0b000,
        SIGM_F = 0b000,
        //RELU_I_F = 0b100,
        //RELU6_I_F = 0b101,
        //LRELU_I_F = 0b110,
        PRELU_I_F = 0b111,
        GELU_I_F = 0b100,
        TANH_I_F = 0b100,
        SIGM_I_F = 0b100,
        //Activation func7
        //RELU_F7 = 0b0101000,
        //RELU6_F7 = 0b0101000,
        //LRELU_F7 = 0b0101000,
        PRELU_F7 = 0b0101000,
        GELU_F7 = 0b0101010,
        TANH_F7 = 0b0101100,
        SIGM_F7 = 0b0101101,
        //RELU_I_F7 = 0b0101000,
        //RELU6_I_F7 = 0b0101000,
        //LRELU_I_F7 = 0b0101000,
        PRELU_I_F7 = 0b0101000,
        GELU_I_F7 = 0b0101010,
        TANH_I_F7 = 0b0101100,
        SIGM_I_F7 = 0b0101101,
        //Softmax
        ESUM = 0b0001011,
        SOFTMAX = 0b0001011,
        ESUM_I = 0b0001011,
        SOFTMAX_I = 0b0001011,
        //Softmax func3
        ESUM_F = 0b001,
        SOFTMAX_F = 0b010,
        ESUM_I_F = 0b101,
        SOFTMAX_I_F = 0b110,
        //Softmax func7
        ESUM_F7 = 0b0101111,
        SOFTMAX_F7 = 0b0101111,
        ESUM_I_F7 = 0b0101111,
        SOFTMAX_I_F7 = 0b0101111,
        //Pooling
        POOL_M = 0b0001011,
        POOL_A = 0b0001011,
        //POOL_MP = 0b0001011,
        //POOL_AP = 0b0001011,
        //Pooling func3
        POOL_M_F = 0b000,
        POOL_A_F = 0b000,
        //POOL_MP_F = 0b001,
        //POOL_AP_F = 0b001,
        //Pooling func7
        POOL_M_F7 = 0b0110000,
        POOL_A_F7 = 0b0110001,
        //POOL_MP_F7 = 0b0110000,
        //POOL_AP_F7 = 0b0110001,
        //DMA
        LOAD = 0b0001011,
        LOAD_3D =0b0001011,
        STORE = 0b0001011,
        STORE_3D= 0b0001011,
        COPY = 0b0001011,
        LOAD_SVR = 0b0001011,
        STORE_SVR = 0b0001011,
        MCAST_S2L = 0b0001011,
        MCAST_G2S = 0b0001011,
        MCAST_S2S = 0b0001011,
        COPY_MEM = 0b0001011,
        //DMA func3
        LOAD_F = 0b000,
        STORE_F = 0b001,
        COPY_F = 0b010,
        LOAD_SVR_F = 0b000,
        STORE_SVR_F = 0b001,
        LOAD_3D_F = 0b100,
        STORE_3D_F = 0b101,
        MCAST_S2L_F = 0b000,
        MCAST_G2S_F = 0b000,
        MCAST_S2S_F = 0b010,
        COPY_MEM_F = 0b011,
        //DMA func7
        LOAD_F7 = 0b1000000,
        STORE_F7 = 0b1000000,
        COPY_F7 = 0b1000000,
        LOAD_3D_F7 = 0b1000001,
        STORE_3D_F7 = 0b1000001,
        LOAD_SVR_F7 = 0b1000001,
        STORE_SVR_F7 = 0b1000001,
        MCAST_S2L_F7 = 0b1000010,
        MCAST_G2S_F7 = 0b1000100,
        MCAST_S2S_F7 = 0b1000100,
        COPY_MEM_F7 = 0b1000100,
        //SPR
        RDSPR = 0b0001011,
        WRSPR = 0b0001011,
        OPSET = 0b0001011,
        CPSVR = 0b0001011,
        MVSVR = 0b0001011,
        //SPR func3
        RDSPR_F = 0b000,
        WRSPR_F = 0b000,
        OPSET_F = 0b000,
        CPSVR_F = 0b000,
        MVSVR_F = 0b000,
        //SPR func7
        RDSPR_F7 = 0b1001000,
        WRSPR_F7 = 0b1001001,
        OPSET_F7 = 0b1001010,
        CPSVR_F7 = 0b1001011,
        MVSVR_F7 = 0b1001100,

        //CREDIT
        CREDIT_LD = 0b0001011,
        CREDIT_LD_CHK = 0b0001011,
        CREDIT_ST = 0b0001011,
        CREDIT_ST_CHK = 0b0001011,
        //CREDIT func3
        CREDIT_LD_F = 0b000,
        CREDIT_LD_CHK_F = 0b000,
        CREDIT_ST_F = 0b000,
        CREDIT_ST_CHK_F = 0b000,
        //CREDIT func7
        CREDIT_LD_F7 = 0b1010000,
        CREDIT_LD_CHK_F7 = 0b1010010,
        CREDIT_ST_F7 = 0b1010001,
        CREDIT_ST_CHK_F7 = 0b1010011,

        //memory operation
        TPOSE = 0b0001011,
        FILL = 0b0001011,
        //memory func3
        TPOSE_F = 0b000,
        FILL_F = 0b000,
        //memory func3
        TPOSE_F7 = 0b0111000,
        FILL_F7 = 0b0111001,
        //mcrun
        MEXEC = 0b0001011,
        //mcrun func3
        MEXEC_F = 0b000,
        //mcrn func7
        MEXEC_F7 = 0b1110000,
        //mbar = nop in ISS
        MBAR = 0b0001011,
        MBAR_F = 0b000,
        MBAR_F7 = 0b1110100,

        //
        MSYNC = 0b0001011,
        MSYNC_F = 0b000,
        MSYNC_F7 = 0b1110101,

        EOM = 0b0001011,
        EOM_F = 0b000,
        EOM_F7 = 0b1110111,

        //Sync
        BAR = 0b0001011,
        WAIT = 0b0001011,
        HALT = 0b0001011,
        INTR = 0b0001011,
        FLUSH = 0b0001011,
        //Sync func3
        BAR_F = 0b000,
        WAIT_F = 0b000,
        HALT_F = 0b000,
        INTR_F = 0b000,
        FLUSH_F = 0b000,
        //Sync func7
        BAR_F7 = 0b1111000,
        WAIT_F7 = 0b1111001,
        HALT_F7 = 0b1111111,
        INTR_F7 = 0b1111011,
        FLUSH_F7 = 0b1111100,
        //Warp operation
        START_T = 0b0101011,
        END_T = 0b0101011,
        START_S = 0b0101011,
        END_S = 0b0101011,
        START_P = 0b0101011,
        END_P = 0b0101011,
        SPLIT = 0b0101011,
        JOIN = 0b0101011,
        //Warp func3
        START_T_F = 0b000,
        END_T_F = 0b001,
        START_S_F = 0b010,
        END_S_F = 0b011,
        START_P_F = 0b110,
        END_P_F = 0b111,
        SPLIT_F = 0b100,
        JOIN_F = 0b101
    
    } GTX_Codes;

    
    template<typename T>
    class GTX_extension : public extension_base<T> {
    public:
        using extension_base<T>::extension_base;

        using signed_T = typename std::make_signed<T>::type;
        using unsigned_T = typename std::make_unsigned<T>::type;

        GTX_extension(const T &instr, Registers<T> *register_bank,
                       CPUMemintf *mem_interface, NSU *nsu_i) :
                extension_base<T>(instr, register_bank, mem_interface), nsu(nsu_i) {
            is_loop = false;

            set_command = false;


        }
        
        //[[nodiscard]] inline std::uint32_t get_funct7() const override {
        //    return static_cast<std::uint32_t>(this->m_instr.range(31, 25));
        //}
        // [[nodiscard]] inline std::uint32_t get_funct3() const override {
        //    return static_cast<std::uint32_t>(this->m_instr.range(14, 12));
        //}
        inline std::uint32_t get_funct7() const {
            return this->m_instr.range(31, 25);
        }
        
        [[nodiscard]] inline std::uint32_t get_rs2() const override {
            return static_cast<std::uint32_t>(this->m_instr.range(24, 20));
        }
        [[nodiscard]] inline std::uint32_t get_rs1() const override {
            return static_cast<std::uint32_t>(this->m_instr.range(19, 15));
        }
        [[nodiscard]] inline std::uint32_t get_rd() const override {
            return static_cast<std::uint32_t>(this->m_instr.range(11, 7));
        }
        signed_T get_imm_I() const;
        inline unsigned_T opcode() const  {
            return this->m_instr.range(6,0);
        }

        inline std::uint32_t get_imm() const{
            return this -> m_instr.range(31,20);
        }

        void set_monitor(bool enable, uint32_t warptime){
            this -> monitoring = enable;
            this -> warp_time = warptime;
        }

        [[nodiscard]] op_GTX_Codes decode() const{    
            switch(opcode()){
                case 0b0001011 :
                    switch(get_funct7()){
                        case MM_F7 :
                            switch(this->get_funct3()){
                                case MM_F :
                                    return OP_GTX_MM;
                                    break;
                                case MM_S_F :
                                    return OP_GTX_MM_S;
                                    break;
                                case MM_O_F :
                                    return OP_GTX_MM_O;
                                    break;
                                case MM_V_F :
                                    return OP_GTX_MM_V;
                                    break;
                                case MM_T_F :
                                    return OP_GTX_MM_T;
                                    break;
                                default :
                                    return OP_GTX_ERROR;
                                    break;
                            }
                            break;
                        case MMC_F7 :
                            switch(this->get_funct3()){
                                case MMC_F :
                                    return OP_GTX_MMC;
                                    break;
                                case MMC_S_F:
                                    return OP_GTX_MMC_S;
                                    break;
                                case MMC_O_F:
                                    return OP_GTX_MMC_O;
                                    break;
                                case MMC_V_F:
                                    return OP_GTX_MMC_V;
                                    break;
                                case MMC_T_F:
                                    return OP_GTX_MMC_T;
                                    break;
                                default :
                                    return OP_GTX_ERROR;
                                    break;
                            }
                            break;
                        case IM2COL_N_F7:
                                if(this->get_funct3() == IM2COL_N_F){
                                    return OP_GTX_IM2COL_N;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case IM2COL_D_F7:
                                if(this->get_funct3() == IM2COL_D_F){
                                    return OP_GTX_IM2COL_D;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case ADD_VS_F7:
                                switch(this->get_funct3()){
                                    case ADD_VS_F:
                                        return OP_GTX_ADD_VS;
                                        break;
                                    case ADD_IS_F:
                                        return OP_GTX_ADD_IS;
                                        break;
                                    case SUB_VS_F:
                                        return OP_GTX_SUB_VS;
                                        break;
                                    case SUB_IS_F:
                                        return OP_GTX_SUB_IS;
                                        break;
                                    case MUL_VS_F:
                                        return OP_GTX_MUL_VS;
                                        break;
                                    case MUL_IS_F:
                                        return OP_GTX_MUL_IS;
                                        break;
                                    case DIV_VS_F:
                                        return OP_GTX_DIV_VS;
                                        break;
                                    case DIV_IS_F:
                                        return OP_GTX_DIV_IS;
                                        break;
                                }
                                break;
                        case FMADD_VS_F7:
                                switch(this->get_funct3()){
                                    case FMADD_VS_F:
                                        return OP_GTX_FMADD_VS;
                                        break;
                                    case FMADD_IS_F:
                                        return OP_GTX_FMADD_IS;
                                        break;
                                }
                                break;
                        case MAX_VS_F7:
                                switch(this->get_funct3()){
                                    case MAX_VS_F:
                                        return OP_GTX_MAX_VS;
                                        break;
                                    case MAX_IS_F:
                                        return OP_GTX_MAX_IS;
                                        break;
                                    case MIN_VS_F:
                                        return OP_GTX_MIN_VS;
                                        break;
                                    case MIN_IS_F:
                                        return OP_GTX_MIN_IS;
                                        break;
                                }
                                break;

                        case ADD_VV_F7:
                                switch(this->get_funct3()){
                                    case ADD_VV_F:
                                        return OP_GTX_ADD_VV;
                                        break;
                                    case ADD_II_F:
                                        return OP_GTX_ADD_II;
                                        break;
                                    case SUB_VV_F:
                                        return OP_GTX_SUB_VV;
                                        break;
                                    case SUB_II_F:
                                        return OP_GTX_SUB_II;
                                        break;
                                    case MUL_VV_F:
                                        return OP_GTX_MUL_VV;
                                        break;
                                    case MUL_II_F:
                                        return OP_GTX_MUL_II;
                                        break;
                                    case DIV_VV_F:
                                        return OP_GTX_DIV_VV;
                                        break;
                                    case DIV_II_F:
                                        return OP_GTX_DIV_II;
                                        break;
                                }
                                break;

                        case DOT_VVS_F7:
                                switch(this->get_funct3()){
                                    case DOT_VVS_F:
                                        return OP_GTX_DOT_VVS;
                                        break;
                                    case DOT_IIS_F:
                                        return OP_GTX_DOT_IIS;
                                        break;
                                    case SUM_VS_F:
                                        return OP_GTX_SUM_VS;
                                        break;
                                    case SUM_IS_F:
                                        return OP_GTX_SUM_IS;
                                        break;
                                }
                                break;
                        case FMADD_VVV_F7:
                                switch(this->get_funct3()){
                                    case FMADD_VVV_F:
                                        return OP_GTX_FMADD_VVV;
                                        break;
                                    case FMADD_III_F:
                                        return OP_GTX_FMADD_III;
                                        break;
                                }      
                        case SQRT_V_F7:
                                switch(this->get_funct3()){
                                    case SQRT_V_F:
                                        return OP_GTX_SQRT_V;
                                        break;
                                    case SQRT_I_F:
                                        return OP_GTX_SQRT_I;
                                        break;
                                    case EXP_V_F:
                                        return OP_GTX_EXP_V;
                                        break;
                                    case EXP_I_F:
                                        return OP_GTX_EXP_I;
                                        break;
                                    case LN_V_F:
                                        return OP_GTX_LN_V;
                                        break;
                                    case LN_I_F:
                                        return OP_GTX_LN_I;
                                        break;
                                }
                                break;
                        case ABS_V_F7:
                                switch(this->get_funct3()){
                                    case ABS_V_F:
                                        return OP_GTX_ABS_V;
                                        break;
                                    case ABS_I_F:
                                        return OP_GTX_ABS_I;
                                        break;
                                    case NEG_V_F:
                                        return OP_GTX_NEG_V;
                                        break;
                                    case NEG_I_F:
                                        return OP_GTX_NEG_I;
                                        break;
                                    case SIGN_V_F:
                                        return OP_GTX_SIGN_V;
                                        break;
                                    case SIGN_I_F:
                                        return OP_GTX_SIGN_I;
                                        break;
                                    case STEP_V_F:
                                        return OP_GTX_STEP_V;
                                        break;
                                    case STEP_I_F:
                                        return OP_GTX_STEP_I;
                                        break;
                                }
                        case CEIL_V_F7:
                                switch(this->get_funct3()){
                                    case CEIL_V_F:
                                        return OP_GTX_CEIL_V;
                                        break;
                                    case CEIL_I_F:
                                        return OP_GTX_CEIL_I;
                                        break;
                                    case TRUNC_V_F:
                                        return OP_GTX_TRUNC_V;
                                        break;
                                    case TRUNC_I_F:
                                        return OP_GTX_TRUNC_I;
                                        break;
                                    case FLOOR_V_F:
                                        return OP_GTX_FLOOR_V;
                                        break;
                                    case FLOOR_I_F:
                                        return OP_GTX_FLOOR_I;
                                        break;
                                    case RNE_V_F:
                                        return OP_GTX_RNE_V;
                                        break;
                                    case RNE_I_F:
                                        return OP_GTX_RNE_I;
                                        break;
                                }
                        case CLAMP_MIN_F7:
                                switch(this->get_funct3()){
                                    case CLAMP_MIN_F:
                                        return OP_GTX_CLAMP_MIN;
                                        break;
                                    case CLAMP_MAX_F:
                                        return OP_GTX_CLAMP_MAX;
                                        break;
                                    case ACCUM_V_F:
                                        return OP_GTX_ACCUM;
                                        break;
                                    case ARANGE_V_F:
                                        return OP_GTX_ARANGE;
                                        break;
                                    case AND_II_F:
                                        return OP_GTX_AND_II;
                                        break;
                                    case OR_II_F:
                                        return OP_GTX_OR_II;
                                        break;
                                    case NOT_I_F:
                                        return OP_GTX_NOT_I;
                                        break;
                                    case SHIFT_I_F:
                                        return OP_GTX_SHIFT_I;
                                        break;
                                }

                        case SCVT_QH_F7:
                                switch(this->get_funct3()){
                                case SCVT_QH_F :
                                    return OP_GTX_SCVT_QH;
                                    break;
                                case SCVT_HQ_F :
                                    return OP_GTX_SCVT_HQ;
                                    break;
                                default :
                                    return OP_GTX_ERROR;
                                    break;
                            }
                            break;
                        case SCVT_IH_F7:
                                switch(this->get_funct3()){
                                case SCVT_IH_F :
                                    return OP_GTX_SCVT_IH;
                                    break;
                                case SCVT_HI_F :
                                    return OP_GTX_SCVT_HI;
                                    break;
                                default:
                                    return OP_GTX_ERROR;
                                    break;
                            }
                        case SCVT_HN_F7:
                                switch(this->get_funct3()){
                                case SCVT_HN_F :
                                    return OP_GTX_SCVT_HN;
                                    break;
                                default:
                                    return OP_GTX_ERROR;
                                    break;
                            }           
                        case FCVT_SH_F7:
                                switch(this->get_funct3()){
                                case FCVT_SH_F :
                                    return OP_GTX_FCVT_SH;
                                    break;
                                case FCVT_HS_F :
                                    return OP_GTX_FCVT_HS;
                                    break;
                                default:
                                    return OP_GTX_ERROR;
                                    break;
                            }
                            break;
                        case FCVT_DH_F7:
                                switch(this->get_funct3()){
                                case FCVT_DH_F :
                                    return OP_GTX_FCVT_DH;
                                    break;
                                case FCVT_HD_F :
                                    return OP_GTX_FCVT_HD;
                                    break;
                                default:
                                    return OP_GTX_ERROR;
                                    break;
                            }
                            break;
                        case 0b0101000:
                                switch(this->get_funct3()){
                                //case RELU_F :
                                //    return OP_GTX_RELU;
                                //    break;
                                //case RELU6_F:
                                //    return OP_GTX_RELU6;
                                //    break;
                                //case LRELU_F:
                                //    return OP_GTX_LRELU;
                                //    break;
                                case PRELU_F:
                                    return OP_GTX_PRELU;
                                    break;
                                //case RELU_I_F:
                                //    return OP_GTX_RELU_I;
                                //    break;
                                //case RELU6_I_F:
                                //    return OP_GTX_RELU6_I;
                                //    break;
                                //case LRELU_I_F:
                                //    return OP_GTX_LRELU_I;
                                //    break;
                                case PRELU_I_F:
                                    return OP_GTX_PRELU_I;
                                    break;
                                default:
                                    return OP_GTX_ERROR;
                                    break;
                            }
                            break;
                        case GELU_F7:
                                switch(this->get_funct3()){
                                case GELU_F :
                                    return OP_GTX_GELU;
                                    break;
                                case GELU_I_F:
                                    return OP_GTX_GELU_I;
                                    break;
                                default:
                                    return OP_GTX_ERROR;
                                    break;
                            }
                            break;
                        case TANH_F7:
                                switch(this->get_funct3()){
                                case TANH_F :
                                    return OP_GTX_TANH;
                                    break;
                                case TANH_I_F:
                                    return OP_GTX_TANH_I;
                                    break;
                                default:
                                    return OP_GTX_ERROR;
                                    break;
                            }
                            break;
                        case SIGM_F7:
                                switch(this->get_funct3()){
                                case SIGM_F :
                                    return OP_GTX_SIGM;
                                    break;
                                case SIGM_I_F:
                                    return OP_GTX_SIGM_I;
                                    break;
                                default:
                                    return OP_GTX_ERROR;
                                    break;
                            }
                            break;
                        case 0b0101111:
                                switch(this->get_funct3()){
                                case ESUM_F :
                                    return OP_GTX_ESUM;
                                    break;
                                case SOFTMAX_F:
                                    return OP_GTX_SOFTMAX;
                                    break;
                                case ESUM_I_F:
                                    return OP_GTX_ESUM_I;
                                    break;
                                case SOFTMAX_I_F:
                                    return OP_GTX_SOFTMAX_I;
                                    break;
                                default:
                                    return OP_GTX_ERROR;
                                    break;
                            }
                            break;
                        case POOL_M_F7:
                                if(this->get_funct3() == POOL_M_F){
                                    return OP_GTX_POOL_M;
                                }//else if(this->get_funct3() == POOL_MP_F){
                                //    return OP_GTX_POOL_MP;}
                                else{ return OP_GTX_ERROR;}
                            break;
                        case POOL_A_F7:
                                if(this->get_funct3() == POOL_A_F){
                                    return OP_GTX_POOL_A;
                                }//else if(this->get_funct3() == POOL_AP_F){
                                //    return OP_GTX_POOL_AP;}
                                else{ return OP_GTX_ERROR;}
                            break;
                        case LOAD_F7:
                                switch(this->get_funct3()){
                                case LOAD_F :
                                    return OP_GTX_LOAD;
                                    break;
                                case STORE_F :
                                    return OP_GTX_STORE;
                                    break;
                                case COPY_F :
                                    return OP_GTX_COPY;
                                    break;
                                default:
                                    return OP_GTX_ERROR;
                                    break;
                            }
                            break;
                        case LOAD_3D_F7:
                            switch(this->get_funct3()){
                                case LOAD_3D_F :
                                    return OP_GTX_LOAD_3D;
                                    break;
                                case STORE_3D_F :
                                    return OP_GTX_STORE_3D;
                                    break;
                                case LOAD_SVR_F:
                                    return OP_GTX_LOAD_SVR;
                                    break;
                                case STORE_SVR_F:
                                    return OP_GTX_STORE_SVR;
                                    break;
                                default:
                                    return OP_GTX_ERROR;
                                    break;
                            }
                            break;
                        
                        case COPY_MEM_F7:
                            switch(this->get_funct3()){
                                case MCAST_G2S_F :
                                    return OP_GTX_MCAST_G2S;
                                    break;
                                case MCAST_S2S_F :
                                    return OP_GTX_MCAST_S2S;
                                    break;
                                case COPY_MEM_F :
                                    return OP_GTX_COPY_MEM;
                                    break;
                                default:
                                    return OP_GTX_ERROR;
                                    break;
                            }
                        case MCAST_S2L_F7:
                                if(this->get_funct3() == MCAST_S2L_F){
                                    return OP_GTX_MCAST_S2L;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case RDSPR_F7:
                                switch(this->get_funct3()){
                                case RDSPR_F :
                                    return OP_GTX_RDSPR;
                                    break;
                                default:
                                    return OP_GTX_ERROR;
                                    break;
                            }
                            break;
                        case WRSPR_F7:
                                if(this->get_funct3() == WRSPR_F){
                                    return OP_GTX_WRSPR;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case OPSET_F7:
                                if(this->get_funct3() == OPSET_F){
                                    return OP_GTX_OPSET;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case CPSVR_F7:
                                if(this->get_funct3() == CPSVR_F){
                                    return OP_GTX_CPSVR;
                                }else{ return OP_GTX_ERROR;}
                        case MVSVR_F7:
                                if(this->get_funct3() == MVSVR_F){
                                    return OP_GTX_MVSVR;
                                }else{return OP_GTX_ERROR;}
                        case CREDIT_LD_F7:
                                if(this->get_funct3() == CREDIT_LD_F){
                                    return OP_GTX_CREDIT_LD;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case CREDIT_LD_CHK_F7:
                                if(this->get_funct3() == CREDIT_LD_CHK_F){
                                    return OP_GTX_CREDIT_LD_CHK;
                                }else { return OP_GTX_ERROR;}
                            break;
                        case CREDIT_ST_F7:
                                if(this->get_funct3() == CREDIT_ST_F){
                                    return OP_GTX_CREDIT_ST;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case CREDIT_ST_CHK_F7:
                                if(this->get_funct3() == CREDIT_ST_CHK_F){
                                    return OP_GTX_CREDIT_ST_CHK;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case MEXEC_F7:
                                if(this->get_funct3() == MEXEC_F){
                                    return OP_GTX_MEXEC;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case MBAR_F7:
                                if(this->get_funct3() == MBAR_F){
                                    return OP_GTX_MBAR;
                                }else { return OP_GTX_ERROR;}
                            break;
                        case MSYNC_F7:
                                if(this->get_funct3() == MSYNC_F){
                                    return OP_GTX_MSYNC;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case EOM_F7:
                                if(this->get_funct3() == EOM_F){
                                    return OP_GTX_EOM;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case TPOSE_F7:
                                if(this->get_funct3() == TPOSE_F){
                                    return OP_GTX_TPOSE;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case FILL_F7:
                                if(this->get_funct3() == FILL_F){
                                    return OP_GTX_FILL;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        
                        case BAR_F7:
                                if(this->get_funct3() == BAR_F){
                                    return OP_GTX_BAR;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case WAIT_F7:
                                if(this->get_funct3() == WAIT_F){
                                    return OP_GTX_WAIT;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case INTR_F7:
                                if(this->get_funct3() == INTR_F){
                                    return OP_GTX_INTR;
                                }else{ return OP_GTX_ERROR;}
                            break;
                        case FLUSH_F7:
                                if(this->get_funct3() == FLUSH_F){
                                    return OP_GTX_FLUSH;
                                }else{ return OP_GTX_FLUSH;}     
                        case HALT_F7:
                                if(this->get_funct3() == HALT_F){
                                    return OP_GTX_HALT;
                                }else{ return OP_GTX_ERROR;}
                            break;                                     
                    }
                    break;
                case 0b0101011 :
                    switch(this->get_funct3()){
                        case START_T_F :
                            return OP_GTX_START_T;
                            break;
                        case END_T_F:
                            return OP_GTX_END_T;
                            break;
                        case START_S_F:
                            return OP_GTX_START_S;
                            break;
                        case END_S_F:
                            return OP_GTX_END_S;
                            break;
                        case START_P_F:
                            return OP_GTX_START_P;
                            break;
                        case END_P_F:
                            return OP_GTX_END_P;
                            break;
                        case SPLIT_F:
                            return OP_GTX_SPLIT;
                            break;
                        case JOIN_F:
                            return OP_GTX_JOIN;
                            break;
                        default:
                            return OP_GTX_ERROR;
                            break;
                    }
                    break;
                default:
                    return OP_GTX_ERROR;
                    break;

            }
            return OP_GTX_ERROR;
        }

        
        union BitsToFloat
        {
            uint32_t u32;
            float f;
        };
        union BitsToDouble
        {
            uint64_t u64;
            double d;
        };

        uint64_t Exec_FCVT_S_F(uint64_t rs1_data){//fp16 to 32

            uint64_t data;
            uint16_t a,b;
            BitsToFloat da, db;
            
            data = rs1_data;
            a = data;
            b = data>>32;
            da.f = fp16_to_32(a);
            db.f = fp16_to_32(b);
            data = ((uint64_t)db.u32 <<32) | ((uint64_t)da.u32);
            return data;
        }

        uint64_t Exec_FCVT_F_S(uint64_t rs1_data){
            uint64_t data;
            uint16_t a,b;
            BitsToFloat da, db;
            
        
            data = rs1_data;
            da.u32 = data;
            db.u32 = data>>32;
            a = fp32_to_16(da.f);
            b = fp32_to_16(db.f);
            data = ((uint64_t)b <<32) | ((uint64_t)a);
            return data;
        }

        uint64_t  Exec_FCVT_D_F(uint64_t rs1_data){
            uint64_t data;
            uint16_t a;
            BitsToDouble da;
            
            data = rs1_data;
            a = data;
            da.d = (double)fp16_to_32(a);

            data = da.u64;
            return data;
        }

        uint64_t  Exec_FCVT_F_D(uint64_t rs1_data){

            uint64_t data;
            uint16_t a;
            BitsToDouble da;
            
            da.u64 = rs1_data;
            a = fp32_to_16((float)da.d);

            data = (uint64_t)a;
            return data;
        }



        void Exec_GTX_WSPLIT(){
            this -> regs -> incPC();
            this -> nsu -> wsplit();
            this -> regs -> reg_copy();
            if(is_loop){
                LOG(LOG_WARNING) <<"[WARNING] WSPLIT FAILED (already in loop)"<<std::endl;
            }else{
                is_loop = true;
                warp_cnt = warp_time;
                PC_copy = this->regs->getPC();
                if(monitoring){
                    uint64_t sp = this -> regs -> getValue(2);
                    this ->mem_intf -> fst_routine = true;
                    this -> mem_intf -> monitoring = true;
                    this -> mem_intf -> high_stack  = sp;
                    this -> mem_intf -> low_stack = sp;
                }
            }
        }

        void Exec_GTX_WJOIN(){
            this -> regs -> incPC();
        
            if(!is_loop){
                LOG(LOG_ERROR) << "[ERROR]WJOIN FAILED (not in loop)"<<std::endl;
                sc_core::sc_stop();
            }else{
                if(!this->mem_intf->fst_routine && this->mem_intf->monitoring){
                    this->mem_intf->WriteMonitor();
                }
                if(warp_time > 0 && warp_cnt > 0){
                    this->regs->setPC(PC_copy);
                    this->nsu->wjoin();
                    this->regs->reg_restore();
                    this->mem_intf->fst_routine = false;
                    warp_cnt--;
                }else if(monitoring && this->mem_intf->fst_routine){
                    if(warp_time != 0){warp_cnt--;}
                    this->regs->setPC(PC_copy);
                    this->nsu->wjoin();
                    this->regs->reg_restore();
                    this->mem_intf->fst_routine = false;
                    is_monitoring = true;
                }else{
                    this->mem_intf->reset_vector();
                    
                    this->mem_intf->monitoring = false;
                    is_loop = false;
                    this->perf->cycleInc(nsu->loop_exec_time);
                    is_monitoring = false;
                    if(gtx_command_dump){
                        this-> nsu->dump_gtx_command(0,0,0,0,OP_GTX_MSYNC);
                    }
                }

            }
            

        }


        bool exec_instruction(Instruction &inst, op_GTX_Codes code){
            bool PC_not_affected = true;
            unsigned int rs1, rs2, rd;
            uint64_t rs1_data, rs2_data, rd_data;
            uint32_t imm_data;

            this -> setInstr(inst.getInstr());
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();
            rd = this->get_rd();
            imm_data = this->get_imm();
            rs1_data = this->regs->getValue(rs1);
            rs2_data = this->regs->getValue(rs2);
            

            switch(code){
                case OP_GTX_MM:
                case OP_GTX_MM_S:
                case OP_GTX_MM_O:
                case OP_GTX_MM_V:
                case OP_GTX_MM_T:
                case OP_GTX_MMC:
                case OP_GTX_MMC_S:
                case OP_GTX_MMC_O:
                case OP_GTX_MMC_V:
                case OP_GTX_MMC_T:
                //case OP_GTX_CONV_NM:
                //case OP_GTX_CONV_DW:
                //case OP_GTX_CONV_DS:
                case OP_GTX_IM2COL_N:
                case OP_GTX_IM2COL_D:
                case OP_GTX_ADD_VS:
                case OP_GTX_SUB_VS:
                case OP_GTX_MUL_VS:
                case OP_GTX_DIV_VS:
                case OP_GTX_FMADD_VS:
                case OP_GTX_ADD_VV:
                case OP_GTX_SUB_VV:
                case OP_GTX_MUL_VV:
                case OP_GTX_DIV_VV:
                case OP_GTX_FMADD_VVV:
                case OP_GTX_SQRT_V:
                case OP_GTX_EXP_V:
                case OP_GTX_LN_V:
                case OP_GTX_ABS_V:
                case OP_GTX_NEG_V:
                case OP_GTX_SIGN_V:
                case OP_GTX_STEP_V:
                case OP_GTX_CEIL_V:
                case OP_GTX_TRUNC_V:
                case OP_GTX_FLOOR_V:
                case OP_GTX_RNE_V:
                case OP_GTX_CLAMP_MIN:
                case OP_GTX_CLAMP_MAX:
                case OP_GTX_ACCUM:
                case OP_GTX_ARANGE:
                case OP_GTX_SCVT_QH:
                case OP_GTX_SCVT_HQ:
                case OP_GTX_SCVT_IH:
                case OP_GTX_SCVT_HI:
                case OP_GTX_SCVT_HN:
                case OP_GTX_PRELU:
                case OP_GTX_GELU:
                case OP_GTX_TANH:
                case OP_GTX_SIGM:
                case OP_GTX_SOFTMAX:
                case OP_GTX_POOL_M:
                case OP_GTX_POOL_A:
                case OP_GTX_TPOSE:
                case OP_GTX_FILL:
                case OP_GTX_LOAD:
                case OP_GTX_STORE:
                case OP_GTX_LOAD_3D:
                case OP_GTX_STORE_3D:
                case OP_GTX_COPY:
                case OP_GTX_STORE_SVR:
                case OP_GTX_MCAST_S2L:
                case OP_GTX_WRSPR:
                case OP_GTX_CPSVR:
                case OP_GTX_MVSVR:
                case OP_GTX_CREDIT_LD:
                case OP_GTX_CREDIT_LD_CHK:
                case OP_GTX_CREDIT_ST:
                case OP_GTX_CREDIT_ST_CHK:
                    nsu->decode_msp_op(rs1_data,rs2_data,rs3_data,rs4_data,code);
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command = false;
                    break;
                case OP_GTX_MCAST_G2S:
                case OP_GTX_MCAST_S2S:
                    if(is_loop){
                        LOG(LOG_WARNING) << "[WARNING] mcast.x2s must be out of split join\n";
                    }else{
                        nsu->decode_msp_op(rs1_data,rs2_data,rs3_data,rs4_data,code);
                    }
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command = false;
                    break;
                case OP_GTX_COPY_MEM:
                    if(is_loop){
                        LOG(LOG_WARNING) << "[WARNING] copy.mem must be out of split join\n";
                    }else{
                        nsu->decode_msp_op(rs1_data,rs2_data,rs3_data,rs4_data,code);
                    }
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command = false;
                    break;
                case OP_GTX_MAX_VS:
                case OP_GTX_MIN_VS:
                case OP_GTX_ADD_IS:
                case OP_GTX_SUB_IS:
                case OP_GTX_MUL_IS:
                case OP_GTX_DIV_IS:
                case OP_GTX_FMADD_IS:
                case OP_GTX_MAX_IS:
                case OP_GTX_MIN_IS:
                case OP_GTX_DOT_VVS:
                case OP_GTX_SUM_VS:
                case OP_GTX_ADD_II:
                case OP_GTX_SUB_II:
                case OP_GTX_MUL_II:
                case OP_GTX_DIV_II:
                case OP_GTX_DOT_IIS:
                case OP_GTX_FMADD_III:
                case OP_GTX_SUM_IS:
                case OP_GTX_SQRT_I:
                case OP_GTX_EXP_I:
                case OP_GTX_LN_I:
                case OP_GTX_ABS_I:
                case OP_GTX_NEG_I:
                case OP_GTX_SIGN_I:
                case OP_GTX_STEP_I:
                case OP_GTX_CEIL_I:
                case OP_GTX_TRUNC_I:
                case OP_GTX_FLOOR_I:
                case OP_GTX_RNE_I:
                case OP_GTX_AND_II:
                case OP_GTX_OR_II:
                case OP_GTX_NOT_I:
                case OP_GTX_SHIFT_I:
                case OP_GTX_PRELU_I:
                case OP_GTX_GELU_I:
                case OP_GTX_TANH_I:
                case OP_GTX_SIGM_I:
                case OP_GTX_ESUM:
                case OP_GTX_ESUM_I:
                case OP_GTX_SOFTMAX_I:
                case OP_GTX_LOAD_SVR:
                    rd_data = nsu->decode_msp_op(rs1_data,rs2_data,rs3_data,rs4_data,code);
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command = false;
                    break;
                case OP_GTX_RDSPR: 
                    if(is_loop){
                        LOG(LOG_WARNING) << "[WARNING] RDSPR must be out of split join!\n";
                    }else{
                    rd_data = nsu->decode_msp_op(rs1_data,rs2_data,rs3_data,rs4_data,code);
                    this->regs->setValue(rd,rd_data);
                    }
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command = false;
                    break;
                case OP_GTX_MEXEC:
                    if(is_loop){
                        LOG(LOG_WARNING) << "[WARNING] mexec must be out of split join\n";
                    }else{
                        rd_data = nsu->decode_msp_op(rs1_data,rs2_data,rs3_data,rs4_data,code);
                        this->regs->setValue(rd,rd_data);
                    }
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command = false;
                    break;
                case OP_GTX_MSYNC:
                    LOG(LOG_WARNING) << "[WARNING] MSYNC should be gernerated by ISS\n";
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command = false;
                    break;
                case OP_GTX_EOM:
                    LOG(LOG_WARNING) << "[WARNING] EOM should be generated by ISS\n";
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command = false;
                    break;
                case OP_GTX_START_T:
                case OP_GTX_END_T:
                case OP_GTX_START_S:
                case OP_GTX_END_S:
                case OP_GTX_START_P:
                case OP_GTX_END_P:
                    nsu->decode_msp_op(rs1_data,imm_data ,rs3_data,rs4_data,code);
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command = false;
                    break;
                case OP_GTX_OPSET:
                    if((rs1_data & 0x1) == 0x1){
                        rs4_data = rs2_data;
                        set_command = true;
                    }else if((rs1_data ) == 0x0){
                        rs3_data = rs2_data;
                        set_command = true;
                    }
                    break;
                case OP_GTX_FCVT_SH: 
                    if(is_loop){
                        LOG(LOG_WARNING) <<"[WARNING] fcvt.sh must be out of split join\n";
                    }else{
                        rd_data = Exec_FCVT_S_F(rs1_data);
                        this->regs->setValue(rd,rd_data);
                        this->perf->cycleInc(FCVT_S_F_CYC);
                    }
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command =false;
                    break;
                case OP_GTX_FCVT_HS:
                    if(is_loop){
                        LOG(LOG_WARNING) <<"[WARNING] fcvt.hs must be out of split join\n";
                    }else{
                        rd_data = Exec_FCVT_F_S(rs1_data);
                        this->regs->setValue(rd,rd_data);
                        this->perf->cycleInc(FCVT_F_S_CYC);
                    }
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command =false;
                    break;
                case OP_GTX_FCVT_DH:
                    if(is_loop){
                        LOG(LOG_WARNING) <<"[WARNING] fcvt.dh must be out of split join\n";
                    }else{
                        rd_data = Exec_FCVT_D_F(rs1_data);
                        this->regs->setValue(rd,rd_data);
                        this->perf->cycleInc(FCVT_D_F_CYC);
                    }
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command =false;
                    break;
                case OP_GTX_FCVT_HD:
                    if(is_loop){
                        LOG(LOG_WARNING) <<"[WARNING] fcvt.hd must be out of split join\n";
                    }else{
                        rd_data = Exec_FCVT_F_D(rs1_data);
                        this->regs->setValue(rd,rd_data);
                        this->perf->cycleInc(FCVT_F_D_CYC);
                    }
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command =false;
                    break;
                case OP_GTX_SPLIT:
                    PC_not_affected = false;
                    Exec_GTX_WSPLIT();
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command =false;
                    break;
                case OP_GTX_JOIN:
                    Exec_GTX_WJOIN();
                   
                    PC_not_affected = false;
                    rs3_data = 0;
                    rs4_data = 0;
                    set_command =false;
                    break;
                case OP_GTX_MBAR: // nop in simulation
                case OP_GTX_BAR: // nop in simulation
                case OP_GTX_WAIT: // nop in simulation
                case OP_GTX_HALT:
                case OP_GTX_FLUSH:
                    if(is_loop){
                        LOG(LOG_WARNING) << "[WARNING] GTX SYNC COMMAND(bar, wait , flush....) can't cast in split join\n";
                    }
                    break;
                case OP_GTX_INTR:
                    break;
                default: 
                    break;

            }
            return PC_not_affected;
        }
        public:
            bool monitoring;
            bool set_command;
            uint32_t warp_time;
            uint64_t rs3_data;
            uint64_t rs4_data;
            int warp_cnt;
     
        private:
            NSU *nsu;
            T PC_copy;
            bool is_loop; // wsplit wjoin



    };
}
#endif