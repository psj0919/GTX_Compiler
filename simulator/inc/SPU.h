#ifndef SPU_H
#define SPU_H

#define SC_INCLUDE_DYNAMIC_PROCESSES

#include <cmath>
#include "systemc"
#include "config.h"
#include "tlm.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"
#include "Real_cal.h"

#include "BASE_ISA.h"
#include "MemoryInterface.h"
#include "Performance.h"
#include "Registers.h"
#include "float_conv.h"
#include "GTX_perf_param.h"

namespace riscv_tlm {
   

    #define LSPR_SPM_ADDRA (0x900)
    #define LSPR_SPM_ADDRB (0x901)
    #define LSPR_SPM_ADDRC (0x902)
    #define LSPR_SPM_ADDRR (0x903)

    //H/W Feature & Debug
    //#define LSPR_SPU_CONTROL (0x0x800B00)
    //#define LSPR_SPU_STATUS (0x800B20)
    //#define LSPR_WG_ID (0x800B30)
    
  
    typedef enum {
        //matrix Multiplication{FUNC7, FUNC3}
        OP_GTX_MM       = 0x2,
        OP_GTX_MM_S     = 0x0,
        OP_GTX_MM_O     = 0x1,
        OP_GTX_MM_V     = 0x3,
        OP_GTX_MM_T     = 0x7,
        OP_GTX_MMC      = 0xA,
        OP_GTX_MMC_S    = 0x8,
        OP_GTX_MMC_O    = 0x9,
        OP_GTX_MMC_V    = 0xB,
        OP_GTX_MMC_T    = 0xF,
        //convolution
        OP_GTX_IM2COL_N = 0x40,
        OP_GTX_IM2COL_D = 0x48,
        //scalar calculation
        OP_GTX_ADD_VS   = 0x80,
        OP_GTX_SUB_VS   = 0x81,
        OP_GTX_MUL_VS   = 0x82,
        OP_GTX_DIV_VS   = 0x83,
        OP_GTX_FMADD_VS = 0x88,
        OP_GTX_MAX_VS   = 0X98,
        OP_GTX_MIN_VS   = 0X99,
        OP_GTX_ADD_IS   = 0X84,
        OP_GTX_SUB_IS   = 0X85,
        OP_GTX_MUL_IS   = 0X86,
        OP_GTX_DIV_IS   = 0X87,
        OP_GTX_FMADD_IS = 0X8C,
        OP_GTX_MAX_IS   = 0X9C,
        OP_GTX_MIN_IS   = 0X9D,
        //vector calculation
        OP_GTX_ADD_VV   = 0XC0,
        OP_GTX_SUB_VV   = 0XC1,
        OP_GTX_MUL_VV   = 0XC2,
        OP_GTX_DIV_VV   = 0XC3,
        OP_GTX_DOT_VVS  = 0XD0,
        OP_GTX_FMADD_VVV= 0XC8,
        OP_GTX_SUM_VS   = 0XD1,
        OP_GTX_SQRT_V   = 0XE0,
        OP_GTX_EXP_V    = 0XE1,
        OP_GTX_LN_V     = 0xE2,
        OP_GTX_ABS_V    = 0xE8,
        OP_GTX_NEG_V    = 0xE9,
        OP_GTX_SIGN_V   = 0xEA,
        OP_GTX_STEP_V   = 0xEB,
        OP_GTX_CEIL_V   = 0xF0,
        OP_GTX_TRUNC_V  = 0xF1,
        OP_GTX_FLOOR_V  = 0xF2,
        OP_GTX_RNE_V    = 0xF3,
        OP_GTX_CLAMP_MIN = 0xF8,
        OP_GTX_CLAMP_MAX = 0xF9,
        OP_GTX_ACCUM    = 0xFA,
        OP_GTX_ARANGE   = 0xFB,
        OP_GTX_ADD_II   = 0XC4,
        OP_GTX_SUB_II   = 0XC5,
        OP_GTX_MUL_II   = 0XC6,
        OP_GTX_DIV_II   = 0XC7,
        OP_GTX_DOT_IIS  = 0XD4,
        OP_GTX_FMADD_III= 0XCC,
        OP_GTX_SUM_IS   = 0XD5,
        OP_GTX_SQRT_I   = 0XE4,
        OP_GTX_EXP_I    = 0XE5,
        OP_GTX_LN_I     = 0xE6,
        OP_GTX_ABS_I    = 0xEC,
        OP_GTX_NEG_I    = 0xED,
        OP_GTX_SIGN_I   = 0xEE,
        OP_GTX_STEP_I   = 0xEF,
        OP_GTX_CEIL_I   = 0xF4,
        OP_GTX_TRUNC_I  = 0xF5,
        OP_GTX_FLOOR_I  = 0xF6,
        OP_GTX_RNE_I    = 0xF7,
        OP_GTX_AND_II   = 0xFC,
        OP_GTX_OR_II    = 0xFD,
        OP_GTX_NOT_I    = 0xFE,
        OP_GTX_SHIFT_I  = 0xFF,
        //Format conversion
        OP_GTX_SCVT_QH  = 0X100,
        OP_GTX_SCVT_HQ  = 0X101,
        OP_GTX_SCVT_IH  = 0X108,
        OP_GTX_SCVT_HI  = 0X109,
        OP_GTX_SCVT_HN  = 0X111,
        OP_GTX_FCVT_SH  = 0X120,
        OP_GTX_FCVT_HS  = 0X121,
        OP_GTX_FCVT_DH  = 0X128,
        OP_GTX_FCVT_HD  = 0X129,
        //Activation

        OP_GTX_PRELU    = 0X143,
        OP_GTX_GELU     = 0X150,
        OP_GTX_TANH     = 0X160,
        OP_GTX_SIGM     = 0X168,
        OP_GTX_PRELU_I  = 0X147,
        OP_GTX_GELU_I   = 0X154,
        OP_GTX_TANH_I   = 0X164,
        OP_GTX_SIGM_I   = 0X16C,
        //Softmax
        OP_GTX_ESUM     = 0X179,
        OP_GTX_SOFTMAX  = 0X17A,
        OP_GTX_ESUM_I   = 0X17D,
        OP_GTX_SOFTMAX_I= 0X17E,
        //Pooling
        OP_GTX_POOL_M   = 0X180,
        OP_GTX_POOL_A   = 0X188,
        //DMA
        OP_GTX_LOAD     = 0X200,
        OP_GTX_LOAD_3D  = 0X20C,
        OP_GTX_STORE    = 0X201,
        OP_GTX_STORE_3D = 0X20D,
        OP_GTX_COPY     = 0X202,
        OP_GTX_LOAD_SVR = 0X208,
        OP_GTX_STORE_SVR= 0X209,
        OP_GTX_MCAST_S2L= 0X210,
        OP_GTX_MCAST_G2S= 0X220,
        OP_GTX_MCAST_S2S= 0X222,
        OP_GTX_COPY_MEM = 0X223,
        //SPR
        OP_GTX_RDSPR    = 0X240,
        OP_GTX_WRSPR    = 0X248,
        OP_GTX_OPSET    = 0X250,
        OP_GTX_CPSVR    = 0x258,
        OP_GTX_MVSVR    = 0x260,
        //CREDIT
        OP_GTX_CREDIT_LD= 0X280,
        OP_GTX_CREDIT_LD_CHK,
        OP_GTX_CREDIT_ST= 0X288,
        OP_GTX_CREDIT_ST_CHK = 0X298,
        //memory operation
        OP_GTX_TPOSE    = 0X1C0,
        OP_GTX_FILL     = 0X1C8,
        //
        OP_GTX_MEXEC    = 0X380,
        OP_GTX_MBAR     = 0x3A0,
        OP_GTX_MSYNC    = 0x3A8,
        OP_GTX_EOM      = 0x3B8,
        //Sync
        OP_GTX_BAR      = 0X3C0,
        OP_GTX_WAIT     = 0X3C8,
        OP_GTX_INTR     = 0X3D8,
        OP_GTX_FLUSH    = 0X3E0,
        OP_GTX_HALT     = 0X3F8,
        //Warp operation{0XFF, FUNCT3}
        OP_GTX_START_T  = 0XFF0,
        OP_GTX_END_T    = 0XFF1,
        OP_GTX_START_S  = 0XFF2,
        OP_GTX_END_S    = 0XFF3,
        OP_GTX_START_P  = 0XFF6,
        OP_GTX_END_P    = 0XFF7,
        OP_GTX_SPLIT    = 0XFF4,
        OP_GTX_JOIN     = 0XFF5,

        OP_GTX_ERROR    = 0XFFF


    } op_GTX_Codes;

    union BitsToFloat
    {
        uint32_t u32;
        float f;
    };

    class SPU : sc_core::sc_module  {
    public:

        /* Constructors */  
        explicit SPU(sc_core::sc_module_name const &name, int id);

        SPU() noexcept = delete;
        SPU(const SPU& other) noexcept = delete;
        SPU(SPU && other) noexcept = delete;
        SPU& operator=(const SPU& other) noexcept = delete;
        SPU& operator=(SPU&& other) noexcept = delete;

        /* Destructors */
        ~SPU() override = default;

        //tlm_utils::simple_initiator_socket<SPU> l2_socket;
        //tlm_utils::simple_initiator_socket<SPU> l1_socket;

        void dump_gtx_command(uint64_t rs1, uint64_t rs2, uint64_t rs3, uint64_t opsel, op_GTX_Codes msp_op);

        int getID();
        void setID(int id);

        uint64_t getLSPR(int lspr);

        uint64_t decode_spu_command(uint64_t rs1, uint64_t rs2, uint64_t rs3, uint64_t rs4, op_GTX_Codes spu_op);

        void setLSPR(int lspr, uint64_t value);
        void DMA_SPM(uint32_t l2_addr, uint32_t spm_addr, uint32_t stride, uint16_t length, uint16_t height, bool isld, bool isreload);
        void DMA_SPM_3D(uint32_t l2_addr, uint32_t spm_addr, uint32_t stride, uint16_t length, uint16_t height, uint32_t write_stride, uint16_t depth,bool isld);
        //MXE FUNCTIONS
        void rst_accum();
        void MM(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size);
        void MM_S(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size);
        void MM_O(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size, uint16_t l0_addr);
        void MM_V(uint16_t col_A_size, uint16_t l0_addr);
        void MM_T(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size);
        void MMC(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size);
        void MMC_S(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size);
        void MMC_O(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size, uint16_t l0_addr);
        void MMC_V(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size, uint16_t l0_addr);
        void MMC_T(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size);

        //void NMCONV(uint16_t kernel_size, uint16_t dialate, uint16_t stride, uint16_t num_channel, uint16_t num_kernel, uint16_t row_A_size, uint16_t col_A_size, uint16_t out_M_row, uint16_t out_M_col);
        //void DWCONV(uint16_t kernel_size, uint16_t dialate, uint16_t stride, uint16_t num_channel, uint16_t num_kernel, uint16_t row_A_size, uint16_t col_A_size, uint16_t out_M_row, uint16_t out_M_col);
        //void DSCONV(uint16_t kernel_size, uint16_t dialate, uint16_t stride, uint16_t num_channel, uint16_t num_kernel, uint16_t row_A_size, uint16_t col_A_size, uint16_t out_M_row, uint16_t out_M_col,uint64_t mxe_data);
        void IM2COL_N(uint16_t kernel_size, uint16_t dialate, uint16_t stride, uint16_t num_channel, uint16_t row_A_size, uint16_t col_A_size);
        void IM2COL_D(uint16_t kernel_size, uint16_t dialate, uint16_t stride, uint16_t num_channel, uint16_t row_A_size, uint16_t col_A_size);


        //SDE FUNCTIONS

        //void SADD(uint16_t scalar_value, uint16_t vector_size);
        //void SSUB(uint16_t scalar_value, uint16_t vector_size);
        //void SMUL(uint16_t scalar_value, uint16_t vector_size);
        //void SDIV(uint16_t scalar_value, uint16_t vector_size);
        void SASMD(uint16_t scalar_value, uint32_t vector_size, int funct);
        void SFMADD(uint16_t scalar_value0, uint16_t scalar_value1, uint32_t vector_size);
        void SMAX(uint16_t prev_max, uint32_t vector_size, uint16_t l0_addr);
        void SMIN(uint16_t prev_min, uint32_t vector_size, uint16_t l0_addr);
        //uint64_t SADD_IMM(uint16_t scalar_value, uint16_t input_data[4]);
        //uint64_t SSUB_IMM(uint16_t scalar_value, uint16_t input_data[4]);
        //uint64_t SMUL_IMM(uint16_t scalar_value, uint16_t input_data[4]);
        //uint64_t SDIV_IMM(uint16_t scalar_value, uint16_t input_data[4]);
        void SASMD_IMM(uint16_t scalar_value, uint16_t input_addr, uint16_t l0_addr, int funct);
        void SFMADD_IMM(uint16_t scalar_value0, uint16_t scalar_value1, uint16_t input_addr, uint16_t l0_addr);
        void SMAX_IMM(uint16_t prev_max, uint16_t input_data, uint16_t l0_addr);
        void SMIN_IMM(uint16_t prev_min, uint16_t input_data, uint16_t l0_addr);

        //void VADD(uint16_t vector_size);
        //void VSUB(uint16_t vector_size);
        //void VMUL(uint16_t vector_size);
        //void VDIV(uint16_t vector_size);
        void VASMD(uint32_t vector_size, int funct);
        //void VMM(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size);
        void VFMADD(uint32_t vector_size);
        void VFUNC(uint32_t vector_size,uint16_t op2_sel, int funct);
        void CLAMP(uint32_t vector_size, uint16_t scalar_value, int funct);
        void ARANGE(uint32_t vector_size, uint16_t start_value, uint16_t step);
        //uint64_t VADD_IMM(uint16_t input_b[4], uint16_t input_a[4]);
        //uint64_t VSUB_IMM(uint16_t input_b[4], uint16_t input_a[4]);
        //uint64_t VMUL_IMM(uint16_t input_b[4], uint16_t input_a[4]);
        //uint64_t VDIV_IMM(uint16_t input_b[4], uint16_t input_a[4]);
        void VASMD_IMM(uint16_t r_addr, uint16_t input_b, uint16_t input_a, int funct);
        void VFMADD_IMM(uint16_t rs1, uint16_t rs2, uint16_t rs3);
        void VFUNC_IMM(uint16_t iaddr, uint16_t raddr, uint16_t op2_sel, int funct);
        void BITWISE_IMM(uint16_t r_addr, uint16_t b_addr, uint16_t a_addr, int funct);

        void SCVT_QH(uint32_t vector_size, uint16_t scale, uint16_t offset);
        void SCVT_HQ(uint32_t vector_size, uint16_t scale, uint16_t offset);
        void SCVT_IH(uint32_t vector_size, uint16_t scale, uint16_t offset);
        void SCVT_HI(uint32_t vecotr_size, uint16_t scale, uint16_t offset);
        void SCVT_HN(uint32_t vecotr_size, uint16_t scale, uint16_t offset);

        void PRELU(uint16_t slope, uint32_t vector_size);
        void GELU(uint32_t vector_size);
        void TANH(uint32_t vector_size);
        void SIGM(uint32_t vector_size);
        void PRELU_IMM(uint16_t l0_addr, uint16_t slope, uint16_t input_addr);
        void GELU_IMM(uint16_t i_addr, uint16_t r_addr);
        void TANH_IMM(uint16_t i_addr, uint16_t r_addr);
        void SIGM_IMM(uint16_t i_addr, uint16_t r_addr);

        void ESUM(uint16_t max_value, uint16_t accum_data, uint32_t vector_size, uint16_t l0_addr);
        void SOFTMAX(uint16_t max_value, uint16_t esum_value, uint32_t vector_size);
        void ESUM_IMM(uint16_t max_value, uint16_t accum_data, uint16_t input_data, uint16_t l0_addr);
        void SOFTMAX_IMM(uint16_t max_value, uint16_t esum_value, uint16_t i_addr, uint16_t l0_addr);

        void POOL_M(uint16_t row_k, uint16_t col_k, uint16_t row_stride, uint16_t col_stride, uint16_t row_in, uint16_t col_in, uint16_t row_out, uint16_t col_out);
        void POOL_A(uint16_t row_k, uint16_t col_k, uint16_t row_stride, uint16_t col_stride,uint16_t k_value, uint16_t row_in, uint16_t col_in, uint16_t row_out, uint16_t col_out);

        void check_start(){
            exec_cycle = 1;
        }
        
        uint_fast64_t check_cycle(){
            return exec_cycle;
        }


        void rs_select(uint16_t *rs_arr, uint16_t rs_sel){
            uint64_t  temp;
            switch(rs_sel>>7){//rs2_only
                    case 2://0
                        temp =0;
                        std::memcpy(rs_arr,&temp, sizeof(uint64_t));
                        break;
                    case 3: //spu_gpr
                        int sub_addr = (rs_sel & 0x3);
                        int addr = ((rs_sel>>2)&0x1F);
                        uint64_t a,b;
                        a = l0_intf->readDataMem(addr*32 + sub_addr*8, 4);
                        b = l0_intf->readDataMem(addr*32 + sub_addr*8 + 4, 4);
                        temp = (b<<32) | a;
                        std::memcpy(rs_arr,&temp, sizeof(uint64_t));
                        break;              
                }
        };


    public:
        MemoryInterface *l0_intf;
        MemoryInterface *l1_intf;
        MemoryInterface *l2_intf;
        int datatype;
        uint_fast64_t curr_cycle;
        void dump_LSPR();
    private:
        Performance *perf;
        uint_fast64_t exec_cycle;
        //tlm_utils::tlm_quantumkeeper *m_qk;
        std::unordered_map<unsigned int, uint64_t> LSPR;
        //std::uint32_t mxe_accum[16][16];
        float mxe_accum;//for mm.o mmc.o
        std::string tracker_name;
        std::uint64_t exec_time;
        int ID;
        //int WID;
        void initLSPR();


    };
}

#endif