#ifndef TMU_H
#define TMU_H

#include "systemc"
#include "tlm.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"

#include "BASE_ISA.h"
#include "SPU.h"
#include "MemoryInterface.h"
#include "Performance.h"
#include "Registers.h"
namespace riscv_tlm {


    #define NSPR_THREAD_MASK (0x400)
    #define NSPR_SHARED_MASK (0x401)
    
    #define NSPR_TYPE (0x402)
    #define NSPR_OP_MODE (0x403)

    #define NSPR_CLEAR (0x700)
    #define NSPR_SDLE_STATUS (0x780)
    #define NSPR_CREDIT_COUNT (0x781)
    #define NSPR_CREDIT_ERROR (0x782)
    

    class TMU : sc_core::sc_module  {
        public:
        
            explicit TMU(sc_core::sc_module_name const &name, riscv_tlm::SPU *spu_i[SPU_NUM/TMU_NUM],int id);
            TMU() noexcept = delete;
            TMU(const TMU& other) noexcept = delete;
            TMU(TMU && other) noexcept = delete;
            TMU& operator=(const TMU& other) noexcept = delete;
            TMU& operator=(TMU&& other) noexcept = delete;

            void dump_gtx_command(uint64_t rs1, uint64_t rs2, uint64_t rs3, uint64_t opsel, op_GTX_Codes msp_op);

            void setNSPR(uint16_t gspr, uint64_t value);

            uint64_t getNSPR(uint16_t gspr);

            void DMA_2D(uint64_t rs1, uint64_t rs2, uint64_t rs3, int funct);

            void fill(uint64_t rs1, uint64_t rs2, uint64_t rs3);
        
            void mcast(uint64_t rs1, uint64_t rs2, uint64_t rs3);
        
            void Transpose(uint64_t rs1,uint64_t rs2, uint64_t rs3);

            uint64_t run_spu(uint64_t rs1, uint64_t rs2, uint64_t rs3, uint64_t rs4,op_GTX_Codes msp_op, int id);

            uint64_t run_gdmac(uint64_t rs1, uint64_t rs2, uint64_t rs3, uint64_t rs4, op_GTX_Codes msp_op, int id);

            void dump_NSPR();

            bool chk_ldcrd();
            
            bool chk_stcrd();

            void check_start(){
                exec_cycle = 1;
            }
            int_fast64_t check_cycle(){
                return exec_cycle;
            }


        public:
            MemoryInterface *mem_intf;
            MemoryInterface *l1_intf[SPU_NUM/TMU_NUM];
            MemoryInterface *l2_intf;
            SPU *spu[SPU_NUM/TMU_NUM];
            int_fast64_t curr_cycle;
            int ID;

        private:
            Performance *perf;
            int_fast64_t exec_cycle;
            //tlm_utils::tlm_quantumkeeper *m_qk;   
            std::unordered_map<unsigned int,uint64_t> NSPR;
            std::string tracker_file;



            void initNSPR();

    };

}


#endif