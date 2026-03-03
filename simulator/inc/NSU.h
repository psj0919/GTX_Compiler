
#ifndef NSU_H
#define NSU_H

#define SC_INCLUDE_DYNAMIC_PROCESSES

#include <queue>
#include "systemc"
#include "tlm.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"

#include "BASE_ISA.h"
#include "TMU.h"
#include "SPU.h"
#include "MemoryInterface.h"
#include "Performance.h"
#include "Registers.h"

namespace riscv_tlm {
    #define GSPR_GTX_RUN (0x000)
    #define GSPR_GTX_OPERAND1 (0x001)
    #define GSPR_GTX_OPERAND2 (0x002)
    #define GSPR_GTX_OPERAND3 (0x003)
    //#define GSPR_GTX_OPERAND4 (0x004)
    #define GSPR_GTX_OPCODE (0x004)
 

    #define GSPR_STACK_INFO (0x010)
    #define GSPR_STACK_SAVE (0x011)



    //#define GSPR_RESULT (0x080)
//
    //#define GSPR_NUMBER (0x08E)
    //#define GSPR_ID (0x08F)
//
    //#define GSPR_OP3 (0x0F0)
    //#define GSPR_OP4 (0x0F1)
//
    //#define GSPR_CREDIT_CONTROL (0x300)
    //
    //#define GSPR_SPU_BUSY (0x380)
    //#define GSPR_NEXT_BUSY (0x381)
    //#define GSPR_LD_CREDIT_COUNT (0x390)
    //#define GSPR_ST_CREDIT_COUNT (0x391)
    //#define GSPR_LD_CREDIT_ERROR (0x392)
    //#define GSPR_ST_CREDIT_ERROR (0x393)

    
    

    struct Inst_info {
        uint64_t rs1_data;
        uint64_t rs2_data;
        uint64_t rs3_data;
        uint64_t rs4_data;
        op_GTX_Codes code;

    };

    class NSU : sc_core::sc_module  {
    public:

        /* Constructors */
        explicit NSU(sc_core::sc_module_name const &name, SPU *spu_i[SPU_NUM], TMU *tmu_i[TMU_NUM]);

        NSU() noexcept = delete;
        NSU(const NSU& other) noexcept = delete;
        NSU(NSU && other) noexcept = delete;
        NSU& operator=(const NSU& other) noexcept = delete;
        NSU& operator=(NSU&& other) noexcept = delete;

        /* Destructors */
        ~NSU() override = default;


        uint64_t getGSPR(uint16_t gspr);

        uint64_t gtx_run();

        uint64_t decode_msp_op(uint64_t rs1, uint64_t rs2, uint64_t rs3, uint64_t rs4 ,op_GTX_Codes msp_op);

        void setGSPR(uint16_t gspr, uint64_t value);

        void mvmem(uint64_t rs1, uint64_t rs2, uint64_t rs3);

        void mcast_s2s(uint64_t rs1, uint64_t rs2, uint64_t rs3);

        void mcast_g2s(uint64_t rs1, uint64_t rs2, uint64_t rs3);

        void wsplit();

        void wjoin();
        
        void startp(uint64_t rs1, uint64_t rs2);
    
        void endp(uint64_t rs1, uint64_t rs2);

        void starts(uint64_t rs1, uint64_t rs2);

        void ends(uint64_t rs1, uint64_t rs2);

        void startt(uint64_t rs1, uint64_t rs2);

        void endt(uint64_t rs1, uint64_t rs2);


        void credit_ld_i(uint64_t rs1, uint64_t rs2);

        void credit_st_i(uint64_t rs1, uint64_t rs2);

        void credit_ld_chk(uint64_t rs1, uint64_t rs2);

        void credit_st_chk(uint64_t rs1, uint64_t rs2);

        uint64_t mexec(uint64_t rs1);

        void dump_gtx_command(uint64_t rs1, uint64_t rs2, uint64_t rs3, uint64_t opsel,op_GTX_Codes msp_op);


        void push_nsu_queue(op_GTX_Codes code,uint64_t rs1_data, uint64_t rs2_data, uint64_t rs3_data, uint64_t rs4_data ){
            Inst_info data;
            data.code = code;
            data.rs1_data = rs1_data;
            data.rs2_data = rs2_data;
            data.rs3_data = rs3_data;
            data.rs4_data = rs4_data;
            nsu_queue.push(data);

        }

        void push_tmu_queue(op_GTX_Codes code,uint64_t rs1_data, uint64_t rs2_data, uint64_t rs3_data, uint64_t rs4_data ,int id){
            Inst_info data;
            data.code = code;
            data.rs1_data = rs1_data;
            data.rs2_data = rs2_data;
            data.rs3_data = rs3_data;
            data.rs4_data = rs4_data;
            tmu_queue[id].push(data);

        }
        void push_spu_queue(op_GTX_Codes code,uint64_t rs1_data, uint64_t rs2_data, uint64_t rs3_data, uint64_t rs4_data, int id){
            Inst_info data;
            data.code = code;
            data.rs1_data = rs1_data;
            data.rs2_data = rs2_data;
            data.rs3_data = rs3_data;
            data.rs4_data = rs4_data;

            spu_queue[id].push(data);

        }

        void pop_nsu_queue(){
            Inst_info data;
            int n_id, g_id;

            int size = nsu_queue.size();

            op_GTX_Codes code;
            use_nsu_queue = false;
            for(int i = 0; i < size; i++){
                data = nsu_queue.front();
                nsu_queue.pop();
                decode_msp_op(data.rs1_data, data.rs2_data, data.rs3_data, data.rs4_data,data.code);
            }
        }

        void pop_tmu_queue(int id){
            Inst_info data;
            int n_id, g_id;
            n_id = id / (GDMAC_NUM/TMU_NUM);
            g_id = id % (GDMAC_NUM/TMU_NUM);


            int size = tmu_queue[id].size();

            op_GTX_Codes code;
            use_tmu_queue[id] = false;
            for(int i = 0; i < size; i++){
                data = tmu_queue[id].front();
                tmu_queue[id].pop();
                bool is_pl, is_sl, is_tl;
                int tid, sid;
                is_pl = is_ploop;
                is_sl = is_sloop;
                is_tl = is_tloop;
                tid = tmu_id;
                sid = curr_id;
                //
                is_ploop = true;
                is_sloop = true;
                is_tloop = false;
                tmu_id = id;   
                curr_id = 0;     
                decode_msp_op(data.rs1_data, data.rs2_data, data.rs3_data, data.rs4_data, data.code);
                tmu_id = tid;
                curr_id = sid;
                is_ploop = is_pl;
                is_sloop = is_sl;
                is_tloop = is_tl;

            }
        }
        void pop_spu_queue(int id){
            Inst_info data;
            int size;
            int n_id, w_id;
            n_id = id / (SPU_NUM / TMU_NUM);
            w_id = id % (SPU_NUM / TMU_NUM);
            size = spu_queue[id].size();
            op_GTX_Codes code;
            use_spu_queue[id] = false;
            for (int i = 0 ; i < size; i++){
                data = spu_queue[id].front();
                spu_queue[id].pop();
                bool is_pl, is_sl, is_tl;
                int tid,cid;
                is_pl = is_ploop;
                is_sl = is_sloop;
                is_tl = is_tloop;
                tid = tmu_id;
                cid = curr_id;
                if((data.code == OP_GTX_CREDIT_LD)|| (data.code == OP_GTX_CREDIT_ST)||(data.code == OP_GTX_CREDIT_ST_CHK)){
                    is_ploop = true;
                    is_sloop = false;
                    is_tloop = true;
                    tmu_id = n_id;
                    curr_id = w_id;
                    decode_msp_op(data.rs1_data, data.rs2_data, data.rs3_data, data.rs4_data,data.code);
                    is_ploop = is_pl;
                    is_sloop = is_sl;
                    is_tloop = is_tl;
                    tmu_id = tid;
                    curr_id = cid;
                }else{
                    this->tmu[n_id]->run_spu(data.rs1_data, data.rs2_data, data.rs3_data, data.rs4_data, data.code, w_id);
                }
            }
        }

        


    public:
        MemoryInterface *mem_intf;
        MemoryInterface *l2_intf[TMU_NUM];
        
        int credit_ld[SPU_NUM];
        int credit_st[SPU_NUM];
        void dump_GSPR();
        int curr_id; // gid or wid
        int tmu_id;
        bool use_nsu_queue;
        bool use_tmu_queue[TMU_NUM];
        bool use_spu_queue[SPU_NUM];
        TMU *tmu[TMU_NUM];
        SPU *spu[SPU_NUM];

        int_fast64_t loop_exec_time;
    

    private:
        Performance *perf;
        //tlm_utils::tlm_quantumkeeper *m_qk;   
        std::unordered_map<unsigned int,uint64_t> GSPR;


        bool is_mexec;
        
        bool is_sloop;
        bool is_ploop;
        bool is_tloop;

        std::array<uint8_t, 0x4000> STK_SRAM{};

        op_GTX_Codes nsu_codes;
        uint64_t nsu_rs1;
        uint64_t nsu_rs2;
        uint64_t nsu_rs3;
        uint64_t nsu_rs4;
        
        bool nfull_flag[SPU_NUM];
        bool tfull_flag[SPU_NUM]; //start tmu queue because of load max credit 
        int  tfull_id[SPU_NUM];
        bool sfull_flag[SPU_NUM]; //start spu queue because of store max credit
        bool ncredit_flag[SPU_NUM];
        bool tcredit_flag[SPU_NUM]; //start tmu queue because store credit not exist
        int  tcredit_id[SPU_NUM];
        bool scredit_flag[SPU_NUM];
        std::queue<Inst_info> spu_queue[SPU_NUM];
        std::queue<Inst_info> tmu_queue[TMU_NUM];
        std::queue<Inst_info> nsu_queue;
        std::queue<Inst_info> empty;
    
        void initGSPR();

    };
}
#endif