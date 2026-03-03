#include "SPU.h"
#include <unistd.h>
#include <sched.h>
#include <thread>
#include <omp.h>

extern "C"
{
#include <cblas.h>
#include <openblas_config.h>
}

static inline void set_gemm_threads(int nthreads)
{
    if (nthreads <= 0)
        return;
    omp_set_num_threads(nthreads);
    openblas_set_num_threads(nthreads);
}

static inline int get_max_usable_threads()
{
    // 1) cgroup/affinity까지 고려 (Linux)

    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0)
    {
        int n = CPU_COUNT(&set);
        if (n > 0)
            return n;
    }
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc > 0)
        return static_cast<int>(nproc);

    // 3) 표준 라이브러리 fallback
    unsigned hc = std::thread::hardware_concurrency();
    return hc ? static_cast<int>(hc) : 1;
}
namespace riscv_tlm
{

    SPU::SPU(sc_core::sc_module_name const &name, int id) : sc_module(name), ID(id)
    {
        initLSPR();
        perf = Performance::getInstance();
        l0_intf = new MemoryInterface((std::string(name) + "_l0").c_str());
        l1_intf = new MemoryInterface((std::string(name) + "_l1").c_str());
        l2_intf = new MemoryInterface((std::string(name) + "_l2").c_str());

        exec_cycle = 0;
        mxe_accum = 0;
        tracker_name = "./tracker/TRACE_SPU" + std::to_string(ID);
        if(gtx_tracker){
            std::ofstream ofs(tracker_name, std::ios::out | std::ios::trunc);
        }
        exec_time = 0;
    }
    void SPU::dump_gtx_command(uint64_t rs1, uint64_t rs2, uint64_t rs3, uint64_t opsel, op_GTX_Codes msp_op){
        std::ofstream ofs(tracker_name,std::ios::out | std::ios::app);
        if(!ofs) {
            LOG(LOG_WARNING) << "[WARNING] FAILED TO OPEN COMMAND DUMPFILE\n";
            return ;
        }
        ofs << std::hex << std::setfill('0') << std::setw(16) << rs1 << " " 
                                             << std::setw(16) << rs2 << " " 
                                             << std::setw(16) << rs3 << " " 
                                             << std::setw( 3) << opsel << " " 
                                             << std::setw( 3) << msp_op << std::endl;

    }

    uint64_t SPU::decode_spu_command(uint64_t rs1, uint64_t rs2, uint64_t rs3, uint64_t rs4, op_GTX_Codes spu_op)
    {
        uint64_t result = 0;

        uint16_t rs1_arr[4];
        uint16_t rs2_arr[4];

        uint16_t rs3_arr[4];
        uint16_t rs4_arr[3];
        std::memcpy(rs1_arr, &rs1, sizeof(uint64_t));
        std::memcpy(rs2_arr, &rs2, sizeof(uint64_t));
        std::memcpy(rs3_arr, &rs3, sizeof(uint64_t));
        rs4_arr[0] = rs4;

        if(gtx_tracker){
            dump_gtx_command(rs1,rs2,rs3,rs4,spu_op);
        }

        //result = LSPR[LSPR_RESULT];

        switch (spu_op)
        {
        case OP_GTX_MM:
            MM(rs1_arr[0], rs1_arr[1], rs1_arr[3]);
            break;
        case OP_GTX_MM_S:
            MM_S(rs1_arr[0], rs1_arr[1], rs1_arr[3]);
            break;
        case OP_GTX_MM_O:
            MM_O(1, rs1_arr[1], 1,rs3_arr[0]);
            break;
        case OP_GTX_MM_V:
            MM_V(rs1_arr[1], rs3_arr[0]);
            break;
        case OP_GTX_MM_T:
            MM_T(rs1_arr[0], rs1_arr[1], rs1_arr[3]);
            break;
        case OP_GTX_MMC:
            MMC(rs1_arr[0], rs1_arr[1], rs1_arr[3]);
            break;
        case OP_GTX_MMC_S:
            MMC_S(rs1_arr[0], rs1_arr[1], rs1_arr[3]);
            break;
        case OP_GTX_MMC_O:
            MMC_O(1, rs1_arr[1], 1, rs3_arr[0]);
            break;
        case OP_GTX_MMC_V:
            MMC_V(1, rs1_arr[1], 1, rs3_arr[0]);
            break;
        case OP_GTX_MMC_T:
            MMC_T(rs1_arr[0], rs1_arr[1], rs1_arr[3]);
            break;
        case OP_GTX_IM2COL_N:
            IM2COL_N((0xF & rs2_arr[0]), ((0xF00 & rs2_arr[0]) >> 8), rs2_arr[1], rs2_arr[2], rs1_arr[0], rs1_arr[1]);
            break;
        case OP_GTX_IM2COL_D:
            IM2COL_D((0xF & rs2_arr[0]), ((0xF00 & rs2_arr[0]) >> 8), rs2_arr[1], rs2_arr[2], rs1_arr[0], rs1_arr[1]);
            break;
        case OP_GTX_ADD_VS:
            rs_select(rs2_arr, rs4_arr[0]); // check rs4
            SASMD(rs2_arr[0], rs1, 0);
            break;
        case OP_GTX_SUB_VS:
            rs_select(rs2_arr, rs4_arr[0]); // check rs4
            SASMD(rs2_arr[0], rs1, 1);
            break;
        case OP_GTX_MUL_VS:
            rs_select(rs2_arr, rs4_arr[0]); // check rs4
            SASMD(rs2_arr[0], rs1, 2);
            break;
        case OP_GTX_DIV_VS:
            rs_select(rs2_arr, rs4_arr[0]); // check rs4
            SASMD(rs2_arr[0], rs1, 3);
            break;
        case OP_GTX_FMADD_VS:
            rs_select(rs2_arr, rs4_arr[0]); // check rs4
            SFMADD(rs2_arr[0], rs2_arr[1], rs1);
            break;
        case OP_GTX_MAX_VS:
            rs_select(rs2_arr, rs4_arr[0]); // check rs4
            SMAX(rs2_arr[0], rs1,rs3_arr[0]);
            break;
        case OP_GTX_MIN_VS:
            rs_select(rs2_arr, rs4_arr[0]); // check rs4
            SMIN(rs2_arr[0], rs1,rs3_arr[0]);
            break;
        case OP_GTX_ADD_IS:
            rs_select(rs2_arr, rs4_arr[0]);
            SASMD_IMM(rs2_arr[0], rs1_arr[0],rs3_arr[0], 0);
            break;
        case OP_GTX_SUB_IS:
            rs_select(rs2_arr, rs4_arr[0]);
            SASMD_IMM(rs2_arr[0], rs1_arr[0],rs3_arr[0], 1);
            break;
        case OP_GTX_MUL_IS:
            rs_select(rs2_arr, rs4_arr[0]);
            SASMD_IMM(rs2_arr[0], rs1_arr[0],rs3_arr[0], 2);
            break;
        case OP_GTX_DIV_IS:
            rs_select(rs2_arr, rs4_arr[0]);
            SASMD_IMM(rs2_arr[0], rs1_arr[0],rs3_arr[0], 3);
            break;
        case OP_GTX_FMADD_IS:
            rs_select(rs2_arr, rs4_arr[0]);
            SFMADD_IMM(rs2_arr[0], rs2_arr[1], rs1_arr[0], rs3_arr[0]);
            break;
        case OP_GTX_MAX_IS:
            rs_select(rs2_arr, rs4_arr[0]);
            SMAX_IMM(rs2_arr[0], rs1_arr[0],rs3_arr[0]);
            break;
        case OP_GTX_MIN_IS:
            rs_select(rs2_arr, rs4_arr[0]);
            SMIN_IMM(rs2_arr[0], rs1_arr[0],rs3_arr[0]);
            break;
        case OP_GTX_ADD_VV:
            VASMD(rs1 & 0xFFFFFF, 0);
            break;
        case OP_GTX_SUB_VV:
            VASMD(rs1 & 0xFFFFFF, 1);
            break;
        case OP_GTX_MUL_VV:
            VASMD(rs1 & 0xFFFFFF, 2);
            break;
        case OP_GTX_DIV_VV:
            VASMD(rs1 & 0xFFFFFF, 3);
            break;
        case OP_GTX_FMADD_VVV:
            VFMADD(rs1 & 0xFFFFFF);
            break;
        case OP_GTX_SQRT_V:
            VFUNC(rs1 & 0xFFFFFF,0,0);
            break;
        case OP_GTX_EXP_V:
            VFUNC(rs1 & 0xFFFFFF,rs2_arr[0],1);
            break;
        case OP_GTX_LN_V:
            VFUNC(rs1 & 0xFFFFFF,rs2_arr[0],2);
            break;
        case OP_GTX_ABS_V:
            VFUNC(rs1 & 0xFFFFFF,0,3);
            break;
        case OP_GTX_NEG_V:
            VFUNC(rs1 & 0xFFFFFF,0,4);
            break;
        case OP_GTX_SIGN_V:
            VFUNC(rs1 & 0xFFFFFF,0,5);
            break;
        case OP_GTX_STEP_V:
            VFUNC(rs1 & 0xFFFFFF,0,6);
            break;
        case OP_GTX_CEIL_V:
            VFUNC(rs1 & 0xFFFFFF,0,7);
            break;
        case OP_GTX_TRUNC_V:
            VFUNC(rs1 & 0xFFFFFF,0,8);
            break;
        case OP_GTX_FLOOR_V:
            VFUNC(rs1 & 0xFFFFFF,0,9);
            break;
        case OP_GTX_RNE_V:
            VFUNC(rs1 & 0xFFFFFF,0,10);
            break;
        case OP_GTX_CLAMP_MAX:
            rs_select(rs2_arr, rs4_arr[0]);
            CLAMP(rs1 & 0xFFFFFF, rs2_arr[0], 0);
            break;
        case OP_GTX_CLAMP_MIN:
            rs_select(rs2_arr, rs4_arr[0]);
            CLAMP(rs1 & 0xFFFFFF, rs2_arr[0], 1);
            break;
        case OP_GTX_ACCUM:
            VFUNC(rs1 & 0xFFFFFF,rs2_arr[0], 11);
            break;
        case OP_GTX_ARANGE:
            ARANGE(rs1 & 0xFFFFFF, rs2_arr[0], rs2_arr[1]);
            break;
        case OP_GTX_ADD_II:
            VASMD_IMM(rs3_arr[0], rs2_arr[0], rs1_arr[0], 0);
            break;
        case OP_GTX_SUB_II:
            VASMD_IMM(rs3_arr[0], rs2_arr[0], rs1_arr[0], 1);
            break;
        case OP_GTX_MUL_II:
            VASMD_IMM(rs3_arr[0], rs2_arr[0], rs1_arr[0], 2);
            break;
        case OP_GTX_DIV_II:
            VASMD_IMM(rs3_arr[0], rs2_arr[0], rs1_arr[0], 3);
            break;
        case OP_GTX_FMADD_III:
            VFMADD_IMM(rs1_arr[0], rs2_arr[0], rs3_arr[0]);
            break;
        case OP_GTX_SQRT_I:
            VFUNC_IMM(rs1_arr[0], rs3_arr[0],0, 0);
            break;
        case OP_GTX_EXP_I:
            VFUNC_IMM(rs1_arr[0], rs3_arr[0],rs2_arr[0], 1);
            break;
        case OP_GTX_LN_I:
            VFUNC_IMM(rs1_arr[0], rs3_arr[0],rs2_arr[0], 2);
            break;
        case OP_GTX_ABS_I:
            VFUNC_IMM(rs1_arr[0], rs3_arr[0],0, 3);
            break;
        case OP_GTX_NEG_I:
            VFUNC_IMM(rs1_arr[0], rs3_arr[0],0, 4);
            break;
        case OP_GTX_SIGN_I:
            VFUNC_IMM(rs1_arr[0], rs3_arr[0],0, 5);
            break;
        case OP_GTX_STEP_I:
            VFUNC_IMM(rs1_arr[0], rs3_arr[0],0, 6);
            break;
        case OP_GTX_CEIL_I:
            VFUNC_IMM(rs1_arr[0], rs3_arr[0],0, 7);
            break;
        case OP_GTX_TRUNC_I:
            VFUNC_IMM(rs1_arr[0], rs3_arr[0],0, 8);
            break;
        case OP_GTX_FLOOR_I:
            VFUNC_IMM(rs1_arr[0], rs3_arr[0],0, 9);
            break;
        case OP_GTX_RNE_I:
            VFUNC_IMM(rs1_arr[0], rs3_arr[0],0, 10);
            break;
        case OP_GTX_AND_II:
            BITWISE_IMM(rs3_arr[0], rs2_arr[0], rs1_arr[0], 0);
            break;
        case OP_GTX_OR_II:
            BITWISE_IMM(rs3_arr[0], rs2_arr[0], rs1_arr[0], 1);
            break;
        case OP_GTX_NOT_I:
            BITWISE_IMM(rs3_arr[0], rs2_arr[0], rs1_arr[0], 2);
            break;
        case OP_GTX_SHIFT_I:
            BITWISE_IMM(rs3_arr[0], rs2_arr[0], rs1_arr[0], 3);
            break;
        case OP_GTX_SCVT_QH:
            rs_select(rs2_arr, rs4_arr[0]);
            SCVT_QH(rs1 & 0xFFFFFF, rs2_arr[0], rs2_arr[1]);
            break;
        case OP_GTX_SCVT_HQ:
            rs_select(rs2_arr, rs4_arr[0]);
            SCVT_HQ(rs1 & 0xFFFFFF, rs2_arr[0], rs2_arr[1]);
            break;
        case OP_GTX_SCVT_IH:
            rs_select(rs2_arr, rs4_arr[0]);
            SCVT_IH(rs1 & 0xFFFFFF, rs2_arr[0], rs2_arr[1]);
            break;
        case OP_GTX_SCVT_HI:
            rs_select(rs2_arr, rs4_arr[0]);
            SCVT_HI(rs1 & 0xFFFFFF, rs2_arr[0], rs2_arr[1]);
            break;
        case OP_GTX_SCVT_HN:
            rs_select(rs2_arr, rs4_arr[0]);
            SCVT_HN(rs1 & 0xFFFFFF, rs2_arr[0], rs2_arr[1]);
            break;
        case OP_GTX_PRELU:
            rs_select(rs2_arr, rs4_arr[0]);
            PRELU(rs2_arr[0], rs1 & 0xFFFFFF);
            break;
        case OP_GTX_GELU:
            GELU(rs1 & 0xFFFFFF);
            break;
        case OP_GTX_TANH:
            TANH(rs1 & 0xFFFFFF);
            break;
        case OP_GTX_SIGM:
            SIGM(rs1 & 0xFFFFFF);
            break;
        case OP_GTX_PRELU_I:
            rs_select(rs2_arr, rs4_arr[0]);
            PRELU_IMM(rs3_arr[0], rs2_arr[0], rs1_arr[0]);
            break;
        case OP_GTX_GELU_I:
            GELU_IMM(rs1_arr[0],rs3_arr[0]);
            break;
        case OP_GTX_TANH_I:
            TANH_IMM(rs1_arr[0],rs3_arr[0]);
            break;
        case OP_GTX_SIGM_I:
            SIGM_IMM(rs1_arr[0],rs3_arr[0]);
            break;
        case OP_GTX_ESUM:
            rs_select(rs2_arr, rs4_arr[0]);
            ESUM(rs2_arr[0], rs2_arr[1], rs1 & 0xFFFFFF, rs3_arr[0]);
            break;
        case OP_GTX_SOFTMAX:
            rs_select(rs2_arr, rs4_arr[0]);
            SOFTMAX(rs2_arr[0], rs2_arr[1], rs1 & 0xFFFFFF);
            break;
        case OP_GTX_ESUM_I:
            rs_select(rs2_arr, rs4_arr[0]);
            ESUM_IMM(rs2_arr[0], rs2_arr[1], rs1_arr[0], rs3_arr[0]);
            break;
        case OP_GTX_SOFTMAX_I:
            rs_select(rs2_arr, rs4_arr[0]);
            SOFTMAX_IMM(rs2_arr[0], rs2_arr[1], rs1_arr[0],rs3_arr[0]);
            break;
        case OP_GTX_POOL_M:
            POOL_M(rs2_arr[0] & 0xFF, ((rs2_arr[0] & 0xFF00) >> 8), ((rs2_arr[1] & 0xFF)), ((rs2_arr[1] & 0xFF00) >> 8), rs1_arr[0], rs1_arr[1], rs1_arr[2], rs1_arr[3]);
            break;
        case OP_GTX_POOL_A:
            POOL_A(rs2_arr[0] & 0xFF, ((rs2_arr[0] & 0xFF00) >> 8), ((rs2_arr[1] & 0xFF)), ((rs2_arr[1] & 0xFF00) >> 8), rs2_arr[2], rs1_arr[0], rs1_arr[1], rs1_arr[2], rs1_arr[3]);
            break;
        case OP_GTX_LOAD:
            DMA_SPM(rs1 >> 27, rs1 & 0x7FFFFFF, rs2, rs2_arr[2], rs2_arr[3], true, false);
            break;
        case OP_GTX_STORE:
            DMA_SPM(rs1 >> 27, rs1 & 0x7FFFFFF, rs2, rs2_arr[2], rs2_arr[3], false, false);
            break;
        case OP_GTX_COPY:
            DMA_SPM(rs1 >> 32, rs1 & 0x7FFFFFF, rs2, rs2_arr[2], rs2_arr[3], false, true);
            break;
        case OP_GTX_LOAD_SVR:
            uint32_t laddr, saddr;
            uint32_t rdata;
            laddr = rs1 & 0xFFFFFF;
            saddr = (rs2 & 0x1f)*32;
            for (int j = 0; j < 8; j++)
            {
                rdata = l1_intf->readDataMem(laddr, 4);
                l0_intf->writeDataMem(saddr, rdata, 4);
                laddr += 4;
                saddr += 4;
            }
            
            curr_cycle = LQW_EXEC_CYCLE;
            this->exec_cycle += curr_cycle;
            break;
        case OP_GTX_STORE_SVR:
            saddr = rs1 & 0xFFFFFF;
            laddr = (rs2 & 0x1f)*32;
            for (int j = 0; j < 8; j++)
            {
                rdata = l0_intf->readDataMem(laddr, 4);
                l1_intf->writeDataMem(saddr, rdata, 4);
                saddr += 4;
                laddr += 4;
            }
            curr_cycle = LQW_EXEC_CYCLE;
            this->exec_cycle += curr_cycle;
            break;
        case OP_GTX_LOAD_3D:
            LOG(LOG_WARNING) << "[WARNING] This command is no longer supported\n";
            //DMA_SPM_3D(rs1 >> 32, rs1, rs2, rs2_arr[2], rs2_arr[3], rs3, rs3_arr[2], true);
            break;
        case OP_GTX_STORE_3D:
            LOG(LOG_WARNING) << "[WARNING] This command is no longer supported\n";
            //DMA_SPM_3D(rs1 >> 32, rs1, rs2, rs2_arr[2], rs2_arr[3], rs3, rs3_arr[2], false);
            break;
        case OP_GTX_RDSPR:
            laddr = rs1_arr[0] & 0xFFF;
            if(laddr >= 0x900){
                result = getLSPR(rs1_arr[0] & 0xFFF);
            }else if(laddr >= 0x800){
                result = (uint64_t)(l0_intf->readDataMem((laddr&0xFF)*8+4, 4)) << 32;
                result = result | (l0_intf->readDataMem((laddr&0xFF)*8, 4));
            }else{
                LOG(LOG_WARNING) << "[WARNING] RDSPR BAD ADDRESS\n";
            }

            curr_cycle = RDSPR_EXEC_CYCLE;
            this->exec_cycle += curr_cycle;
            break;
        case OP_GTX_WRSPR:
            uint64_t temp;
            rs_select(rs2_arr, rs4_arr[0]);
            laddr = rs1_arr[0] & 0xFFF;
            std::memcpy(&temp, &rs2_arr, sizeof(uint64_t));

            uint64_t pdata, wdata;
            // strobe
            uint64_t wstrb;
            uint64_t wstrbn;
            wstrbn = rs1 & 0xFF0000;
            wstrb = 0;
            for (int i = 0; i < 8; i++)
            {
                if (((wstrbn >> (7-i)) & 0x10000) != 0x10000)
                {
                    wstrb = (wstrb | 0xFF);
                }
                if (i < 7)
                {
                    wstrb = wstrb << 8;
                }
            }

            wdata = (temp & wstrb);          
            if(laddr >= 0x900){
                pdata = (getLSPR(laddr) & ~wstrb);
                setLSPR(laddr, (pdata | wdata));
            }else {
                pdata = (l0_intf->readDataMem((laddr&0xFF)*8,4) | ((uint64_t)(l0_intf->readDataMem((laddr&0xFF)*8+4, 4))<<32)) & ~wstrb;
                l0_intf->writeDataMem((laddr&0xFF)*8,(pdata | wdata),4);
                l0_intf->writeDataMem((laddr&0xFF)*8 + 4,((pdata | wdata)>>32),4);
            }
            // setLSPR(rs1_arr[0] & 0xFFF,temp);
            curr_cycle = WRSPR_EXEC_CYCLE;
            this->exec_cycle += curr_cycle;

            break;
        case OP_GTX_CPSVR:
            int byte_num ;
            uint64_t pattern;
            uint64_t l0data;
            pattern = 0;
            switch(rs2_arr[0]){
                case 0:
                    l0data = (l0_intf->readDataMem((rs1_arr[0]&0x1F)*32, 1)) & 0xFF;
                    for(int i =0 ; i < 8; i++){
                        pattern = (pattern << 8) | l0data;
                    }
                    break;
                case 1:
                    l0data = l0_intf->readDataMem((rs1_arr[0]&0x1F)*32, 2);
                    for(int i = 0 ; i < 4; i++){
                        pattern = (pattern << 16) | l0data;
                    }
                    break;
                case 2:
                    l0data = l0_intf->readDataMem((rs1_arr[0]&0x1F)*32, 4);
                    for(int i= 0 ; i < 2 ; i++){
                        pattern = (pattern << 32) |l0data;
                    }
                    break;
                case 3:
                    l0data = l0_intf->readDataMem((rs1_arr[0]&0x1F)*32 + 4, 4);
                    pattern = (l0data<<32);
                    l0data = l0_intf->readDataMem((rs1_arr[0]&0x1F)*32, 4);
                    pattern = pattern |l0data;
                    break;
                default:
                    LOG(LOG_WARNING) << "[WARNING] cpsvr wrong byte size !\n";
                    break;
            }
            for(int i = 0 ;i < 4 ; i++){
                l0_intf->writeDataMem((rs1_arr[0]&0x1F)*32 + 8*i, pattern,4);
                l0_intf->writeDataMem((rs1_arr[0]&0x1F)*32 + 8*i +4, (pattern >> 32), 4);
            }
            break;
        case OP_GTX_MVSVR:
            for(int i= 0 ; i < 32 ; i++){
                if((((uint32_t)0x1 << i) & rs3) != ((uint32_t)0x1 << i)){
                    l0data = l0_intf->readDataMem((rs1_arr[0]&0x1F)*32 + i, 1);
                    l0_intf->writeDataMem((rs2_arr[0]&0x1F)*32 + i,l0data ,1);
                }
            }
            break;
        default:
            LOG(LOG_ERROR) << "[ERROR] unknown command -  SPU\n";
            LOG(LOG_ERROR) << "[ERROR] opcode " << spu_op << std::endl;
            break;
        }
        if(L1_dump_mode && (ID == L1_mode_id)){
            exec_time++;
            if(spu_exec_time == exec_time){
                is_L1_dump = true;
                LOG(LOG_INFO) << "\n============================================="<<std::endl;
                LOG(LOG_INFO) << "[INFO] SIMULATION FIN BY L1 DUMP MODE, Exiting..\n";
                LOG(LOG_INFO) << "=============================================\n"<<std::endl;
                sc_core::sc_stop();
            }
        }

        //LSPR[LSPR_RESULT] = result;
        return result;
    }

    void SPU::DMA_SPM(uint32_t l2_addr, uint32_t spm_addr, uint32_t stride, uint16_t length, uint16_t height, bool isld, bool isreload)
    {
        std::uint32_t rdata;
        MemoryInterface *src_intf;
        MemoryInterface *dst_intf;
        std::uint32_t src_addr, dst_addr;
        std::uint32_t ar_addr, aw_addr;
        std::uint32_t wstride;
        std::string src_mem, dst_mem, funct;
        int rem;
        src_addr = isld ? l2_addr : spm_addr;
        dst_addr = isld ? spm_addr : l2_addr;
        src_intf = isreload ? l1_intf : (isld ? l2_intf : l1_intf);
        dst_intf = isreload ? l1_intf : (isld ? l1_intf : l2_intf);
        wstride = isld ? length : stride;
        stride = isld ? stride : length;

        funct = isreload ? "COPY " : (isld ? "LOAD " : "STORE ");
        src_mem = isreload ? "L1 MEMORY" : (isld ? "L2 MEMORY" : "L1 MEMORY");
        dst_mem = isreload ? "L1_MEMORY" : (isld ? "L1 MEMORY" : "L2 MEMORY");
        LOG(LOG_DEBUG) << "[DEBUG] " << funct << src_mem << "(0x" << std::hex << src_addr << ")->" << dst_mem << "(0x" << dst_addr << ")\n";

        // simple DMA function
        rem = length % 4;
        for (int i = 0; i < height; i++)
        {
            ar_addr = src_addr;
            aw_addr = dst_addr;
            for (int j = 0; j < length / 4; j++)
            {
                rdata = src_intf->readDataMem(src_addr, 4);
                dst_intf->writeDataMem(dst_addr, rdata, 4);
                src_addr = src_addr + 4;
                dst_addr = dst_addr + 4;
                if (j % 8 == 0)
                {
                    perf->dataMemoryRead();
                    perf->dataMemoryWrite();
                }
            }
            if (rem != 0)
            {
                rdata = src_intf->readDataMem(src_addr, rem);
                dst_intf->writeDataMem(dst_addr, rdata, rem);
                if (length % 32 != 0)
                {
                    perf->dataMemoryRead();
                    perf->dataMemoryWrite();
                }
                src_addr = src_addr + rem;
                dst_addr = dst_addr + rem;
            }
            src_addr = ar_addr + stride;
            dst_addr = aw_addr + wstride;
        }
        curr_cycle = DMAC_EXEC_CYCLE(length, height);
        this->exec_cycle += curr_cycle;
    }

    void SPU::DMA_SPM_3D(uint32_t l2_addr, uint32_t spm_addr, uint32_t stride, uint16_t length, uint16_t height, uint32_t write_stride, uint16_t depth, bool isld)
    {
        uint64_t dma_3d;
        for (int i = 0; i < depth; i++)
        {
            DMA_SPM(l2_addr, spm_addr, stride, length, height, isld, false);
            l2_addr += write_stride;
            spm_addr += (length * height);
        }
    }

    // void SPU::rst_accum(){
    //     for(int i =0 ; i < 16; i++){
    //         for(int j = 0; j < 16; j++){
    //             this->mxe_accum[i][j] = 0;
    //         }
    //     }
    // }
    void SPU::MM(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size)
    {
        //int num_threads = get_max_usable_threads();
        //set_gemm_threads(num_threads);
        set_gemm_threads(MAX_THREAD);
        uint32_t a_addr = LSPR[LSPR_SPM_ADDRA];
        uint32_t b_addr = LSPR[LSPR_SPM_ADDRB];
        uint32_t c_addr = LSPR[LSPR_SPM_ADDRC];
        uint32_t r_addr = LSPR[LSPR_SPM_ADDRR];

        if((row_A_size == 0) ||(col_A_size == 0) || (col_B_size == 0)){
            LOG(LOG_WARNING) << "[WARNING] MM matrix size 0 !!!(nothing to do)\n";
            return ;
        }
        const int M = static_cast<int>(row_A_size);
        const int K = static_cast<int>(col_A_size);
        const int N = static_cast<int>(col_B_size);

        // 1) A, B를 FP32 버퍼로 준비 (Row-Major)
        std::vector<float> A(M * K);
        std::vector<float> B(K * N);
        if(datatype == 2){
            LOG(LOG_WARNING) << "[WARNING] MM not support int8 input\n";
            return;
        }
        // A 로드 및 형변환
        if (datatype == 2)
        { // int8
#pragma omp parallel for
            for (int i = 0; i < M; ++i)
            {
                for (int k = 0; k < K; ++k)
                {
                    uint16_t data = this->l1_intf->readDataMem(a_addr + (k + K * i), 1);
                    A[i * K + k] = int8_to_32(static_cast<uint8_t>(data));
                    if (k % 32 == 0)
                    {
                        perf->dataMemoryRead();
                        perf->dataMemoryRead();
                    }
                }
            }
#pragma omp parallel for
            for (int k = 0; k < N; ++k)
            {
                for (int j = 0; j < K; ++j)
                {
                    uint16_t datb = this->l1_intf->readDataMem(b_addr + (j + k * K), 1);
                    B[j * N + k] = int8_to_32(static_cast<uint8_t>(datb));
                    if (k % 32 == 0)
                    {
                        perf->dataMemoryRead();
                        perf->dataMemoryRead();
                    }
                }
            }
        }
        else if (datatype == 0)
        { // fp8
#pragma omp parallel for
            for (int i = 0; i < M; ++i)
            {
                for (int k = 0; k < K; ++k)
                {
                    uint16_t data = this->l1_intf->readDataMem(a_addr + (k + K * i), 1);
                    A[i * K + k] = fp8_to_32(static_cast<uint8_t>(data));
                    if (k % 16 == 0)
                    {
                        perf->dataMemoryRead();
                        perf->dataMemoryRead();
                    }
                }
            }
#pragma omp parallel for
            for (int k = 0; k < N; ++k)
            {
                for (int j = 0; j < K; ++j)
                {
                    uint16_t datb = this->l1_intf->readDataMem(b_addr + (j + K * k), 1);
                    B[j * N + k] = fp8_to_32(static_cast<uint8_t>(datb));
                    if (k % 16 == 0)
                    {
                        perf->dataMemoryRead();
                        perf->dataMemoryRead();
                    }
                }
            }
        }
        else if (datatype == 1)
        { // fp16
#pragma omp parallel for
            for (int i = 0; i < M; ++i)
            {
                for (int k = 0; k < K; ++k)
                {
                    uint16_t data = this->l1_intf->readDataMem(a_addr + (k + K * i) * 2, 2);
                    A[i * K + k] = fp16_to_32(data);
                    if (k % 16 == 0)
                    {
                        perf->dataMemoryRead();
                        perf->dataMemoryRead();
                    }
                }
            }
#pragma omp parallel for
            for (int k = 0; k < N; ++k)
            {
                for (int j = 0; j < K; ++j)
                {
                    uint16_t datb = this->l1_intf->readDataMem(b_addr + (j + K * k) * 2, 2);
                    B[j * N + k] = fp16_to_32(datb);
                    if (k % 16 == 0)
                    {
                        perf->dataMemoryRead();
                        perf->dataMemoryRead();
                    }
                }
            }
        }
        else
        {
            LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n";
            return;
        }

        // 2) SGEMM: C = A(MxK) * B(KxN)
        std::vector<float> C(M * N, 0.0f);
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Row-major, NoTrans * NoTrans
        // lda = K, ldb = N, ldc = N  (row-major에서 각각 행당 리드 수)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, alpha,
                    A.data(), K,
                    B.data(), N,
                    beta,
                    C.data(), N);

// 3) 결과 FP16으로 변환하여 메모리에 저장 (기존과 동일한 레이아웃)
#pragma omp parallel for
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float s = C[i * N + j];
                if(datatype == 2){
                    int32_t sum_int = s;
                    this->l1_intf->writeDataMem(c_addr + (j + N * i) * 4, sum_int, 4);
                }else{
                    uint16_t sum_fp16 = fp32_to_16(s);
                    this->l1_intf->writeDataMem(r_addr + (j + N * i) * 2, sum_fp16, 2);
                }
                if (j % 16 == 0)
                {
                    perf->dataMemoryWrite();
                    perf->dataMemoryWrite();
                }
            }
        }
        // rst_accum(); // 필요 시 유지
        curr_cycle = MM_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
        this->exec_cycle += MM_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
    }

   

    void SPU::MM_S(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size)
    {
        if((row_A_size == 0) ||(col_A_size == 0) || (col_B_size == 0)){
            LOG(LOG_WARNING) << "[WARNING] MM.S matrix size 0 !!!(nothing to do)\n";
            return ;
        }
        //int num_threads = get_max_usable_threads();
        //set_gemm_threads(num_threads);
        set_gemm_threads(MAX_THREAD);

        uint64_t spm_addr;
        uint32_t a_addr, b_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        b_addr = LSPR[LSPR_SPM_ADDRB];
        r_addr = LSPR[LSPR_SPM_ADDRC];

        // 메모리 할당 (행렬 데이터를 저장할 배열)
        std::vector<float> matrix_A(row_A_size * col_A_size);
        std::vector<float> matrix_B(col_A_size * col_B_size);
        std::vector<float> matrix_C(row_A_size * col_B_size);

// 방법 1: OpenBLAS 사용 (더 빠름)
// 데이터 로드 (OpenMP로 병렬화)
#pragma omp parallel for
        for (int i = 0; i < row_A_size; i++)
        {
            for (int k = 0; k < col_A_size; k++)
            {
                uint16_t data;
                switch (datatype)
                {
                case 2:
                    data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                    matrix_A[i * col_A_size + k] = int8_to_32(data);
                    if (k % 32 == 0)
                        {
                            perf->dataMemoryRead();
                            perf->dataMemoryRead();
                        }
                    break;
                case 0:
                    data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                    matrix_A[i * col_A_size + k] = fp8_to_32(data);
                    if (k % 32 == 0)
                        {
                            perf->dataMemoryRead();
                            perf->dataMemoryRead();
                        }
                    break;
                case 1:
                    data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i) * 2, 2);
                    matrix_A[i * col_A_size + k] = fp16_to_32(data);
                    if (k % 16 == 0)
                        {
                            perf->dataMemoryRead();
                            perf->dataMemoryRead();
                        }
                    break;
                default:
                    LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n";
                    break;
                }
            }
        }

#pragma omp parallel for
        for (int k = 0; k < col_A_size; k++)
        {
            for (int l = 0; l < col_B_size; l++)
            {
                uint16_t datb;
                switch (datatype)
                {
                case 2:
                    datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size), 1);
                    matrix_B[k * col_B_size + l] = int8_to_32(datb);
                    break;
                case 0:
                    datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size), 1);
                    matrix_B[k * col_B_size + l] = fp8_to_32(datb);
                    break;
                case 1:
                    datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size) * 2, 2);
                    matrix_B[k * col_B_size + l] = fp16_to_32(datb);
                    break;
                default:
                    LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n";
                    break;
                }
            }
        }

        // OpenBLAS를 사용한 행렬 곱셈
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    row_A_size, col_B_size, col_A_size,
                    1.0f, matrix_A.data(), col_A_size,
                    matrix_B.data(), col_B_size,
                    0.0f, matrix_C.data(), col_B_size);

// 결과 저장 (OpenMP로 병렬화)
#pragma omp parallel for
        for (int i = 0; i < row_A_size; i++)
        {
            for (int l = 0; l < col_B_size; l++)
            {
                
                BitsToFloat s;
                s.f = matrix_C[i * col_B_size + l];
                if(datatype == 2){
                    int32_t sum_int = s.f;
                    this->l1_intf->writeDataMem(r_addr + (l + col_B_size * i) * 4, sum_int, 4);
                }else{
                    this->l1_intf->writeDataMem(r_addr + (l + col_B_size * i) * 4, s.u32, 4);
                }
                if(l%(8) == 0) perf->dataMemoryWrite();
            }
        }
        curr_cycle = MM_S_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
        this->exec_cycle += MM_S_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
    }


    void SPU::MM_O(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size,uint16_t l0_addr)
    {
        if((row_A_size == 0) ||(col_A_size == 0) || (col_B_size == 0)){
            LOG(LOG_WARNING) << "[WARNING] MM.O matrix size 0 !!!(nothing to do)\n";
            return ;
        }
        //int num_threads = get_max_usable_threads();
        //set_gemm_threads(num_threads);
        set_gemm_threads(MAX_THREAD);

        uint64_t spm_addr;
        uint32_t a_addr, b_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        b_addr = LSPR[LSPR_SPM_ADDRB];
        r_addr = LSPR[LSPR_SPM_ADDRC];

        // 메모리 할당 (행렬 데이터를 저장할 배열)
        std::vector<float> matrix_A(col_A_size);
        std::vector<float> matrix_B(col_A_size);
        std::vector<float> matrix_C(1);

// 방법 1: OpenBLAS 사용 (더 빠름)
// 데이터 로드 (OpenMP로 병렬화)`
#pragma omp parallel for
        for (int k = 0; k < col_A_size; k++)
        {
            uint16_t data;
            switch (datatype)
            {
            case 2:
                data = this->l1_intf->readDataMem(a_addr + k, 1);
                matrix_A[k] = int8_to_32(data);
                if (k % 32 == 0)
                        {
                            perf->dataMemoryRead();
                        }
                break;
            case 0:
                data = this->l1_intf->readDataMem(a_addr + k, 1);
                matrix_A[k] = fp8_to_32(data);
                if (k % 32 == 0)
                        {
                            perf->dataMemoryRead();
                        }
                break;
            case 1:
                data = this->l1_intf->readDataMem(a_addr + k * 2, 2);
                matrix_A[k] = fp16_to_32(data);
                if (k % 16 == 0)
                        {
                            perf->dataMemoryRead();
                            perf->dataMemoryRead();
                        }
                break;
            default:
                LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n";
                break;
            }
            matrix_B[k] = 1;
        }

        // OpenBLAS를 사용한 행렬 곱셈
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    1, 1, col_A_size,
                    1.0f, matrix_A.data(), col_A_size,
                    matrix_B.data(), 1,
                    0.0f, matrix_C.data(), 1);


        BitsToFloat s;
        s.f = matrix_C[0];
        //this->l1_intf->writeDataMem(r_addr , s.u32, 4);
        mxe_accum = matrix_C[0];
        if(datatype == 2){
            int32_t sum_int = s.f;
            this->l0_intf->writeDataMem(l0_addr*32 , sum_int , 4);
            for(int i = 1; i < 8 ; i++){
                this-> l0_intf->writeDataMem(l0_addr*32 +4*i, 0x0, 4);
            }
        }else{
            this->l0_intf->writeDataMem(l0_addr*32 , fp32_to_16(s.f), 2);
            for(int i = 1; i < 16 ; i++){
                this-> l0_intf->writeDataMem(l0_addr*32 +2*i, 0x0, 2);
            }
        }
        perf->dataMemoryWrite();

        curr_cycle = MM_S_EXEC_CYCLE(1, col_A_size, 1);
        this->exec_cycle += MM_S_EXEC_CYCLE(1, col_A_size, 1);
    }
void SPU::MM_V(uint16_t col_A_size, uint16_t l0_addr)
    {
        if((col_A_size == 0)){
            LOG(LOG_WARNING) << "[WARNING] MM.V matrix size 0 !!!(nothing to do)\n";
            return ;
        }
        //int num_threads = get_max_usable_threads();
        //set_gemm_threads(num_threads);
        set_gemm_threads(MAX_THREAD);

        uint64_t spm_addr;
        uint32_t a_addr, b_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        b_addr = LSPR[LSPR_SPM_ADDRB];
        r_addr = LSPR[LSPR_SPM_ADDRC];

        // 메모리 할당 (행렬 데이터를 저장할 배열)
        std::vector<float> matrix_A(col_A_size);
        std::vector<float> matrix_B(col_A_size);
        std::vector<float> matrix_C(1);

// 방법 1: OpenBLAS 사용 (더 빠름)
// 데이터 로드 (OpenMP로 병렬화)
#pragma omp parallel for

            for (int k = 0; k < col_A_size; k++)
            {
                uint16_t data;
                switch (datatype)
                {
                case 2:
                    data = this->l1_intf->readDataMem(a_addr + k, 1);
                    matrix_A[k] = int8_to_32(data);
                    if (k % 32 == 0)
                        {
                            perf->dataMemoryRead();
                            perf->dataMemoryRead();
                        }
                    break;
                case 0:
                    data = this->l1_intf->readDataMem(a_addr + k , 1);
                    matrix_A[k] = fp8_to_32(data);
                    if (k % 32 == 0)
                        {
                            perf->dataMemoryRead();
                            perf->dataMemoryRead();
                        }
                    break;
                case 1:
                    data = this->l1_intf->readDataMem(a_addr + k * 2, 2);
                    matrix_A[k] = fp16_to_32(data);
                    if (k % 16 == 0)
                        {
                            perf->dataMemoryRead();
                            perf->dataMemoryRead();
                        }
                    break;
                default:
                    LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n";
                    break;
                }
            }
    

#pragma omp parallel for
        for (int k = 0; k < col_A_size; k++)
        {
                uint16_t datb;
                switch (datatype)
                {
                case 2:
                    datb = this->l1_intf->readDataMem(b_addr + k , 1);
                    matrix_B[k ] = int8_to_32(datb);
                    break;
                case 0:
                    datb = this->l1_intf->readDataMem(b_addr + k , 1);
                    matrix_B[k ] = fp8_to_32(datb);
                    break;
                case 1:
                    datb = this->l1_intf->readDataMem(b_addr + k * 2, 2);
                    matrix_B[k ] = fp16_to_32(datb);
                    break;
                default:
                    LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n";
                    break;
                }
        }

        // OpenBLAS를 사용한 행렬 곱셈
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    1, 1, col_A_size,
                    1.0f, matrix_A.data(), col_A_size,
                    matrix_B.data(), 1,
                    0.0f, matrix_C.data(), 1);

// 결과 저장 (OpenMP로 병렬화)
        BitsToFloat s;
        s.f = matrix_C[0];
        //this->l1_intf->writeDataMem(r_addr, s.u32, 4);
        mxe_accum = matrix_C[0];
        if(datatype == 2){
            int32_t sum_int = s.f;
            this->l0_intf->writeDataMem(l0_addr*32, sum_int, 4);
            for(int i = 1; i < 8 ; i++){
                this-> l0_intf->writeDataMem(l0_addr*32 +4*i, 0x0, 4);
            }
        }else{
            this->l0_intf->writeDataMem(l0_addr*32, fp32_to_16(s.f), 2);
            for(int i = 1; i < 16 ; i++){
                this-> l0_intf->writeDataMem(l0_addr*32 +2*i, 0x0, 2);
            }
        }
        perf->dataMemoryWrite();

        curr_cycle = MM_S_EXEC_CYCLE(1, col_A_size, 1);
        this->exec_cycle += MM_S_EXEC_CYCLE(1, col_A_size, 1);
    }
void SPU::MM_T(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size) {
    if((row_A_size == 0) ||(col_A_size == 0) || (col_B_size == 0)){
            LOG(LOG_WARNING) << "[WARNING] MM.T matrix size 0 !!!(nothing to do)\n";
            return ;
        }
    // ---- 주소/치수 ----
    const uint32_t a_addr = LSPR[LSPR_SPM_ADDRA];   // A base
    const uint32_t b_addr = LSPR[LSPR_SPM_ADDRB];   // B base
    const uint32_t c_addr = LSPR[LSPR_SPM_ADDRC];
    const uint32_t r_addr = LSPR[LSPR_SPM_ADDRR];   // R (출력)

    const int M = static_cast<int>(row_A_size);  // A: MxK
    const int K = static_cast<int>(col_A_size);
    const int N = static_cast<int>(col_B_size);  // B: KxN

    // ---- FP32 행렬 버퍼 ----
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    // C^T를 직접 계산할 버퍼 (N x M, RowMajor)
    std::vector<float> CT(N * M);

    // ---- A/B 패킹 (원본 형식 -> FP32) ----
    // A(i,k) -> A[i*K + k]
    // B(k,j) -> B[k*N + j]
    // 메모리에서 읽는 바이트 수: int8/fp8=1바이트, fp16=2바이트
    const int a_step = (datatype == 1) ? 2 : 1; // fp16=2, 나머지=1
    const int b_step = (datatype == 1) ? 2 : 1;

    // A 패킹
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            uint32_t off = a_addr + ( (datatype==1) ? ((k + K*i) * 2u) : (k + K*i) );
            uint16_t raw = this->l1_intf->readDataMem(off, a_step);
            float a;
            switch (datatype) {
                case 2:  a = int8_to_32(static_cast<uint8_t>(raw)); break; // int8
                case 0:  a = fp8_to_32 (static_cast<uint8_t>(raw)); break; // fp8
                case 1:  a = fp16_to_32(raw);                         break; // fp16
                default: a = 0.0f; LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n"; break;
            }
            A[i*K + k] = a;

            // (선택) 성능 카운터
            if ((k & ((datatype==1)?15:31)) == 0) {
                perf->dataMemoryRead();
                perf->dataMemoryRead();
            }
        }
    }

    // B 패킹
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            uint32_t off = b_addr + ( (datatype==1) ? ((k + j * K) * 2u) : (k + j * K) );
            uint16_t raw = this->l1_intf->readDataMem(off, b_step);
            float b;
            switch (datatype) {
                case 2:  b = int8_to_32(static_cast<uint8_t>(raw)); break;
                case 0:  b = fp8_to_32 (static_cast<uint8_t>(raw)); break;
                case 1:  b = fp16_to_32(raw);                        break;
                default: b = 0.0f; LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n"; break;
            }
            B[k*N + j] = b;

            if ((k & ((datatype==1)?15:31)) == 0) {
                perf->dataMemoryRead();
                perf->dataMemoryRead();
            }
        }
    }

    // ---- GEMM: CT = (A*B)^T = B^T * A^T ----
    // RowMajor에서 Trans,Trans로 호출하면 결과를 N×M (row-major) 버퍼 CT에 바로 생성
    //  CT(NxM) = B^T(NxK) * A^T(KxM)
    //  lda(N), ldb(K), ldc(M) 주의!
    const float alpha = 1.0f, beta = 0.0f;

    // (선택) BLAS 스레드 수 조절: set_gemm_threads(n) 같은 헬퍼가 있다면 사용
    //int num_threads = get_max_usable_threads();
    //set_gemm_threads(num_threads); // OpenBLAS: openblas_set_num_threads 등
    set_gemm_threads(MAX_THREAD);


    cblas_sgemm(CblasRowMajor,
                CblasTrans, CblasTrans,
                /* m */ N,          /* n */ M,          /* k */ K,
                alpha,
                /* A= B (transposed) */ B.data(), /* lda */ N,
                /* B= A (transposed) */ A.data(), /* ldb */ K,
                beta,
                /* C= CT */ CT.data(), /* ldc */ M);

    // ---- 결과 쓰기: 시뮬레이터 R 메모리에 (전치된 형태) ----
    // 기존 MM_T는 r_addr + (i + row_A_size*l)*2 로 썼음 → 이는 (l,i) 순서(=전치)와 일치
    #pragma omp parallel for schedule(static)
    for (int l = 0; l < N; ++l) {        // CT is N x M
        for (int i = 0; i < M; ++i) {
            float s = CT[l*M + i];       // (l,i)
            uint16_t out = fp32_to_16(s);
            if(datatype == 2){
                int32_t sum_int = s;
                this->l1_intf->writeDataMem(r_addr + (i + M * l) * 4, sum_int, 4);
                
            }else{
                this->l1_intf->writeDataMem(r_addr + (i + M * l) * 2u, out, 2);
            }
            if ((l & 15) == 0) perf->dataMemoryWrite();
        }
    }

    // 사이클/통계
    curr_cycle = MM_T_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
    this->exec_cycle += curr_cycle;
}


// MMC 함수 (행렬곱 + 바이어스 추가, fp16 출력)
void SPU::MMC(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size)
    {

        if((row_A_size == 0) ||(col_A_size == 0) || (col_B_size == 0)){
            LOG(LOG_WARNING) << "[WARNING] MMC matrix size 0 !!!(nothing to do)\n";
            return ;
        }
        //int num_threads = get_max_usable_threads();
        //set_gemm_threads(num_threads);
        set_gemm_threads(MAX_THREAD);

        uint32_t a_addr, b_addr, c_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        b_addr = LSPR[LSPR_SPM_ADDRB];
        c_addr = LSPR[LSPR_SPM_ADDRC];
        r_addr = LSPR[LSPR_SPM_ADDRR];

        std::vector<float> matrix_A(row_A_size * col_A_size);
        std::vector<float> matrix_B(col_A_size * col_B_size);
        std::vector<float> matrix_C(row_A_size * col_B_size);
        std::vector<float> bias_C(row_A_size * col_B_size);

        if(datatype == 2){
            LOG(LOG_WARNING) << "[WARNING] MMC not support int8 input\n";
            return;
        }

// A, B 행렬 로드
#pragma omp parallel for collapse(2)
        for (int i = 0; i < row_A_size; i++)
        {
            for (int k = 0; k < col_A_size; k++)
            {
                uint16_t data;
                switch (datatype)
                {
                case 2:
                    data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                    matrix_A[i * col_A_size + k] = int8_to_32(data);
                    break;
                case 0:
                    data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                    matrix_A[i * col_A_size + k] = fp8_to_32(data);
                    break;
                case 1:
                    data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i) * 2, 2);
                    matrix_A[i * col_A_size + k] = fp16_to_32(data);
                    break;
                }
            }
        }

#pragma omp parallel for collapse(2)
        for (int k = 0; k < col_A_size; k++)
        {
            for (int l = 0; l < col_B_size; l++)
            {
                uint16_t datb;
                switch (datatype)
                {
                case 2:
                    datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size), 1);
                    matrix_B[k * col_B_size + l] = int8_to_32(datb);
                    break;
                case 0:
                    datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size), 1);
                    matrix_B[k * col_B_size + l] = fp8_to_32(datb);
                    break;
                case 1:
                    datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size) * 2, 2);
                    matrix_B[k * col_B_size + l] = fp16_to_32(datb);
                    break;
                }
            }
        }

// 바이어스 로드
#pragma omp parallel for collapse(2)
        for (int i = 0; i < row_A_size; i++)
        {
            for (int l = 0; l < col_B_size; l++)
            {
                BitsToFloat datc;
                datc.u32 = this->l1_intf->readDataMem(c_addr + (l + col_B_size * i) * 4, 4);
                bias_C[i * col_B_size + l] = datc.f;
            }
        }

        // OpenBLAS 행렬 곱셈
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    row_A_size, col_B_size, col_A_size,
                    1.0f, matrix_A.data(), col_A_size,
                    matrix_B.data(), col_B_size,
                    0.0f, matrix_C.data(), col_B_size);

// 바이어스 추가 및 fp16 변환하여 저장
#pragma omp parallel for collapse(2)
        for (int i = 0; i < row_A_size; i++)
        {
            for (int l = 0; l < col_B_size; l++)
            {
                float result = matrix_C[i * col_B_size + l] + bias_C[i * col_B_size + l];
                if(datatype == 2){
                    int32_t sum_int = result;
                    this->l1_intf->writeDataMem(c_addr + (l + col_B_size * i) * 4, sum_int, 4);
                }
                else{
                    uint16_t sum = fp32_to_16(result);
                    this->l1_intf->writeDataMem(r_addr + (l + col_B_size * i) * 2, sum, 2);
                }
            }
        }

        curr_cycle = MMC_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
        this->exec_cycle += MMC_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
    }

    // MMC_S 함수 (행렬곱 + 바이어스 추가, fp32 출력)
    void SPU::MMC_S(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size)
    {
        if((row_A_size == 0) ||(col_A_size == 0) || (col_B_size == 0)){
            LOG(LOG_WARNING) << "[WARNING] MMC.S matrix size 0 !!!(nothing to do)\n";
            return ;
        }
        //int num_threads = get_max_usable_threads();
        //set_gemm_threads(num_threads);
        set_gemm_threads(MAX_THREAD);

        uint32_t a_addr, b_addr, c_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        b_addr = LSPR[LSPR_SPM_ADDRB];
        c_addr = LSPR[LSPR_SPM_ADDRC];
        r_addr = LSPR[LSPR_SPM_ADDRR];
        std::vector<float> matrix_A(row_A_size * col_A_size);
        std::vector<float> matrix_B(col_A_size * col_B_size);
        std::vector<float> matrix_C(row_A_size * col_B_size);
        std::vector<float> bias_C(row_A_size * col_B_size);

// A, B 행렬 로드
#pragma omp parallel for collapse(2)
        for (int i = 0; i < row_A_size; i++)
        {
            for (int k = 0; k < col_A_size; k++)
            {
                uint16_t data;
                switch (datatype)
                {
                case 2:
                    data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                    matrix_A[i * col_A_size + k] = int8_to_32(data);
                    break;
                case 0:
                    data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                    matrix_A[i * col_A_size + k] = fp8_to_32(data);
                    break;
                case 1:
                    data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i) * 2, 2);
                    matrix_A[i * col_A_size + k] = fp16_to_32(data);
                    break;
                }
            }
        }

#pragma omp parallel for collapse(2)
        for (int k = 0; k < col_A_size; k++)
        {
            for (int l = 0; l < col_B_size; l++)
            {
                uint16_t datb;
                switch (datatype)
                {
                case 2:
                    datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size), 1);
                    matrix_B[k * col_B_size + l] = int8_to_32(datb);
                    break;
                case 0:
                    datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size), 1);
                    matrix_B[k * col_B_size + l] = fp8_to_32(datb);
                    break;
                case 1:
                    datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size) * 2, 2);
                    matrix_B[k * col_B_size + l] = fp16_to_32(datb);
                    break;
                }
            }
        }

// 바이어스 로드
#pragma omp parallel for collapse(2)
        for (int i = 0; i < row_A_size; i++)
        {
            for (int l = 0; l < col_B_size; l++)
            {
                BitsToFloat datc;
                float res;
                switch(datatype){
                    case 2:
                        datc.u32 = this->l1_intf->readDataMem(c_addr + (l + col_B_size * i) * 4, 4);
                        res = datc.u32;
                        bias_C[i * col_B_size + l] = res;
                        break;
                    default:
                        datc.u32 = this->l1_intf->readDataMem(c_addr + (l + col_B_size * i) * 4, 4);
                        bias_C[i * col_B_size + l] = datc.f;
                        break;
                }
            }
        }

        // OpenBLAS 행렬 곱셈
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    row_A_size, col_B_size, col_A_size,
                    1.0f, matrix_A.data(), col_A_size,
                    matrix_B.data(), col_B_size,
                    0.0f, matrix_C.data(), col_B_size);

// 바이어스 추가 및 fp32로 저장
#pragma omp parallel for collapse(2)
        for (int i = 0; i < row_A_size; i++)
        {
            for (int l = 0; l < col_B_size; l++)
            {

                BitsToFloat s;
                s.f = matrix_C[i * col_B_size + l] + bias_C[i * col_B_size + l];
                if(datatype == 2){
                    int32_t sum_int = s.f;
                    this->l1_intf->writeDataMem(c_addr + (l + col_B_size * i) * 4, sum_int, 4);
                }else{
                    this->l1_intf->writeDataMem(c_addr + (l + col_B_size * i) * 4, s.u32, 4);
                }
            }
        }

        curr_cycle = MMC_S_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
        this->exec_cycle += MMC_S_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
    }

    // MMC_O: Matrix-Vector 곱셈 + 바이어스 (B 행렬이 모두 1)
    void SPU::MMC_O(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size, uint16_t l0_addr)
    {
        if((row_A_size == 0) ||(col_A_size == 0) || (col_B_size == 0)){
            LOG(LOG_WARNING) << "[WARNING] MMC.O matrix size 0 !!!(nothing to do)\n";
            return ;
        }
        uint32_t a_addr, b_addr, c_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        b_addr = LSPR[LSPR_SPM_ADDRB];
        c_addr = LSPR[LSPR_SPM_ADDRC];
        r_addr = LSPR[LSPR_SPM_ADDRR];

        uint32_t total_operations = row_A_size * col_B_size * col_A_size;

        // OpenBLAS 사용 버전 (중간 크기 이상)
        if (total_operations >= 1024)
        {
            std::vector<float> matrix_A(row_A_size * col_A_size);
            std::vector<float> ones_vector(col_A_size, 1.0f);
            std::vector<float> bias_vector(row_A_size * col_B_size);
            std::vector<float> result_vector(row_A_size * col_B_size);

// A 행렬 로드 (OpenMP 병렬화)
#pragma omp parallel for collapse(2)
            for (int i = 0; i < row_A_size; i++)
            {
                for (int k = 0; k < col_A_size; k++)
                {
                    uint16_t data;
                    switch (datatype)
                    {
                    case 2:
                        data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                        matrix_A[i * col_A_size + k] = int8_to_32(data);
                        break;
                    case 0:
                        data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                        matrix_A[i * col_A_size + k] = fp8_to_32(data);
                        break;
                    case 1:
                        data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i) * 2, 2);
                        matrix_A[i * col_A_size + k] = fp16_to_32(data);
                        break;
                    default:
                        LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n";
                        matrix_A[i * col_A_size + k] = 0.0f;
                        break;
                    }
                }
            }

// 바이어스 로드 (OpenMP 병렬화)
#pragma omp parallel for collapse(2)
            for (int i = 0; i < row_A_size; i++)
            {
                for (int l = 0; l < col_B_size; l++)
                {
                    BitsToFloat datc;
                    //datc.u32 = this->l1_intf->readDataMem(c_addr + (i * col_B_size + l) * 4, 4);
                    datc.f = mxe_accum;
                    bias_vector[i * col_B_size + l] = datc.f;
                }
            }

// OpenBLAS를 사용한 행별 벡터 곱셈 (각 행 × [1,1,1,...,1])
#pragma omp parallel for
            for (int i = 0; i < row_A_size; i++)
            {
                for (int l = 0; l < col_B_size; l++)
                {
                    // 각 (i,l)에 대해 A의 i번째 행과 ones_vector의 내적 계산
                    float sum = cblas_sdot(col_A_size,
                                           &matrix_A[i * col_A_size], 1,
                                           ones_vector.data(), 1);
                    result_vector[i * col_B_size + l] = sum + bias_vector[i * col_B_size + l];
                }
            }

// 결과 저장 (fp16 변환)
#pragma omp parallel for collapse(2)
            for (int i = 0; i < row_A_size; i++)
            {
                for (int l = 0; l < col_B_size; l++)
                {
                    BitsToFloat sum;
                    sum.f = fp32_to_16(result_vector[i * col_B_size + l]);
                    //this->l1_intf->writeDataMem(r_addr + (l + col_B_size * i) * 4, sum.u32, 4);
                    mxe_accum = result_vector[i * col_B_size + l];
                    if(datatype == 2){
                        int32_t sum_int = sum.f;
                        this->l0_intf->writeDataMem(l0_addr*32 + (l + col_B_size * i) * 4, sum_int, 4);
                        for(int x = 1; x < 8 ; x++){
                            this-> l0_intf->writeDataMem(l0_addr*32 +4*x, 0x0, 4);
                        }
                    }else{
                        this->l0_intf->writeDataMem(l0_addr*32 + (l + col_B_size * i) * 2, fp32_to_16(sum.f), 2);
                        for(int x = 1; x < 16 ; x++){
                            this-> l0_intf->writeDataMem(l0_addr*32 +2*x, 0x0, 2);
                        }
                    }
                }
            }

            // 성능 카운터 업데이트
            for (int i = 0; i < row_A_size; i++)
            {
                for (int l = 0; l < col_B_size; l++)
                {
                    for (int k = 0; k < col_A_size; k++)
                    {
                        if (k % (datatype == 1 ? 16 : 32) == 0)
                        {
                            perf->codeMemoryRead();
                        }
                    }
                    if (l % 16 == 0)
                        perf->codeMemoryWrite();
                    if (l % 8 == 0)
                        perf->codeMemoryRead();
                }
            }
        }
        else if (total_operations < 1024)
        {
            // 작은 데이터: 원본 순차 처리
            uint16_t sum, data;
            BitsToFloat datc;
            float s, a;
            int t = 0;

            for (int i = 0; i < row_A_size; i++)
            {
                for (int l = 0; l < col_B_size; l++)
                {
                    //datc.u32 = this->l1_intf->readDataMem(c_addr + t, 4);
                    datc.f = mxe_accum;
                    s = 0;

                    for (int k = 0; k < col_A_size; k++)
                    {
                        switch (datatype)
                        {
                        case 2:
                            data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                            a = int8_to_32(data);
                            if (k % 32 == 0)
                                perf->codeMemoryRead();
                            break;
                        case 0:
                            data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                            a = fp8_to_32(data);
                            if (k % 32 == 0)
                                perf->codeMemoryRead();
                            break;
                        case 1:
                            data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i) * 2, 2);
                            a = fp16_to_32(data);
                            if (k % 16 == 0)
                                perf->codeMemoryRead();
                            break;
                        default:
                            LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n";
                            break;
                        }
                        s += a; // b = 1이므로 a*1 = a
                    }
                    s += datc.f;
                    BitsToFloat s_bit;
                    s_bit.f = s;
                    sum = fp32_to_16(s);
                    //this->l1_intf->writeDataMem(r_addr + (l + col_B_size * i) * 4, s_bit.u32, 4);
                    mxe_accum = s;
                    if(datatype == 2){
                        int32_t sum_int = s;
                        this->l0_intf->writeDataMem(l0_addr*32 + (l + col_B_size * i) * 4, sum_int, 4);
                        for(int x = 1; x < 8 ; x++){
                            this-> l0_intf->writeDataMem(l0_addr*32 +4*x, 0x0, 4);
                        }
                    }else{
                        this->l0_intf->writeDataMem(l0_addr*32 + (l + col_B_size * i) * 2, sum, 2);
                        for(int x = 1; x < 16 ; x++){
                            this-> l0_intf->writeDataMem(l0_addr*32 +2*x, 0x0, 2);
                        }
                    }
                    if (l % 16 == 0)
                        perf->codeMemoryWrite();
                    if (l % 8 == 0)
                        perf->codeMemoryRead();
                    t += 4;
                }
            }
        }
        else if (total_operations < 8192)
        {
// 중간 크기: 외부 루프 병렬화
#pragma omp parallel
            {
                uint16_t sum, data;
                BitsToFloat datc;
                float s, a;

#pragma omp for collapse(2) schedule(static)
                for (int i = 0; i < row_A_size; i++)
                {
                    for (int l = 0; l < col_B_size; l++)
                    {
                        int t = (i * col_B_size + l) * 4;
                        //datc.u32 = this->l1_intf->readDataMem(c_addr + t, 4);
                        datc.f = mxe_accum;
                        s = 0;
/*
// 내부 루프 벡터화
#pragma omp simd reduction(+ : s)
                        for (int k = 0; k < col_A_size; k++)
                        {
                            switch (datatype)
                            {
                            case 2:
                                data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                                a = int8_to_32(data);
                                if (k % 32 == 0)
                                {
#pragma omp critical
                                    perf->codeMemoryRead();
                                }
                                break;
                            case 0:
                                data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                                a = fp8_to_32(data);
                                if (k % 32 == 0)
                                {
#pragma omp critical
                                    perf->codeMemoryRead();
                                }
                                break;
                            case 1:
                                data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i) * 2, 2);
                                a = fp16_to_32(data);
                                if (k % 16 == 0)
                                {
#pragma omp critical
                                    perf->codeMemoryRead();
                                }
                                break;
                            default:
                                LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n";
                                a = 0;
                                break;
                            }
                            s += a;
                        }
                        */
                        int read_cnt32 = 0;   // 32B마다 read 카운트
                        int read_cnt16 = 0;   // 16B마다 read 카운트 (datatype==1용)

                        #pragma omp simd reduction(+ : s, read_cnt32, read_cnt16)
                        for (int k = 0; k < col_A_size; k++) {
                            switch (datatype) {
                            case 2:
                                data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                                a = int8_to_32(data);
                                if (k % 32 == 0) read_cnt32 += 1;   //  카운터만 증가
                                break;
                            case 0:
                                data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                                a = fp8_to_32(data);
                                if (k % 32 == 0) read_cnt32 += 1;   // 
                                break;
                            case 1:
                                data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i) * 2, 2);
                                a = fp16_to_32(data);
                                if (k % 16 == 0) read_cnt16 += 1;   // 
                                break;
                            default:
                                LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n";
                                a = 0;
                                break;
                            }
                            s += a;
                        }

                        // SIMD 루프 끝난 뒤(스칼라 영역) 카운터 반영
                        for (int t = 0; t < read_cnt32; ++t) perf->codeMemoryRead();
                        for (int t = 0; t < read_cnt16; ++t) perf->codeMemoryRead();


                        s += datc.f;
                        BitsToFloat s_bit;
                        s_bit.f = s;
                        sum = fp32_to_16(s);
                        //this->l1_intf->writeDataMem(r_addr + (l + col_B_size * i) * 4, s_bit.u32, 4);
                        mxe_accum = s;
                        if(datatype == 2){
                            int32_t sum_int = s;
                            this->l0_intf->writeDataMem(l0_addr*32 + (l + col_B_size * i) * 4, sum_int, 4);
                            for(int x = 1; x < 8 ; x++){
                                this-> l0_intf->writeDataMem(l0_addr*32 +4*x, 0x0, 4);
                            }
                        }else{
                            this->l0_intf->writeDataMem(l0_addr*32 + (l + col_B_size * i) * 2, sum, 2);
                            for(int x = 1; x < 16 ; x++){
                                this-> l0_intf->writeDataMem(l0_addr*32 +2*x, 0x0, 2);
                            }
                        }
                        if (l % 16 == 0)
                        {
#pragma omp critical
                            perf->codeMemoryWrite();
                        }
                        if (l % 8 == 0)
                        {
#pragma omp critical
                            perf->codeMemoryRead();
                        }
                    }
                }
            }
        }
        else
        {
            // 큰 데이터: 블록 기반 병렬화 + 캐시 최적화
            const int BLOCK_SIZE = 64;

#pragma omp parallel
            {
                uint16_t sum, data;
                BitsToFloat datc;
                float s, a;

#pragma omp for collapse(2) schedule(dynamic)
                for (int bi = 0; bi < row_A_size; bi += BLOCK_SIZE)
                {
                    for (int bl = 0; bl < col_B_size; bl += BLOCK_SIZE)
                    {

                        int i_max = (bi + BLOCK_SIZE < row_A_size) ? bi + BLOCK_SIZE : row_A_size;
                        int l_max = (bl + BLOCK_SIZE < col_B_size) ? bl + BLOCK_SIZE : col_B_size;

                        // 블록 내부 처리
                        for (int i = bi; i < i_max; i++)
                        {
                            for (int l = bl; l < l_max; l++)
                            {
                                int t = (i * col_B_size + l) * 4;
                                //datc.u32 = this->l1_intf->readDataMem(c_addr + t, 4);
                                datc.f = mxe_accum;
                                s = 0;

                                for (int k = 0; k < col_A_size; k++)
                                {
                                    switch (datatype)
                                    {
                                    case 2:
                                        data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                                        a = int8_to_32(data);
                                        if (k % 32 == 0)
                                        {
#pragma omp critical
                                            perf->codeMemoryRead();
                                        }
                                        break;
                                    case 0:
                                        data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                                        a = fp8_to_32(data);
                                        if (k % 32 == 0)
                                        {
#pragma omp critical
                                            perf->codeMemoryRead();
                                        }
                                        break;
                                    case 1:
                                        data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i) * 2, 2);
                                        a = fp16_to_32(data);
                                        if (k % 16 == 0)
                                        {
#pragma omp critical
                                            perf->codeMemoryRead();
                                        }
                                        break;
                                    default:
                                        LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n";
                                        a = 0;
                                        break;
                                    }
                                    s += a;
                                }
                                s += datc.f;
                                BitsToFloat s_bit;
                                s_bit.f = s;
                                sum = fp32_to_16(s);
                                //this->l1_intf->writeDataMem(r_addr + (l + col_B_size * i) * 4, s_bit.u32, 4);
                                mxe_accum = s;
                                if(datatype == 2){
                                    int32_t sum_int = s;
                                    this->l0_intf->writeDataMem(l0_addr*32 + (l + col_B_size * i) * 4, sum_int, 4);
                                    for(int x = 1; x < 8 ; x++){
                                        this-> l0_intf->writeDataMem(l0_addr*32 +4*x, 0x0, 4);
                                    }
                                }else{
                                    this->l0_intf->writeDataMem(l0_addr*32 + (l + col_B_size * i) * 2, sum, 2);
                                    for(int x = 1; x < 16 ; x++){
                                        this-> l0_intf->writeDataMem(l0_addr*32 +2*x, 0x0, 2);
                                    }
                                }

                                if (l % 16 == 0)
                                {
#pragma omp critical
                                    perf->codeMemoryWrite();
                                }
                                if (l % 8 == 0)
                                {
#pragma omp critical
                                    perf->codeMemoryRead();
                                }
                            }
                        }
                    }
                }
            }
        }

        curr_cycle = MMC_O_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
        this->exec_cycle += MMC_O_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
    }

    void SPU::MMC_V(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size, uint16_t l0_addr)
    {
        if((row_A_size == 0) ||(col_A_size == 0) || (col_B_size == 0)){
            LOG(LOG_WARNING) << "[WARNING] MMC.S matrix size 0 !!!(nothing to do)\n";
            return ;
        }
        //int num_threads = get_max_usable_threads();
        //set_gemm_threads(num_threads);
        set_gemm_threads(MAX_THREAD);

        uint32_t a_addr, b_addr, c_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        b_addr = LSPR[LSPR_SPM_ADDRB];
        c_addr = LSPR[LSPR_SPM_ADDRC];
        r_addr = LSPR[LSPR_SPM_ADDRR];
        std::vector<float> matrix_A(row_A_size * col_A_size);
        std::vector<float> matrix_B(col_A_size * col_B_size);
        std::vector<float> matrix_C(row_A_size * col_B_size);
        std::vector<float> bias_C(row_A_size * col_B_size);

// A, B 행렬 로드
#pragma omp parallel for collapse(2)
        for (int i = 0; i < row_A_size; i++)
        {
            for (int k = 0; k < col_A_size; k++)
            {
                uint16_t data;
                switch (datatype)
                {
                case 2:
                    data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                    matrix_A[i * col_A_size + k] = int8_to_32(data);
                    break;
                case 0:
                    data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                    matrix_A[i * col_A_size + k] = fp8_to_32(data);
                    break;
                case 1:
                    data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i) * 2, 2);
                    matrix_A[i * col_A_size + k] = fp16_to_32(data);
                    break;
                }
            }
        }

#pragma omp parallel for collapse(2)
        for (int k = 0; k < col_A_size; k++)
        {
            for (int l = 0; l < col_B_size; l++)
            {
                uint16_t datb;
                switch (datatype)
                {
                case 2:
                    datb = this->l1_intf->readDataMem(b_addr + (k * col_B_size + l), 1);
                    matrix_B[k * col_B_size + l] = int8_to_32(datb);
                    break;
                case 0:
                    datb = this->l1_intf->readDataMem(b_addr + (k * col_B_size + l), 1);
                    matrix_B[k * col_B_size + l] = fp8_to_32(datb);
                    break;
                case 1:
                    datb = this->l1_intf->readDataMem(b_addr + (k * col_B_size + l) * 2, 2);
                    matrix_B[k * col_B_size + l] = fp16_to_32(datb);
                    break;
                }
            }
        }

// 바이어스 로드
#pragma omp parallel for collapse(2)
        for (int i = 0; i < row_A_size; i++)
        {
            for (int l = 0; l < col_B_size; l++)
            {
                BitsToFloat datc;
                datc.f = mxe_accum;
                //datc.u32 = this->l1_intf->readDataMem(c_addr + (l + col_B_size * i) * 4, 4);
                bias_C[i * col_B_size + l] = datc.f;
            }
        }

        // OpenBLAS 행렬 곱셈
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    row_A_size, col_B_size, col_A_size,
                    1.0f, matrix_A.data(), col_A_size,
                    matrix_B.data(), col_B_size,
                    0.0f, matrix_C.data(), col_B_size);

// 바이어스 추가 및 fp32로 저장
#pragma omp parallel for collapse(2)
        for (int i = 0; i < row_A_size; i++)
        {
            for (int l = 0; l < col_B_size; l++)
            {
                BitsToFloat s;
                s.f = matrix_C[i * col_B_size + l] + bias_C[i * col_B_size + l];
                mxe_accum = s.f;
                //this->l1_intf->writeDataMem(c_addr + (l + col_B_size * i) * 4, s.u32, 4);
                if(datatype == 2){
                    int32_t sum_int = s.f;
                    this->l0_intf->writeDataMem(l0_addr*32 +(l + col_B_size * i) * 4, sum_int, 4);
                    for(int x = 1; x < 8 ; x++){
                        this-> l0_intf->writeDataMem(l0_addr*32 +4*x, 0x0, 4);
                    }
                }else{
                    this->l0_intf->writeDataMem(l0_addr*32 +(l + col_B_size * i) * 2, fp32_to_16(s.f), 2);
                    for(int x = 1; x < 16 ; x++){
                        this-> l0_intf->writeDataMem(l0_addr*32 +2*x, 0x0, 2);
                    }
                }
            }
        }

        curr_cycle = MMC_S_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
        this->exec_cycle += MMC_S_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
    }


    // MMC_T: 전치된 출력으로 행렬 곱셈 + 바이어스
    void SPU::MMC_T(uint16_t row_A_size, uint16_t col_A_size, uint16_t col_B_size)
    {
        if((row_A_size == 0) ||(col_A_size == 0) || (col_B_size == 0)){
            LOG(LOG_WARNING) << "[WARNING] MMC.T matrix size 0 !!!(nothing to do)\n";
            return ;
        }
        uint32_t a_addr, b_addr, c_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        b_addr = LSPR[LSPR_SPM_ADDRB];
        c_addr = LSPR[LSPR_SPM_ADDRC];
        r_addr = LSPR[LSPR_SPM_ADDRR];

        uint32_t total_operations = row_A_size * col_B_size * col_A_size;

        if (total_operations < 1024)
        {
            // 작은 데이터: 원본 순차 처리
            uint16_t sum, data, datb;
            BitsToFloat datc;
            float s, a, b;
            int t = 0;

            for (int i = 0; i < row_A_size; i++)
            {
                for (int l = 0; l < col_B_size; l++)
                {
                    datc.u32 = this->l1_intf->readDataMem(c_addr + t, 4);
                    s = 0;

                    for (int k = 0; k < col_A_size; k++)
                    {
                        switch (datatype)
                        {
                        case 2:
                            data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                            datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size), 1);
                            a = int8_to_32(data);
                            b = int8_to_32(datb);
                            if (k % 32 == 0)
                            {
                                perf->dataMemoryRead();
                                perf->dataMemoryRead();
                            }
                            break;
                        case 0:
                            data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                            datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size), 1);
                            a = fp8_to_32(data);
                            b = fp8_to_32(datb);
                            if (k % 32 == 0)
                            {
                                perf->dataMemoryRead();
                                perf->dataMemoryRead();
                            }
                            break;
                        case 1:
                            data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i) * 2, 2);
                            datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size) * 2, 2);
                            a = fp16_to_32(data);
                            b = fp16_to_32(datb);
                            if (k % 16 == 0)
                            {
                                perf->dataMemoryRead();
                                perf->dataMemoryRead();
                            }
                            break;
                        default:
                            LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n";
                            break;
                        }
                        s += (a * b);
                    }
                    float t_s;
                    switch(datatype){
                        case 2:
                            t_s = datc.u32;
                            s += t_s;
                            break;
                        default:
                            s += datc.f;
                            break;
                    }
                    sum = fp32_to_16(s);
                    // 전치된 출력: (i,l) -> (l,i)
                    if(datatype == 2){
                        int32_t sum_int = s;
                        this->l1_intf->writeDataMem(r_addr + (i + row_A_size * l) * 4, sum_int, 4);
                    }else{
                        this->l1_intf->writeDataMem(r_addr + (i + row_A_size * l) * 2, sum, 2);
                    }
                    if (l % 16 == 0)
                        perf->codeMemoryWrite();
                    if (l % 8 == 0)
                        perf->codeMemoryRead();
                    t += 4;
                }
            }
        }
        else if (total_operations < 8192)
        {
// 중간 크기: 외부 루프 병렬화
#pragma omp parallel
            {
                uint16_t sum, data, datb;
                BitsToFloat datc;
                float s, a, b;

#pragma omp for collapse(2) schedule(static)  
                for (int i = 0; i < row_A_size; i++)
                {
                    for (int l = 0; l < col_B_size; l++)
                    {
                        int t = (i * col_B_size + l) * 4;
                        datc.u32 = this->l1_intf->readDataMem(c_addr + t, 4);
                        s = 0;

                        for (int k = 0; k < col_A_size; k++)
                        {
                            switch (datatype)
                            {
                            case 2:
                                data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                                datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size), 1);
                                a = int8_to_32(data);
                                b = int8_to_32(datb);
                                if (k % 32 == 0)
                                {
#pragma omp critical
                                    {
                                        perf->dataMemoryRead();
                                        perf->dataMemoryRead();
                                    }
                                }
                                break;
                            case 0:
                                data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                                datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size), 1);
                                a = fp8_to_32(data);
                                b = fp8_to_32(datb);
                                if (k % 32 == 0)
                                {
#pragma omp critical
                                    {
                                        perf->dataMemoryRead();
                                        perf->dataMemoryRead();
                                    }
                                }
                                break;
                            case 1:
                                data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i) * 2, 2);
                                datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size) * 2, 2);
                                a = fp16_to_32(data);
                                b = fp16_to_32(datb);
                                if (k % 16 == 0)
                                {
#pragma omp critical
                                    {
                                        perf->dataMemoryRead();
                                        perf->dataMemoryRead();
                                    }
                                }
                                break;
                            default:
                                LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n";
                                a = b = 0;
                                break;
                            }
                            s += (a * b);
                        }
                        float t_s;
                        switch(datatype){
                            case 2:
                                t_s = datc.u32;
                                s += t_s;
                                break;
                            default:
                                s += datc.f;
                                break;
                        }
                        sum = fp32_to_16(s);
                        // 전치된 출력
                        if(datatype == 2){
                            int32_t sum_int = s;
                            this->l1_intf->writeDataMem(r_addr + (i + row_A_size * l) * 4, sum_int, 4);
                        }else{
                            this->l1_intf->writeDataMem(r_addr + (i + row_A_size * l) * 2, sum, 2);
                        }

                        if (l % 16 == 0)
                        {
#pragma omp critical
                            perf->codeMemoryWrite();
                        }
                        if (l % 8 == 0)
                        {
#pragma omp critical
                            perf->codeMemoryRead();
                        }
                    }
                }
            }
        }
        else
        {
            // 큰 데이터: 블록 기반 병렬화 + 캐시 최적화
            const int BLOCK_SIZE = 64;

#pragma omp parallel
            {
                uint16_t sum, data, datb;
                BitsToFloat datc;
                float s, a, b;

#pragma omp for collapse(2) schedule(dynamic)
                for (int bi = 0; bi < row_A_size; bi += BLOCK_SIZE)
                {
                    for (int bl = 0; bl < col_B_size; bl += BLOCK_SIZE)
                    {

                        int i_max = (bi + BLOCK_SIZE < row_A_size) ? bi + BLOCK_SIZE : row_A_size;
                        int l_max = (bl + BLOCK_SIZE < col_B_size) ? bl + BLOCK_SIZE : col_B_size;

                        // 블록 내부 처리 (전치된 출력을 고려한 순서)
                        for (int l = bl; l < l_max; l++)
                        {
                            for (int i = bi; i < i_max; i++)
                            {
                                int t = (i * col_B_size + l) * 4;
                                datc.u32 = this->l1_intf->readDataMem(c_addr + t, 4);
                                s = 0;

                                for (int k = 0; k < col_A_size; k++)
                                {
                                    switch (datatype)
                                    {
                                    case 2:
                                        data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                                        datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size), 1);
                                        a = int8_to_32(data);
                                        b = int8_to_32(datb);
                                        if (k % 32 == 0)
                                        {
#pragma omp critical
                                            {
                                                perf->dataMemoryRead();
                                                perf->dataMemoryRead();
                                            }
                                        }
                                        break;
                                    case 0:
                                        data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i), 1);
                                        datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size), 1);
                                        a = fp8_to_32(data);
                                        b = fp8_to_32(datb);
                                        if (k % 32 == 0)
                                        {
#pragma omp critical
                                            {
                                                perf->dataMemoryRead();
                                                perf->dataMemoryRead();
                                            }
                                        }
                                        break;
                                    case 1:
                                        data = this->l1_intf->readDataMem(a_addr + (k + col_A_size * i) * 2, 2);
                                        datb = this->l1_intf->readDataMem(b_addr + (k + l * col_A_size) * 2, 2);
                                        a = fp16_to_32(data);
                                        b = fp16_to_32(datb);
                                        if (k % 16 == 0)
                                        {
#pragma omp critical
                                            {
                                                perf->dataMemoryRead();
                                                perf->dataMemoryRead();
                                            }
                                        }
                                        break;
                                    default:
                                        LOG(LOG_WARNING) << "[WARNING] Unknown Data Type!!\n";
                                        a = b = 0;
                                        break;
                                    }
                                    s += (a * b);
                                }
                                float t_s;
                                switch(datatype){
                                    case 2:
                                        t_s = datc.u32;
                                        s += t_s;
                                        break;
                                    default:
                                        s += datc.f;
                                        break;
                                }
                                sum = fp32_to_16(s);
                                // 전치된 출력으로 쓰기 (캐시 친화적 순서)
                                if(datatype == 2){
                                    int32_t sum_int = s;
                                    this->l1_intf->writeDataMem(r_addr + (i + row_A_size * l) * 4, sum_int, 4);
                                }else{
                                    this->l1_intf->writeDataMem(r_addr + (i + row_A_size * l) * 2, sum, 2);
                                }

                                if (l % 16 == 0)
                                {
#pragma omp critical
                                    perf->codeMemoryWrite();
                                }
                                if (l % 8 == 0)
                                {
#pragma omp critical
                                    perf->codeMemoryRead();
                                }
                            }
                        }
                    }
                }
            }
        }

        curr_cycle = MMC_T_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
        this->exec_cycle += MMC_T_EXEC_CYCLE(row_A_size, col_A_size, col_B_size);
    }

   
   void SPU::IM2COL_D(uint16_t kernel_size,
                       uint16_t dialate,
                       uint16_t stride,
                       uint16_t num_channel,
                       uint16_t row_A_size,
                       uint16_t col_A_size)
{
    // ---- dialate → 유효 커널 크기 산출 (원래 로직 유지) ----
    uint32_t byte_num = (datatype == 1)?2 : 1; 
    uint16_t filt_size;
    switch (dialate) {
    case 0:
        LOG(LOG_WARNING) << "[WARNING] dialation set to 0 im2col ignored - IM2COL.D\n";
        return;
    case 1:
        filt_size = kernel_size;
        break;
    case 2:
        filt_size = 5;
        if (kernel_size != 3) {
            kernel_size = 3;
            LOG(LOG_WARNING) << "[WARNING] kernel size must set 3 when dialation used - IM2COL.D\n";
        }
        break;
    case 3:
        filt_size = 7;
        if (kernel_size != 3) {
            kernel_size = 3;
            LOG(LOG_WARNING) << "[WARNING] kernel size must set 3 when dialation used - IM2COL.D\n";
        }
        break;
    default:
        LOG(LOG_WARNING) << "[WARNING] dialation too big im2col ignored - IM2COL.D\n";
        return;
    }
    if(((row_A_size - kernel_size)%stride) != 0){
        LOG(LOG_WARNING) << "[WARNING] Input size does not match kernel size and stride! - IM2COL.D\n";
        return;
    }

    const uint32_t a_addr = LSPR[LSPR_SPM_ADDRA]; // dst (im2col)
    const uint32_t r_addr = LSPR[LSPR_SPM_ADDRR]; // src (feature map)

    // ---- 출력 타일 개수 (j,k 슬라이딩 횟수) ----
    // (j + filt_size) <= row_A_size  => j = 0, stride, ..., row_A_size - filt_size
    // (k + filt_size) <= col_A_size  => k = 0, stride, ..., col_A_size - filt_size
    const int out_h = (row_A_size >= filt_size)
                      ? ( (row_A_size - filt_size) / stride + 1 )
                      : 0;
    const int out_w = (col_A_size >= filt_size)
                      ? ( (col_A_size - filt_size) / stride + 1 )
                      : 0;

    if (out_h <= 0 || out_w <= 0) {
        // 유효 출력 없음
        curr_cycle = IM2COL_EXEC_CYCLE(kernel_size, stride, row_A_size, col_A_size);
        this->exec_cycle += curr_cycle;
        return;
    }

    const int M = static_cast<int>(num_channel);
    const int Kh = static_cast<int>(kernel_size);
    const int Kw = static_cast<int>(kernel_size);
    const int K2 = Kh * Kw;

    // perf 카운터를 병렬부에서 직접 호출하면 오버헤드/경쟁이 큼.
    // 스레드 로컬 합계를 모아 나중에 한 번에 반영합니다.
    long perf_read_cnt  = 0;
    long perf_write_cnt = 0;

    // ---- 병렬화 포인트: 채널 × (출력 위치 oh, ow) ----
    // 각 (i, oh, ow) 슬라이딩 창에 대해 커널 내 (l, m)을 직렬로 복사
    // 목적지 오프셋은 (i,oh,ow,l,m)로 수학적으로 계산 → 전역 t 불필요
    #pragma omp parallel for collapse(3) reduction(+:perf_read_cnt, perf_write_cnt) schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {

                const int j = oh * stride; // 원래의 j
                const int k = ow * stride; // 원래의 k

                // 원 코드의 perf 카운터 트리거: (k % 16 == 0) 일 때 read/write
                if ( (k & 15) == 0 ) {
                    ++perf_read_cnt;
                    ++perf_write_cnt;
                }

                // (l, m) 내에서 커널 요소를 순차 복사
                for (int l = 0; l < Kh; ++l) {
                    for (int m = 0; m < Kw; ++m) {
                        // ---- 소스 주소 (원래 식을 그대로 사용) ----
                        // r_addr + ( (row_A*col_A*i) + (col_A*j + k) + (dialate*l*col_A + dialate*m) ) * 2
                        const uint32_t src_index =
                            (static_cast<uint32_t>(row_A_size) * col_A_size) * i
                          + (static_cast<uint32_t>(col_A_size) * j + k)
                          + (static_cast<uint32_t>(dialate) * l * col_A_size + static_cast<uint32_t>(dialate) * m);
                        const uint32_t src_addr = r_addr + src_index * byte_num;

                        const uint16_t rdata = this->l1_intf->readDataMem(src_addr, byte_num);

                        // ---- 목적지 주소: 전역 t 대신 (i, oh, ow, l, m)로 선형화 ----
                        // 원래 t 증가 순서는 i→j→k→l→m
                        // dst_index = ((((i*out_h + oh)*out_w + ow) * K2) + (l*Kw + m))
                        const uint32_t dst_index =
                            ( ( (i * out_h + oh) * out_w + ow ) * K2 )
                          + ( l * Kw + m );
                        const uint32_t dst_addr = a_addr + dst_index * byte_num;
                        this->l1_intf->writeDataMem(dst_addr, rdata, byte_num);
                    }
                }
            }
        }
    }

    // perf 누적 반영(직렬 구간에서 수행)
    for (long c = 0; c < perf_read_cnt;  ++c) perf->codeMemoryRead();
    for (long c = 0; c < perf_write_cnt; ++c) perf->codeMemoryWrite();

    curr_cycle = IM2COL_EXEC_CYCLE(kernel_size, stride, row_A_size, col_A_size);
    this->exec_cycle += curr_cycle;
    }

    // 최적화된 IM2COL 함수
    void SPU::IM2COL_N(uint16_t kernel_size, uint16_t dialate, uint16_t stride, uint16_t num_channel,
                     uint16_t row_A_size, uint16_t col_A_size)
    {
        uint32_t byte_num = (datatype == 1)?2 : 1; 
        // 파라미터 검증 및 필터 크기 계산
        uint16_t filt_size;
        switch (dialate)
        {
        case 0:
            LOG(LOG_WARNING) << "[WARNING] dialation set to 0 im2col ignored - IM2COL.N\n";
            return;
        case 1:
            filt_size = kernel_size;
            break;
        case 2:
            filt_size = 5;
            if (kernel_size != 3)
            {
                kernel_size = 3;
                LOG(LOG_WARNING) << "[WARNING] kernel size must set 3 when dialation used - IM2COL.N\n";
            }
            break;
        case 3:
            filt_size = 7;
            if (kernel_size != 3)
            {
                kernel_size = 3;
                LOG(LOG_WARNING) << "[WARNING] kernel size must set 3 when dialation used - IM2COL.N\n";
            }
            break;
        default:
            LOG(LOG_WARNING) << "[WARNING] dialation too big im2col ignored - IM2COL.N\n";
            return;
        }
        if(((row_A_size - kernel_size)%stride) != 0){
            LOG(LOG_WARNING) << "[WARNING] Input size does not match kernel size and stride! - IM2COL.N\n";
            return;
        }
        uint32_t a_addr = LSPR[LSPR_SPM_ADDRA];
        uint32_t r_addr = LSPR[LSPR_SPM_ADDRR];

        uint16_t ksq = kernel_size * kernel_size;
        uint32_t out_rows = (row_A_size - filt_size) / stride + 1;
        uint32_t out_cols = (col_A_size - filt_size) / stride + 1;
        uint32_t total_patches = out_rows * out_cols;
        uint32_t total_elements = total_patches * num_channel * ksq;

        if (total_elements < 1024)
        {
            // 작은 데이터: 원본 순차 처리
            int t = 0;
            for (int j = 0; (j + filt_size) <= row_A_size; j += stride)
            {
                for (int k = 0; (k + filt_size) <= col_A_size; k += stride)
                {
                    for (int i = 0; i < num_channel; i++)
                    {
                        for (int l = 0; l < kernel_size; l++)
                        {
                            for (int m = 0; m < kernel_size; m++)
                            {
                                uint32_t src_addr = r_addr + ((row_A_size * col_A_size * i) +
                                                              (col_A_size * j + k) +
                                                              (dialate * l * col_A_size + dialate * m)) *
                                                                 byte_num;
                                uint16_t rdata = this->l1_intf->readDataMem(src_addr, byte_num);
                                this->l1_intf->writeDataMem(a_addr + t, rdata, byte_num);
                                t += byte_num;
                            }
                        }
                        if (k % 16 == 0)
                        {
                            perf->codeMemoryRead();
                            perf->codeMemoryWrite();
                        }
                    }
                }
            }
        }
        else if (total_elements < 8192)
        {
// 중간 크기: 외부 루프 병렬화
#pragma omp parallel
            {
#pragma omp for collapse(2) schedule(static)
                for (int j_idx = 0; j_idx < out_rows; j_idx++)
                {
                    for (int k_idx = 0; k_idx < out_cols; k_idx++)
                    {
                        int j = j_idx * stride;
                        int k = k_idx * stride;
                        int patch_base = (j_idx * out_cols + k_idx) * num_channel * ksq;

                        for (int i = 0; i < num_channel; i++)
                        {
                            int channel_base = patch_base + i * ksq;
                            int elem_idx = 0;

                            for (int l = 0; l < kernel_size; l++)
                            {
                                for (int m = 0; m < kernel_size; m++)
                                {
                                    uint32_t src_addr = r_addr + ((row_A_size * col_A_size * i) +
                                                                  (col_A_size * j + k) +
                                                                  (dialate * l * col_A_size + dialate * m)) *
                                                                     byte_num;
                                    uint16_t rdata = this->l1_intf->readDataMem(src_addr, byte_num);
                                    this->l1_intf->writeDataMem(a_addr + (channel_base + elem_idx) * byte_num, rdata, byte_num);
                                    elem_idx++;
                                }
                            }

                            if (k_idx % 16 == 0)
                            {
#pragma omp critical
                                {
                                    perf->codeMemoryRead();
                                    perf->codeMemoryWrite();
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            // 큰 데이터: 블록 기반 캐시 최적화
            const int BLOCK_SIZE = 32;

#pragma omp parallel
            {
#pragma omp for collapse(3) schedule(dynamic)
                for (int bi = 0; bi < out_rows; bi += BLOCK_SIZE)
                {
                    for (int bk = 0; bk < out_cols; bk += BLOCK_SIZE)
                    {
                        for (int i = 0; i < num_channel; i++)
                        {

                            int i_max = (bi + BLOCK_SIZE < out_rows) ? bi + BLOCK_SIZE : out_rows;
                            int k_max = (bk + BLOCK_SIZE < out_cols) ? bk + BLOCK_SIZE : out_cols;

                            // 블록 내부 처리 (캐시 친화적 순서)
                            for (int j_idx = bi; j_idx < i_max; j_idx++)
                            {
                                for (int k_idx = bk; k_idx < k_max; k_idx++)
                                {
                                    int j = j_idx * stride;
                                    int k = k_idx * stride;
                                    int patch_base = (j_idx * out_cols + k_idx) * num_channel * ksq;
                                    int channel_base = patch_base + i * ksq;
                                    int elem_idx = 0;

                                    // 커널 순회 (벡터화 가능)
                                    for (int l = 0; l < kernel_size; l++)
                                    {
                                        for (int m = 0; m < kernel_size; m++)
                                        {
                                            uint32_t src_addr = r_addr + ((row_A_size * col_A_size * i) +
                                                                          (col_A_size * j + k) +
                                                                          (dialate * l * col_A_size + dialate * m)) *
                                                                             byte_num;
                                            uint16_t rdata = this->l1_intf->readDataMem(src_addr, byte_num);
                                            this->l1_intf->writeDataMem(a_addr + (channel_base + elem_idx) * byte_num, rdata, byte_num);
                                            elem_idx++;
                                        }
                                    }
                                }
                            }

#pragma omp critical
                            {
                                int ops_count = ((i_max - bi) * (k_max - bk)) / 16;
                                for (int op = 0; op < ops_count; op++)
                                {
                                    perf->codeMemoryRead();
                                    perf->codeMemoryWrite();
                                }
                            }
                        }
                    }
                }
            }
        }

        curr_cycle = IM2COL_EXEC_CYCLE(kernel_size, stride, row_A_size, col_A_size);
        this->exec_cycle += curr_cycle;
    }

    void SPU::SASMD(uint16_t scalar_value, uint32_t vector_size, int funct)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;

        a_addr = LSPR[LSPR_SPM_ADDRA];

        r_addr = LSPR[LSPR_SPM_ADDRR];

        uint16_t data, datr;
        float a, b, s;
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i * 2, 2);
            if (i % 16 == 0)
                perf->codeMemoryRead();
            a = fp16_to_32(data);
            b = fp16_to_32(scalar_value);
            switch (funct)
            {
            case 0:
                s = a + b;
                break; // sadd
            case 1:
                s = a - b;
                break; // ssub
            case 2:
                s = a * b;
                break; // smul
            case 3:
                s = a / b;
                break; // sdiv
            }

            datr = fp32_to_16(s);
            this->l1_intf->writeDataMem(r_addr + i * 2, datr, 2);
            if (i % 16 == 0)
                perf->codeMemoryWrite();
        }
        if (funct == 3)
        {
            curr_cycle = SADD_EXEC_CYCLE(vector_size);
            this->exec_cycle += SADD_EXEC_CYCLE(vector_size);
        }
        else
        {
            curr_cycle = SDIV_EXEC_CYCLE(vector_size);
            this->exec_cycle += SDIV_EXEC_CYCLE(vector_size);
        }
    }

    void SPU::SFMADD(uint16_t scalar_value0, uint16_t scalar_value1, uint32_t vector_size)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        r_addr = LSPR[LSPR_SPM_ADDRR];

        uint16_t data, datr;
        float a, b, c, s;
        b = fp16_to_32(scalar_value0);
        c = fp16_to_32(scalar_value1);
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i * 2, 2);
            if (i % 16 == 0)
                perf->codeMemoryRead();
            if (i % 16 == 0)
                perf->codeMemoryWrite();

            a = fp16_to_32(data);
            s = a * b + c;
            datr = fp32_to_16(s);
            this->l1_intf->writeDataMem(r_addr + i * 2, datr, 2);
        }
        curr_cycle = SADD_EXEC_CYCLE(vector_size);
        this->exec_cycle += SADD_EXEC_CYCLE(vector_size);
    }

    void SPU::SMAX(uint16_t prev_max, uint32_t vector_size, uint16_t l0_addr)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;

        a_addr = LSPR[LSPR_SPM_ADDRA];

        r_addr = LSPR[LSPR_SPM_ADDRR];

        uint16_t data, datr;
        float a, b;
        b = fp16_to_32(prev_max);
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i * 2, 2);
            if (i % 16 == 0)
                perf->codeMemoryRead();
            a = fp16_to_32(data);
            b = (b > a) ? b : a;
        }
        datr = fp32_to_16(b);
        //LSPR[LSPR_RESULT] = datr;
        l0_intf -> writeDataMem(l0_addr*32, datr, 2);
        for(int x = 1; x < 16 ; x++){
            this-> l0_intf->writeDataMem(l0_addr*32 +2*x, 0x0, 2);
        }
        curr_cycle = SMAX_EXEC_CYCLE(vector_size);
        this->exec_cycle += SMAX_EXEC_CYCLE(vector_size);
    }

    void SPU::SMIN(uint16_t prev_min, uint32_t vector_size, uint16_t l0_addr)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];

        r_addr = LSPR[LSPR_SPM_ADDRR];

        uint16_t data, datr;
        float a, b;
        b = fp16_to_32(prev_min);
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i * 2, 2);
            if (i % 16 == 0)
                perf->codeMemoryRead();

            a = fp16_to_32(data);
            b = (b < a) ? b : a;
        }
        datr = fp32_to_16(b);
        //LSPR[LSPR_RESULT] = datr;
        l0_intf->writeDataMem(l0_addr*32, datr, 2);
        for(int x = 1; x < 16 ; x++){
            this-> l0_intf->writeDataMem(l0_addr*32 +2*x, 0x0, 2);
        }
        curr_cycle = SMAX_EXEC_CYCLE(vector_size);
        this->exec_cycle += SMAX_EXEC_CYCLE(vector_size);
    }

    void SPU::SASMD_IMM(uint16_t scalar_value, uint16_t input_addr,uint16_t l0_addr, int funct)
    {
        float a[16];
        float r[16];
        uint16_t result[16];
        float b;
        b = fp16_to_32(scalar_value);
        for (int i = 0; i < 16; i++)
        {
            a[i] = fp16_to_32(l0_intf->readDataMem(input_addr*32 + 2*i , 2));
            switch (funct)
            {
            case 0:
                r[i] = a[i] + b;
                break;
            case 1:
                r[i] = a[i] - b;
                break;
            case 2:
                r[i] = a[i] * b;
                break;
            case 3:
                r[i] = a[i] / b;
                break;
            }
            result[i] = fp32_to_16(r[i]);
            l0_intf ->writeDataMem(l0_addr*32 + i*2,result[i], 2 );
        }
        //rd = ((uint64_t)result[3] << 48) | ((uint64_t)result[2] << 32) | ((uint64_t)result[1] << 16) | ((uint64_t)result[0]);
        //LSPR[LSPR_RESULT] = rd;
        if (funct == 3)
        {
            curr_cycle = SDIV_IMM_EXEC_CYCLE;
            this->exec_cycle += SDIV_IMM_EXEC_CYCLE;
        }
        {
            curr_cycle = SADD_IMM_EXEC_CYCLE;
            this->exec_cycle += SADD_IMM_EXEC_CYCLE;
        }
    }

    void SPU::SFMADD_IMM(uint16_t scalar_value0, uint16_t scalar_value1, uint16_t input_addr, uint16_t l0_addr)
    {
        float a[16];
        float r[16];
        uint16_t result[16];
        float b, c;
        b = fp16_to_32(scalar_value0);
        c = fp16_to_32(scalar_value1);
        for (int i = 0; i < 16; i++)
        {
            a[i] = fp16_to_32(l0_intf->readDataMem(input_addr*32 +2*i, 2));
            r[i] = a[i] * b + c;
            result[i] = fp32_to_16(r[i]);
            l0_intf ->writeDataMem(l0_addr*32 + i*2,result[i], 2 );
        }
        curr_cycle = SADD_IMM_EXEC_CYCLE;
        this->exec_cycle += SADD_IMM_EXEC_CYCLE;
    }

    void SPU::SMAX_IMM(uint16_t prev_max, uint16_t input_addr, uint16_t l0_addr)
    {
        float a, b;
        uint16_t result;
        b = fp16_to_32(prev_max);
        for(int i = 0; i < 16 ; i++){
            a = fp16_to_32(l0_intf->readDataMem(input_addr*32 + 2*i , 2));
            b = (b > a) ? b : a;
        }
        result = fp32_to_16(b);
        l0_intf->writeDataMem(l0_addr*32, result,2);
        for(int x = 1; x < 16 ; x++){
            this-> l0_intf->writeDataMem(l0_addr*32 +2*x, 0x0, 2);
        }
        curr_cycle = SMAX_IMM_EXEC_CYCLE;
        this->exec_cycle += SMAX_IMM_EXEC_CYCLE;
    }

    void SPU::SMIN_IMM(uint16_t prev_min, uint16_t input_addr, uint16_t l0_addr)
    {
        float a, b;
        uint16_t result;
        b = fp16_to_32(prev_min);
        for(int i = 0; i < 16 ; i++){
            a = fp16_to_32(l0_intf->readDataMem(input_addr*32 + 2*i , 2));
            b = (b < a) ? b : a;
        }
        result = fp32_to_16(b);
        l0_intf->writeDataMem(l0_addr *32, result,2);
        for(int x = 1; x < 16 ; x++){
            this-> l0_intf->writeDataMem(l0_addr*32 +2*x, 0x0, 2);
        }
        curr_cycle = SMAX_IMM_EXEC_CYCLE;
    }

    void SPU::VASMD(uint32_t vector_size, int funct)
    {
        uint64_t spm_addr;
        uint32_t a_addr, b_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        b_addr = LSPR[LSPR_SPM_ADDRB];
        r_addr = LSPR[LSPR_SPM_ADDRR];

        uint16_t data, datb, datr;
        float a, b, c, r;
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i * 2, 2);
            datb = this->l1_intf->readDataMem(b_addr + i * 2, 2);
            if (i % 16 == 0)
            {
                perf->codeMemoryRead();
                perf->codeMemoryRead();
            }
            a = fp16_to_32(data);
            b = fp16_to_32(datb);
            switch (funct)
            {
            case 0:
                r = a + b;
                break;
            case 1:
                r = a - b;
                break;
            case 2:
                r = a * b;
                break;
            case 3:
                r = a / b;
                break;
            }
            datr = fp32_to_16(r);
            this->l1_intf->writeDataMem(r_addr + i * 2, datr, 2);
            if (i % 16 == 0)
                perf->codeMemoryWrite();
        }
        curr_cycle = SADD_EXEC_CYCLE(vector_size);
        this->exec_cycle += SADD_EXEC_CYCLE(vector_size);
    }


    void SPU::VFMADD(uint32_t vector_size)
    {
        uint64_t spm_addr;
        uint32_t a_addr, b_addr, c_addr, r_addr;

        a_addr = LSPR[LSPR_SPM_ADDRA];
        b_addr = LSPR[LSPR_SPM_ADDRB];
        c_addr = LSPR[LSPR_SPM_ADDRC];
        r_addr = LSPR[LSPR_SPM_ADDRR];

        uint16_t data, datb, datc, datr;
        float a, b, c, r;
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i * 2, 2);
            datb = this->l1_intf->readDataMem(b_addr + i * 2, 2);
            datc = this->l1_intf->readDataMem(c_addr + i * 2, 2);
            if (i % 16 == 0)
            {
                perf->codeMemoryRead();
                perf->codeMemoryRead();
                perf->codeMemoryRead();
            }
            a = fp16_to_32(data);
            b = fp16_to_32(datb);
            c = fp16_to_32(datc);

            r = a * b + c;
            datr = fp32_to_16(r);
            this->l1_intf->writeDataMem(r_addr + i * 2, datr, 2);
            if (i % 16 == 0)
                perf->codeMemoryWrite();
        }
        curr_cycle = SADD_EXEC_CYCLE(vector_size);
        this->exec_cycle += SADD_EXEC_CYCLE(vector_size);
    }

    void SPU::VFUNC(uint32_t vector_size,uint16_t op2_sel, int funct)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        r_addr = LSPR[LSPR_SPM_ADDRR];
        uint16_t data, datr, accum;
        uint16_t stage0[16];
        uint16_t stage1[16];
        uint16_t stage2[16];
        uint16_t stage3[16];
        uint16_t stage4[16];
        float a, r,x;
        r  = 0;
        accum = 0;
        //#pragma omp parallel for
        for (int k = 0; k < vector_size; k++)
        {
            data = this->l1_intf->readDataMem(a_addr + (k * 2), 2);
            if (k % 16 == 0)
            {
                perf->codeMemoryRead();
                perf->codeMemoryWrite();
            }
            a = fp16_to_32(data);
            switch(funct){
                case 0: //sqrt
                    if(!hw_mode){
                        r = std::sqrt(a);
                    }else{
                        r = fp16_to_32(HW_SQRT(data));
                    }
                    curr_cycle = VSQRT_EXEC_CYCLE(vector_size);
                    this->exec_cycle += VSQRT_EXEC_CYCLE(vector_size);
                    break;
                case 1: //exp
                    if(!hw_mode){
                        switch(op2_sel&0x3){
                            case  0: r = std::exp(a); break;
                            case  1: r = std::exp2(a); break;
                            case  2: r = std::pow(3.0,a); break;
                            default: r = std::pow(5.0,a); break;
                        }
                    }else{
                        r = fp16_to_32(HW_EXP(data,(op2_sel & 0x3)));
                    }
                    
                    curr_cycle = VEXP_EXEC_CYCLE(vector_size);
                    this->exec_cycle += VEXP_EXEC_CYCLE(vector_size);
                    break;
                case 2: //ln
                    if(!hw_mode){
                        switch(op2_sel & 0x3){
                            case 0 : r = std::log(a); break;
                            case 1 : r = std::log2(a); break;
                            case 2 : x = 1.0/std::log(3.0);
                                     r = std::log(a)*x; break;
                            default: r = std::log10(a); break;
                        }
                    }else{
                        r = fp16_to_32(HW_LN(data,(op2_sel& 0x3)));
                    }
                    curr_cycle = VLN_EXEC_CYCLE(vector_size);
                    this->exec_cycle += VLN_EXEC_CYCLE(vector_size);
                    break;
                case 3: //abs
                    r = std::abs(a);
                    curr_cycle = VABS_EXEC_CYCLE(vector_size);
                    this->exec_cycle += VABS_EXEC_CYCLE(vector_size);
                    break;
                case 4: //neg
                    r = -a;
                    curr_cycle = VNEG_EXEC_CYCLE(vector_size);
                    this->exec_cycle += VNEG_EXEC_CYCLE(vector_size);
                    break;
                case 5: //sign
                    r = (0 < a) - (a < 0);
                    curr_cycle = VSIGN_EXEC_CYCLE(vector_size);
                    this->exec_cycle += VSIGN_EXEC_CYCLE(vector_size);
                    break;
                case 6: //step
                    r = (a > 0)?  1 : 0;
                    curr_cycle = VSTEP_EXEC_CYCLE(vector_size);
                    this->exec_cycle += VSTEP_EXEC_CYCLE(vector_size); 
                    break;
                case 7: //ceil
                    r = std::ceil(a);
                    curr_cycle = VCEIL_EXEC_CYCLE(vector_size);
                    this->exec_cycle += VCEIL_EXEC_CYCLE(vector_size);
                    break;
                case 8: //trunc
                    r = std::trunc(a);
                    curr_cycle = VTRUNC_EXEC_CYCLE(vector_size);
                    this->exec_cycle += VTRUNC_EXEC_CYCLE(vector_size);
                    break;
                case 9: //floor
                    r = std::floor(a);
                    curr_cycle = VFLOOR_EXEC_CYCLE(vector_size);
                    this->exec_cycle += VFLOOR_EXEC_CYCLE(vector_size);
                    break;
                case 10: //rne
                    r = std::rint(a);
                    curr_cycle = VRNE_EXEC_CYCLE(vector_size);
                    this->exec_cycle += VRNE_EXEC_CYCLE(vector_size);
                    break;
                case 11: //accumsum
                    if(!hw_mode){
                        r += a;
                    }else{//scan tree algorithm
                        int temp;
                        temp = (k+1)%16;
                        switch(temp){
                            case 1: stage0[0] = data; 
                                    datr = fp32_to_16((fp16_to_32(stage0[0]) + fp16_to_32(accum)));
                                    break;
                            case 2: stage1[1] = fp32_to_16((fp16_to_32(data) + fp16_to_32(stage0[0])));
                                    stage0[1] = data; 
                                    datr = fp32_to_16((fp16_to_32(stage1[1]) + fp16_to_32(accum)));
                                    break;
                            case 3: stage1[2] = fp32_to_16((fp16_to_32(data) + fp16_to_32(stage0[1])));
                                    stage0[2] = data; 
                                    stage2[2] = fp32_to_16((fp16_to_32(stage0[0]) + fp16_to_32(stage1[2])));
                                    datr = fp32_to_16((fp16_to_32(stage2[2]) + fp16_to_32(accum)));
                                    break;                                   
                            case 4: stage1[3] = fp32_to_16((fp16_to_32(stage0[2]) + fp16_to_32(data)));
                                    stage0[3] = data;
                                    stage2[3] = fp32_to_16((fp16_to_32(stage1[1]) + fp16_to_32(stage1[3])));
                                    datr = fp32_to_16((fp16_to_32(stage2[3]) + fp16_to_32(accum)));
                                    break;
                            case 5: stage0[4] = data;
                                    stage1[4] = fp32_to_16((fp16_to_32(stage0[3]) + fp16_to_32(data)));
                                    stage2[4] = fp32_to_16((fp16_to_32(stage1[2]) + fp16_to_32(stage1[4])));
                                    stage3[4] = fp32_to_16((fp16_to_32(stage2[4]) + fp16_to_32(stage0[0])));
                                    datr = fp32_to_16((fp16_to_32(stage3[4]) + fp16_to_32(accum)));
                                    break;
                            case 6: stage0[5] = data;
                                    stage1[5] = fp32_to_16((fp16_to_32(stage0[4]) + fp16_to_32(data)));
                                    stage2[5] = fp32_to_16((fp16_to_32(stage1[3]) + fp16_to_32(stage1[5])));
                                    stage3[5] = fp32_to_16((fp16_to_32(stage2[5]) + fp16_to_32(stage1[1])));
                                    datr = fp32_to_16((fp16_to_32(stage3[5]) + fp16_to_32(accum)));
                                    break;
                            case 7: stage0[6] = data;
                                    stage1[6] = fp32_to_16((fp16_to_32(stage0[5]) + fp16_to_32(data)));
                                    stage2[6] = fp32_to_16((fp16_to_32(stage1[4]) + fp16_to_32(stage1[6])));
                                    stage3[6] = fp32_to_16((fp16_to_32(stage2[6]) + fp16_to_32(stage2[2])));
                                    datr = fp32_to_16((fp16_to_32(stage3[6]) + fp16_to_32(accum)));
                                    break;
                            case 8: stage0[7] = data;
                                    stage1[7] = fp32_to_16((fp16_to_32(stage0[6]) + fp16_to_32(data)));
                                    stage2[7] = fp32_to_16((fp16_to_32(stage1[5]) + fp16_to_32(stage1[7])));
                                    stage3[7] = fp32_to_16((fp16_to_32(stage2[7]) + fp16_to_32(stage2[3])));
                                    datr = fp32_to_16((fp16_to_32(stage3[7]) + fp16_to_32(accum)));
                                    break;
                            case 9: stage0[8] = data;
                                    stage1[8] = fp32_to_16((fp16_to_32(stage0[7]) + fp16_to_32(data)));
                                    stage2[8] = fp32_to_16((fp16_to_32(stage1[6]) + fp16_to_32(stage1[8])));
                                    stage3[8] = fp32_to_16((fp16_to_32(stage2[8]) + fp16_to_32(stage2[4])));
                                    stage4[8] = fp32_to_16((fp16_to_32(stage0[0]) + fp16_to_32(stage3[8])));
                                    datr = fp32_to_16((fp16_to_32(stage4[8]) + fp16_to_32(accum)));
                                    break;
                            case 10:stage0[9] = data;
                                    stage1[9] = fp32_to_16((fp16_to_32(stage0[8]) + fp16_to_32(data)));
                                    stage2[9] = fp32_to_16((fp16_to_32(stage1[7]) + fp16_to_32(stage1[9])));
                                    stage3[9] = fp32_to_16((fp16_to_32(stage2[9]) + fp16_to_32(stage2[5])));
                                    stage4[9] = fp32_to_16((fp16_to_32(stage1[1]) + fp16_to_32(stage3[9])));
                                    datr = fp32_to_16((fp16_to_32(stage4[9]) + fp16_to_32(accum)));
                                    break;
                            case 11:stage0[10] = data;
                                    stage1[10] = fp32_to_16((fp16_to_32(stage0[9]) + fp16_to_32(data)));
                                    stage2[10] = fp32_to_16((fp16_to_32(stage1[8]) + fp16_to_32(stage1[10])));
                                    stage3[10] = fp32_to_16((fp16_to_32(stage2[10]) + fp16_to_32(stage2[6])));
                                    stage4[10] = fp32_to_16((fp16_to_32(stage2[2]) + fp16_to_32(stage3[10])));
                                    datr = fp32_to_16((fp16_to_32(stage4[10]) + fp16_to_32(accum)));
                                    break;
                            case 12:stage0[11] = data;
                                    stage1[11] = fp32_to_16((fp16_to_32(stage0[10]) + fp16_to_32(data)));
                                    stage2[11] = fp32_to_16((fp16_to_32(stage1[9]) + fp16_to_32(stage1[11])));
                                    stage3[11] = fp32_to_16((fp16_to_32(stage2[11]) + fp16_to_32(stage2[7])));
                                    stage4[11] = fp32_to_16((fp16_to_32(stage2[3]) + fp16_to_32(stage3[11])));
                                    datr = fp32_to_16((fp16_to_32(stage4[11]) + fp16_to_32(accum)));
                                    break;
                            case 13:stage0[12] = data;
                                    stage1[12] = fp32_to_16((fp16_to_32(stage0[11]) + fp16_to_32(data)));
                                    stage2[12] = fp32_to_16((fp16_to_32(stage1[10]) + fp16_to_32(stage1[12])));
                                    stage3[12] = fp32_to_16((fp16_to_32(stage2[12]) + fp16_to_32(stage2[8])));
                                    stage4[12] = fp32_to_16((fp16_to_32(stage3[4]) + fp16_to_32(stage3[12])));
                                    datr = fp32_to_16((fp16_to_32(stage4[12]) + fp16_to_32(accum)));
                                    break;
                            case 14:stage0[13] = data;
                                    stage1[13] = fp32_to_16((fp16_to_32(stage0[12]) + fp16_to_32(data)));
                                    stage2[13] = fp32_to_16((fp16_to_32(stage1[11]) + fp16_to_32(stage1[13])));
                                    stage3[13] = fp32_to_16((fp16_to_32(stage2[13]) + fp16_to_32(stage2[9])));
                                    stage4[13] = fp32_to_16((fp16_to_32(stage3[5]) + fp16_to_32(stage3[13])));
                                    datr = fp32_to_16((fp16_to_32(stage4[13]) + fp16_to_32(accum)));
                                    break;
                            case 15:stage0[14] = data;
                                    stage1[14] = fp32_to_16((fp16_to_32(stage0[13]) + fp16_to_32(data)));
                                    stage2[14] = fp32_to_16((fp16_to_32(stage1[12]) + fp16_to_32(stage1[14])));
                                    stage3[14] = fp32_to_16((fp16_to_32(stage2[14]) + fp16_to_32(stage2[10])));
                                    stage4[14] = fp32_to_16((fp16_to_32(stage3[6]) + fp16_to_32(stage3[14])));
                                    datr = fp32_to_16((fp16_to_32(stage4[14]) + fp16_to_32(accum)));
                                    break;
                            default:stage0[15] = data;
                                    stage1[15] = fp32_to_16((fp16_to_32(stage0[14]) + fp16_to_32(data)));
                                    stage2[15] = fp32_to_16((fp16_to_32(stage1[13]) + fp16_to_32(stage1[15])));
                                    stage3[15] = fp32_to_16((fp16_to_32(stage2[15]) + fp16_to_32(stage2[11])));
                                    stage4[15] = fp32_to_16((fp16_to_32(stage3[7]) + fp16_to_32(stage3[15])));
                                    datr = fp32_to_16((fp16_to_32(stage4[15]) + fp16_to_32(accum)));
                                    accum = datr;
                                    break;
                        }
                        r = fp16_to_32(datr);
                    }
                    curr_cycle = ACCUM_EXEC_CYCLE(vector_size);
                    this->exec_cycle += ACCUM_EXEC_CYCLE(vector_size);
                    break;
                default:
                    break;
            }
            datr = fp32_to_16(r);
            this->l1_intf->writeDataMem(r_addr + (k * 2), datr, 2);
        }
        
    }

    void SPU::CLAMP(uint32_t vector_size, uint16_t scalar_value, int funct){
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        r_addr = LSPR[LSPR_SPM_ADDRR];
        uint16_t data, datr;
        float a, r, stdval;
        stdval = fp16_to_32(scalar_value);
        //#pragma omp parallel for
        for (int k = 0; k < vector_size; k++)
        {
            data = this->l1_intf->readDataMem(a_addr + (k * 2), 2);
            if (k % 16 == 0)
            {
                perf->codeMemoryRead();
                perf->codeMemoryWrite();
            }
            a = fp16_to_32(data);
            if(funct == 0){
                r = (a > stdval)? stdval : a;
            }else{
                r = (a < stdval)? stdval : a;
            }
            
            datr = fp32_to_16(r);
            this->l1_intf->writeDataMem(r_addr + (k * 2), datr, 2);
        }
    }

    void SPU::ARANGE(uint32_t vector_size, uint16_t start_value, uint16_t step){
        uint64_t spm_addr;
        uint32_t a_addr, r_addr, accum;
        uint16_t stage0[16];
        uint16_t stage1[16];
        uint16_t stage2[16];
        uint16_t stage3[16];
        uint16_t stage4[16];
        a_addr = LSPR[LSPR_SPM_ADDRA];
        r_addr = LSPR[LSPR_SPM_ADDRR];
        uint16_t data, datr;
        float a, r;
        r = fp16_to_32(start_value);
        a = fp16_to_32(step);
        datr = fp32_to_16(r);
        this->l1_intf->writeDataMem(r_addr, datr, 2);
        //#pragma omp parallel for
        if(!hw_mode){
            for (int k = 1; k < vector_size; k++)
            {
                if (k % 16 == 0)
                {
                    //#pragma omp critical
                    perf->codeMemoryRead();
                    perf->codeMemoryWrite();
                }
                r += a;
                datr = fp32_to_16(r);
                this->l1_intf->writeDataMem(r_addr + (k * 2), datr, 2);
            }
        }else{
            accum = 0;
            data = step;
    
            for(int k = 0 ; k < vector_size ; k++){
                        int temp;
                        temp = (k+1)%16;
                        switch(temp){
                            case 1: stage0[0] = (k == 0)? start_value : data; 
                                    datr = fp32_to_16((fp16_to_32(stage0[0]) + fp16_to_32(accum)));
                                    break;
                            case 2: stage1[1] = fp32_to_16((fp16_to_32(data) + fp16_to_32(stage0[0])));
                                    stage0[1] = data; 
                                    datr = fp32_to_16((fp16_to_32(stage1[1]) + fp16_to_32(accum)));
                                    break;
                            case 3: stage1[2] = fp32_to_16((fp16_to_32(data) + fp16_to_32(stage0[1])));
                                    stage0[2] = data; 
                                    stage2[2] = fp32_to_16((fp16_to_32(stage0[0]) + fp16_to_32(stage1[2])));
                                    datr = fp32_to_16((fp16_to_32(stage2[2]) + fp16_to_32(accum)));
                                    break;                                   
                            case 4: stage1[3] = fp32_to_16((fp16_to_32(stage0[2]) + fp16_to_32(data)));
                                    stage0[3] = data;
                                    stage2[3] = fp32_to_16((fp16_to_32(stage1[1]) + fp16_to_32(stage1[3])));
                                    datr = fp32_to_16((fp16_to_32(stage2[3]) + fp16_to_32(accum)));
                                    break;
                            case 5: stage0[4] = data;
                                    stage1[4] = fp32_to_16((fp16_to_32(stage0[3]) + fp16_to_32(data)));
                                    stage2[4] = fp32_to_16((fp16_to_32(stage1[2]) + fp16_to_32(stage1[4])));
                                    stage3[4] = fp32_to_16((fp16_to_32(stage2[4]) + fp16_to_32(stage0[0])));
                                    datr = fp32_to_16((fp16_to_32(stage3[4]) + fp16_to_32(accum)));
                                    break;
                            case 6: stage0[5] = data;
                                    stage1[5] = fp32_to_16((fp16_to_32(stage0[4]) + fp16_to_32(data)));
                                    stage2[5] = fp32_to_16((fp16_to_32(stage1[3]) + fp16_to_32(stage1[5])));
                                    stage3[5] = fp32_to_16((fp16_to_32(stage2[5]) + fp16_to_32(stage1[1])));
                                    datr = fp32_to_16((fp16_to_32(stage3[5]) + fp16_to_32(accum)));
                                    break;
                            case 7: stage0[6] = data;
                                    stage1[6] = fp32_to_16((fp16_to_32(stage0[5]) + fp16_to_32(data)));
                                    stage2[6] = fp32_to_16((fp16_to_32(stage1[4]) + fp16_to_32(stage1[6])));
                                    stage3[6] = fp32_to_16((fp16_to_32(stage2[6]) + fp16_to_32(stage2[2])));
                                    datr = fp32_to_16((fp16_to_32(stage3[6]) + fp16_to_32(accum)));
                                    break;
                            case 8: stage0[7] = data;
                                    stage1[7] = fp32_to_16((fp16_to_32(stage0[6]) + fp16_to_32(data)));
                                    stage2[7] = fp32_to_16((fp16_to_32(stage1[5]) + fp16_to_32(stage1[7])));
                                    stage3[7] = fp32_to_16((fp16_to_32(stage2[7]) + fp16_to_32(stage2[3])));
                                    datr = fp32_to_16((fp16_to_32(stage3[7]) + fp16_to_32(accum)));
                                    break;
                            case 9: stage0[8] = data;
                                    stage1[8] = fp32_to_16((fp16_to_32(stage0[7]) + fp16_to_32(data)));
                                    stage2[8] = fp32_to_16((fp16_to_32(stage1[6]) + fp16_to_32(stage1[8])));
                                    stage3[8] = fp32_to_16((fp16_to_32(stage2[8]) + fp16_to_32(stage2[4])));
                                    stage4[8] = fp32_to_16((fp16_to_32(stage0[0]) + fp16_to_32(stage3[8])));
                                    datr = fp32_to_16((fp16_to_32(stage4[8]) + fp16_to_32(accum)));
                                    break;
                            case 10:stage0[9] = data;
                                    stage1[9] = fp32_to_16((fp16_to_32(stage0[8]) + fp16_to_32(data)));
                                    stage2[9] = fp32_to_16((fp16_to_32(stage1[7]) + fp16_to_32(stage1[9])));
                                    stage3[9] = fp32_to_16((fp16_to_32(stage2[9]) + fp16_to_32(stage2[5])));
                                    stage4[9] = fp32_to_16((fp16_to_32(stage1[1]) + fp16_to_32(stage3[9])));
                                    datr = fp32_to_16((fp16_to_32(stage4[9]) + fp16_to_32(accum)));
                                    break;
                            case 11:stage0[10] = data;
                                    stage1[10] = fp32_to_16((fp16_to_32(stage0[9]) + fp16_to_32(data)));
                                    stage2[10] = fp32_to_16((fp16_to_32(stage1[8]) + fp16_to_32(stage1[10])));
                                    stage3[10] = fp32_to_16((fp16_to_32(stage2[10]) + fp16_to_32(stage2[6])));
                                    stage4[10] = fp32_to_16((fp16_to_32(stage2[2]) + fp16_to_32(stage3[10])));
                                    datr = fp32_to_16((fp16_to_32(stage4[10]) + fp16_to_32(accum)));
                                    break;
                            case 12:stage0[11] = data;
                                    stage1[11] = fp32_to_16((fp16_to_32(stage0[10]) + fp16_to_32(data)));
                                    stage2[11] = fp32_to_16((fp16_to_32(stage1[9]) + fp16_to_32(stage1[11])));
                                    stage3[11] = fp32_to_16((fp16_to_32(stage2[11]) + fp16_to_32(stage2[7])));
                                    stage4[11] = fp32_to_16((fp16_to_32(stage2[3]) + fp16_to_32(stage3[11])));
                                    datr = fp32_to_16((fp16_to_32(stage4[11]) + fp16_to_32(accum)));
                                    break;
                            case 13:stage0[12] = data;
                                    stage1[12] = fp32_to_16((fp16_to_32(stage0[11]) + fp16_to_32(data)));
                                    stage2[12] = fp32_to_16((fp16_to_32(stage1[10]) + fp16_to_32(stage1[12])));
                                    stage3[12] = fp32_to_16((fp16_to_32(stage2[12]) + fp16_to_32(stage2[8])));
                                    stage4[12] = fp32_to_16((fp16_to_32(stage3[4]) + fp16_to_32(stage3[12])));
                                    datr = fp32_to_16((fp16_to_32(stage4[12]) + fp16_to_32(accum)));
                                    break;
                            case 14:stage0[13] = data;
                                    stage1[13] = fp32_to_16((fp16_to_32(stage0[12]) + fp16_to_32(data)));
                                    stage2[13] = fp32_to_16((fp16_to_32(stage1[11]) + fp16_to_32(stage1[13])));
                                    stage3[13] = fp32_to_16((fp16_to_32(stage2[13]) + fp16_to_32(stage2[9])));
                                    stage4[13] = fp32_to_16((fp16_to_32(stage3[5]) + fp16_to_32(stage3[13])));
                                    datr = fp32_to_16((fp16_to_32(stage4[13]) + fp16_to_32(accum)));
                                    break;
                            case 15:stage0[14] = data;
                                    stage1[14] = fp32_to_16((fp16_to_32(stage0[13]) + fp16_to_32(data)));
                                    stage2[14] = fp32_to_16((fp16_to_32(stage1[12]) + fp16_to_32(stage1[14])));
                                    stage3[14] = fp32_to_16((fp16_to_32(stage2[14]) + fp16_to_32(stage2[10])));
                                    stage4[14] = fp32_to_16((fp16_to_32(stage3[6]) + fp16_to_32(stage3[14])));
                                    datr = fp32_to_16((fp16_to_32(stage4[14]) + fp16_to_32(accum)));
                                    break;
                            default:stage0[15] = data;
                                    stage1[15] = fp32_to_16((fp16_to_32(stage0[14]) + fp16_to_32(data)));
                                    stage2[15] = fp32_to_16((fp16_to_32(stage1[13]) + fp16_to_32(stage1[15])));
                                    stage3[15] = fp32_to_16((fp16_to_32(stage2[15]) + fp16_to_32(stage2[11])));
                                    stage4[15] = fp32_to_16((fp16_to_32(stage3[7]) + fp16_to_32(stage3[15])));
                                    datr = fp32_to_16((fp16_to_32(stage4[15]) + fp16_to_32(accum)));
                                    accum = datr;
                                    break;
                        }
                        this->l1_intf->writeDataMem(r_addr + (k * 2), datr, 2);
            }
        }
    }

    void SPU::VASMD_IMM(uint16_t r_addr, uint16_t b_addr, uint16_t a_addr, int funct)
    {

        float a, b, c, r;
        for (int i = 0; i < 16; i++)
        {
            a = fp16_to_32(l0_intf->readDataMem(a_addr*32 + 2*i, 2));
            b = fp16_to_32(l0_intf->readDataMem(b_addr*32 + 2*i, 2));
            switch (funct)
            {
            case 0:
                r = a + b;
                break;
            case 1:
                r = a - b;
                break;
            case 2:
                r = a * b;
                break;
            case 3:
                r = a / b;
                break;
            }
            l0_intf->writeDataMem(r_addr*32 + 2*i, fp32_to_16(r), 2);
        }

        curr_cycle = SADD_IMM_EXEC_CYCLE;
        this->exec_cycle += SADD_IMM_EXEC_CYCLE;
    }


    void SPU::VFMADD_IMM(uint16_t rs1, uint16_t rs2, uint16_t rs3)
    {
        float a, b, c, r;
        uint16_t a_addr, b_addr, c_addr;
        a_addr = rs1&0x1f;
        b_addr = rs2&0x1f;
        c_addr = (rs2>>5)& 0x1f;

        //uint16_t result[4];

        for (int i = 0; i < 16; i++)
        {
            a = fp16_to_32(l0_intf->readDataMem(a_addr*32 + 2*i, 2));
            b = fp16_to_32(l0_intf->readDataMem(b_addr*32 + 2*i, 2));
            c = fp16_to_32(l0_intf->readDataMem(c_addr*32 + 2*i, 2));
            r = a * b + c;
            l0_intf->writeDataMem(rs3*32 + 2*i, fp32_to_16(r),2);
        }

        curr_cycle = SADD_IMM_EXEC_CYCLE;
        this->exec_cycle += SADD_IMM_EXEC_CYCLE;

    }

    void SPU::VFUNC_IMM(uint16_t iaddr, uint16_t raddr,uint16_t op2_sel, int funct)
    {
        float a, r,x;
        uint16_t data;
        for (int i = 0; i < 16; i++)
        {
            data = l0_intf->readDataMem(iaddr*32 + 2*i , 2);
            a = fp16_to_32(data);
            switch(funct){
                case 0: //sqrt
                    if(!hw_mode){
                        r = std::sqrt(a);
                    }else{
                        r = fp16_to_32(HW_SQRT(data));
                    }
                    curr_cycle = VSQRT_IMM_EXEC_CYCLE;
                    this->exec_cycle += VSQRT_IMM_EXEC_CYCLE;
                    break;
                case 1: //exp
                    if(!hw_mode){
                        switch(op2_sel&0x3){
                            case  0: r = std::exp(a); break;
                            case  1: r = std::exp2(a); break;
                            case  2: r = std::pow(3.0,a); break;
                            default: r = std::pow(5.0,a); break;
                        }
                    }else{
                        r = fp16_to_32(HW_EXP(data,(op2_sel&0x3)));
                    }
                    curr_cycle = VEXP_IMM_EXEC_CYCLE;
                    this->exec_cycle += VEXP_IMM_EXEC_CYCLE;
                    break;
                case 2: //ln
                    if(!hw_mode){
                        switch(op2_sel&0x3){
                            case  0: r = std::log(a); break;
                            case  1: r = std::log2(a); break;
                            case  2: x = 1.0/std::log(3.0);
                                     r = std::log(a)*x; break;
                            default: r = std::log10(a); break;
                        }
                    }else{
                        r = fp16_to_32(HW_LN(data,(op2_sel&0x3)));
                    }
                    curr_cycle = VLN_IMM_EXEC_CYCLE;
                    this->exec_cycle += VLN_IMM_EXEC_CYCLE;
                    break;
                case 3: //abs
                    r = std::abs(a);
                    curr_cycle = VABS_IMM_EXEC_CYCLE;
                    this->exec_cycle += VABS_IMM_EXEC_CYCLE;
                    break;
                case 4: //neg
                    r = -a;
                    curr_cycle = VNEG_IMM_EXEC_CYCLE;
                    this->exec_cycle += VNEG_IMM_EXEC_CYCLE;
                    break;
                case 5: //sign
                    r = (a > 0) - (a < 0);
                    curr_cycle = VSIGN_IMM_EXEC_CYCLE;
                    this->exec_cycle += VSIGN_IMM_EXEC_CYCLE;
                    break;
                case 6: //step
                    r = (a > 0)? 1 : 0;
                    curr_cycle = VSTEP_IMM_EXEC_CYCLE;
                    this->exec_cycle += VSTEP_IMM_EXEC_CYCLE;
                    break;
                case 7: //ceil
                    r = std::ceil(a);
                    curr_cycle = VCEIL_IMM_EXEC_CYCLE;
                    this->exec_cycle += VCEIL_IMM_EXEC_CYCLE;
                    break;
                case 8: //trunc
                    r = std::trunc(a);
                    curr_cycle = VTRUNC_IMM_EXEC_CYCLE;
                    this->exec_cycle += VTRUNC_IMM_EXEC_CYCLE;
                    break;
                case 9: //floor
                    r = std::floor(a);
                    curr_cycle = VFLOOR_IMM_EXEC_CYCLE;
                    this->exec_cycle += VFLOOR_IMM_EXEC_CYCLE;
                    break;
                case 10: //rne
                    r = std::rint(a);
                    curr_cycle = VRNE_IMM_EXEC_CYCLE;
                    this->exec_cycle += VRNE_IMM_EXEC_CYCLE;
                    break;
                default:
                    break;
            }
            l0_intf->writeDataMem(raddr*32 + 2*i, fp32_to_16(r), 2);
        }
    }

    void SPU::BITWISE_IMM(uint16_t r_addr, uint16_t b_addr, uint16_t a_addr, int funct)
    {
        uint64_t rd;
        //uint16_t result[16];
        uint16_t a, b, c, r;
        for (int i = 0; i < 16; i++)
        {
            a = l0_intf->readDataMem(a_addr*32 + 2*i, 2);
            switch (funct)
            {
            case 0: //and
                b = l0_intf->readDataMem(b_addr*32 + 2*i, 2);
                r = a & b;
                curr_cycle = AND_IMM_EXEC_CYCLE;
                this->exec_cycle += AND_IMM_EXEC_CYCLE;
                break;
            case 1: //or
                b = l0_intf->readDataMem(b_addr*32 + 2*i, 2);
                r = a | b;
                curr_cycle = OR_IMM_EXEC_CYCLE;
                this->exec_cycle += OR_IMM_EXEC_CYCLE;
                break;
            case 2: //not
                r = ~a;
                curr_cycle = NOT_IMM_EXEC_CYCLE;
                this->exec_cycle += NOT_IMM_EXEC_CYCLE;
                break;
            case 3: //shift
                int shft_amt;
                shft_amt = b_addr & 0xF;
                if((b_addr & 0x10) == 0x10){
                    r = a << shft_amt;
                }else{
                    r = a >> shft_amt;
                }
                curr_cycle = SHIFT_IMM_EXEC_CYCLE;
                this->exec_cycle += SHIFT_IMM_EXEC_CYCLE;
                break;
            }
            l0_intf->writeDataMem(r_addr*32 + 2*i, r, 2);
        }

    }


    void SPU::SCVT_QH(uint32_t vector_size, uint16_t scale, uint16_t offset)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        r_addr = LSPR[LSPR_SPM_ADDRR];

        int16_t data;
        uint8_t datr;
        float a;
        float sc, os;
        sc = fp16_to_32(scale);
        os = fp16_to_32(offset);
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i * 2, 2);
            if (i % 16 == 0)
                perf->codeMemoryRead();
            a = fp16_to_32(data);
            a = a * sc + os;
            data = fp32_to_16(a);
            datr = fp16_to_8(data);
            this->l1_intf->writeDataMem(r_addr + i, datr, 1);
            if (i % 32 == 0)
                perf->codeMemoryWrite();
        }
        curr_cycle = SCVT_EXEC_CYCLE(vector_size);
        this->exec_cycle += SCVT_EXEC_CYCLE(vector_size);
    }
    void SPU::SCVT_HQ(uint32_t vector_size, uint16_t scale, uint16_t offset)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        r_addr = LSPR[LSPR_SPM_ADDRR];

        int8_t data;
        uint16_t datr;
        float a;
        float sc, os;
        sc = fp16_to_32(scale);
        os = fp16_to_32(offset);
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i, 1);
            if (i % 32 == 0)
                perf->codeMemoryRead();

            a = fp8_to_32(data);
            a = (a * sc) + os;
            datr = fp32_to_16(a);
            this->l1_intf->writeDataMem(r_addr + i * 2, datr, 2);
            if (i % 16 == 0)
                perf->codeMemoryRead();
        }
        curr_cycle = SCVT_EXEC_CYCLE(vector_size);
        this->exec_cycle += SCVT_EXEC_CYCLE(vector_size);
    }
    void SPU::SCVT_IH(uint32_t vector_size, uint16_t scale, uint16_t offset)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        r_addr = LSPR[LSPR_SPM_ADDRR];

        int16_t data, datr;
        float a;
        float sc, os;
        sc = fp16_to_32(scale);
        os = fp16_to_32(offset);
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i * 2, 2);
            if (i % 16 == 0)
                perf->codeMemoryRead();
            a = fp16_to_32(data);
            a = (a * sc) + os;
            data = fp32_to_16(a);
            datr = fp16_to_i8(data);
            this->l1_intf->writeDataMem(r_addr + i, datr, 1);
            if (i % 32 == 0)
                perf->codeMemoryWrite();
        }
        curr_cycle = SCVT_EXEC_CYCLE(vector_size);
        this->exec_cycle += SCVT_EXEC_CYCLE(vector_size);
    }
    void SPU::SCVT_HI(uint32_t vector_size, uint16_t scale, uint16_t offset)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        r_addr = LSPR[LSPR_SPM_ADDRR];

        int8_t data;
        uint16_t datr;
        float a;
        float sc, os;
        sc = fp16_to_32(scale);
        os = fp16_to_32(offset);
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i, 1);
            if (i % 32 == 0)
                perf->codeMemoryRead();

            a = static_cast<float>(data);
            a = (a * sc) + os;
            datr = fp32_to_16(a);
            this->l1_intf->writeDataMem(r_addr + i * 2, datr, 2);
            if (i % 16 == 0)
                perf->codeMemoryWrite();
        }
        curr_cycle = SCVT_EXEC_CYCLE(vector_size);
        this->exec_cycle += SCVT_EXEC_CYCLE(vector_size);
    }
    void SPU::SCVT_HN(uint32_t vector_size, uint16_t scale, uint16_t offset)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        r_addr = LSPR[LSPR_SPM_ADDRR];

        int32_t data;
        int16_t datr;
        float a;
        float sc, os;
        sc = fp16_to_32(scale);
        os = fp16_to_32(offset);
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i * 4, 4);
            if (i % 8 == 0)
                perf->codeMemoryRead();

            a = static_cast<float>(data);
            a = (a * sc) + os;
            datr = fp32_to_16(a);
            this->l1_intf->writeDataMem(r_addr + i * 2, datr, 2);
            if (i % 16 == 0)
                perf->codeMemoryWrite();
        }
        curr_cycle = SCVT_EXEC_CYCLE(vector_size);
        this->exec_cycle += SCVT_EXEC_CYCLE(vector_size);
    }

 

    void SPU::PRELU(uint16_t slope, uint32_t vector_size)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRR];
        r_addr = LSPR[LSPR_SPM_ADDRA];

        uint16_t data, datr;
        float a, r;
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i * 2, 2);
            if (i % 16 == 0)
                perf->codeMemoryRead();

            a = fp16_to_32(data);
            r = (a < 0) ? fp16_to_32(slope) * a : a;
            datr = fp32_to_16(r);
            this->l1_intf->writeDataMem(r_addr + i * 2, datr, 2);
            if (i % 16 == 0)
                perf->codeMemoryWrite();
        }
        curr_cycle = PRELU_EXEC_CYCLE(vector_size);
        this->exec_cycle += PRELU_EXEC_CYCLE(vector_size);
    }

    void SPU::GELU(uint32_t vector_size)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRR];
        r_addr = LSPR[LSPR_SPM_ADDRA];

        uint16_t data, datr;
        float a, r;
        float sqrt_val, x_cubed;
        sqrt_val = std::sqrt(2.0f / M_PI);
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i * 2, 2);
            if (i % 16 == 0)
                perf->codeMemoryRead();
            if(!hw_mode){
                a = fp16_to_32(data);
                x_cubed = a * a * a;
                r = 0.5f * a * (1.0f + std::tanh(sqrt_val * (a + 0.044715f * x_cubed)));
                datr = fp32_to_16(r);
            }else{
                datr = HW_GELU(data);
            }
            this->l1_intf->writeDataMem(r_addr + i * 2, datr, 2);
            if (i % 16 == 0)
                perf->codeMemoryWrite();
        }
        curr_cycle = TANH_EXEC_CYCLE(vector_size);
        this->exec_cycle += TANH_EXEC_CYCLE(vector_size);
    }

    void SPU::TANH(uint32_t vector_size)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRR];
        r_addr = LSPR[LSPR_SPM_ADDRA];

        uint16_t data, datr;
        float a, r;
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i * 2, 2);
            if (i % 16 == 0)
                perf->codeMemoryRead();
            if(!hw_mode){
                a = fp16_to_32(data);
                r = std::tanh(a);
                datr = fp32_to_16(r);
            }else{
                datr = HW_TANH(data);
            }

            this->l1_intf->writeDataMem(r_addr + i * 2, datr, 2);
            if (i % 16 == 0)
                perf->codeMemoryWrite();
        }
        curr_cycle = TANH_EXEC_CYCLE(vector_size);
        this->exec_cycle += TANH_EXEC_CYCLE(vector_size);
    }

    void SPU::SIGM(uint32_t vector_size)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRR];
        r_addr = LSPR[LSPR_SPM_ADDRA];

        uint16_t data, datr;
        float a, r;
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i * 2, 2);
            if (i % 16 == 0)
                perf->codeMemoryRead();
            if(!hw_mode){
                a = fp16_to_32(data);
                r = 1.0f / (1.0f + std::exp(-a));
                datr = fp32_to_16(r);
            }else{
                datr = HW_SIGM(data);
            }
            this->l1_intf->writeDataMem(r_addr + i * 2, datr, 2);
            if (i % 16 == 0)
                perf->codeMemoryWrite();
        }
        curr_cycle = TANH_EXEC_CYCLE(vector_size);
        this->exec_cycle += TANH_EXEC_CYCLE(vector_size);
    }


    void SPU::PRELU_IMM(uint16_t l0_addr, uint16_t slope, uint16_t input_addr)
    {
        uint64_t result;
        float a, r;
        for (int i = 0; i < 16; i++)
        {
            a = fp16_to_32(l0_intf->readDataMem(input_addr*32 + 2*i, 2));
            r = (a > 0) ? a : a * fp16_to_32(slope);
            l0_intf->writeDataMem(l0_addr*32+ 2*i, fp32_to_16(r) , 2);
        }
        //result = ((uint64_t)result_data[3] << 48) | ((uint64_t)result_data[2] << 32) | ((uint64_t)result_data[1] << 16) | ((uint64_t)result_data[0]);
        //LSPR[LSPR_RESULT] = result;
        curr_cycle = PRELU_IMM_CYCLE;
        this->exec_cycle += PRELU_IMM_CYCLE;

    }

 
    void SPU::GELU_IMM(uint16_t i_addr, uint16_t r_addr)
    {
        float a, r;
        float sqrt, x_cubed;
        uint16_t data;

        sqrt = std::sqrt(2.0f / M_PI);
        for (int i = 0; i < 16; i++)
        {
            if(!hw_mode){
                a = fp16_to_32(l0_intf->readDataMem(i_addr*32+2*i, 2));
                x_cubed = a * a * a;
                r = 0.5f * a * (1.0f + std::tanh(sqrt * (a + 0.044715f * x_cubed)));
                l0_intf->writeDataMem(r_addr*32 + 2*i, fp32_to_16(r), 2);
            }else{
                data = l0_intf->readDataMem(i_addr*32+2*i, 2);
                data = HW_GELU(data);
                l0_intf->writeDataMem(r_addr*32 + 2*i, data, 2);
            }
        }
        //result = ((uint64_t)result_data[3] << 48) | ((uint64_t)result_data[2] << 32) | ((uint64_t)result_data[1] << 16) | ((uint64_t)result_data[0]);
        //LSPR[LSPR_RESULT] = result;
        curr_cycle = TANH_IMM_CYCLE;
        this->exec_cycle += TANH_IMM_CYCLE;

    }

    void SPU::TANH_IMM(uint16_t i_addr, uint16_t r_addr)
    {
        float a, r;
        uint16_t data;
        for (int i = 0; i < 16; i++)
        {
            if(!hw_mode){
                a = fp16_to_32(l0_intf->readDataMem(i_addr*32+2*i, 2));
                r = std::tanh(a);
                l0_intf->writeDataMem(r_addr*32+2*i, fp32_to_16(r), 2);
            }else{
                data = l0_intf->readDataMem(i_addr*32+2*i, 2);
                data = HW_TANH(data);
                l0_intf->writeDataMem(r_addr*32 + 2*i, data, 2);
            }
        }
        //result = ((uint64_t)result_data[3] << 48) | ((uint64_t)result_data[2] << 32) | ((uint64_t)result_data[1] << 16) | ((uint64_t)result_data[0]);
        //LSPR[LSPR_RESULT] = result;
        curr_cycle = TANH_IMM_CYCLE;
        this->exec_cycle += TANH_IMM_CYCLE;

    }

    void SPU::SIGM_IMM(uint16_t i_addr, uint16_t r_addr)
    {
        float a, r;
        uint16_t data;
        for (int i = 0; i < 16; i++)
        {
            if(!hw_mode){
                a = fp16_to_32(l0_intf->readDataMem(i_addr*32 + 2*i,2));
                r = 1.0f / (1.0f + std::exp(-a));
                l0_intf->writeDataMem(r_addr*32 + 2*i, fp32_to_16(r), 2);
            }else{
                data = l0_intf->readDataMem(i_addr*32+2*i, 2);
                data = HW_SIGM(data);
                l0_intf->writeDataMem(r_addr*32 + 2*i, data, 2);
            }
        }
        //result = ((uint64_t)result_data[3] << 48) | ((uint64_t)result_data[2] << 32) | ((uint64_t)result_data[1] << 16) | ((uint64_t)result_data[0]);
        //LSPR[LSPR_RESULT] = result;
        curr_cycle = TANH_IMM_CYCLE;
        this->exec_cycle += TANH_IMM_CYCLE;

    }

    void SPU::ESUM(uint16_t max_value, uint16_t accum_data, uint32_t vector_size, uint16_t l0_addr)
    {
        uint64_t spm_addr, rd;
        uint16_t temp;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        r_addr = LSPR[LSPR_SPM_ADDRR];
        uint16_t prev_input;
        uint16_t st1_result[8];
        uint16_t st2_result[4];
        uint16_t st3_result[2];
        uint16_t st4_result;
        

        uint16_t data, datr, msum;
        float a, max, accum, r;
        max = fp16_to_32(max_value);
        accum = fp16_to_32(accum_data);
        r = accum;
        if(!hw_mode){
            //#pragma omp parallel for
            for (int i = 0; i < vector_size; i++)
            {
                data = this->l1_intf->readDataMem(a_addr + i * 2, 2);
                if (i % 16 == 0)
                    perf->codeMemoryRead();

                a = fp16_to_32(data);
                r += std::exp(a - max);
            }
            datr = fp32_to_16(r);
        }else{
            datr = accum_data;
            for(int i = 0; i < vector_size; i++){
                if (i % 16 == 0)
                    perf->codeMemoryRead();
                data = this->l1_intf->readDataMem(a_addr + i*2 , 2);
                r = fp16_to_32(data) - fp16_to_32(max_value);
                data = HW_SM_EXP(fp32_to_16(r));
                //add tree
                temp = (i+1)%16;
                switch(temp){
                    case  1: 
                            prev_input = data; 
                            msum = data;
                            break;
                    case  2:
                            msum = fp32_to_16(fp16_to_32(data) + fp16_to_32(prev_input));
                            st1_result[0]= msum;
                            prev_input = data;
                            break;
                    case  3:
                            msum = fp32_to_16(fp16_to_32(data) + fp16_to_32(st1_result[0]));
                            prev_input = data;
                            break;
                    case  4:
                            st1_result[1] = fp32_to_16(fp16_to_32(data) + fp16_to_32(prev_input));
                            st2_result[0] = fp32_to_16(fp16_to_32(st1_result[0]) + fp16_to_32(st1_result[1]));
                            msum = st2_result[0];
                            prev_input = data;
                            break;
                    case  5:
                            msum = fp32_to_16(fp16_to_32(st2_result[0]) + fp16_to_32(data));
                            prev_input = data;
                            break;
                    case  6:
                            st1_result[2] = fp32_to_16(fp16_to_32(data) + fp16_to_32(prev_input));
                            msum = fp32_to_16(fp16_to_32(st2_result[0]) + fp16_to_32(st1_result[2]));
                            prev_input = data;
                            break;
                    case  7:
                            msum = fp32_to_16(fp16_to_32(st1_result[2]) + fp16_to_32(data));
                            msum = fp32_to_16(fp16_to_32(st2_result[0]) + fp16_to_32(msum));
                            prev_input = data;
                            break;
                    case  8:
                            st1_result[3] = fp32_to_16(fp16_to_32(data) + fp16_to_32(prev_input));
                            st2_result[1] = fp32_to_16(fp16_to_32(st1_result[2]) + fp16_to_32(st1_result[3]));
                            st3_result[0] = fp32_to_16(fp16_to_32(st2_result[1]) + fp16_to_32(st2_result[0]));
                            msum = st3_result[0];
                            prev_input = data;
                            break;
                    case  9:
                            msum = fp32_to_16(fp16_to_32(st3_result[0]) + fp16_to_32(data));
                            prev_input = data;
                            break;
                    case 10:
                            st1_result[4] = fp32_to_16(fp16_to_32(data) + fp16_to_32(prev_input));
                            msum = fp32_to_16(fp16_to_32(st3_result[0]) + fp16_to_32(st1_result[4]));
                            prev_input = data;
                            break;
                    case 11:
                            msum = fp32_to_16(fp16_to_32(st1_result[4]) + fp16_to_32(data));
                            msum = fp32_to_16(fp16_to_32(st3_result[0]) + fp16_to_32(msum));
                            prev_input = data;
                            break;
                    case 12:
                            st1_result[5] = fp32_to_16(fp16_to_32(data) + fp16_to_32(prev_input));
                            st2_result[2] = fp32_to_16(fp16_to_32(st1_result[4]) + fp16_to_32(st1_result[5]));
                            msum = fp32_to_16(fp16_to_32(st3_result[0]) + fp16_to_32(st2_result[2]));

                            prev_input = data;
                            break;
                    case 13:
                            msum = fp32_to_16(fp16_to_32(data) + fp16_to_32(st2_result[2]));
                            msum = fp32_to_16(fp16_to_32(st3_result[0]) + fp16_to_32(msum));
                            prev_input = data;
                            break;
                    case 14:
                            st1_result[6] = fp32_to_16(fp16_to_32(data) + fp16_to_32(prev_input));
                            msum = fp32_to_16(fp16_to_32(st1_result[6]) + fp16_to_32(st2_result[2]));
                            msum = fp32_to_16(fp16_to_32(st3_result[0]) + fp16_to_32(msum));
                            prev_input = data;
                            break;
                    case 15:
                            msum = fp32_to_16(fp16_to_32(st1_result[6]) + fp16_to_32(data));
                            msum = fp32_to_16(fp16_to_32(st2_result[2]) + fp16_to_32(msum));
                            msum = fp32_to_16(fp16_to_32(st3_result[0]) + fp16_to_32(msum));
                            prev_input = data;
                            break;
                    default:
                            st1_result[7] = fp32_to_16(fp16_to_32(data) + fp16_to_32(prev_input));
                            st2_result[3] = fp32_to_16(fp16_to_32(st1_result[6]) + fp16_to_32(st1_result[7]));
                            st3_result[1] = fp32_to_16(fp16_to_32(st2_result[2]) + fp16_to_32(st2_result[3]));
                            st4_result    = fp32_to_16(fp16_to_32(st3_result[1]) + fp16_to_32(st3_result[0]));
                            datr          = fp32_to_16(fp16_to_32(datr) + fp16_to_32(st4_result));
                            msum = 0;
                            break;
                }
                if(i == (vector_size -1)){
                    datr = fp32_to_16(fp16_to_32(msum) + fp16_to_32(datr));
                }
            }
        }
        rd = ((uint64_t)datr << 16) | ((uint64_t)max_value);
        l0_intf->writeDataMem(l0_addr*32 , rd, 4);
        for(int x = 1; x < 8 ; x++){
            this-> l0_intf->writeDataMem(l0_addr*32 +4*x, 0x0, 4);
        }
        //LSPR[LSPR_RESULT] = rd;
        curr_cycle = ESUM_EXEC_CYCLE(vector_size);
        this->exec_cycle += ESUM_EXEC_CYCLE(vector_size);

    }

    void SPU::SOFTMAX(uint16_t max_value, uint16_t esum_value, uint32_t vector_size)
    {
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        r_addr = LSPR[LSPR_SPM_ADDRR];
        uint16_t data, datr, ln_max;
        float a, max, ln, r;
        max = fp16_to_32(max_value);
        ln = std::log(fp16_to_32(esum_value));
        if(hw_mode){
            ln_max = HW_SM_LN(esum_value, max_value);
        }
        //#pragma omp parallel for
        for (int i = 0; i < vector_size; i++)
        {
            data = this->l1_intf->readDataMem(a_addr + i * 2, 2);
            if (i % 16 == 0)
                perf->codeMemoryRead();
            if(!hw_mode){
                a = fp16_to_32(data);
                r = std::exp(a - max - ln);
                datr = fp32_to_16(r);
            }else{
                datr = fp32_to_16( fp16_to_32(data)-fp16_to_32(ln_max));
                datr = HW_SM_EXP(datr);
            }
            this->l1_intf->writeDataMem(r_addr + i * 2, datr, 2);
            if (i % 16 == 0)
                perf->codeMemoryWrite();
        }
        curr_cycle = SOFTMAX_EXEC_CYCLE(vector_size);
        this->exec_cycle += SOFTMAX_EXEC_CYCLE(vector_size);
    }

    void SPU::ESUM_IMM(uint16_t max_value, uint16_t accum_data, uint16_t i_addr, uint16_t l0_addr)
    {
        uint16_t datr,data;
        uint16_t hw_in[16];
        uint32_t rd;
        float a, max, accum, r;
        max = fp16_to_32(max_value);
        accum = fp16_to_32(accum_data);
        r = accum;
        for(int i = 0; i < 16 ; i++){
            data =l0_intf->readDataMem(i_addr*32 + i*2, 2);
            a = fp16_to_32(data);
            if(!hw_mode){
                r += std::exp(a - max);
            }else{
                r = fp16_to_32(data) - fp16_to_32(max_value);
                hw_in[i] = HW_SM_EXP(fp32_to_16(r));
            }
        }
        if(!hw_mode){
            datr = fp32_to_16(r);
        }else{
            for(int i = 0; i < 8 ; i++){
                hw_in[i] = fp32_to_16(fp16_to_32(hw_in[2*i]) + fp16_to_32(hw_in[2*i+1]));
            }
            for(int i = 0; i < 4 ; i++){
                hw_in[i] = fp32_to_16(fp16_to_32(hw_in[2*i]) + fp16_to_32(hw_in[2*i+1]));
            }
            for(int i = 0; i < 2 ; i++){
                hw_in[i] = fp32_to_16(fp16_to_32(hw_in[2*i]) + fp16_to_32(hw_in[2*i+1]));
            }
            datr = fp32_to_16(fp16_to_32(hw_in[0]) + fp16_to_32(hw_in[1]));
        }

        
        rd = ((uint64_t)datr << 16) | ((uint64_t)max_value);
        l0_intf->writeDataMem(l0_addr*32, rd, 4);
        for(int x = 1; x < 8 ; x++){
            this-> l0_intf->writeDataMem(l0_addr*32 +4*x, 0x0, 4);
        }
        //LSPR[LSPR_RESULT] = rd;
        curr_cycle = ESUM_IMM_CYCLE;
        this->exec_cycle += ESUM_IMM_CYCLE;
    }

    void SPU::SOFTMAX_IMM(uint16_t max_value, uint16_t esum_value, uint16_t i_addr, uint16_t l0_addr)
    {
        float a, max, ln, r;
        max = fp16_to_32(max_value);
        uint16_t data, datr, ln_max;
        if(hw_mode){
            ln_max = HW_SM_LN(esum_value, max_value);
        }
        ln = std::log(fp16_to_32(esum_value));
        for (int i = 0; i < 16; i++)
        {
            if(!hw_mode){
                a = fp16_to_32(l0_intf->readDataMem(i_addr*32+2*i, 2));
                r = std::exp(a - max - ln);
                l0_intf->writeDataMem(l0_addr*32+2*i, fp32_to_16(r), 2);
            }else{
                data = l0_intf->readDataMem(i_addr*32+2*i, 2);
                datr = fp32_to_16( fp16_to_32(data)-fp16_to_32(ln_max));
                datr = HW_SM_EXP(datr);
                l0_intf->writeDataMem(l0_addr*32+2*i, datr, 2);
            }
        }
        curr_cycle = SOFTMAX_IMM_CYCLE;
        this->exec_cycle += SOFTMAX_IMM_CYCLE;

    }
/*
    void SPU::POOL_M(uint16_t row_k, uint16_t col_k, uint16_t row_stride, uint16_t col_stride, uint16_t row_in, uint16_t col_in, uint16_t row_out, uint16_t col_out)
    {
        float a, b;
        uint16_t rdata;
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        uint16_t wdata;
        int wp = 0;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        r_addr = LSPR[LSPR_SPM_ADDRR];

        if (((row_stride * (row_out - 1) + row_k) != row_in) || ((col_stride * (col_out - 1) + col_k) != col_in))
        {
            LOG(LOG_WARNING) << "[WARNING] POOLING FAILED : pooling input error" << std::endl;
            return;
        }
        
        for (int i = 0; i <= row_in - row_k; i = i + row_stride)
        {
            for (int j = 0; j <= col_in - col_k; j = j + col_stride)
            {
                rdata = this->l1_intf->readDataMem(a_addr + (i * col_in + j) * 2, 2);
                if (j % 16 == 0)
                    perf->codeMemoryRead();
                a = fp16_to_32(rdata);
                for (int k = 0; k < row_k; k++)
                {
                    for (int l = 0; l < col_k; l++)
                    {
                        rdata = this->l1_intf->readDataMem(a_addr + (i * col_in + j + (k * col_in) + l) * 2, 2);
                        b = fp16_to_32(rdata);
                        a = (a > b) ? a : b;
                    }
                }
                wdata = fp32_to_16(a);
                this->l1_intf->writeDataMem(r_addr + wp, wdata, 2);
                if (j % 16 == 0)
                    perf->codeMemoryWrite();

                wp += 2;
            }
        }
        curr_cycle = MPOOL_EXEC_CYCLE(row_k, col_k, row_in, col_in, row_s, col_s);
        this->exec_cycle += MPOOL_EXEC_CYCLE(row_k, col_k, row_in, col_in, row_s, col_s);
    }*/
    void SPU::POOL_M(uint16_t row_k, uint16_t col_k,
                     uint16_t row_stride, uint16_t col_stride,
                     uint16_t row_in, uint16_t col_in,
                     uint16_t row_out, uint16_t col_out)
{
    const uint32_t a_addr = LSPR[LSPR_SPM_ADDRA]; // 입력 feature map (fp16)
    const uint32_t r_addr = LSPR[LSPR_SPM_ADDRR]; // 출력 (fp16)

    // 파라미터 검증 (원래 로직 유지)
    if (((row_stride * (row_out - 1) + row_k) != row_in) ||
        ((col_stride * (col_out - 1) + col_k) != col_in))
    {
        LOG(LOG_WARNING) << "[WARNING] POOLING FAILED : pooling input error" << std::endl;
        return;
    }

    // perf 카운터를 병렬부에서 직접 호출(락/경쟁)하지 말고, 스레드 로컬 합계를 reduction으로 모은 뒤
    // 마지막에 직렬로 한 번에 반영
    long perf_read_cnt  = 0;
    long perf_write_cnt = 0;

    // 출력 좌표 (oh, ow)를 전체 병렬 작업 단위로 사용
    // 각 (oh, ow)에 대해 내장 커널 창(row_k x col_k)을 순회해 max를 구함
    #pragma omp parallel for collapse(2) schedule(static) reduction(+:perf_read_cnt, perf_write_cnt)
    for (int oh = 0; oh < static_cast<int>(row_out); ++oh) {
        for (int ow = 0; ow < static_cast<int>(col_out); ++ow) {
            // 입력 상의 좌상단 위치
            const int i0 = oh * row_stride;
            const int j0 = ow * col_stride;

            // (선택) 원 코드와 유사하게 16열마다 perf 카운터를 증가
            if ((j0 & 15) == 0) { ++perf_read_cnt; ++perf_write_cnt; }

            // 초기값: 창의 (0,0)
            uint32_t base0 = a_addr + static_cast<uint32_t>((i0 * col_in + j0) * 2);
            uint16_t r0    = this->l1_intf->readDataMem(base0, 2);
            float amax     = fp16_to_32(r0);

            // 창 내에서 최대값 탐색
            for (int ki = 0; ki < static_cast<int>(row_k); ++ki) {
                const int ii = i0 + ki;
                const int row_base = ii * col_in;
                for (int kj = 0; kj < static_cast<int>(col_k); ++kj) {
                    const int jj = j0 + kj;
                    const uint32_t src_addr = a_addr + static_cast<uint32_t>((row_base + jj) * 2);
                    const uint16_t rdata = this->l1_intf->readDataMem(src_addr, 2);
                    const float val = fp16_to_32(rdata);
                    if (val > amax) amax = val;
                }
            }

            // 결과 저장: 출력은 (oh, ow) 순서로 연속 저장
            const uint16_t wdata = fp32_to_16(amax);
            const uint32_t dst_addr = r_addr + static_cast<uint32_t>(((oh * col_out + ow) * 2));
            this->l1_intf->writeDataMem(dst_addr, wdata, 2);
        }
    }

    // perf 누적을 직렬 구간에서 반영
    for (long c = 0; c < perf_read_cnt;  ++c) perf->codeMemoryRead();
    for (long c = 0; c < perf_write_cnt; ++c) perf->codeMemoryWrite();

    // 사이클 갱신 (원 코드에 오탈자 있었음: row_s/col_s → row_stride/col_stride)
    curr_cycle = MPOOL_EXEC_CYCLE(row_k, col_k, row_in, col_in, row_stride, col_stride);
    this->exec_cycle += curr_cycle;
}
void SPU::POOL_A(uint16_t row_k, uint16_t col_k,
                     uint16_t row_stride, uint16_t col_stride,
                     uint16_t k_value,        // fp16로 들어온 평균 스케일 (ex: 1/(k*k))
                     uint16_t row_in, uint16_t col_in,
                     uint16_t row_out, uint16_t col_out)
{
    const uint32_t a_addr = LSPR[LSPR_SPM_ADDRA]; // 입력 feature map (fp16)
    const uint32_t r_addr = LSPR[LSPR_SPM_ADDRR]; // 출력 feature map (fp16)

    // 파라미터 검증 (원 코드와 동일)
    if (((row_stride * (row_out - 1) + row_k) != row_in) ||
        ((col_stride * (col_out - 1) + col_k) != col_in))
    {
        LOG(LOG_WARNING) << "[WARNING] POOLING FAILED : pooling input error" << std::endl;
        return;
    }

    // 평균 스케일 (원 코드 유지: 요소마다 b * avg를 누적)
    const float avg = fp16_to_32(k_value);

    // perf 카운터는 병렬부에서 직접 호출하지 말고, 스레드 로컬 합계 → 끝에 일괄 반영
    long perf_read_cnt  = 0;
    long perf_write_cnt = 0;

    // 출력 좌표를 전체 작업으로 병렬화
    // 출력 (oh,ow)에 대해 창(row_k x col_k)을 돌며 합을 구하고 avg를 곱해 저장
    #pragma omp parallel for collapse(2) schedule(static) reduction(+:perf_read_cnt, perf_write_cnt)
    for (int oh = 0; oh < static_cast<int>(row_out); ++oh) {
        for (int ow = 0; ow < static_cast<int>(col_out); ++ow) {
            const int i0 = oh * row_stride;   // 입력의 윗쪽 시작 행
            const int j0 = ow * col_stride;   // 입력의 왼쪽 시작 열

            // 원 코드의 패턴 유지: 열 기준 16단위마다 read/write 카운트 1씩
            if ((j0 & 15) == 0) { ++perf_read_cnt; ++perf_write_cnt; }

            float acc = 0.0f;

            // 창 내 합산
            for (int ki = 0; ki < static_cast<int>(row_k); ++ki) {
                const int ii = i0 + ki;
                const int row_base = ii * col_in;
                for (int kj = 0; kj < static_cast<int>(col_k); ++kj) {
                    const int jj = j0 + kj;
                    const uint32_t src_addr =
                        a_addr + static_cast<uint32_t>((row_base + jj) * 2);
                    const uint16_t rdata = this->l1_intf->readDataMem(src_addr, 2);
                    const float b = fp16_to_32(rdata);
                    acc += (b * avg);   // 원 코드 로직 그대로
                }
            }

            // 결과 저장: 출력은 (oh,ow) 순서로 연속
            const uint16_t wdata = fp32_to_16(acc);
            const uint32_t dst_addr =
                r_addr + static_cast<uint32_t>(((oh * col_out + ow) * 2));
            this->l1_intf->writeDataMem(dst_addr, wdata, 2);
        }
    }

    // perf 누적 반영(직렬 구간)
    for (long c = 0; c < perf_read_cnt;  ++c) perf->codeMemoryRead();
    for (long c = 0; c < perf_write_cnt; ++c) perf->codeMemoryWrite();

    // 사이클 갱신 (원 코드의 파라미터명 오탈자 row_s/col_s → row_stride/col_stride로 수정)
    curr_cycle = APOOL_EXEC_CYCLE(row_k, col_k, row_in, col_in, row_stride, col_stride);
    this->exec_cycle += curr_cycle;
}
/*
    void SPU::POOL_A(uint16_t row_k, uint16_t col_k, uint16_t row_stride, uint16_t col_stride, uint16_t k_value, uint16_t row_in, uint16_t col_in, uint16_t row_out, uint16_t col_out)
    {
        float a, b, avg;
        uint16_t rdata;
        uint64_t spm_addr;
        uint32_t a_addr, r_addr;
        uint16_t wdata;

        int wp = 0;
        a_addr = LSPR[LSPR_SPM_ADDRA];
        r_addr = LSPR[LSPR_SPM_ADDRR];

        avg = fp16_to_32(k_value);

        if (((row_stride * (row_out - 1) + row_k) != row_in) || ((col_stride * (col_out - 1) + col_k) != col_in))
        {
            LOG(LOG_WARNING) << "[WARNING] POOLING FAILED : pooling input error" << std::endl;
            return;
        }
        for (int i = 0; i <= row_in - row_k; i = i + row_stride)
        {
            for (int j = 0; j <= col_in - col_k; j = j + col_stride)
            {
                a = 0;
                if (j % 16 == 0)
                    perf->codeMemoryRead();
                if (j % 16 == 0)
                    perf->codeMemoryWrite();

                for (int k = 0; k < row_k; k++)
                {
                    for (int l = 0; l < col_k; l++)
                    {
                        rdata = this->l1_intf->readDataMem(a_addr + (i * col_in + j + (k * col_in) + l) * 2, 2);
                        b = fp16_to_32(rdata);
                        a += (b * avg);
                    }
                }
                wdata = fp32_to_16(a);
                this->l1_intf->writeDataMem(r_addr + wp, wdata, 2);
                wp += 2;
            }
        }
        curr_cycle = APOOL_EXEC_CYCLE(row_k, col_k, row_in, col_in, row_s, col_s);
        this->exec_cycle += APOOL_EXEC_CYCLE(row_k, col_k, row_in, col_in, row_s, col_s);
    }
*/
    void SPU::setLSPR(int lspr, uint64_t value)
    {
        LSPR[lspr] = value;
    }

    uint64_t SPU::getLSPR(int lspr)
    {
        uint64_t ret_value;

        ret_value = LSPR[lspr];
        return ret_value;
    }

    int SPU::getID()
    {
        return this->ID;
    }

    void SPU::dump_LSPR()
    {
        LOG(LOG_INFO) << " ************************************\n";
        LOG(LOG_INFO) << "           LSPR REGSITERS\n";
        LOG(LOG_INFO) << " ====================================\n";
        LOG(LOG_INFO) << " [LSPR_SPU_ADDRA     ] : 0x" << std::hex << std::uppercase << LSPR[LSPR_SPM_ADDRA] << std::endl;
        LOG(LOG_INFO) << " [LSPR_SPU_ADDRB     ] : 0x" << std::hex << std::uppercase << LSPR[LSPR_SPM_ADDRB] << std::endl;
        LOG(LOG_INFO) << " [LSPR_SPU_ADDRC     ] : 0x" << std::hex << std::uppercase << LSPR[LSPR_SPM_ADDRC] << std::endl;
        LOG(LOG_INFO) << " [LSPR_SPU_ADDRR     ] : 0x" << std::hex << std::uppercase << LSPR[LSPR_SPM_ADDRR] << std::endl;
        LOG(LOG_INFO) << " ************************************\n";
    }

    void SPU::initLSPR()
    {
        return;
    }
}