#include "NSU.h"
#include "GTX_extension.h"
#include "config.h"

namespace riscv_tlm {
    NSU::NSU(sc_core::sc_module_name const &name, SPU *spu_i[SPU_NUM], TMU *tmu_i[TMU_NUM]) : sc_module(name){
        initGSPR();
        mem_intf = new MemoryInterface((std::string(name) + "_mem").c_str());
        perf = Performance::getInstance();
        for(int i = 0 ; i < SPU_NUM ; i++){
            
            spu[i] = spu_i[i];
            credit_ld[i] = 0;
            credit_st[i] = 0;            
        }

        for(int i = 0; i < TMU_NUM; i++){
            l2_intf[i] = new MemoryInterface((std::string(name) + "_l2_" + std::to_string(i)).c_str());
            tmu[i] = tmu_i[i];
        }
        if(gtx_command_dump){
            std::ofstream ofs(cmd_dfname, std::ios::out | std::ios::trunc);
        }
        curr_id = 0;

    };

    uint64_t NSU::getGSPR(uint16_t gspr) {
            uint64_t ret_value;
            if(gspr != GSPR_GTX_RUN){
                ret_value = GSPR[gspr]; 
            }
            
            return ret_value;
    }

    void NSU::setGSPR(uint16_t gspr, uint64_t value) {
            uint64_t  t_msk;
            uint64_t rd;

            switch(gspr){
                case GSPR_GTX_RUN:
                    rd = gtx_run();
                    break;
                default:
                    GSPR[gspr] = value;
                    break;
            }
        
    }

    uint64_t NSU::gtx_run(){
        uint64_t opcode, rs1, rs2,rs3,rs4, rd;
        uint16_t func7, func3,wid;
        bool is_broad; 
        bool tflag, sflag, pflag;
        uint16_t sid, nid;
        op_GTX_Codes spu_op;
        opcode = GSPR[GSPR_GTX_OPCODE];
        rs1 = GSPR[GSPR_GTX_OPERAND1];
        rs2 = GSPR[GSPR_GTX_OPERAND2];
        rs3 = GSPR[GSPR_GTX_OPERAND3];

        tflag = (((opcode>>28) & 0x1) == 0x1);
        sflag = (((opcode>>29) & 0x1) == 0x1);
        pflag = (((opcode>>30) & 0x1) == 0x1);
        nid = (((opcode>>38) & 0x3f));
        sid = (((opcode>>32) & 0x3f));
        spu_op = (op_GTX_Codes)(opcode & 0x3FF);
        rs4 = ((opcode>>16) & 0xFFF);
        

        if(is_ploop || is_sloop || is_tloop){
            LOG(LOG_WARNING)<<"[WARNING] RUN with GPR FAILED (must used out of start end)\n";
        }
        is_tloop = tflag;
        is_sloop = sflag;
        is_ploop = pflag;
        curr_id = sid;
        tmu_id = nid;
        rd = decode_msp_op(rs1, rs2, rs3, rs4, spu_op);  
        is_tloop = false;
        is_sloop = false;
        is_ploop = false;

        return rd;
    }

    void NSU::mvmem(uint64_t rs1, uint64_t rs2, uint64_t rs3){
            uint64_t l_addr,d_addr;
            uint32_t stride;
            uint16_t length, height;
            std::uint32_t rdata;
            uint64_t src_addr,dst_addr;
            uint64_t ddr_high;
            MemoryInterface *src_intf;
            MemoryInterface *dst_intf;
            uint64_t ar_addr, aw_addr;
            std::uint64_t wstride, broad_cast;
            std::uint32_t stride_high, w_high, r_high;
            


            bool is_mcast;
            int rem;
            is_mcast = false;


            l_addr = (rs1 & 0x1FFFFFFFFF);
            d_addr = (rs3 & 0x1FFFFFFFFF);
            stride = (rs2 & 0xFFFFFFFF);
            wstride = (((rs1>>48) & 0xFFFF)) | (((rs3>>32)&0xFFFF0000));
            length = ((rs2>>32) & 0xFFFF);
            height = ((rs2>>48) & 0xFFFF);
            src_addr = l_addr;
            dst_addr = d_addr;
            src_intf = mem_intf;
            dst_intf = mem_intf;

            rem =  length %4;
            //#pragma omp parallel for
            for(int i = 0 ; i < height ; i++){
                ar_addr = src_addr;
                aw_addr = dst_addr;
                for(int j = 0 ; j < length/4 ; j++){
                    rdata  = src_intf -> readDataMem(src_addr, 4);
                    dst_intf -> writeDataMem(dst_addr, rdata, 4);
                    

                    src_addr = src_addr + 4;
                    dst_addr = dst_addr + 4;
                    if(j%8 == 0) {
                        perf -> dataMemoryRead();
                        perf -> dataMemoryWrite();
                    }
                    
                }
                if(rem!= 0){
                    rdata = src_intf -> readDataMem(src_addr ,rem);
                    dst_intf -> writeDataMem(dst_addr,rdata, rem);
                    
                    src_addr = src_addr + rem;
                    dst_addr = dst_addr + rem;
                    if(length%32 != 0){
                        perf->dataMemoryRead();
                        perf -> dataMemoryWrite();
                    }

                }
                src_addr = ar_addr + stride;
                dst_addr = aw_addr + wstride;
                

            }
            perf->cycleInc(NSU_MVMEM_CYCLE(length, height));
            
        }
    void NSU::mcast_s2s(uint64_t rs1, uint64_t rs2, uint64_t rs3){
            uint64_t l_addr,d_addr;
            uint32_t stride;
            uint16_t length, height;
            std::uint32_t rdata;
            uint64_t src_addr,dst_addr;
            uint64_t ddr_high,spu_sel;
            bool target_nest_sel;
            MemoryInterface *src_intf;
            MemoryInterface *dst_intf;
            uint64_t ar_addr, aw_addr;
            std::uint64_t wstride, broad_cast;
            std::uint32_t stride_high, w_high, r_high;
            int src_id;
            uint64_t target_tmu;
            l_addr = (rs1 & 0x7FFFFFF);
            d_addr = ((rs1>>27) & 0x7FFFFFF);
            stride = (rs2 & 0xFFFFFFFF);
            wstride = (rs3 & 0xFFFFFFFF);
            length = ((rs2>>32) & 0xFFFF);
            height = ((rs2>>48) & 0xFFFF);
            src_id = ((rs1>>56) & 0x3F);
            target_nest_sel = ((rs1>>63)&0x1 == 0x1);
            target_tmu = (target_nest_sel)? rs3 & 0xFFFFFFFF00000000:((rs3>>32) & 0xFFFFFFFF);
            src_addr = l_addr;
            dst_addr = d_addr;
            int rem;
                src_intf = l2_intf[src_id];
                //simple DMA function
                rem =  length %4;
                //#pragma omp parallel for
                for(int x = 0 ; x < height ; x++){
                    ar_addr = src_addr;
                    aw_addr = dst_addr;
                    for(int j = 0 ; j < length/4 ; j++){
                        rdata  = src_intf -> readDataMem(src_addr, 4);
                        if(j%8 == 0) perf -> dataMemoryRead();
                        for(int k = 0; k < TMU_NUM ;k++){
                            if(((target_tmu>>k) & 0x1) == 0x1){
                                l2_intf[k] -> writeDataMem(dst_addr, rdata,4);
                                if(j%8 == 0) perf->dataMemoryWrite();
                            }
                        }
                        src_addr = src_addr + 4;
                        dst_addr = dst_addr + 4;

                    }
                    if(rem!= 0){
                        rdata = src_intf -> readDataMem(src_addr ,rem);
                        if(length%32 != 0) perf -> dataMemoryRead();
                        for(int k = 0; k < TMU_NUM ;k++){
                            if(((target_tmu>>k) & 0x1) == 0x1){
                                l2_intf[k] -> writeDataMem(dst_addr, rdata,rem);
                                if(length%32 != 0)perf->dataMemoryWrite();
                            }
                        }

                        src_addr = src_addr + rem;
                        dst_addr = dst_addr + rem;
                    }
                    src_addr = ar_addr + stride;
                    dst_addr = aw_addr + wstride;
            }
            perf->cycleInc(NSU_MCAST_S2S_CYCLE(length, height));

        }
    void NSU::mcast_g2s(uint64_t rs1, uint64_t rs2, uint64_t rs3){
            uint64_t l_addr,d_addr;
            uint32_t stride;
            uint16_t length, height;
            std::uint32_t rdata;
            uint64_t src_addr,dst_addr;
            uint64_t ddr_high,tmu_sel;
            MemoryInterface *src_intf;
            MemoryInterface *dst_intf;
            uint64_t ar_addr, aw_addr;
            std::uint64_t wstride, broad_cast;
            std::uint32_t stride_high, w_high, r_high;
            src_addr = ((rs1>>27) & 0x1FFFFFFFFF);
            dst_addr = (rs1 & 0x7FFFFFF);
            stride = (rs2 & 0xFFFFFFFF);
            length = ((rs2>>32) & 0xFFFF);
            height = ((rs2>>48) & 0xFFFF);

            int rem;
            wstride = length;
            
            src_intf = mem_intf;
            dst_intf = l2_intf[0];

            //simple DMA function
            rem =  length %4;
            //#pragma omp parallel for
            for(int i = 0 ; i < height ; i++){
                ar_addr = src_addr;
                aw_addr = dst_addr;
                for(int j = 0 ; j < length/4 ; j++){
                    rdata  = src_intf -> readDataMem(src_addr, 4);
                    if(j%8 == 0) perf -> dataMemoryRead();
                    tmu_sel = rs3;
                    for(int k = 0; k < TMU_NUM ;k++){
                        if((tmu_sel & 0x1) == 0x1){
                            l2_intf[k] -> writeDataMem(dst_addr, rdata,4);
                            if(j%8==0) perf->dataMemoryWrite();
                        }
                        tmu_sel = tmu_sel>>1;
                    }
                    src_addr = src_addr + 4;
                    dst_addr = dst_addr + 4;
                    
                }
                if(rem!= 0){
                    rdata = src_intf -> readDataMem(src_addr ,rem);
                    if(length%32 != 0)perf -> dataMemoryRead();
                    tmu_sel = rs3;
                    for(int k = 0; k < TMU_NUM ;k++){
                        if((tmu_sel & 0x1) == 0x1){
                            l2_intf[k] -> writeDataMem(dst_addr, rdata,rem);
                            if(length%32 != 0)perf->dataMemoryWrite();
                        }
                        tmu_sel = tmu_sel>>1;
                    }
                    
                    src_addr = src_addr + rem;
                    dst_addr = dst_addr + rem;
                }
                src_addr = ar_addr + stride;
                dst_addr = aw_addr + wstride;

            }
            perf->cycleInc(NSU_MCAST_G2S_CYCLE(length, height));

        }

    void NSU::wsplit(){
        bool stack_recovery_en;
        bool stack_save_target; //true L2spm : false DDR
        MemoryInterface *dst_intf;
        uint32_t stack_save_addr;
        uint32_t stack_pointer;
        uint32_t stack_size;
        uint64_t CSR_DATA,rdata;
        CSR_DATA = GSPR[GSPR_STACK_INFO];
        stack_pointer = CSR_DATA&0xFFFFFFFFF;
        stack_size    = CSR_DATA>>48;
        CSR_DATA = GSPR[GSPR_STACK_SAVE];
        stack_save_target = (((CSR_DATA>>40) & 0x1) == 0x1);
        stack_save_addr   = CSR_DATA & 0xFFFFFFFFF;
        stack_recovery_en = (((CSR_DATA>>48) & 0x1) == 0x1);
        //if(stack_size > stack_pointer){stack_pointer = 0;}
        //else{stack_pointer -= stack_size ;}
        dst_intf = mem_intf;
        if(stack_recovery_en){
            int rem ;
            rem = stack_size%8;
            //if(!stack_save_target){
            for(int j = 0 ; j < stack_size/4 ; j++){
                rdata  = mem_intf -> readDataMem(stack_pointer, 4);
                if(j%8 == 0) perf -> dataMemoryRead();
                dst_intf -> writeDataMem(stack_save_addr, rdata, 4);
                if(j%8 == 0) perf -> dataMemoryWrite();
                stack_pointer = stack_pointer + 4;
                stack_save_addr = stack_save_addr + 4;
            }
            if(rem!= 0){
                rdata = mem_intf -> readDataMem(stack_pointer ,rem);
                if(stack_size%32 != 0)perf -> dataMemoryRead();
                dst_intf -> writeDataMem(stack_save_addr,rdata, rem);
                if(stack_size%32 != 0)perf -> dataMemoryWrite();
                stack_pointer = stack_pointer + rem;
                stack_save_addr = stack_save_addr + rem;
            }
            /*}else {
                for(int j = 0; j < stack_size; j++){
                    rdata = mem_intf -> readDataMem(stack_pointer, 1);
                    if(j%32 == 0) perf -> dataMemoryRead();
                    STK_SRAM[stack_save_addr] = rdata;
                    stack_pointer++;
                    stack_save_addr++;
                }
            }*/
        }    
        //reset execution time value to 0
        for(int i = 0; i < TMU_NUM ;i++){
            tmu[i]->check_start();
        }
        for(int j = 0 ; j < SPU_NUM; j++){
            spu[j]->check_start();
        }
    }

    void NSU::wjoin(){
        bool stack_recovery_en;
        bool stack_save_target; //true L2spm : false DDR
        MemoryInterface *src_intf;
        uint32_t stack_save_addr;
        uint32_t stack_pointer;
        uint32_t stack_size;
        uint64_t CSR_DATA,rdata;
        CSR_DATA = GSPR[GSPR_STACK_INFO];
        stack_pointer = CSR_DATA&0xFFFFFFFFF;
        stack_size    = CSR_DATA>>48;
        CSR_DATA = GSPR[GSPR_STACK_SAVE];
        stack_save_target = (((CSR_DATA>>40) & 0x1) == 0x1);
        stack_save_addr   = CSR_DATA & 0xFFFFFFFFF;
        stack_recovery_en = (((CSR_DATA>>48) & 0x1) == 0x1);
        //if(stack_size > stack_pointer){stack_pointer = 0;}
        //else{stack_pointer -= stack_size;}

        src_intf =  mem_intf;
        if(stack_recovery_en){
            if(!stack_save_target){
            int rem ;
            rem = stack_size%8;
            for(int j = 0 ; j < stack_size/4 ; j++){
                rdata  = src_intf -> readDataMem(stack_save_addr, 4);
                if(j%8 == 0) perf -> dataMemoryRead();
                mem_intf -> writeDataMem(stack_pointer, rdata, 4);
                if(j%8 == 0) perf -> dataMemoryWrite();
                stack_pointer = stack_pointer + 4;
                stack_save_addr = stack_save_addr + 4;
            }
            if(rem!= 0){
                rdata = src_intf -> readDataMem(stack_save_addr ,rem);
                if(stack_size%32 != 0)perf -> dataMemoryRead();
                mem_intf -> writeDataMem(stack_pointer ,rdata, rem);
                if(stack_size%32 != 0)perf -> dataMemoryWrite();
                stack_pointer = stack_pointer + rem;
                stack_save_addr = stack_save_addr + rem;
            }
            }else{
                for(int j = 0; j < stack_size; j++){
                    rdata = STK_SRAM[stack_save_addr];
                    mem_intf -> writeDataMem(stack_pointer, rdata, 1);
                    if(stack_size%32 != 0)perf -> dataMemoryWrite();
                    stack_pointer++;
                    stack_save_addr++;
                }
            }
        }    
        for(int i= 0; i < TMU_NUM ; i++){
            if(!tmu_queue[i].empty()){
                LOG(LOG_ERROR) << "[ERROR] COMMAND not finished (can't escape wjoin)\n";
                if(is_monitoring) {
                    tmu_queue[i].swap(empty);
                }else{
                    sc_core::sc_stop();
                }
            }
        }
        for(int j = 0; j < SPU_NUM; j++){
            if(!spu_queue[j].empty()){
                LOG(LOG_ERROR) << "[ERROR] COMMAND not finished (can't escape wjoin)\n";
                if(is_monitoring){
                    spu_queue[j].swap(empty);
                }else{
                    sc_core::sc_stop();
                }
            }
        }
        //check execution time
        loop_exec_time = 0;
        for(int i =0 ; i < TMU_NUM; i++){
            int_fast64_t curr_exec, spu_exec, max_spu_exec;
            max_spu_exec = 0;
            curr_exec = tmu[i]->check_cycle();
            for(int j = 0; j < (SPU_NUM/TMU_NUM) ; j++){
                spu_exec = spu[(SPU_NUM/TMU_NUM)*i +j]->check_cycle();
                max_spu_exec = (spu_exec > max_spu_exec)? spu_exec : max_spu_exec;
            }
            curr_exec += max_spu_exec;
            loop_exec_time = (loop_exec_time > curr_exec)? loop_exec_time: curr_exec;
        }
        
    }

    void NSU::startp(uint64_t rs1, uint64_t rs2){
        uint32_t id;
        if((rs2 & 0x400) == 0x400){
            id = rs2 & (0x3f);
        }else{
            id = rs1;
        }
        
        this-> tmu_id = id;
        if(id >= TMU_NUM){
            LOG(LOG_ERROR) << "[ERROR] NO MATCHING NEST ID FOUND!! \n";
            if(is_monitoring){
                tmu_id = 0;
            }else{
                sc_core::sc_stop();  
                return;
            }
        }
        if(is_ploop){
            LOG(LOG_ERROR) << "[ERROR] tmu warp start error already in warp";
        }else{
            is_ploop  = true;
        }
    }
    void NSU::endp(uint64_t rs1, uint64_t rs2){
        uint32_t id;
        if((rs2 & 0x400) == 0x400){
            id = rs2 & (0x3f);
        }else{
            id = rs1;
        }
        if((tmu_id == id) && is_ploop){
            //end success
            is_ploop = false;
        }else {
            LOG(LOG_ERROR) << "[ERROR] tmu warp end error\n" ;
            if(is_monitoring){
                is_ploop == false;
                return;
            }
            sc_core::sc_stop();
        }
    }

    void NSU::starts(uint64_t rs1, uint64_t rs2){
        uint32_t id;
        if((rs2 & 0x400) == 0x400){
            id = rs2 & (0x3f);
        }else{
            id = rs1;
        }
        if(id >= GDMAC_NUM){
            LOG(LOG_ERROR) << "[ERROR] NO MATCHING SHARED ID FOUND!! \n";
            if(is_monitoring){
                id = 0;
            }else{
                sc_core::sc_stop();
                return;
            }
        }
        if(this->is_sloop){
            LOG(LOG_ERROR) << "[ERROR] gwarp start error already in warp\n";
            sc_core::sc_stop();
        }else if(!is_ploop){
            LOG(LOG_ERROR) << "[ERROR] starts ends must included in startp endp (must set tmu number)\n";
            sc_core::sc_stop();
        }else{
            this->is_sloop  = true;
        }
    }
    void NSU::ends(uint64_t rs1, uint64_t rs2){
        uint32_t id;
        if((rs2 & 0x400) == 0x400){
            id = rs2 & (0x3f);
        }else{
            id = rs1;
        }
        if(this->is_sloop && is_ploop){
            //end success
            this->is_sloop = false;
        }else {
            LOG(LOG_ERROR) << "[ERROR] gwarp end error\n" ;
        }
        
    }



    void NSU::startt(uint64_t rs1, uint64_t rs2){
        uint32_t id;
        if((rs2 & 0x400) == 0x400){
            id = rs2 & (0x3f);
        }else{
            id = rs1;
        }
        if(id >= SPU_NUM){
            LOG(LOG_ERROR) << "[ERROR] NO MATCHING SPU ID FOUND!! \n";
            if(is_monitoring){
                id = 0;
            }else{
                sc_core::sc_stop();
                return;
            }
        }
        this -> curr_id = id;
        if(is_tloop){
            LOG(LOG_ERROR) << "[ERROR] warp start error already in warp\n";
            sc_core::sc_stop();
        }else if(!is_ploop){
            LOG(LOG_ERROR) << "[ERROR] startt endt must included in startp endp (must set tmu number)\n";
            sc_core::sc_stop();
        }else{
            is_tloop = true;
        }
    }

    void NSU::endt(uint64_t rs1, uint64_t rs2){
        uint32_t id;
        if((rs2 & 0x800) == 0x400){
            id = rs2 & (0x3f);
        }else{
            id = rs1;
        }
        if((curr_id == id) && is_tloop && is_ploop){
            //end success
            is_tloop = false;
        }else {
            if(is_monitoring) is_tloop = false;
            LOG(LOG_ERROR) << "[ERROR] warp end error\n" ;
        }
        
    }

    void NSU::credit_ld_i(uint64_t rs1, uint64_t rs2 ){
        uint64_t temp;
        int id;
        if(!is_ploop){//credit_ld_inc NSU
        /*
            if(!use_nsu_queue){
                for(int i = 0; i < TMU_NUM; i++){
                    temp  = 0x1 << i;
                    if((rs2 & temp) == temp){
                        for(int j= 0 ; j < (SPU_NUM/TMU_NUM); j++){
                            temp  = 0x1 << j;
                            if((rs1 & temp) == temp){
                                id = (SPU_NUM/TMU_NUM)*i + j;
                                if(tmu[i]->chk_ldcrd()){
                                credit_ld[id]++;
                                LOG(LOG_DEBUG) << "[DEBUG] LOAD  CREDIT Increased - SPU UID : "<< id<< " (current value = " << credit_ld[id] << ")\n";
                                if(scredit_flag[id]){
                                    scredit_flag[id] = false;
                                    pop_spu_queue(id);
                                }
                                }
                            }
                        }
                    }
                }
            }else{
                push_nsu_queue(nsu_codes, nsu_rs1,nsu_rs2,nsu_rs3,nsu_rs4);
            }*/
            LOG(LOG_WARNING) << "[WARNING] credit command must in plan\n";
        }else if(is_sloop){//credit_ld_inc TMU
            int gid = (GDMAC_NUM/TMU_NUM)*tmu_id ;
            if(!use_tmu_queue[gid]){
                id = (SPU_NUM/TMU_NUM)*tmu_id;
                for(int i = 0 ; i < SPU_NUM/TMU_NUM; i++){
                    temp = 0x1 << i;
                    if((rs1&temp) == temp){
                        int tid;
                        tid = id + i;
                        if(tmu[tmu_id]->chk_ldcrd()){
                        credit_ld[tid]++;
                        this->tmu[tmu_id]->setNSPR(0x781,((this->tmu[tmu_id]->getNSPR(0x781)) | ((0x1) << i)));
                        LOG(LOG_DEBUG) << "[DEBUG] LOAD  CREDIT Increased - SPU UID : "<< tid<< " (current value = " << credit_ld[tid] << ")\n";
                        if(scredit_flag[tid]){
                            scredit_flag[tid] = false;
                            pop_spu_queue(tid);
                        }
                        }
                    }
                }
            }else{
                push_tmu_queue(nsu_codes, nsu_rs1,nsu_rs2,nsu_rs3,nsu_rs4, gid);
            }
        }else if(is_tloop){ // credit_ld_dec SPUis_tloop
            int wid = (SPU_NUM /TMU_NUM)*tmu_id + curr_id;
            if(!use_spu_queue[wid]){
                id = (tmu_id)*(SPU_NUM/TMU_NUM) + curr_id;
                if(tmu[tmu_id]->chk_ldcrd()){
                    if(credit_ld[id]>0){
                        credit_ld[id]--;
                        if(credit_ld[id] == 0){
                            this->tmu[tmu_id]->setNSPR(0x781,((this->tmu[tmu_id]->getNSPR(0x781)) & (~((0x1) << curr_id))));
                        }
                        LOG(LOG_DEBUG) << "[DEBUG] LOAD  CREDIT Decreased - SPU UID : "<< id<< " (current value = " << credit_ld[id] << ")\n";
                        if(tfull_flag[id]){
                            tfull_flag[id] = false;
                            pop_tmu_queue((tfull_id[id]));
                        }else if(nfull_flag[id]){
                            nfull_flag[id] = false;
                            pop_nsu_queue();
                        }
                    }else{
                        use_spu_queue[id] = true;
                        scredit_flag[id] = true;
                        push_spu_queue(nsu_codes, nsu_rs1, nsu_rs2, nsu_rs3, nsu_rs4, id);
                    }
                }
            }else{
                push_spu_queue(nsu_codes, nsu_rs1, nsu_rs2, nsu_rs3, nsu_rs4, wid);
            }
            
        }else{
            LOG(LOG_ERROR) << "[ERROR] LOAD CREDIT ERROR\n";
            if(!is_monitoring)sc_core::sc_stop();
        }

    }
    //will delete ld chk
    void NSU::credit_ld_chk(uint64_t rs1, uint64_t rs2){
        //uint64_t credit_reg;
        //uint16_t ld_max;
        //int id;
        //uint64_t temp;
        //credit_reg = (((GSPR[GSPR_STACK_SAVE]>>37) & 0x1) == 0x1);
        //ld_max = (credit_reg == 0x1)? 2 : 1;
        if(!is_ploop){
            /*
            if(!use_nsu_queue){
            for(int i = 0; i < TMU_NUM; i++){
                    temp  = 0x1 << i;
                    credit_reg = (tmu[i]->getNSPR(NSPR_OP_MODE)&0x1);
                    ld_max = (credit_reg == 0x1)? 2 : 1;
                    if((rs2 & temp) == temp){
                        for(int j= 0 ; j < (SPU_NUM/TMU_NUM); j++){
                            if(credit_ld[id]>= ld_max){

                                use_nsu_queue = true;
                                nfull_flag[id] = true;
                                push_nsu_queue(nsu_codes,nsu_rs1,nsu_rs2,nsu_rs3,nsu_rs4);
                            }
                        }
                    }
            }
            }else{
                push_nsu_queue(nsu_codes, nsu_rs1,nsu_rs2,nsu_rs3,nsu_rs4);
            }
            */
           LOG(LOG_WARNING) << "[WARNING] credit must in plan\n";
        }else if(is_tloop){
            if(use_spu_queue[tmu_id*(SPU_NUM/TMU_NUM)+curr_id ]){
                        push_spu_queue(OP_GTX_CREDIT_ST_CHK,rs1,rs2,0,0, tmu_id*(SPU_NUM/TMU_NUM)+curr_id);
                }else{
                if(!((credit_ld[tmu_id*(SPU_NUM/TMU_NUM) + curr_id] != 0)||!(tmu[tmu_id]->chk_ldcrd()))){
                    scredit_flag[tmu_id*(SPU_NUM/TMU_NUM) + curr_id] = true;
                    use_spu_queue[tmu_id*(SPU_NUM/TMU_NUM) + curr_id]  = true;
                    push_spu_queue(OP_GTX_CREDIT_ST_CHK,rs1,rs2,0,0, tmu_id*(SPU_NUM/TMU_NUM) + curr_id);
                }
            }
        }/*else if(is_sloop){
            int gid = (GDMAC_NUM/TMU_NUM)*tmu_id ;
            if(!use_tmu_queue[gid]){
            id = (SPU_NUM/TMU_NUM)*tmu_id;
            credit_reg = (tmu[tmu_id]->getNSPR(NSPR_OP_MODE)&0x1);
            ld_max = (credit_reg == 0x1)? 2 : 1;
            for(int i = 0 ; i < SPU_NUM/TMU_NUM; i++){
                temp = 0x1 << i;
                if((rs1&temp) == temp){
                    int tid;
                    tid = id + i;
                    if(credit_ld[tid]>= ld_max){

                        use_tmu_queue[tmu_id] = true;
                        tfull_flag[tid] =true;
                        tfull_id[tid] = tmu_id;
                        push_tmu_queue(nsu_codes,nsu_rs1,nsu_rs2,nsu_rs3,nsu_rs4,tmu_id);
                    }
                }
            }
            }else{
                push_tmu_queue(nsu_codes, nsu_rs1, nsu_rs2, nsu_rs3, nsu_rs4, gid);
            }
        }*/else{
            LOG(LOG_ERROR) << "[ERROR] LOAD CHECK ERROR\n";
            if(!is_monitoring) sc_core::sc_stop();
        }
    }

    void NSU::credit_st_i(uint64_t rs1, uint64_t rs2){
        uint64_t temp;
        int id;
        if(!is_ploop){//credit_st_dec NSU
        /*
            if(!use_nsu_queue){
            for(int i = 0; i < TMU_NUM; i++){
                temp  = 0x1 << i;
                if((rs2 & temp) == temp){
                    for(int j= 0 ; j < (SPU_NUM/TMU_NUM); j++){
                        temp  = 0x1 << j;
                        if((rs1 & temp) == temp){
                            id = (SPU_NUM/TMU_NUM)*i + j;
                            if(tmu[i]->chk_stcrd()){
                                credit_st[id]--;
                                LOG(LOG_DEBUG) << "[DEBUG] STORE CREDIT Increased - SPU UID : "<< id<< " (current value = " << credit_st[id] << ")\n";
                                if(sfull_flag[id]){
                                    sfull_flag[id] = false;
                                    pop_spu_queue(id);
                                }
                            }
                        }
                    }
                }
            }
            }else {
                push_nsu_queue(nsu_codes, nsu_rs1, nsu_rs2, nsu_rs3, nsu_rs4);
            }*/
            LOG(LOG_WARNING) << "[WARNING] credit command must in plan loop\n";
        }else if(is_sloop){//credit_st_dec TMU
            int gid = (GDMAC_NUM/TMU_NUM)*tmu_id;
            if(!use_tmu_queue[gid]){
            id = (SPU_NUM/TMU_NUM)*tmu_id;
            for(int i = 0 ; i < SPU_NUM/TMU_NUM; i++){
                temp = 0x1 << i;
                if((rs1&temp) == temp){
                    int tid;
                    tid = id + i;
                    if(tmu[tmu_id]->chk_stcrd()){
                        if(credit_st[tid] == 0){
                            LOG(LOG_ERROR) << "[ERROR] credit already 0 but program try to decrease st credit!!(plz check code) - ID : " << tid << std::endl;
                        }
                        credit_st[tid]--;
                        if(credit_st[id] == 0){
                            this->tmu[tmu_id]->setNSPR(0x781,((this->tmu[tmu_id]->getNSPR(0x781)) & (~((0x1) << (i+16)))));
                        }
                        LOG(LOG_DEBUG) << "[DEBUG] STORE CREDIT Decreased - SPU UID : "<< tid<< " (current value = " << credit_st[tid] << ")\n";
                        if(sfull_flag[tid]){
                            sfull_flag[tid] = false;
                            pop_spu_queue(tid);
                        }
                    }
                }
            }
            }else{
                push_tmu_queue(nsu_codes, nsu_rs1, nsu_rs2, nsu_rs3, nsu_rs4, gid);
            }
        }else if(is_tloop){//credit st_inc SPU
            int wid = (SPU_NUM/TMU_NUM)*tmu_id + curr_id;
            if(!use_spu_queue[wid]){
            uint64_t max_st_credit,credit_reg;
             

            id = (tmu_id)*(SPU_NUM/TMU_NUM) + curr_id;
            credit_reg = (tmu[tmu_id]->getNSPR(NSPR_OP_MODE)&0x1);
            max_st_credit = (credit_reg == 0x1)? 2 : 1;   
            
            if(tmu[tmu_id]->chk_stcrd()){
                if(credit_st[id]<max_st_credit){
                    credit_st[id]++;
                    this->tmu[tmu_id]->setNSPR(0x781,((this->tmu[tmu_id]->getNSPR(0x781)) | (((0x1) << (curr_id+16)))));
                    LOG(LOG_DEBUG) << "[DEBUG] STORE CREDIT Increased - SPU UID : "<< id<< " (current value = " << credit_st[id] << ")\n";
                    if(tcredit_flag[id]){
                        tcredit_flag[id] = false;
                        pop_tmu_queue((tcredit_id[id]));
                    }else if(ncredit_flag[id]){
                        ncredit_flag[id] = false;
                        pop_nsu_queue();
                    }
                }else{
                    use_spu_queue[id] = true;
                    sfull_flag[id] = true;
                    push_spu_queue(nsu_codes, nsu_rs1, nsu_rs2, nsu_rs3, nsu_rs4, id);
                }
            }
            }else{
                push_spu_queue(nsu_codes, nsu_rs1, nsu_rs2, nsu_rs3, nsu_rs4, wid);
            }
            
        }else{
            LOG(LOG_ERROR) << "[ERROR] STORE CREDIT ERROR\n";
            if(!is_monitoring) sc_core::sc_stop();
        }
        
    }

    void NSU::credit_st_chk(uint64_t rs1, uint64_t rs2){
        uint64_t credit_reg;

        int id;
        uint64_t temp;

        if(!is_ploop){// is store exist?> NSU
            LOG(LOG_WARNING) << "[WARNING] CREDIT CHECK MUST CAST IN SHARED LOOP\n";
        }else if(is_sloop){
            int gid = (GDMAC_NUM/TMU_NUM)*tmu_id ;
            if(!use_tmu_queue[gid]){
            id = (SPU_NUM/TMU_NUM)*tmu_id;
            for(int i = 0 ; i < SPU_NUM/TMU_NUM; i++){
                temp = 0x1 << i;
                if((rs1&temp) == temp){
                    int tid;
                    tid = id + i;
                    if(tmu[tmu_id]->chk_stcrd()){          
                        if(credit_st[tid] == 0){
                            use_tmu_queue[tmu_id] = true;
                            tcredit_flag[tid] =true;
                            tcredit_id[tid] = tmu_id;
                            push_tmu_queue(OP_GTX_CREDIT_ST_CHK,rs1,rs2,0,0,tmu_id);
                            return;
                        }
                    }
                }
            }
            }else{
                push_tmu_queue(nsu_codes, nsu_rs1, nsu_rs2, nsu_rs3, nsu_rs4, gid);
            }
        }else{
            LOG(LOG_ERROR) << "[ERROR] STORE CHECK ERROR\n";
            if(!is_monitoring) sc_core::sc_stop();
        }
    }

    
    void NSU::dump_gtx_command(uint64_t rs1, uint64_t rs2, uint64_t rs3, uint64_t opsel, op_GTX_Codes msp_op){
        std::string fp;
        fp = cmd_dfname;
        if((msp_op &0xFF0) == 0xFF0) {return;}
        std::ofstream ofs(fp,std::ios::out | std::ios::app);
        if(!ofs) {
            LOG(LOG_WARNING) << "[WARNING] FAILED TO OPEN COMMAND DUMPFILE\n";
            return ;
        }
        uint64_t rs4;
        rs4 = ((uint64_t)tmu_id << 28) | ((uint64_t)curr_id << 22) | (is_ploop << 21) | (is_sloop << 20) | (is_tloop << 19) | (opsel << 10) | msp_op;
/*
        ofs << std::uppercase <<std::hex << std::setfill('0') 
            << std::setw(16) << rs1 << ' '
            << std::setw(16) << rs2 << ' '
            << std::setw(16) << rs3 << ' '
            << std::setw( 3) << msp_op << ' '
            << std::setw( 3) << opsel << ' '
            << is_tloop << ' '
            << is_sloop << ' '
            << is_ploop << ' '
            << curr_id << ' '
            << tmu_id << ' ' << std::endl;
*/
        ofs <<std::hex << std::setfill('0') 
            << std::setw(16) << rs4 
            << std::setw(16) << rs3 
            << std::setw(16) << rs2 
            << std::setw(16) << rs1 << std::endl;
        return ;
        
    }

    uint64_t NSU::decode_msp_op(uint64_t rs1, uint64_t rs2, uint64_t rs3 , uint64_t rs4, op_GTX_Codes msp_op){
        bool is_broadcast = false;
        uint64_t mask;
        uint64_t rd;
        
        nsu_codes = msp_op;
        nsu_rs1 = rs1;
        nsu_rs2 = rs2;
       
        nsu_rs3 = rs3;
        nsu_rs4 = rs4;

        if(!is_ploop && (msp_op != OP_GTX_START_P)){
            is_broadcast = true;
        }

        switch(msp_op){
            case OP_GTX_MCAST_S2L:
                if(is_broadcast){
                        for(int i = 0; i < TMU_NUM ; i++){
                            for(int j = 0; j < GDMAC_NUM/TMU_NUM ; j++){
                                if(!use_tmu_queue[i*(GDMAC_NUM/TMU_NUM) + j]){
                                    mask  = rs3 >> ((SPU_NUM/TMU_NUM)*i);
                                    this->tmu[i] -> run_gdmac(rs1,rs2,mask,rs4,msp_op,j);
                                }else{
                                    push_tmu_queue(msp_op,rs1,rs2,rs3,rs4,i*(GDMAC_NUM/TMU_NUM) + j);
                                }
                            }
                        }
                        perf->cycleInc(tmu[0]->curr_cycle);
                }else{
                    if(is_sloop){
                        LOG(LOG_WARNING) << "[WARNING] MCAST.S2L COMMAND IN SHARED!!\n";
                    }else if(!is_tloop  & !is_sloop){//& !use_tmu_queue[tmu_id]){
                        this->tmu[tmu_id]->run_gdmac(rs1,rs2,rs3,rs4,msp_op,0);
                        
                    }else{
                        LOG(LOG_WARNING) << "[WARNING] MCAST.S2L COMMAND IN THREAD!!\n";
                    }
                }
                break;
            case OP_GTX_TPOSE: 
            case OP_GTX_FILL:
                if(is_broadcast){
                        for(int i = 0; i < TMU_NUM ; i++){
                            for(int j = 0; j < GDMAC_NUM/TMU_NUM ; j++){
                                if(!use_tmu_queue[i*(GDMAC_NUM/TMU_NUM) + j]){
                                        this->tmu[i] -> run_gdmac(rs1,rs2,rs3,rs4,msp_op,j);
                                }else{
                                    push_tmu_queue(msp_op,rs1,rs2,rs3,rs4,i*(GDMAC_NUM/TMU_NUM) + j);
                                }
                            }
                        }
                        perf->cycleInc(tmu[0]->curr_cycle);
                }else{
                    if(is_sloop){
                        if(!use_tmu_queue[tmu_id]){
                            rd = this-> tmu[tmu_id] -> run_gdmac(rs1,rs2,rs3,rs4,msp_op, 0);
                        }else{
                            push_tmu_queue(msp_op,rs1,rs2,rs3,rs4,tmu_id);
                        }
                    }else if(!is_tloop & !use_tmu_queue[tmu_id]){
                        this->tmu[tmu_id]->run_gdmac(rs1,rs2,rs3,rs4,msp_op,0);
                    }else{
                        LOG(LOG_WARNING) << "[WARNING] SHARED COMMAND NOT IN SHARED!!\n";
                    }
                }
                break;
            case OP_GTX_RDSPR:
                uint16_t spr_addr, spu_id, nest_id;
                if(!is_ploop){
                    spr_addr = rs1 & 0xFFF;
                    spu_id = (rs1>>16)& 0x3F;
                    nest_id = (rs1>>24)& 0x3F;

                    if(spr_addr < 0x400){
                        rd = getGSPR(spr_addr);
                    }else if(spr_addr <= 0x7FF){
                        rd = this->tmu[nest_id] -> run_gdmac(rs1,rs2,rs3,rs4,msp_op,0);
                    }else if(spr_addr <= 0xBFF){
                        rd = this->tmu[nest_id] -> run_spu(rs1,rs2,rs3,rs4,msp_op,spu_id);
                    }else{
                        LOG(LOG_WARNING) << "[WARNING] RDSPR BAD ADDRESS\n";
                    }
                }else if(this->is_sloop){
                    LOG(LOG_WARNING) << "[WARNING] can't use rdspr in start - send (use rdspr in pstart - pend or tstart - tend)\n";
                }else if(this->is_tloop){
                    rd = this->tmu[tmu_id] -> run_spu(rs1,rs2,rs3,rs4,msp_op,curr_id);
                }else{
                    rd = this->tmu[tmu_id] -> run_gdmac(rs1,rs2,rs3,rs4,msp_op,0);
                }
                break;
            case OP_GTX_WRSPR:
                spr_addr = rs1 & 0xFFF;
                if(!is_ploop){
                    if(spr_addr < 0x400){
                        uint64_t pdata, wdata;
                        //strobe
                        uint64_t wstrb = 0;
                        uint64_t wstrbn = rs1 & 0xFF0000;
                        for(int i = 0; i < 8 ;i++){
                            if(((wstrbn>>(7-i)) & 0x10000) != 0x10000){
                                wstrb = (wstrb | 0xFF); 
                            }
                            if(i < 7){
                                wstrb = wstrb << 8;
                            }
                        }
                        pdata = (getGSPR(spr_addr) & ~wstrb);
                        wdata = (rs2 & wstrb);
                        setGSPR(spr_addr,(pdata | wdata));
                    }else if(spr_addr <= 0x7FF){
                        for(int i =0; i < TMU_NUM; i++){
                            if(((rs3>>i)&0x1) != 0x1){
                                rd = this->tmu[i] -> run_gdmac(rs1,rs2,rs3,rs4,msp_op,0);
                            }
                        }
                    }else if(spr_addr <= 0xBFF){
                        for(int i = 0; i < SPU_NUM ; i++){
                            if(((rs3>>i)&0x1) != 0x1){
                                rd = this->tmu[i/(SPU_NUM/TMU_NUM)] -> run_spu(rs1,rs2,rs3,rs4,msp_op,(i%(SPU_NUM/TMU_NUM)));
                            }
                        }
                    }else{
                        LOG(LOG_WARNING) << "[WARNING] WRSPR BAD ADDRESS\n";
                    }
                }else if(this->is_tloop){
                    this->tmu[tmu_id]->run_spu(rs1,rs2,rs3,rs4,msp_op,curr_id);
                }else if(this->is_sloop){
                    LOG(LOG_WARNING) << "[WARNING] can't use wrspr in start - send (use wrspr in pstart - pend or tstart - tend)\n";
                }else{
                    if(spr_addr >= 0x400){
                        if(spr_addr < 0x800){
                            for(int i =0; i < TMU_NUM; i++){
                                if(((rs3>>i)&0x1) != 0x1){
                                    this->tmu[tmu_id]->run_gdmac(rs1,rs2,rs3,rs4,msp_op,0);
                                }
                            }
                        }else if(spr_addr < 0xC00){
                            for(int i = 0 ; i < SPU_NUM/TMU_NUM ; i++){
                                if(((rs3>>i)&0x1) != 0x1){
                                    this->tmu[tmu_id]->run_spu(rs1,rs2,rs3,rs4,msp_op,i);
                                }
                            }
                        }else{
                            LOG(LOG_WARNING) << "[WARNING] WRSPR BAD ADDRESS\n";
                        }
                    }
                }
                break;
            case OP_GTX_LOAD:
                if(!is_ploop){
                    LOG(LOG_ERROR) << "[ERROR] COMMAND MUST INCLUDED IN START END\n";
                }else if(this->is_sloop){
                    if(use_tmu_queue[tmu_id]){
                        push_tmu_queue(msp_op,rs1,rs2,rs3,rs4, tmu_id );
                    }else{
                        tmu[tmu_id]-> run_gdmac(rs1,rs2,rs3,rs4,msp_op,0);
                    }
                }else if(is_tloop){
                    if(use_spu_queue[tmu_id*(SPU_NUM/TMU_NUM)+curr_id ]){
                        push_spu_queue(msp_op,rs1,rs2,rs3,rs4, tmu_id*(SPU_NUM/TMU_NUM)+curr_id);
                    }else{
                        //if((credit_ld[tmu_id*(SPU_NUM/TMU_NUM) + curr_id] != 0)||!(tmu[tmu_id]->chk_ldcrd())){
                            tmu[tmu_id]-> run_spu(rs1,rs2,rs3,rs4,msp_op,curr_id);
                        //}else{
                        //    scredit_flag[tmu_id*(SPU_NUM/TMU_NUM) + curr_id] = true;
                        //    use_spu_queue[tmu_id*(SPU_NUM/TMU_NUM) + curr_id]  = true;
                        //    push_spu_queue(msp_op,rs1,rs2,rs3,rs4, tmu_id*(SPU_NUM/TMU_NUM) + curr_id);
                        //}
                    }
                }else{
                    if(!use_tmu_queue[tmu_id]){
                        tmu[tmu_id] -> run_gdmac(rs1,rs2,rs3,rs4,msp_op,0);
                    }else{
                        LOG(LOG_ERROR) << "[ERROR] COMMAND CANNOT START WITH ONLY NLOOP\n";
                    }
                }
                break;
            case OP_GTX_STORE:
            case OP_GTX_COPY:
                if(!is_ploop){
                    LOG(LOG_ERROR) << "[ERROR] COMMAND MUST INCLUDED IN START END\n";
                }else if(is_sloop){
                    if(use_tmu_queue[tmu_id*(GDMAC_NUM/TMU_NUM) ]){
                        push_tmu_queue(msp_op,rs1,rs2,rs3,rs4, tmu_id*(GDMAC_NUM/TMU_NUM) );
                    }else{
                        tmu[tmu_id]-> run_gdmac(rs1,rs2,rs3,rs4,msp_op,0);
                    }
                }else if(is_tloop){
                    if(use_spu_queue[tmu_id*(SPU_NUM/TMU_NUM) + curr_id ]){
                        push_spu_queue(msp_op,rs1,rs2,rs3,rs4, tmu_id*(SPU_NUM/TMU_NUM)+curr_id);

                    }else{

                        tmu[tmu_id]-> run_spu(rs1,rs2,rs3,rs4,msp_op,curr_id);
                    }
                }else{
                    if(!use_tmu_queue[tmu_id]){
                        tmu[tmu_id]->run_gdmac(rs1,rs2,rs3,rs4,msp_op,0);
                    }else{
                        LOG(LOG_ERROR) << "[ERROR] COMMAND CANNOT START WITH ONLY NLOOP\n";
                    }
                }
                break;
            

            case OP_GTX_CREDIT_LD:
                this->credit_ld_i(rs1,rs2);
                break;
            case OP_GTX_CREDIT_LD_CHK:
                //this->credit_ld_chk(rs1,rs2);
                break;
            case OP_GTX_CREDIT_ST:
                this->credit_st_i(rs1,rs2);
                break;
            case OP_GTX_CREDIT_ST_CHK:
                if(is_sloop){
                    this->credit_st_chk(rs1,rs2);
                }else if(is_tloop){
                    this->credit_ld_chk(rs1,rs2);
                }else{
                    LOG(LOG_ERROR) << "[ERROR] credit check error - credit_chk only available in shared loop(store check) or thread loop(load check) \n";
                }
                break;

            case OP_GTX_MCAST_G2S:/*
                if(use_nsu_queue){
                    push_nsu_queue(msp_op, rs1,rs2,rs3,rs4);
                }else{
                    mcast_g2s(rs1,rs2,rs3);
                }*/
                mcast_g2s(rs1,rs2,rs3);
                break;
            case OP_GTX_MCAST_S2S:/*
                if(use_nsu_queue){
                    push_nsu_queue(msp_op, rs1,rs2,rs3,rs4);
                }else{
                    mcast_s2s(rs1,rs2,rs3);
                }*/
                mcast_s2s(rs1,rs2,rs3);
                break;
            case OP_GTX_COPY_MEM:/*
                if(use_nsu_queue){
                    push_nsu_queue(msp_op, rs1,rs2,rs3,rs4);
                }else{
                    mvmem(rs1,rs2,rs3);
                }*/
                mvmem(rs1,rs2,rs3);
                break;
            case OP_GTX_START_T://imm = rs2
                startt(rs1,rs2);
                break;
            case OP_GTX_END_T:
                endt(rs1,rs2);
                break;
            case OP_GTX_START_S:
                starts(rs1,rs2);
                break;
            case OP_GTX_END_S:
                ends(rs1,rs2);
                break;
            case OP_GTX_START_P:
                startp(rs1,rs2);
                break;
            case OP_GTX_END_P:
                endp(rs1,rs2);
                break;
            case OP_GTX_MSYNC:
                //nop in ISS
                break;
            case OP_GTX_MEXEC:
                mexec(rs1);
                break;
            default:// SPU COMMAND
                    if(is_broadcast){
                        for(int i = 0; i < TMU_NUM ; i++){
                            for(int j = 0; j < SPU_NUM/TMU_NUM ; j++){
                                if(!use_spu_queue[i*(SPU_NUM/TMU_NUM) + j]){
                                    this->tmu[i] -> run_spu(rs1,rs2,rs3,rs4,msp_op,j);
                                }else{
                                    push_spu_queue(msp_op, rs1,rs2,rs3,rs4 ,i*(SPU_NUM/TMU_NUM) + j);
                                    
                                }

                            }
                        }
                        perf->cycleInc(spu[0]->curr_cycle);
                    }else if(is_ploop && !is_sloop && !is_tloop){//code area 4 
                        uint64_t t = 0x1;

                            for(int j = 0; j < SPU_NUM/TMU_NUM ; j++){
                                if(!use_spu_queue[tmu_id*(SPU_NUM/TMU_NUM) + j]){
                                    this->tmu[tmu_id] -> run_spu(rs1,rs2,rs3,rs4,msp_op,j);
                                }else{
                                    push_spu_queue(msp_op, rs1,rs2,rs3,rs4 ,tmu_id*(SPU_NUM/TMU_NUM) + j);
                                    
                                }
                            }
                        
                    }else{
                        if(!use_spu_queue[tmu_id*(GDMAC_NUM/TMU_NUM) + curr_id]){
                            rd = this-> tmu[tmu_id] -> run_spu(rs1,rs2,rs3,rs4,msp_op, curr_id);
                        }else{
                            push_spu_queue(msp_op,rs1,rs2,rs3,rs4,tmu_id*(GDMAC_NUM/TMU_NUM) + curr_id);
                        }
                    }
                break;
           
        }
        //command dump
        if(gtx_command_dump){
            if(is_tloop && !(use_spu_queue[curr_id])){
                dump_gtx_command(rs1,rs2,rs3,rs4,msp_op);
            }else if(is_sloop && !(use_tmu_queue[tmu_id])){
                dump_gtx_command(rs1,rs2,rs3,rs4,msp_op);
            }else if(!is_sloop && ! is_tloop){
                dump_gtx_command(rs1,rs2,rs3,rs4,msp_op);
            }
        }
        //this->spu[id]->setSCSR(SCSR_RESULT_DATA,rd);
        return rd;
    }
    uint64_t NSU::mexec(uint64_t rs1){
            uint64_t raddr;
            uint64_t rdata[4];
            //uint32_t funct7, funct3;
            uint32_t opcode ;
            uint16_t spu_id;
            uint64_t result,status;
            uint16_t nest_id;
            uint16_t op_sel;
            op_GTX_Codes instr;
            bool lflag;
            bool sflag;
            bool gflag;
            status = 0;
            raddr = rs1;
            is_mexec = true;
            while(is_mexec){
                for(int i= 0; i < 4; i++){
                    rdata[i]  = mem_intf -> readDataMem(raddr, 4);
                    rdata[i]  = ((uint64_t)mem_intf->readDataMem(raddr+4, 4)<<32)|rdata[i];
                    if(i%4 == 0) perf -> dataMemoryRead();
                    raddr += 8;
                }
                //funct7 = ((rdata[3]>>3) & 0x7F);
                //funct3 = (rdata[3] & 0x7);
                opcode = (rdata[3] & 0x3FF);
                instr = static_cast<op_GTX_Codes>(opcode);
                if(opcode == 0x3b8){
                    is_mexec = false;
                    return status;
                }
                op_sel = (rdata[3]>>10)&0xFFF;
                spu_id = ((rdata[3]>>22) & 0x3F);
                nest_id = ((rdata[3]>>28)& 0x3F);
                lflag = ((rdata[3]>>19) & 0x1) == 0x1;
                sflag = ((rdata[3]>>20) & 0x1) == 0x1;
                gflag = ((rdata[3]>>21) & 0x1) == 0x1;
                tmu_id = nest_id;
                curr_id = spu_id;
                is_sloop = sflag;
                is_ploop = gflag;
                is_tloop = lflag;
                result = decode_msp_op(rdata[0], rdata[1], rdata[2], op_sel, instr);
                is_sloop = false;
                is_ploop = false;
                is_tloop = false; 
                status = (status > result)? status : result;
            }

        return status;
    }

    void NSU::dump_GSPR(){
        std::cout << " ************************************\n";
        std::cout << "           GSPR REGSITERS\n";
        std::cout << " ====================================\n";
        std::cout << " [GSPR_GTX_OPERAND1      ] : 0x" <<std::hex << std::uppercase<< GSPR[GSPR_GTX_OPERAND1] << std::endl;
        std::cout << " [GSPR_GTX_OPERAND2      ] : 0x" <<std::hex << std::uppercase<< GSPR[GSPR_GTX_OPERAND2] << std::endl;
        std::cout << " [GSPR_GTX_OPERAND3      ] : 0x" <<std::hex << std::uppercase<< GSPR[GSPR_GTX_OPERAND3] << std::endl;
        std::cout << " [GSPR_GTX_OPCODE        ] : 0x" <<std::hex << std::uppercase<< GSPR[GSPR_GTX_OPCODE] << std::endl;
        std::cout << " [GSPR_STACK_INFO        ] : 0x" <<std::hex << std::uppercase<< GSPR[GSPR_STACK_INFO] << std::endl;
        std::cout << " [GSPR_STACK_SAVE        ] : 0x" <<std::hex << std::uppercase<< GSPR[GSPR_STACK_SAVE] << std::endl;
        std::cout << " ************************************\n";
    }

    void NSU::initGSPR(){
        
        return;

        //GSPR[GSPR_GDMAC_CONTROL] = CONTROL_L_DTYPE;
        //GSPR[GSPR_SPU_CONTROL] = DRC_MSP_EN | DRC_STQ_EN;
    }

}
