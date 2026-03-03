#include "TMU.h"


namespace riscv_tlm {
        
        TMU::TMU(sc_core::sc_module_name const &name, riscv_tlm::SPU *spu_i[SPU_NUM/TMU_NUM], int id) :sc_module(name){
            mem_intf = new MemoryInterface((std::string(name) + "_mem").c_str());
            l2_intf = new MemoryInterface((std::string(name) + "_l2").c_str());

            perf = Performance::getInstance();
            for(int i = 0 ; i < SPU_NUM/TMU_NUM ; i++){
                l1_intf[i] = new MemoryInterface((std::string(name) + "_l1_" + std::to_string(i)).c_str());
                spu[i] = spu_i[i];
            }
            NSPR[NSPR_TYPE] = 0x1;
            NSPR[NSPR_OP_MODE] = 0x300;
            exec_cycle = 0;
            ID = id;
            for(int i = 0 ; i < SPU_NUM/TMU_NUM;i++){
                spu[i]->datatype = 0x1;
            }
            tracker_file = "./tracker/TRACE_SMU" + std::to_string(ID);
            if(gtx_tracker){
                std::ofstream ofs(tracker_file, std::ios::out | std::ios::trunc);
            }
        
        }
            
            void TMU::dump_NSPR(){
                LOG(LOG_INFO) << " ************************************\n";
                LOG(LOG_INFO) << "           NSPR REGSITERS\n";
                LOG(LOG_INFO) << " ====================================\n";
                LOG(LOG_INFO) << " [NSPR_THREAD_MASK  ] : 0x" <<std::hex << std::uppercase<< NSPR[NSPR_THREAD_MASK] << std::endl;
                LOG(LOG_INFO) << " [NSPR_SHARED_MASK  ] : 0x" <<std::hex << std::uppercase<< NSPR[NSPR_SHARED_MASK] << std::endl;
                LOG(LOG_INFO) << " [NSPR_TYPE         ] : 0x" <<std::hex << std::uppercase<< NSPR[NSPR_TYPE] << std::endl;
                LOG(LOG_INFO) << " [NSPR_OP_MODE      ] : 0x" <<std::hex << std::uppercase<< NSPR[NSPR_OP_MODE] << std::endl;
                //LOG(LOG_INFO) << " [NSPR_SDLE_STATUS  ] : 0x" <<std::hex << std::uppercase<< NSPR[NSPR_SDLE_STATUS] << std::endl;
                LOG(LOG_INFO) << " [NSPR_CREDIT_COUNT ] : 0x" <<std::hex << std::uppercase<< NSPR[NSPR_CREDIT_COUNT] << std::endl;
                LOG(LOG_INFO) << " [NSPR_CREDIT_ERROR ] : 0x" <<std::hex << std::uppercase<< NSPR[NSPR_CREDIT_ERROR] << std::endl; 
                LOG(LOG_INFO) << " ************************************\n";
                return;
            }
            void TMU::dump_gtx_command(uint64_t rs1, uint64_t rs2, uint64_t rs3, uint64_t opsel, op_GTX_Codes msp_op){
                std::ofstream ofs(tracker_file,std::ios::out | std::ios::app);
                if(!ofs) {
                    LOG(LOG_WARNING) << "[WARNING] FAILED TO OPEN COMMAND DUMPFILE\n";
                    return ;
                }
                if (msp_op != OP_GTX_RDSPR && msp_op != OP_GTX_WRSPR ){
                    ofs << std::hex << std::setfill('0') << std::setw(16) << rs1 << " " 
                                             << std::setw(16) << rs2 << " " 
                                             << std::setw(16) << rs3 << " " 
                                             << std::setw( 3) << opsel << " " 
                                             << std::setw( 3) << msp_op << std::endl;
                }

            }

            void TMU::setNSPR(uint16_t nspr, uint64_t value){
                uint64_t  t_msk;
                uint64_t rd;

                NSPR[nspr] = value;
                if(nspr == NSPR_TYPE){
                    for(int i = 0 ; i < SPU_NUM/TMU_NUM;i++){
                        spu[i]->datatype = (value&0x3);
                    }
                }


            }
            
            uint64_t TMU::getNSPR(uint16_t nspr) {
                uint64_t ret_value;

                ret_value = NSPR[nspr]; 

            
                return ret_value;
            }


            void TMU::DMA_2D(uint64_t rs1, uint64_t rs2, uint64_t rs3, int funct){
                uint64_t src_addr;
                uint64_t dst_addr;
                uint32_t stride,wstride;
                uint16_t length, height;
                uint64_t ar_addr, aw_addr;
                uint64_t rdata;
                MemoryInterface *src_intf;
                MemoryInterface *dst_intf;

                
                switch(funct){
                    case 0: // ldspm.g
                        src_addr = ((rs1>>27) & 0x1FFFFFFFFF);
                        dst_addr = (rs1 & 0x7FFFFFF);
                        stride = (rs2& 0xFFFFFFFF);
                        wstride = (rs3 & 0xFFFFFFFF);
                        length = ((rs2>>32) & 0xFFFF);
                        height = ((rs2>>48) & 0xFFFF);
                        //wstride = length;
                        src_intf = mem_intf;
                        dst_intf = l2_intf;
                        LOG(LOG_DEBUG) << "[DEBUG] LOAD MAIN MEMORY(0x"<< std::hex<<src_addr<<")->L2 MEMORY(0x"<<dst_addr<<")\n";
                        break;
                    case 1: // stspm.g
                        dst_addr = ((rs1>>27) & 0x1FFFFFFFFF);
                        src_addr = (rs1 & 0x7FFFFFF);
                        wstride = (rs2& 0xFFFFFFFF);
                        stride = (rs3 & 0xFFFFFFFF);
                        length = ((rs2>>32) & 0xFFFF);
                        //stride = length;
                        height = ((rs2>>48) & 0xFFFF);
                        src_intf = l2_intf;
                        dst_intf = mem_intf;
                        LOG(LOG_DEBUG) << "[DEBUG] STORE L2 MEMORY(0x"<< std::hex<<src_addr<<")->MAIN MEMORY(0x"<<dst_addr<<")\n";
                        break;
                    default: //mvspm.g = stspm.s
                        dst_addr = ((rs1>>32) & 0xFFFFFFFF);
                        src_addr = (rs1 & 0xFFFFFFF);
                        stride = (rs2& 0xFFFFFFFF);
                        wstride = (rs3 & 0xFFFFFFFF);
                        length = ((rs2>>32) & 0xFFFF);
                        height = ((rs2>>48) & 0xFFFF);
                        src_intf = l2_intf;
                        dst_intf = l2_intf;
                        LOG(LOG_DEBUG) << "[DEBUG] COPY L2 MEMORY(0x"<< std::hex<<src_addr<<")->L2 MEMORY(0x"<<dst_addr<<")\n";
                        break;
                }
                int rem = length % 4;
                //#pragma omp parallel for
                for(int i = 0 ; i < height ; i++){
                    ar_addr = src_addr;
                    aw_addr = dst_addr;
                    for(int j = 0 ; j < length/4 ; j++){
                        rdata  = src_intf -> readDataMem(src_addr, 4);
                        dst_intf -> writeDataMem(dst_addr, rdata, 4);
                        if(j%8 == 0){
                            perf->dataMemoryRead();
                            perf->dataMemoryWrite();
                        }
                        src_addr = src_addr + 4;
                        dst_addr = dst_addr + 4;
                        
                    }
                    if(rem!= 0){
                        rdata = src_intf -> readDataMem(src_addr ,rem);
                        dst_intf -> writeDataMem(dst_addr,rdata, rem);
                        if(length%32){
                            perf->dataMemoryRead();
                            perf->dataMemoryWrite();
                        }
                        src_addr = src_addr + rem;
                        dst_addr = dst_addr + rem;
                    }
                    src_addr = ar_addr + stride;
                    dst_addr = aw_addr + wstride;
    
                }
                curr_cycle =SMU_EXEC_CYCLE(length, height);
                exec_cycle += SMU_EXEC_CYCLE(length, height);
            }

            
        
            void TMU::mcast(uint64_t rs1, uint64_t rs2, uint64_t rs3){
                uint64_t l_addr,d_addr;
                uint32_t stride;
                uint16_t length, height;
                uint32_t rdata;
                uint64_t src_addr,dst_addr;
                uint64_t ddr_high;
                MemoryInterface *src_intf;
                MemoryInterface *dst_intf;
                uint64_t ar_addr, aw_addr;
                uint64_t wstride, broad_cast;
                uint32_t stride_high, w_high, r_high;
                uint64_t spu_sel;
                src_addr = ((rs1>>32) & 0xFFFFFFFF);
                dst_addr = (rs1 & 0xFFFFFF);
                stride = (rs2 & 0xFFFFFFFF);
                length = ((rs2>>32) & 0xFFFF);
                height = ((rs2>>48) & 0xFFFF);

                int rem;
                wstride = length;

                src_intf = l2_intf;
                dst_intf = l1_intf[0];

                spu_sel = NSPR[NSPR_THREAD_MASK];
                //simple DMA function
                rem =  length %4;
                //#pragma omp parallel for
                for(int i = 0 ; i < height ; i++){
                    ar_addr = src_addr;
                    aw_addr = dst_addr;
                    for(int j = 0 ; j < length/4 ; j++){
                        rdata  = src_intf -> readDataMem(src_addr, 4);
                        if(j%8 != 0) perf->dataMemoryRead();
                        spu_sel = rs3;
                        for(int k = 0; k < (SPU_NUM/TMU_NUM) ;k++){
                            if((spu_sel & 0x1) == 0x1){
                                l1_intf[k] -> writeDataMem(dst_addr, rdata,4);
                                if(j%8 != 0) perf->dataMemoryWrite();
                            }
                            spu_sel = spu_sel>>1;
                        }
                        src_addr = src_addr + 4;
                        dst_addr = dst_addr + 4;

                    }
                    if(rem!= 0){
                        rdata = src_intf -> readDataMem(src_addr ,rem);
                        if(length%32 != 0)perf->dataMemoryRead();
                        spu_sel = rs3;
                        for(int k = 0; k < (SPU_NUM/TMU_NUM) ;k++){
                            if((spu_sel & 0x1) == 0x1){
                                l1_intf[k] -> writeDataMem(dst_addr, rdata,rem);
                                if(length%32 != 0) perf->dataMemoryWrite();
                            }
                            spu_sel = spu_sel>>1;
                        }

                        src_addr = src_addr + rem;
                        dst_addr = dst_addr + rem;
                    }
                    src_addr = ar_addr + stride;
                    dst_addr = aw_addr + wstride;

                }
                curr_cycle = MCAST_EXEC_CYCLE(length,height);
                exec_cycle += MCAST_EXEC_CYCLE(length,height);

        }
        

            void TMU::Transpose(uint64_t rs1, uint64_t rs2, uint64_t rs3){
        
                    uint16_t dim0,dim1,dim2,dim0_size,dim1_size,dim2_size;
                    uint64_t src_addr, dst_addr;
                    uint16_t rw_dir;
                    uint16_t dtype;
                    bool is16;
                    uint32_t wstride;

                    MemoryInterface *src_intf;
                    MemoryInterface *dst_intf;

                    src_addr = (rs1 & 0x1FFFFFFFFF);
                    dst_addr = (rs3 & 0x1FFFFFFFFF);
                    dim0 = rs2&0x3;
                    dim1 = (rs2&0x30)>>4;
                    dim2 = (rs2&0x300)>>8;
                    dim0_size = ((rs2>>16)&0xFFFF);
                    dim1_size = ((rs2>>32)&0xFFFF);
                    dim2_size = ((rs2>>48)&0xFFFF);
                    rw_dir = ((rs1>>48) & 0x3);
                    dtype = (0x1 & (rs1>>56));
                    is16 = (dtype != 0);
                    src_intf = ((0x2 & rw_dir) == 0) ? l2_intf : mem_intf;
                    dst_intf = ((0x1 & rw_dir) == 0) ? l2_intf : mem_intf;
                   
                    uint32_t dim2_s, dim1_s, dim0_s;//dim stride
                    uint32_t dim2_n, dim1_n, dim0_n;

                    switch(dim0){
                        case 0:
                            if(dim1 == 1 && dim2 == 2){
                                dim2_s = 1;
                                dim1_s = dim2_size;
                                dim0_s = dim1_size*dim2_size;
                                dim2_n = dim2_size;
                                dim1_n = dim1_size;
                                dim0_n = dim0_size;
                            }else if(dim1 == 2 && dim2 == 1){
                                dim2_s = dim2_size;
                                dim1_s = 1;
                                dim0_s = dim1_size*dim2_size;
                                dim2_n = dim1_size;
                                dim1_n = dim2_size;
                                dim0_n = dim0_size;
                            }else {return;}
                            break;
                        case 1:
                            if(dim1 == 0 && dim2 ==2){
                                dim2_s = 1;
                                dim1_s = dim1_size*dim2_size;
                                dim0_s = dim2_size;
                                dim2_n = dim2_size;
                                dim1_n = dim0_size;
                                dim0_n = dim1_size;
                            }else if(dim1 == 2 && dim2 ==0){
                                dim2_s = dim1_size*dim2_size;
                                dim1_s = 1;
                                dim0_s = dim2_size;
                                dim2_n = dim0_size;
                                dim1_n = dim2_size;
                                dim0_n = dim1_size;
                            } else{return;}
                            break;
                        case 2: 
                            if(dim1 == 1 && dim2 ==0){
                                dim2_s = dim1_size * dim2_size;
                                dim1_s = dim2_size;
                                dim0_s = 1;
                                dim2_n = dim0_size;
                                dim1_n = dim1_size;
                                dim0_n = dim2_size;
                            }else if(dim1 == 0 && dim2 ==1){
                                dim2_s = dim2_size;
                                dim1_s = dim1_size * dim2_size;
                                dim0_s = 1;
                                dim2_n = dim1_size;
                                dim1_n = dim0_size;
                                dim0_n = dim2_size;
                            }else{return;}
                            break;
                        default: return; break;
                    }            
        
                    uint32_t data;
                    int byte_size;
                    byte_size = (is16)? 2:1;
                    uint32_t r_p, w_p;
                    w_p = 0;
                    //#pragma omp parallel for collapse(3)
                    for(int i = 0; i <dim0_n; i++){
                        for(int j = 0; j < dim1_n ; j++){
                            for(int k =0; k < dim2_n ; k++){
                                r_p = (dim0_s *i + dim1_s * j + dim2_s *k) *byte_size;
                                data = src_intf -> readDataMem(src_addr + r_p , byte_size);            
                                dst_intf -> writeDataMem(dst_addr + w_p, data, byte_size);
                                w_p = w_p + byte_size;
                                if(k % (32/byte_size) == 0) {
                                    perf->dataMemoryRead();
                                    perf->dataMemoryWrite();
                                }
                            }
                            
                        }
                    }
                    curr_cycle = TRANSPOSE_EXEC_CYCLE(dim0_n,dim1_n,dim2_n);
                    exec_cycle += TRANSPOSE_EXEC_CYCLE(dim0_n,dim1_n,dim2_n);

    }           


            void TMU::fill(uint64_t rs1, uint64_t rs2, uint64_t rs3){
                    uint32_t stride;
                    uint64_t dst_addr;
                    uint16_t rw_dir,length, height;
                    uint64_t pattern,ddr_high;
                    MemoryInterface *dst_intf;
                    uint32_t nxt_addr;
                    dst_addr = (rs1 & 0x1FFFFFFFFF);
                    stride = (rs2 & 0x1FFFFFFFF);
                    rw_dir = ((rs1>>48) & 0x1);
                    length = ((rs2>>32) & 0xFFFF);
                    height = ((rs2>>48) & 0xFFFF);


                    nxt_addr = dst_addr + stride;
                    pattern = rs3;

                    dst_intf = (rw_dir == 0)? l2_intf : mem_intf;

                    int rem = length%8;
                    //#pragma omp parallel for
                    for(int i = 0 ; i < height ; i++){
                        for(int j = 0 ; j < length/8 ; j++){
                            uint32_t wdata;
                            dst_intf -> writeDataMem(dst_addr, pattern, 4);
                            wdata = pattern>>32;
                            dst_intf -> writeDataMem(dst_addr+4, wdata, 4);
                            if(j%4 == 0)perf->dataMemoryWrite();

                            dst_addr = dst_addr + 8;
                        }
                        if(rem){
                            uint64_t temp;
                            temp = pattern;
                            for(int k = 0; k < rem; k++){
                                dst_intf -> writeDataMem(dst_addr,temp, 1);
                                dst_addr++;
                                temp = temp >> 8;
                            }
                            if(length%32 != 0) perf->dataMemoryWrite();
                        }
                        dst_addr = nxt_addr;
                        nxt_addr += stride;         
                    }
                    curr_cycle =FILL_EXEC_CYCLE(length, height);
                    exec_cycle += FILL_EXEC_CYCLE(length, height);
                }

            bool TMU::chk_ldcrd(){
                if((NSPR[NSPR_OP_MODE] & 0x100) == 0x100){
                    return true;
                }
                return false;
            }
            bool TMU::chk_stcrd(){
                if((NSPR[NSPR_OP_MODE] & 0x200) == 0x200){
                    return true;
                }
                return false;
            }

            uint64_t TMU::run_spu(uint64_t rs1, uint64_t rs2, uint64_t rs3, uint64_t rs4,op_GTX_Codes msp_op, int id){
                if((((NSPR[NSPR_THREAD_MASK]>>id)&0x1) == 0x1) && (msp_op != OP_GTX_WRSPR)){
                    //LOG(LOG_DEBUG) << "[DEBUG] ignore spu command - masked(check mask bit) \n";
                    return 0;
                }
                return this->spu[id]->decode_spu_command(rs1,rs2,rs3,rs4,msp_op);
            }

            uint64_t TMU::run_gdmac(uint64_t rs1, uint64_t rs2, uint64_t rs3, uint64_t rs4, op_GTX_Codes msp_op, int id){
                uint64_t result;
                bool is_masked;
                is_masked = ((NSPR[NSPR_SHARED_MASK] & 0x1) == 0x1);
                if(is_masked &&(msp_op != OP_GTX_WRSPR)){
                    LOG(LOG_DEBUG) << "[DEBUG] ignore tmu command - masked(check mask bit)\n";
                    return 0;
                }

                if(gtx_tracker){
                    dump_gtx_command(rs1,rs2,rs3,rs4,msp_op);
                }
                switch(msp_op){
                    case OP_GTX_TPOSE:
                        Transpose(rs1,rs2,rs3);
                        break;
                    case OP_GTX_FILL:
                        fill(rs1,rs2,rs3);
                        break;
                    case OP_GTX_MCAST_S2L:
                        mcast(rs1,rs2,rs3);
                        break;
                    case OP_GTX_LOAD:
                        DMA_2D(rs1,rs2,rs3,0);
                        break;
                    case OP_GTX_STORE:
                        DMA_2D(rs1,rs2,rs3,1);
                        break;
                    case OP_GTX_COPY:
                        DMA_2D(rs1,rs2,rs3,2);
                        break;
                    case OP_GTX_RDSPR:
                        uint16_t nsu_addr, spu_id;
                        nsu_addr = (rs1& 0xFFF);
                        spu_id = (rs1>>16)& 0x3F;
                        if(nsu_addr < 0x400){
                            LOG(LOG_WARNING) << "[WARNING] RDSPR BAD ADDRESS\n";
                        }else if(nsu_addr <= 0x7FF){
                            result = getNSPR(nsu_addr);
                        }else if(nsu_addr <= 0xBFF){
                            result = run_spu(rs1,rs2,rs3,rs4,msp_op,spu_id);
                        }else{
                            LOG(LOG_WARNING) << "[WARNING] RDSPR BAD ADDRESS\n";
                        }
                        curr_cycle = RDSPR_EXEC_CYCLE;
                        exec_cycle += RDSPR_EXEC_CYCLE;
                        break;
                    case OP_GTX_WRSPR:
                        nsu_addr = (rs1 & 0xFFF);
                        if(nsu_addr < 0x400){
                            LOG(LOG_WARNING) << "[WARNING] WRSPR BAD ADDRESS\n";
                        }else if(nsu_addr <= 0x7FF){
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
                                pdata = (getNSPR(nsu_addr) & ~wstrb);
                                wdata = (rs2 & wstrb);
                                setNSPR(nsu_addr,(pdata | wdata));
                        }
                        
                        /*else if(nsu_addr <= 0xBFF){
                            for(int i = 0 ;i  < SPU_NUM/TMU_NUM; i++){
                                if((spu_mask& 0x1) == 0x0){     
                                    run_spu(rs1,rs2,rs3,rs4,msp_op,i);
                                }
                                spu_mask >> 1;
                            }
                        }*/else{
                            LOG(LOG_WARNING) << "[WARNING] WRSPR BAD ADDRESS - " << std::hex << nsu_addr <<std::endl;
                        }
                        curr_cycle = WRSPR_EXEC_CYCLE;
                        exec_cycle += WRSPR_EXEC_CYCLE;
                        break;
                    default:
                        LOG(LOG_ERROR) << "[ERROR] UNKNOWN COMMAND - TMU\n";
                        std::cout << "opcode " << msp_op << std::endl;
                        if(!is_monitoring) sc_core::sc_stop();
                        break;
                }
                return result;

            }
            

}