
// SPDX-License-Identifier: GPL-3.0-or-later
#include "CPU.h"

namespace riscv_tlm {

    CPURV64::CPURV64(sc_core::sc_module_name const &name, BaseType PC, bool debug, NSU *nsu_b, bool enable, uint32_t warp_time, riscv_tlm::peripherals::VMachine *vm) :
            CPU(name, debug,vm), INSTR(0), nsu(nsu_b) {

        register_bank = new Registers<BaseType>();
        mem_intf = new CPUMemintf(name);
        register_bank->setPC(PC);
        register_bank->setValue(Registers<BaseType>::sp, (Memory::SIZE / 4) - 1);

        int_cause = 0;
        instr_bus.register_invalidate_direct_mem_ptr(this, &CPURV64::invalidate_direct_mem_ptr);

        base_inst = new BASE_ISA<BaseType>(0, register_bank, mem_intf);
        c_inst = new C_extension<BaseType>(0, register_bank, mem_intf);
        m_inst = new M_extension<BaseType>(0, register_bank, mem_intf);
        a_inst = new A_extension<BaseType>(0, register_bank, mem_intf);
        f_inst = new F_extension<BaseType>(0, register_bank, mem_intf);
        gtx_inst = new GTX_extension<BaseType>(0, register_bank, mem_intf, nsu);

        gtx_inst -> set_monitor(enable, warp_time);

        base_inst->wait_intr = false;
        base_inst->stop_wait = false;

        base_inst->vm = vm;
         
        trans.set_data_ptr(reinterpret_cast<unsigned char *>(&INSTR));
    }

    CPURV64::~CPURV64() {
        delete register_bank;
        delete mem_intf;
        delete base_inst;
        delete c_inst;
        delete m_inst;
        delete a_inst;
        delete gtx_inst;
        delete m_qk;
    }

    bool CPURV64::cpu_process_IRQ() {
        BaseType csr_temp;
        bool ret_value = false;

        if (interrupt) {
            if(base_inst->wait_intr){
                base_inst->stop_wait = true;
            }
            csr_temp = register_bank->getCSR(CSR_MSTATUS);
            if ((csr_temp & MSTATUS_MIE) == 0) {
                return ret_value;
            }

            csr_temp = register_bank->getCSR(CSR_MIP);

            // Fix: int_cause contains the interrupt number (e.g., 7 for timer)
            // Need to set the corresponding bit in MIP (e.g., bit 7 = 0x80)
            uint64_t int_bit = 1ULL << (int_cause & 0x1f);
            if ((csr_temp & int_bit) == 0) {
                csr_temp |= int_bit;
                register_bank->setCSR(CSR_MIP, csr_temp);
            }

                /* updated MEPC register */
                BaseType old_pc = register_bank->getPC();
                register_bank->setCSR(CSR_MEPC, old_pc);

                /* update MCAUSE register */
                register_bank->setCSR(CSR_MCAUSE,((uint64_t)0x8000000000000000 | int_cause) );


                /* set new PC address */
                BaseType new_pc = register_bank->getCSR(CSR_MTVEC);
                //new_pc = new_pc & 0xFFFFFFFC; // last two bits always to 0
                register_bank->setPC(new_pc);
                
                csr_temp = register_bank->getCSR(CSR_MSTATUS);
                register_bank->setCSR(CSR_MSTATUS, (csr_temp | MSTATUS_MPIE) & ~MSTATUS_MIE);// set mie 0 mpie 1

                ret_value = true;
                interrupt = false;
                irq_already_down = false;
        } else {
            if (!irq_already_down) {
                csr_temp = register_bank->getCSR(CSR_MIP);
                csr_temp &= ~MIP_MEIP;
                register_bank->setCSR(CSR_MIP, csr_temp);
                irq_already_down = true;
            }
        }

        return ret_value;
    }

    bool CPURV64::CPU_step() {

        /* Get new PC value */
        if (dmi_ptr_valid) {
            /* if memory_offset at Memory module is set, this won't work */
            std::memcpy(&INSTR, dmi_ptr + register_bank->getPC(), 4);
        } else {
            sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
            tlm::tlm_dmi dmi_data;
            trans.set_address(register_bank->getPC());
            instr_bus->b_transport(trans, delay);

            if (trans.is_response_error()) {
                LOG(LOG_ERROR) << "[ERROR] CPU step failed at read address from memory\n";
                if(!is_monitoring) sc_core::sc_stop();
            }

            if (trans.is_dmi_allowed()) {
                dmi_ptr_valid = instr_bus->get_direct_mem_ptr(trans, dmi_data);
                if (dmi_ptr_valid) {
                    dmi_ptr = dmi_data.get_dmi_ptr();
                }
            }
        }
        if(is_regdump && (register_bank ->getPC() == reg_dump_addr)){
            register_bank -> dump_regtofile(is_regaddr);
        }

        perf->codeMemoryRead();
        inst.setInstr(INSTR);
        bool breakpoint = false;


        base_inst->setInstr(INSTR);
        auto deco = base_inst->decode();
        if (deco != OP_ERROR) {
            auto PC_not_affected = base_inst->exec_instruction(inst, &breakpoint, deco);

            if (PC_not_affected) {
                register_bank->incPC();
            }
        } else {
            // C extension is only valid when lower 2 bits are NOT 0b11
            // 0b11 indicates a 32-bit instruction, not a compressed instruction
            bool is_compressed = (INSTR & 0x3) != 0x3;

            if (is_compressed) {
                c_inst->setInstr(INSTR);
                auto c_deco = c_inst->decode();
                if (c_deco != OP_C_ERROR ) {
                    auto PC_not_affected = c_inst->exec_instruction(inst, &breakpoint, c_deco);
                    if (PC_not_affected) {
                        register_bank->incPCby2();
                    }
                } else {
                    std::cout << "[WARNING] C instruction not implemented yet at PC=0x" << std::hex << register_bank->getPC() << " INSTR=0x" << INSTR << std::dec << std::endl;
                    inst.dump();
                    base_inst->NOP();
                }
            } else {
                m_inst->setInstr(INSTR);
                auto m_deco = m_inst->decode();
                if (m_deco != OP_M_ERROR) {
                    auto PC_not_affected = m_inst->exec_instruction(inst, m_deco);
                    if (PC_not_affected) {
                        register_bank->incPC();
                    }
                } else {
                    gtx_inst->setInstr(INSTR);
                    auto gtx_deco = gtx_inst->decode();
                    if (gtx_deco != OP_GTX_ERROR) {
                        auto PC_not_affected = gtx_inst->exec_instruction(inst, gtx_deco);
                        if (PC_not_affected) {
                            register_bank->incPC();
                        }
                    } else {
                        f_inst -> setInstr(INSTR);
                        auto f_deco = f_inst -> decode();
                        if(f_deco != OP_F_ERROR){
                            auto PC_not_affected = f_inst->exec_instruction(inst, f_deco);
                            if(PC_not_affected){
                                register_bank->incPC();
                            }
                        } else {
                            a_inst -> setInstr(INSTR);
                            auto a_deco = a_inst->decode();
                            if(a_deco != OP_A_ERROR){
                                auto PC_not_affected = a_inst->exec_instruction(inst, a_deco);
                                if(PC_not_affected){
                                   register_bank -> incPC();
                                }
                            } else{
                                LOG(LOG_WARNING) << "[WARNING] Extension not implemented yet" << std::endl;
                                inst.dump();
                                base_inst->NOP();
                            }
                        }
                    }
                }
            }
        }

        if (breakpoint) {
            LOG(LOG_INFO) << "[INFO] Breakpoint set to true\n";
        }

        
        perf->instructionsInc();

        return breakpoint;
    }
    void CPURV64::dump_reg(){
        if(vts_debug){
            this -> register_bank-> dump();
        }
    }
    void CPURV64::dump_fp_reg(){
        if(vts_debug){
            this -> register_bank-> fp_dump();
        }
    }
    void CPURV64::dump_csr_reg(int addr){
        if(vts_debug){
            this -> register_bank-> csr_dump(addr);
        }
    }
    void CPURV64::dump_reg_file(bool is_addr){
        this -> register_bank -> dump_regtofile(is_addr);
    }
    bool CPURV64::check_breakpoint(){
        if(breakpoints.find(this->register_bank->getPC()) != breakpoints.end()){
            return true;
        } else {
            return false;
        }
    }
    void CPURV64::set_breakpoint(std::uint64_t pc_break){
        std::cout << "[DBG-UI] Breakpoint set : 0x" << std::hex << pc_break << std::endl;
        breakpoints.insert(pc_break);
    }
    void CPURV64::delete_breakpoint(std::uint64_t pc_break){
        if(breakpoints.find(pc_break) != breakpoints.end()){
            breakpoints.erase(pc_break);
            std::cout << "[DBG-UI] Delete Breakpoint" << std::endl;
        }else{
            std::cout << "[DBG-UI] No Breakpoint!!" << std::endl;
        }
    }
    void CPURV64::print_breakpoint(){
        std::cout << std::hex << "[DBG-UI] <Breakpoints>"  << std::endl;
        for(const auto& bp : breakpoints){
            std::cout << std::hex << "[DBG-UI] 0x" << bp << std::endl;
        }
    }
    void CPURV64::delete_breakpoint_all(){
        breakpoints.clear();
        std::cout << "[DBG-UI] Clear Breakpoints" << std::endl;
    }

    void CPURV64::call_interrupt(tlm::tlm_generic_payload &m_trans,
                              sc_core::sc_time &delay) {
        interrupt = true;
        if(m_trans.get_command() == tlm::TLM_READ_COMMAND){
            interrupt = false;
        }
        /* Socket caller send a cause (its id) */
        memcpy(&int_cause, m_trans.get_data_ptr(), sizeof(BaseType));

        delay = sc_core::SC_ZERO_TIME;
    }

    std::uint64_t CPURV64::getStartDumpAddress() {
        return register_bank->getValue(Registers<std::uint32_t>::t0);
    }

    std::uint64_t CPURV64::getEndDumpAddress() {
        return register_bank->getValue(Registers<std::uint32_t>::t1);
    }
}