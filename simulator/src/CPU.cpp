/*!         
 \file CPU.cpp
 */
#include "CPU.h"

namespace riscv_tlm {

    SC_HAS_PROCESS(CPU);

    CPU::CPU(sc_core::sc_module_name const &name, bool debug,riscv_tlm::peripherals::VMachine *vm) : sc_module(name), instr_bus("instr_bus"), inst(0),vert_m(vm), default_time(10 ,sc_core::SC_NS) {
        perf = Performance::getInstance();

        m_qk = new tlm_utils::tlm_quantumkeeper();
        m_qk->reset();
        mem_intf = nullptr;
        dmi_ptr_valid = false;



        irq_already_down = false;
        interrupt = false;
        cycle = 0;
        vts_debug = debug;
        is_breakpoint = false;

        irq_line_socket.register_b_transport(this, &CPU::call_interrupt);
        irq_line_socket_1.register_b_transport(this, &CPU::call_interrupt);

        trans.set_command(tlm::TLM_READ_COMMAND);

        trans.set_data_length(4);
        trans.set_streaming_width(4); // = data_length to indicate no streaming
        trans.set_byte_enable_ptr(nullptr); // 0 indicates unused
        trans.set_dmi_allowed(false); // Mandatory initial value
        trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);

        //if (!debug) { //CHANGE DEBUG MODE
        SC_THREAD(CPU_thread);
        //}
    };
    
    
    void CPU::off_debug(){
        this -> vts_debug = false;
    }

    void CPU::set_cycle(int cycles){
        this -> cycle = cycles;
    }

    void CPU::set_perfcheck(){
        this -> perf -> is_perfcheck = true;
    }
    

    void CPU::invalidate_direct_mem_ptr(sc_dt::uint64 start, sc_dt::uint64 end) {
        (void) start;
        (void) end;
        dmi_ptr_valid = false;
    }

    [[noreturn]] void CPU::CPU_thread() {

        while (true) {
            if(vts_debug &&check_breakpoint()){
                if(!is_breakpoint){
                    cycle = 0;
                    std::cout << "[DBG-UI] !!Break Point" << std::endl;
                    is_breakpoint = true;
                } else{
                    is_breakpoint = false;
                }
            }
            if(vts_debug && (cycle == 0)){
                sc_core::sc_pause();//test pause for debug
            } else { 
                if(cycle >0){ cycle--; }
                CPU_step();
                cpu_process_IRQ();
            }

            /* Fixed instruction time to 10 ns (i.e. 100 MHz) */
//#define USE_QK
#ifdef USE_QK
            // Model time used for additional processing
            m_qk->inc(default_time);
            if (m_qk->need_sync()) {
                m_qk->sync();
            }
#else
            sc_core::wait(default_time);
#endif
        } // while(1)
    } // CPU_thread
}