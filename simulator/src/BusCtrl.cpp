
#include "BusCtrl.h"
#include "config.h"


namespace riscv_tlm {

    SC_HAS_PROCESS(BusCtrl);

    BusCtrl::BusCtrl(sc_core::sc_module_name const &name) :
            sc_module(name), cpu_instr_socket("cpu_instr_socket"), cpu_data_socket(
            "cpu_data_socket"), PCIE_data_socket("PCIE_data_socket"), memory_socket("memory_socket"),PCIE_socket("PCIE_socket"), timer_socket("timer_socket"),
            vm_socket("vm_socket")/*, trace_socket(
            "trace_socket") */{
        cpu_instr_socket.register_b_transport(this, &BusCtrl::b_transport);
        cpu_data_socket.register_b_transport(this, &BusCtrl::b_transport);
        PCIE_data_socket.register_b_transport(this,&BusCtrl::b_transport);

        cpu_instr_socket.register_get_direct_mem_ptr(this,
                                                     &BusCtrl::instr_direct_mem_ptr);
        memory_socket.register_invalidate_direct_mem_ptr(this,
                                                         &BusCtrl::invalidate_direct_mem_ptr);
    
        is_vmenable = true;
    }

    void BusCtrl::b_transport(tlm::tlm_generic_payload &trans,
                              sc_core::sc_time &delay) {

        sc_dt::uint64 adr = trans.get_address() / 4;
        
        unsigned char* data_ptr = trans.get_data_ptr();

        switch (adr) {
            //case VIRTUAL_DEVICE_COMMAND_LO / 4:
            //case VIRTUAL_DEVICE_COMMAND_HI / 4:
            //case VIRTUAL_DEVICE_FILEDESC_LO/ 4:
            //case VIRTUAL_DEVICE_FILEDESC_HI/ 4:
            //case VIRTUAL_DEVICE_OPENFLAG_LO/ 4:
            //case VIRTUAL_DEVICE_OPENFLAG_HI/ 4:
            //case VIRTUAL_DEVICE_ADDRESS_LO/  4:
            //case VIRTUAL_DEVICE_ADDRESS_HI/  4:
            //case VIRTUAL_DEVICE_LENGTH_LO/   4:
            //case VIRTUAL_DEVICE_LENGTH_HI/   4:
            //case VIRTUAL_DEVICE_PARAM0_LO/   4:
            //case VIRTUAL_DEVICE_PARAM0_HI/   4:    
            //case VIRTUAL_DEVICE_PARAM1_LO/   4:
            //case VIRTUAL_DEVICE_PARAM1_HI/   4:
            //case VIRTUAL_DEVICE_RESULT_LO/   4:
            //case VIRTUAL_DEVICE_RESULT_HI/   4:   
            //case VIRTUAL_DEVICE_RESULT1_LO/  4:
            //case VIRTUAL_DEVICE_RESULT1_HI/  4:
            //case VIRTUAL_DEVICE_LAST_LO/     4:
            //case VIRTUAL_DEVICE_LAST_HI/     4:
            //    if(is_vmenable){
            //        vm_socket->b_transport(trans,delay);
            //    }else{
            //        timer_socket->b_transport(trans, delay);
            //    }
            //    break;
            case SC_SIM_OUTPORT >> 2:
                for (int i = 0; i < trans.get_data_length(); ++i) {
                    printf("%c",static_cast<unsigned char>(data_ptr[i]));
                }
                fflush(stdout);
                break;
            case TIMERCTRL_MEMORY_ADDRESS /4:
            case TIMERDIV_MEMORY_ADDRESS / 4:
            case TIMER_MEMORY_ADDRESS_HI / 4:
            case TIMER_MEMORY_ADDRESS_LO / 4:
            case TIMERCMP_MEMORY_ADDRESS_HI / 4:
            case TIMERCMP_MEMORY_ADDRESS_LO / 4:
                timer_socket->b_transport(trans, delay);
                break;
            case PCIE_DBI_MSI/4:
            case PCIE_CFG_INTERRUPT_FLAG/4:
            case PCIE_CFG_INTERRUPT_MASK/4:
            case PCIE_CFG_MASKED_STATUS/4:
            case PCIE_INTERRUPT/4:
                PCIE_socket->b_transport(trans, delay);
                break;
            //case TRACE_MEMORY_ADDRESS / 4:
            //    trace_socket->b_transport(trans, delay);
            //    break;
                [[likely]] default:
                memory_socket->b_transport(trans, delay);
                break;
            

        }



        trans.set_response_status(tlm::TLM_OK_RESPONSE);
    }

    bool BusCtrl::instr_direct_mem_ptr(tlm::tlm_generic_payload &gp,
                                       tlm::tlm_dmi &dmi_data) {
        return memory_socket->get_direct_mem_ptr(gp, dmi_data);
    }

    void BusCtrl::invalidate_direct_mem_ptr(sc_dt::uint64 start,
                                            sc_dt::uint64 end) {
        cpu_instr_socket->invalidate_direct_mem_ptr(start, end);
    }
}