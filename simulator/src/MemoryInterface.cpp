


#include "MemoryInterface.h"
#include "config.h"
#include <iostream>
#include <sstream>

namespace riscv_tlm {

    MemoryInterface::MemoryInterface(sc_core::sc_module_name const &name) :
            data_bus(("data_bus" + std::string(name)).c_str()) { }

/**
 * Access data memory to get data
 * @param  addr address to access to
 * @param size size of the data to read in bytes
 * @return data value read
 */
    std::uint32_t MemoryInterface::readDataMem(std::uint64_t addr, int size) {
        std::uint32_t data;
        tlm::tlm_generic_payload trans;
        sc_core::sc_time delay = sc_core::SC_ZERO_TIME;

        trans.set_command(tlm::TLM_READ_COMMAND);
        trans.set_data_ptr(reinterpret_cast<unsigned char *>(&data));
        trans.set_data_length(size);
        trans.set_streaming_width(4); // = data_length to indicate no streaming
        trans.set_byte_enable_ptr(nullptr); // 0 indicates unused
        trans.set_dmi_allowed(false); // Mandatory initial value
        trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
        trans.set_address(addr);

        data_bus->b_transport(trans, delay);

        if (trans.is_response_error() && !is_failed) {
                std::stringstream error_msg;
                error_msg << "Read address : 0x" << std::hex << addr;
                LOG(LOG_ERROR)<< "[ERROR] MEMORY " << error_msg.str().c_str() << std::endl<<std::endl;
                if(!is_monitoring){
                    sc_core::sc_report_handler::set_actions(sc_core::SC_ERROR, sc_core::SC_DO_NOTHING | sc_core::SC_STOP);
                    sc_core::sc_stop();
                }
                is_failed = true;

        }

        return data;
    }

/**
 * Acces data memory to write data
 * @brief
 * @param addr addr address to access to
 * @param data data to write
 * @param size size of the data to write in bytes
 */
    void MemoryInterface::writeDataMem(std::uint64_t addr, std::uint32_t data, int size) {
        tlm::tlm_generic_payload trans;
        sc_core::sc_time delay = sc_core::SC_ZERO_TIME;

        trans.set_command(tlm::TLM_WRITE_COMMAND);
        trans.set_data_ptr(reinterpret_cast<unsigned char *>(&data));
        trans.set_data_length(size);
        trans.set_streaming_width(4); // = data_length to indicate no streaming
        trans.set_byte_enable_ptr(nullptr); // 0 indicates unused
        trans.set_dmi_allowed(false); // Mandatory initial value
        trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
        trans.set_address(addr);

        data_bus->b_transport(trans, delay);

        if (trans.is_response_error() && !is_failed) {
            std::stringstream error_msg;
            error_msg << "Write memory: 0x" << std::hex << addr;
            LOG(LOG_ERROR)<< "[ERROR] MEMORY " << error_msg.str().c_str() << std::endl<<std::endl;
            if(!is_monitoring){
                sc_core::sc_report_handler::set_actions(sc_core::SC_ERROR, sc_core::SC_DO_NOTHING | sc_core::SC_STOP);
                sc_core::sc_stop();
            }
            is_failed = true;

        }
    }


    CPUMemintf::CPUMemintf(sc_core::sc_module_name const &name) :MemoryInterface(name) {
        monitoring  = false;
        fst_routine =false;
    }

    void CPUMemintf::writeDataMem(std::uint64_t addr, std::uint32_t data, int size) {
        tlm::tlm_generic_payload trans;
        sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
        WriteIntf cur_intf;

        if(monitoring){
            cur_intf.addr = addr;
            cur_intf.data = data;
            cur_intf.size = size;
            if(addr > high_stack){
                high_stack = addr;
            }
            if(addr < low_stack){
                low_stack = addr;
            }
            if(fst_routine){
                write_ref.push_back(cur_intf);
            }else{
                write_vect.push_back(cur_intf);
            }
        }

        trans.set_command(tlm::TLM_WRITE_COMMAND);
        trans.set_data_ptr(reinterpret_cast<unsigned char *>(&data));
        trans.set_data_length(size);
        trans.set_streaming_width(4); // = data_length to indicate no streaming
        trans.set_byte_enable_ptr(nullptr); // 0 indicates unused
        trans.set_dmi_allowed(false); // Mandatory initial value
        trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
        trans.set_address(addr);

        data_bus->b_transport(trans, delay);

        if (trans.is_response_error() && is_failed){
            std::stringstream error_msg;
            error_msg << "Write memory: 0x" << std::hex << addr;
            LOG(LOG_ERROR)<< "[ERROR] MEMORY " << error_msg.str().c_str() << std::endl <<std::endl;
            if(!is_monitoring){
                sc_core::sc_report_handler::set_actions(sc_core::SC_ERROR, sc_core::SC_DO_NOTHING | sc_core::SC_STOP);
                sc_core::sc_stop();
            }
            is_failed = true;

        }
    }

    bool CPUMemintf::WriteMonitor(){
        bool is_okay = true;
        if(write_ref.size() != write_vect.size()){
            LOG(LOG_WARNING) << "[WARNING] MONITOR >> Write request number mismatch!!\n"
                      << "Reference write times = " << write_ref.size()
                      << " || current write times = " << write_vect.size()<<std::endl;
            is_okay = false;

        } else{
            for(int i = 0 ; i < write_ref.size() ; i++){
                if(!(write_ref[i] == write_vect[i])){
                    LOG(LOG_WARNING) << "[WARNING] MONITOR >> WRITE MISMATCH OCCURED\n"
                              << "ref loop addr : " <<std::hex<< write_ref[i].addr << " | current loop addr : "<<std::hex << write_vect[i].addr <<std::endl
                              << "ref loop data : " <<std::hex << write_ref[i].data << " | current loop data : "<<std::hex << write_vect[i].data <<std::endl;
                    is_okay = false;
                    
                }
            }
        }
        if(!is_okay){
            LOG(LOG_INFO) << "\n[INFO] GTX STACK RECOVERY REGSITER SETTING RECOMMENDED\n";
            LOG(LOG_INFO) << "[INFO] [stack_info] pointer : 0x" << std::hex << low_stack << " || size : 0x" << std::hex << (high_stack - low_stack) << std::endl;
            LOG(LOG_INFO) << "[INFO] [stack_recovery] stack recovery enable bit should be enable! \n\n" ;
        }
        write_vect.clear();
        return is_okay;
    }

    

    void CPUMemintf::reset_vector(){
        write_vect.clear();
        write_ref.clear();
    }
    
}