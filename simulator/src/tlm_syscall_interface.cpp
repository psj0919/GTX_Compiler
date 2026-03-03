

#include "tlm_syscall_interface.h"
#include <iostream>

namespace riscv_tlm {

TLMSyscallInterface::TLMSyscallInterface() 
    : pk_socket("pk_socket"),
      default_delay(10, sc_core::SC_NS)
{
    // Initialize TLM transaction
    trans.set_command(tlm::TLM_READ_COMMAND);
    trans.set_data_ptr(reinterpret_cast<unsigned char*>(syscall_data));
    trans.set_data_length(sizeof(syscall_data));
    trans.set_streaming_width(sizeof(syscall_data));
    trans.set_byte_enable_ptr(nullptr);
    trans.set_dmi_allowed(false);
    trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
}

long TLMSyscallInterface::execute_syscall(long syscall_num, long arg0, long arg1, long arg2, 
                                         long arg3, long arg4, long arg5, long arg6) {
    // Prepare syscall data
    syscall_data[0] = static_cast<uint64_t>(syscall_num);
    syscall_data[1] = static_cast<uint64_t>(arg0);
    syscall_data[2] = static_cast<uint64_t>(arg1);
    syscall_data[3] = static_cast<uint64_t>(arg2);
    syscall_data[4] = static_cast<uint64_t>(arg3);
    syscall_data[5] = static_cast<uint64_t>(arg4);
    syscall_data[6] = static_cast<uint64_t>(arg5);
    syscall_data[7] = static_cast<uint64_t>(arg6);
    
    // Reset transaction status
    trans.set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
    
    // Execute TLM transaction
    sc_core::sc_time delay = default_delay;
    pk_socket->b_transport(trans, delay);
    
    // Check response
    if (trans.is_response_error()) {
        std::cerr << "[TLM-SYSCALL] Error in system call transaction" << std::endl;
        return -1;
    }
    
    // Return result (stored in syscall_data[0] by proxy kernel)
    return static_cast<long>(syscall_data[0]);
}

} // namespace riscv_tlm
