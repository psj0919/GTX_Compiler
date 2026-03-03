
#ifndef TLM_SYSCALL_INTERFACE_H
#define TLM_SYSCALL_INTERFACE_H

#include "systemc"
#include "tlm.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"
#include <stdint.h>

namespace riscv_tlm {

// System call interface for CPU to communicate with TLM Proxy Kernel
class TLMSyscallInterface {
public:
    TLMSyscallInterface();
    ~TLMSyscallInterface() = default;
    
    // TLM socket to connect to proxy kernel
    tlm_utils::simple_initiator_socket<TLMSyscallInterface> pk_socket;
    
    // System call execution method
    long execute_syscall(long syscall_num, long arg0, long arg1, long arg2, 
                        long arg3, long arg4, long arg5, long arg6);
    
private:
    sc_core::sc_time default_delay;
    tlm::tlm_generic_payload trans;
    uint64_t syscall_data[8];  // syscall_num + 7 arguments
};

// Helper function to be used in CPU instruction execution
inline long tlm_syscall(TLMSyscallInterface* interface, long syscall_num, 
                       long arg0, long arg1, long arg2, long arg3, long arg4, long arg5, long arg6) {
    if (interface) {
        return interface->execute_syscall(syscall_num, arg0, arg1, arg2, arg3, arg4, arg5, arg6);
    }
    return -38;  // ENOSYS
}

} // namespace riscv_tlm

#endif // TLM_SYSCALL_INTERFACE_H
