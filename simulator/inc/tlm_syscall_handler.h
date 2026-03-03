
#ifndef TLM_SYSCALL_HANDLER_H
#define TLM_SYSCALL_HANDLER_H

#include <stdint.h>

namespace riscv_tlm {

// Forward declarations
template<typename T> class BASE_ISA;
template<typename T> class Registers;
class TLMSyscallInterface;

template<typename T>
class TLMSyscallHandler {
public:
    // Handle ECALL instruction for system calls
    static bool handle_ecall(BASE_ISA<T>* isa, Registers<T>* regs, TLMSyscallInterface* syscall_interface);
    
    // Handle EBREAK instruction for debugging
    static bool handle_ebreak(BASE_ISA<T>* isa, Registers<T>* regs);

private:
    // Extract system call arguments from registers
    static void extract_syscall_args(Registers<T>* regs, uint64_t& syscall_num, 
                                    uint64_t args[6]);
    
    // Store system call result in register
    static void store_syscall_result(Registers<T>* regs, int64_t result);
};

} // namespace riscv_tlm



#endif // TLM_SYSCALL_HANDLER_H
