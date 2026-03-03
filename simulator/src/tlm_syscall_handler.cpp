/*!
 \file tlm_syscall_handler.tpp
 \brief System call handler template implementation
 \author TLM Proxy Kernel Team
 \date 2025
 */
// SPDX-License-Identifier: GPL-3.0-or-later

#include "tlm_syscall_interface.h"
#include "tlm_syscall_handler.h"
#include "BASE_ISA.h"
#include "Registers.h"
#include <iostream>

namespace riscv_tlm {

template<typename T>
bool TLMSyscallHandler<T>::handle_ecall(BASE_ISA<T>* isa, Registers<T>* regs, TLMSyscallInterface* syscall_interface) {
    if (!syscall_interface) {
        std::cerr << "[TLM-SYSCALL] No syscall interface available" << std::endl;
        return false;
    }
    
    // Extract system call arguments from registers
    uint64_t syscall_num;
    uint64_t args[6];
    extract_syscall_args(regs, syscall_num, args);
    
    // Execute system call through TLM interface
    int64_t result = syscall_interface->execute_syscall(
        static_cast<long>(syscall_num),
        static_cast<long>(args[0]),
        static_cast<long>(args[1]),
        static_cast<long>(args[2]),
        static_cast<long>(args[3]),
        static_cast<long>(args[4]),
        static_cast<long>(args[5]),
        0  // arg6 not used in most syscalls
    );
    
    // Store result in register a0
    store_syscall_result(regs, result);
    
    return true;
}

template<typename T>
bool TLMSyscallHandler<T>::handle_ebreak(BASE_ISA<T>* isa, Registers<T>* regs) {
    std::cout << "[TLM-SYSCALL] EBREAK instruction encountered" << std::endl;
    
    // For debugging purposes, print register state
    std::cout << "[TLM-SYSCALL] PC: 0x" << std::hex << regs->getPC() << std::dec << std::endl;
    
    // Could trigger debugger or breakpoint handling
    return true;
}

template<typename T>
void TLMSyscallHandler<T>::extract_syscall_args(Registers<T>* regs, uint64_t& syscall_num, uint64_t args[6]) {
    // RISC-V calling convention:
    // a7 (x17) = system call number
    // a0-a5 (x10-x15) = arguments
    
    syscall_num = static_cast<uint64_t>(regs->getValue(17)); // a7
    args[0] = static_cast<uint64_t>(regs->getValue(10));     // a0
    args[1] = static_cast<uint64_t>(regs->getValue(11));     // a1
    args[2] = static_cast<uint64_t>(regs->getValue(12));     // a2
    args[3] = static_cast<uint64_t>(regs->getValue(13));     // a3
    args[4] = static_cast<uint64_t>(regs->getValue(14));     // a4
    args[5] = static_cast<uint64_t>(regs->getValue(15));     // a5
}

template<typename T>
void TLMSyscallHandler<T>::store_syscall_result(Registers<T>* regs, int64_t result) {
    // Store result in a0 (x10)
    regs->setValue(10, static_cast<T>(result));
    
    // For negative results (errors), also set a1 to indicate error
    if (result < 0) {
        regs->setValue(11, static_cast<T>(-1)); // a1 = -1 indicates error
    } else {
        regs->setValue(11, static_cast<T>(0));  // a1 = 0 indicates success
    }
}

} // namespace riscv_tlm
