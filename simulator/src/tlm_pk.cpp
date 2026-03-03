/*!
 \file tlm_pk.cpp
 \brief TLM Proxy Kernel Implementation
 \author TLM Proxy Kernel Team
 \date 2025
 */
// SPDX-License-Identifier: GPL-3.0-or-later

#include "tlm_pk.h"
#include "MemoryInterface.h"
#include "config.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <fstream>
#include <filesystem>
#include <vector>
// Additional system headers for real network communication
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>

namespace riscv_tlm {

SC_HAS_PROCESS(TLMProxyKernel);

TLMProxyKernel::TLMProxyKernel(sc_core::sc_module_name name, uint64_t heap_start) 
    : sc_module(name),
      syscall_socket("syscall_socket"),
      mem_intf(nullptr),
      heap_ptr(heap_start),  // Default heap start
      heap_max(heap_start + 0x10000000),  // Default heap limit
      default_delay(10, sc_core::SC_NS),
      next_pid_counter(2000 + (rand() % 8000)),  // Random start PID between 2000-9999
      current_process_pid(1000)
{
    // Register TLM transport method
    syscall_socket.register_b_transport(this, &TLMProxyKernel::b_transport);
    
    // Initialize file descriptor table
    init_fd_table();
    init_pci_proxy();
    init_process_table();
    
    // Initialize default command line arguments
    argv_storage.clear();
    argv_storage.push_back("riscv_program");  // Default program name
    argc_value = 1;
    cmdline_buffer = "riscv_program";
    
    // Create initial process (the main program)
    current_process_pid = create_process("riscv_program", 1);  // Parent is init (PID 1)
    
    //LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] TLM Proxy Kernel initialized" << std::endl;
}

void TLMProxyKernel::b_transport(tlm::tlm_generic_payload& trans, sc_core::sc_time& delay) {
    // Handle TLM transaction for system calls
    handle_syscall(trans, delay);
}

void TLMProxyKernel::handle_syscall(tlm::tlm_generic_payload& trans, sc_core::sc_time& delay) {
    // Extract system call parameters from TLM payload
    uint64_t* data = reinterpret_cast<uint64_t*>(trans.get_data_ptr());
    
    long syscall_num = static_cast<long>(data[0]);
    long arg0 = static_cast<long>(data[1]);
    long arg1 = static_cast<long>(data[2]);
    long arg2 = static_cast<long>(data[3]);
    long arg3 = static_cast<long>(data[4]);
    long arg4 = static_cast<long>(data[5]);
    long arg5 = static_cast<long>(data[6]);
    
    long result = 0;
    
    print_syscall_info(syscall_num, arg0, arg1, arg2);
    
    // Handle system call
    switch (syscall_num) {
        case SYS_exit:
            result = sys_exit(arg0);
            break;
            
        case SYS_exit_group:
            result = sys_exit_group(arg0);
            break;
            
        case SYS_write:
            result = sys_write(arg0, reinterpret_cast<const char*>(arg1), arg2);
            break;
            
        case SYS_read:
            result = sys_read(arg0, reinterpret_cast<char*>(arg1), arg2);
            break;
        case SYS_open_file:    
        case SYS_open:
            result = sys_open(reinterpret_cast<const char*>(arg0), arg1, arg2);
            break;
            
        case SYS_close:
            result = sys_close(arg0);
            break;
            
        case SYS_lseek:
            result = sys_lseek(arg0, arg1, arg2);
            break;
            
        case SYS_brk:
            result = sys_brk(arg0);
            break;
            
        case SYS_mmap:
            result = sys_mmap(arg0, arg1, arg2, arg3, arg4, arg5);
            break;
            
        case SYS_munmap:
            result = sys_munmap(arg0, arg1);
            break;
            
        case SYS_mprotect:
            result = sys_mprotect(arg0, arg1, arg2);
            break;
            
        case SYS_gettimeofday:
            result = sys_gettimeofday(reinterpret_cast<void*>(arg0), reinterpret_cast<void*>(arg1));
            break;
            
        case SYS_clock_gettime:
            result = sys_clock_gettime(arg0, reinterpret_cast<void*>(arg1));
            break;
            
        case SYS_get_argc:
            result = sys_get_argc();
            break;
            
        case SYS_get_argv:
            result = sys_get_argv(reinterpret_cast<char*>(arg0), arg1);
            break;
            
        case SYS_set_cmdline:
            result = sys_set_cmdline(reinterpret_cast<const char*>(arg0));
            break;
            
        case SYS_list_processes:
            result = sys_list_processes(reinterpret_cast<void*>(arg0), arg1);
            break;
            
        case SYS_stat:
            result = sys_stat(reinterpret_cast<const char*>(arg0), reinterpret_cast<void*>(arg1));
            break;
            
        case SYS_fstat:
            result = sys_fstat(arg0, reinterpret_cast<void*>(arg1));
            break;
            
        case SYS_unlink:
            result = sys_unlink(reinterpret_cast<const char*>(arg0));
            break;
            
        case SYS_getpid:
            result = sys_getpid();
            break;
            
        case SYS_getppid:
            result = sys_getppid();
            break;
            
        // Socket system calls for PCI communication proxy
        case SYS_socket:
            result = sys_socket(arg0, arg1, arg2);
            break;
            
        case SYS_bind:
            result = sys_bind(arg0, reinterpret_cast<const void*>(arg1), arg2);
            break;
            
        case SYS_listen:
            result = sys_listen(arg0, arg1);
            break;
            
        case SYS_accept:
            result = sys_accept(arg0, reinterpret_cast<void*>(arg1), reinterpret_cast<void*>(arg2));
            break;
            
        case SYS_connect:
            result = sys_connect(arg0, reinterpret_cast<const void*>(arg1), arg2);
            break;
            
        case SYS_send:
            result = sys_send(arg0, reinterpret_cast<const void*>(arg1), arg2, arg3);
            break;
            
        case SYS_recv:
            result = sys_recv(arg0, reinterpret_cast<void*>(arg1), arg2, arg3);
            break;
            
        case SYS_sendto:
            result = sys_sendto(arg0, reinterpret_cast<const void*>(arg1), arg2, arg3, 
                               reinterpret_cast<const void*>(arg4), arg5);
            break;
            
        case SYS_recvfrom:
            result = sys_recvfrom(arg0, reinterpret_cast<void*>(arg1), arg2, arg3,
                                 reinterpret_cast<void*>(arg4), reinterpret_cast<void*>(arg5));
            break;
            
        case SYS_shutdown:
            result = sys_shutdown(arg0, arg1);
            break;
            
        case SYS_setsockopt:
            result = sys_setsockopt(arg0, arg1, arg2, reinterpret_cast<const void*>(arg3), arg4);
            break;
            
        case SYS_getsockopt:
            result = sys_getsockopt(arg0, arg1, arg2, reinterpret_cast<void*>(arg3), reinterpret_cast<void*>(arg4));
            break;
        case SYS_fcntl:
            result = sys_fcntl(arg0, arg1, arg2);
            break;
        case SYS_tlm_select:
            result = sys_tlm_select(arg0, reinterpret_cast<void*>(arg1), reinterpret_cast<void*>(arg2), reinterpret_cast<void*>(arg3), reinterpret_cast<void*>(arg4));
            break;
        case SYS_tlm_poll:
            result = sys_tlm_poll(reinterpret_cast<void*>(arg0), arg1, arg2);
            break;
        case SYS_epoll_create:
            result = sys_epoll_create(arg0);
            break;
        case SYS_epoll_ctl:
            result = sys_epoll_ctl(arg0, arg1, arg2, reinterpret_cast<void*>(arg3));
            break;
        case SYS_epoll_wait:
            result = sys_epoll_wait(arg0, reinterpret_cast<void*>(arg1), arg2, arg3);
            break;
            
        default:
            LOG(LOG_WARNING) << "[DEBUG][TLM-PK] Warning unimplemented system call " << syscall_num << std::endl;
    
            
            result = -ENOSYS;
            break;
    }
    
    // Store result back in TLM payload
    data[0] = static_cast<uint64_t>(result);
    
    // Set TLM response
    trans.set_response_status(tlm::TLM_OK_RESPONSE);
    delay += default_delay;
}

long TLMProxyKernel::sys_exit(long code) {
    LOG(LOG_INFO) << "\n=============================================\n";
    LOG(LOG_INFO) << "[INFO][TLM-PK] Process PID=" << current_process_pid << " exiting with code: " << code << std::endl;
    LOG(LOG_INFO) << "=============================================\n\n";
    exitcode = code;
    // Clean up current process from process table
    cleanup_process(current_process_pid);
    
    // In real implementation, this would terminate the simulation
    // For now, we'll just return the exit code
    sc_core::sc_stop();
    return code;
}

long TLMProxyKernel::sys_write(long fd, const char* buf, long count) {
    if (!is_valid_fd(fd)) {
        return -EBADF;
    }
    
    if (fd == STDOUT_FILENO || fd == STDERR_FILENO) {
        // Simulate writing to stdout/stderr
        // 1) Prefer simulator memory read
        // 2) Fallback: if mem_intf absent or read failed, treat buf as host pointer (unit test convenience)
        std::vector<char> buffer(count + 1);  // +1 for null terminator
        bool ok = false;
        if (mem_intf && read_memory(reinterpret_cast<uint64_t>(buf), buffer.data(), count)) {
            ok = true;
        } else if (!mem_intf) {
            // Fallback path: assume direct host pointer
            std::memcpy(buffer.data(), buf, count);
            ok = true;
        }
        if (ok) {
            buffer[count] = '\0';
            std::cout.write(buffer.data(), count);
            std::cout.flush();
            return count;
        } else {
            LOG(LOG_ERROR) << "[DEBUG][TLM-PK] Error Failed to obtain write buffer (addr=0x" 
                          << std::hex << reinterpret_cast<uint64_t>(buf) << std::dec << ")" << std::endl;

            return -EFAULT;  // Bad address
        }
    }
    
    // For regular files, write to actual file system
    if (fd >= 3 && fd < MAX_FDS && fd_table[fd].is_open && !fd_table[fd].is_socket) {
        LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Writing " << count << " bytes to file '" << fd_table[fd].filename << "' (fd " << fd << ")" << std::endl;

        // Read data from simulator memory
        std::vector<char> buffer(count);
        if (read_memory(reinterpret_cast<uint64_t>(buf), buffer.data(), count)) {
            try {
                // Open file for writing at current position
                std::fstream file(fd_table[fd].filename, std::ios::in | std::ios::out | std::ios::binary);
                
                // If file doesn't exist, create it
                if (!file.is_open()) {
                    file.open(fd_table[fd].filename, std::ios::out | std::ios::binary);
                    file.close();
                    file.open(fd_table[fd].filename, std::ios::in | std::ios::out | std::ios::binary);
                }
                
                if (file.is_open()) {
                    // Seek to current position or EOF if O_APPEND
                    const int O_APPEND_FLAG = 0x400;
                    if ((fd_table[fd].open_flags & O_APPEND_FLAG) != 0) {
                        file.seekp(0, std::ios::end);
                        fd_table[fd].position = static_cast<uint64_t>(file.tellp());
                    } else {
                        file.seekp(fd_table[fd].position);
                    }
                    
                    // Write data to file
                    file.write(buffer.data(), count);
                    file.flush();
                    
                    if (file.good()) {
                        fd_table[fd].position += count;
                        file.close();

                        LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Successfully wrote " << count << " bytes to file '" 
                                      << fd_table[fd].filename << "'" << std::endl;

                        return count;
                    } else {
                        
                        LOG(LOG_ERROR) <<"[DEBUG][TLM-PK] Error writing to file '" << fd_table[fd].filename << "'" << std::endl;

                        file.close();
                        return -EIO;
                    }
                } else {
                        LOG(LOG_ERROR) << "[DEBUG][TLM-PK] Error opening file '" << fd_table[fd].filename << "' for writing" << std::endl;

                    return -EACCES;
                }
            } catch (const std::exception& e) {

                LOG(LOG_ERROR) << "[DEBUG][TLM-PK] Error Exception writing to file '" << fd_table[fd].filename << "': " << e.what() << std::endl;
                return -EIO;
            }
        } else {
            return -EFAULT;
        }
    }
    
    // For other file descriptors, simulate file writing
    LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Writing " << count << " bytes to fd " << fd << std::endl;

    return count;
}

long TLMProxyKernel::sys_read(long fd, char* buf, long count) {
    if (!is_valid_fd(fd)) {
        return -EBADF;
    }
    
    if (fd == STDIN_FILENO) {
        // Simulate reading from stdin
        LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Reading from stdin (simulated)" << std::endl;
        // For simulation, return empty read
        return 0;
    }
    
    // For regular files, read from actual file system
    if (fd >= 3 && fd < MAX_FDS && fd_table[fd].is_open && !fd_table[fd].is_socket) {
        LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Reading " << count << " bytes from file '" << fd_table[fd].filename << "' (fd " << fd << ")" << std::endl;
        try {
            // Open file for reading
            std::ifstream file(fd_table[fd].filename, std::ios::binary);
            
            if (file.is_open()) {
                // Seek to current position
                file.seekg(fd_table[fd].position);
                
                // Read data from file
                std::vector<char> buffer(count);
                file.read(buffer.data(), count);
                
                // Get actual bytes read
                long bytes_read = file.gcount();
                
                if (bytes_read > 0) {
                    // Write data to simulator memory
                    if (write_memory(reinterpret_cast<uint64_t>(buf), buffer.data(), bytes_read)) {
                        fd_table[fd].position += bytes_read;
                        file.close();
                        LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Successfully read " << bytes_read << " bytes from file '" 
                                      << fd_table[fd].filename << "'" << std::endl;

                        return bytes_read;
                    } else {
                        file.close();
                        return -EFAULT;
                    }
                } else {
                    // EOF or error
                    file.close();
                    if (file.eof()) {
                        LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] EOF reached for file '" << fd_table[fd].filename << "'" << std::endl;

                        return 0;  // EOF
                    } else {
                        LOG(LOG_ERROR) << "[DEBUG][TLM-PK] Error : reading from file '" << fd_table[fd].filename << "'" << std::endl;
                        return -EIO;
                    }
                }
            } else {
                LOG(LOG_ERROR) << "[DEBUG][TLM-PK] Error : opening file '" << fd_table[fd].filename << "' for reading" << std::endl;
                return -EACCES;
            }
        } catch (const std::exception& e) {
            LOG(LOG_ERROR) << "[DEBUG][TLM-PK] Error : Exception reading from file '" << fd_table[fd].filename << "': " << e.what() << std::endl;
            return -EIO;
        }
    }
    
    // For other file descriptors, simulate file reading
    LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Reading " << count << " bytes from fd " << fd << std::endl;
    return 0;  // Simulate empty file
}

long TLMProxyKernel::sys_open(const char* filename, long flags, long mode) {
    if (!filename) {
        return -EFAULT;
    }
    
    // Read filename from memory using memory interface
    std::string filename_str;
    if (mem_intf) {
        uint64_t filename_addr = reinterpret_cast<uint64_t>(filename);
        
        // Read string until null terminator (max 256 chars for safety)
        for (int i = 0; i < 256; i++) {
            uint8_t ch = mem_intf->readDataMem(filename_addr + i, 1);
            if (ch == 0) break;
            filename_str += static_cast<char>(ch);
        }
    } else {
        LOG(LOG_WARNING) << "[DEBUG][TLM-PK] Warning: No memory interface for open filename reading" << std::endl;

        return -1;
    }
    
    // Find available file descriptor
    for (int i = 3; i < MAX_FDS; i++) {  // Start from 3 (after stdin, stdout, stderr)
        if (!fd_table[i].is_open) {
            fd_table[i].is_open = true;
            fd_table[i].position = 0;
            fd_table[i].filename = filename_str;
            fd_table[i].open_flags = static_cast<int>(flags);
            fd_table[i].is_socket = false;
            fd_table[i].file_data.clear();  // Clear in-memory cache (not used for real files)
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Opened file '" << filename_str << "' with fd " << i << std::endl;

            // Check if file exists for reading operations
            std::ifstream test_file(filename_str);
            if (test_file.is_open()) {
                test_file.close();
                LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] File '" << filename_str << "' exists and is accessible" << std::endl;
                
                // If O_TRUNC requested, truncate immediately
                const int O_TRUNC_FLAG = 0x200; // matches tests/proxy usage 0x241 includes O_TRUNC
                if ((fd_table[i].open_flags & O_TRUNC_FLAG) != 0) {
                    LOG(LOG_DEBUG) <<"[DEBUG][TLM-PK] Truncating file '" << filename_str << "' due to O_TRUNC" << std::endl;
                    std::ofstream trunc_file(filename_str, std::ios::out | std::ios::trunc | std::ios::binary);
                    // truncation occurs upon open; file closed at end of scope
                    fd_table[i].position = 0;
                }
            } else {
                LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] File '" << filename_str << "' does not exist or not accessible - will be created on write" << std::endl;

                // If O_CREAT specified, create empty file now so that subsequent writes append from start
                const int O_CREAT_FLAG = 0x40; // 0100 octal -> 0x40
                if ((fd_table[i].open_flags & O_CREAT_FLAG) != 0) {
                    std::ofstream create_file(filename_str, std::ios::out | std::ios::binary);
                    // created empty file
                }
            }
            
            return i;
        }
    }
    
    return -EMFILE;  // Too many open files
}

long TLMProxyKernel::sys_close(long fd) {
    if (!is_valid_fd(fd) || fd < 3) {  // Don't close stdin/stdout/stderr
        return -EBADF;
    }
    
    if (fd_table[fd].is_open) {
        LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Closing file '" << fd_table[fd].filename << "' (fd " << fd << ")" << std::endl;

        
        // AF_UNIX specific cleanup (deferred earlier modification here)
        if (fd_table[fd].is_unix) {
            if (fd_table[fd].is_unix_server) {
                if (!fd_table[fd].unix_path.empty()) unix_socket_registry.erase(fd_table[fd].unix_path);
                unix_pending_connections.erase(fd);
                if (fd_table[fd].host_sockfd>=0) {
                    ::close(fd_table[fd].host_sockfd);
                    ::unlink(fd_table[fd].unix_path.c_str());
                }
            } else if (fd_table[fd].is_real_socket && fd_table[fd].host_sockfd>=0) {
                ::close(fd_table[fd].host_sockfd);
            }
            int peer = fd_table[fd].peer_fd;
            if (peer>=3 && peer<MAX_FDS && fd_table[peer].peer_fd==fd) fd_table[peer].peer_fd=-1;
            fd_table[fd].unix_msg_queue.clear();
        }
        if (fd_table[fd].is_real_socket && fd_table[fd].host_sockfd>=0 && !fd_table[fd].is_unix) {
            ::close(fd_table[fd].host_sockfd);
        }
        // Clean up file descriptor entry
        fd_table[fd].is_open = false;
        fd_table[fd].filename.clear();
        fd_table[fd].file_data.clear();  // Clear in-memory cache
        fd_table[fd].position = 0;
        fd_table[fd].is_socket = false;
        
        return 0;
    }
    
    return -EBADF;
}

long TLMProxyKernel::sys_brk(long addr) {
    if (addr == 0) {
        // Return current heap pointer
        return static_cast<long>(heap_ptr);
    }
    
    if (addr < heap_ptr) {
        // Shrinking heap
        heap_ptr = static_cast<uint64_t>(addr);
        LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Heap shrunk from 0x" << std::hex << heap_ptr 
                      << " to 0x" << addr << std::dec << std::endl;
        return static_cast<long>(heap_ptr);
    }
    
    if (addr > heap_max) {
        // Out of memory
        return -ENOMEM;
    }
    
    // Growing heap
    uint64_t old_heap = heap_ptr;
    heap_ptr = static_cast<uint64_t>(addr);

    LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Growing heap from 0x" << std::hex << old_heap 
                  << " to 0x" << heap_ptr << std::dec << std::endl;
    
    return static_cast<long>(heap_ptr);
}

long TLMProxyKernel::sys_mmap(long addr, long length, long prot, long flags, long fd, long offset) {
    // Simplified mmap implementation
    uint64_t allocated_addr = allocate_memory(static_cast<uint64_t>(length));
    
    if (allocated_addr == 0) {
        return -ENOMEM;
    }

    LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] mmap allocated 0x" << std::hex << allocated_addr 
              << " size 0x" << length << std::dec << std::endl;

    
    return static_cast<long>(allocated_addr);
}

long TLMProxyKernel::sys_munmap(long addr, long length) {
    bool success = deallocate_memory(static_cast<uint64_t>(addr), static_cast<uint64_t>(length));
    
    if (success) {
        LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] munmap freed 0x" << std::hex << addr 
                  << " size 0x" << length << std::dec << std::endl;
        return 0;
    }
    
    return -EINVAL;
}

long TLMProxyKernel::sys_gettimeofday(void* tv, void* tz) {
    // Simulate getting time of day
    if (tv != nullptr) {
        struct timeval {
            long tv_sec;
            long tv_usec;
        };
        
        // Get simulation time
        sc_core::sc_time current_time = sc_core::sc_time_stamp();
        uint64_t time_us = current_time.to_double() * 1000000;  // Convert to microseconds
        
        timeval time_val;
        time_val.tv_sec = time_us / 1000000;
        time_val.tv_usec = time_us % 1000000;
        
        // Write to user space using memory interface
        if (mem_intf) {
            uint64_t tv_addr = reinterpret_cast<uint64_t>(tv);
            uint8_t* time_bytes = reinterpret_cast<uint8_t*>(&time_val);
            
            for (size_t i = 0; i < sizeof(timeval); i++) {
                mem_intf->writeDataMem(tv_addr + i, time_bytes[i], 1);
            }
        } else {
            LOG(LOG_WARNING) << "[DEBUG][TLM-PK] Warning: No memory interface for gettimeofday" << std::endl;
            return -1;
        }
    }
    
    return 0;
}

void TLMProxyKernel::print_syscall_info(long syscall_num, long arg0, long arg1, long arg2) {

    switch (syscall_num) {
        case SYS_exit:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - exit\n";
            break;
        case SYS_exit_group:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - exit_group\n";
            break;
        case SYS_write:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - write\n";
            break;
        case SYS_read:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - read\n";
            break;
        case SYS_open_file:
        case SYS_open:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - open\n";
            break;
        case SYS_close:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - close\n";
            break;
        case SYS_lseek:   
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - lseek\n";
            break;
        case SYS_brk:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - brk\n";
            break;
        case SYS_mmap:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - mmap\n";
            break;
        case SYS_munmap:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - munmap\n";
            break;
        case SYS_mprotect:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - mprotect\n";
            break;
        case SYS_gettimeofday:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - gettimeofday\n";
            break;
        case SYS_clock_gettime:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - clock_gettime\n";
            break;
        case SYS_get_argc:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - get_argc\n";
            break;
        case SYS_get_argv:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - get_argv\n";
            break;
        case SYS_set_cmdline:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - set_cmdline\n";
            break;
        case SYS_stat:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - stat\n";
            break;
        case SYS_fstat:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - fstat\n";
            break;
        case SYS_unlink:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - unlink\n";
            break;
        case SYS_getpid:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - getpid\n";
            break;
        case SYS_getppid:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - getppid\n";
            break;
        default:
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] syscall - unknown_syscall\n";
            break;
    }
}

bool TLMProxyKernel::is_valid_fd(long fd) {
    return (fd >= 0 && fd < MAX_FDS);
}

void TLMProxyKernel::init_fd_table() {
    for (int i = 0; i < MAX_FDS; i++) {
        fd_table[i].is_open = false;
        fd_table[i].position = 0;
        fd_table[i].filename.clear();
        fd_table[i].is_socket = false;
        fd_table[i].socket_type = 0;
        fd_table[i].socket_domain = 0;
        fd_table[i].is_listening = false;
        fd_table[i].bound_port = 0;
        fd_table[i].peer_address.clear();
        // Initialize new fields for real socket support
        fd_table[i].is_real_socket = false;
        fd_table[i].host_sockfd = -1;
    }
    
    // Initialize standard file descriptors
    fd_table[STDIN_FILENO].is_open = true;
    fd_table[STDIN_FILENO].filename = "stdin";
    
    fd_table[STDOUT_FILENO].is_open = true;
    fd_table[STDOUT_FILENO].filename = "stdout";
    
    fd_table[STDERR_FILENO].is_open = true;
    fd_table[STDERR_FILENO].filename = "stderr";
}

uint64_t TLMProxyKernel::allocate_memory(uint64_t size) {
    // Simple memory allocator
    uint64_t aligned_size = (size + 0xFFF) & ~0xFFF;  // Page align
    
    if (heap_ptr + aligned_size > heap_max) {
        return 0;  // Out of memory
    }
    
    uint64_t allocated_addr = heap_ptr;
    heap_ptr += aligned_size;
    
    // Record allocation
    memory_region region;
    region.start = allocated_addr;
    region.size = aligned_size;
    region.allocated = true;
    memory_map.push_back(region);
    
    return allocated_addr;
}

bool TLMProxyKernel::deallocate_memory(uint64_t addr, uint64_t size) {
    // Find and mark memory as deallocated
    for (auto& region : memory_map) {
        if (region.start == addr && region.size == size && region.allocated) {
            region.allocated = false;
            return true;
        }
    }
    
    return false;
}

void TLMProxyKernel::set_memory_interface(MemoryInterface* intf) {
    mem_intf = intf;
}

bool TLMProxyKernel::read_memory(uint64_t addr, void* data, size_t size) {
    if (!data || size == 0) return false;
    // If no memory interface (unit test mode), treat addr as host pointer
    if (!mem_intf) {
        std::memcpy(data, reinterpret_cast<const void*>(addr), size);
        return true;
    }
    try {
        uint8_t* byte_data = reinterpret_cast<uint8_t*>(data);
        for (size_t i = 0; i < size; i++) byte_data[i] = mem_intf->readDataMem(addr + i, 1);
        return true;
    } catch (...) {
        LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Memory read failed at address 0x" << std::hex << addr << std::dec << std::endl;
        return false;
    }
}

bool TLMProxyKernel::write_memory(uint64_t addr, const void* data, size_t size) {
    if (!data || size == 0) return false;
    if (!mem_intf) {
        std::memcpy(reinterpret_cast<void*>(addr), data, size);
        return true;
    }
    try {
        const uint8_t* byte_data = reinterpret_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; i++) mem_intf->writeDataMem(addr + i, byte_data[i], 1);
        return true;
    } catch (...) {
        LOG(LOG_DEBUG) <<"[DEBUG][TLM-PK] Memory write failed at address 0x" << std::hex << addr << std::dec << std::endl;
        return false;
    }
}

// Command line argument system calls implementation
long TLMProxyKernel::sys_get_argc() {
    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] sys_get_argc() = " << argc_value << std::endl;
    return argc_value;
}

long TLMProxyKernel::sys_list_processes(void* buf, long buf_size) {
    if (!buf || buf_size <= 0) {
        return -EINVAL;
    }
    
    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] sys_list_processes() called with buf_size=" << buf_size << std::endl;

    
    // Build process list string in format: "PID PPID COMMAND\n"
    std::string process_data;
    process_data += "PID\tPPID\tCOMMAND\n";
    
    for (int i = 0; i < MAX_PROCESSES; i++) {
        if (process_table[i].active) {
            process_data += std::to_string(process_table[i].pid);
            process_data += "\t";
            process_data += std::to_string(process_table[i].ppid);
            process_data += "\t";
            process_data += process_table[i].command;
            process_data += "\n";
        }
    }
    
    if (process_data.size() >= static_cast<size_t>(buf_size)) {
        // Buffer too small, return required size
        return static_cast<long>(process_data.size());
    }
    
    // Copy data to user buffer using memory interface
    if (mem_intf) {
        uint64_t buf_addr = reinterpret_cast<uint64_t>(buf);
        for (size_t i = 0; i < process_data.size(); i++) {
            mem_intf->writeDataMem(buf_addr + i, static_cast<uint8_t>(process_data[i]), 1);
        }
        // Add null terminator
        if (process_data.size() < static_cast<size_t>(buf_size)) {
            mem_intf->writeDataMem(buf_addr + process_data.size(), 0, 1);
        }
        
        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Listed " << process_data.size() << " bytes of process data" << std::endl;
    } else {
        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Warning: No memory interface available for process list copy" << std::endl;
        return -1;
    }
    
    return static_cast<long>(process_data.size());
}

long TLMProxyKernel::sys_get_argv(char* buf, long buf_size) {
    if (!buf || buf_size <= 0) {
        return -EINVAL;
    }
    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] sys_get_argv() called with buf_size=" << buf_size << std::endl;
    // Build argv string in format: "arg0\0arg1\0arg2\0...\0"
    std::string argv_data;
    for (const auto& arg : argv_storage) {
        argv_data += arg;
        argv_data += '\0';
    }
    
    if (argv_data.size() >= static_cast<size_t>(buf_size)) {
        // Buffer too small, return required size
        return static_cast<long>(argv_data.size());
    }
    
    // Copy data to user buffer using memory interface
    if (mem_intf) {
        uint64_t buf_addr = reinterpret_cast<uint64_t>(buf);
        for (size_t i = 0; i < argv_data.size(); i++) {
            mem_intf->writeDataMem(buf_addr + i, static_cast<uint8_t>(argv_data[i]), 1);
        }
        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Copied " << argv_data.size() << " bytes of argv data via memory interface" << std::endl;
    } else {
        LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Warning: No memory interface available for argv copy" << std::endl;
        return -1;
    }
    
    return static_cast<long>(argv_data.size());
}

long TLMProxyKernel::sys_set_cmdline(const char* cmdline) {
    if (!cmdline) {
        return -EINVAL;
    }
    
    // Read command line string from user memory using memory interface
    std::string cmdline_str;
    if (mem_intf) {
        uint64_t cmdline_addr = reinterpret_cast<uint64_t>(cmdline);
        
        // Read string until null terminator (max 1024 chars for safety)
        for (int i = 0; i < 1024; i++) {
            uint8_t ch = mem_intf->readDataMem(cmdline_addr + i, 1);
            if (ch == 0) break;
            cmdline_str += static_cast<char>(ch);
        }
    } else {
        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Warning: No memory interface available for cmdline read" << std::endl;
        return -1;
    }

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] sys_set_cmdline(\"" << cmdline_str << "\")" << std::endl;

    // Clear existing arguments
    argv_storage.clear();
    cmdline_buffer = cmdline_str;
    
    // Parse command line into arguments
    std::string current_arg;
    bool in_quotes = false;
    
    for (size_t i = 0; i < cmdline_str.length(); i++) {
        char c = cmdline_str[i];
        
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ' ' && !in_quotes) {
            if (!current_arg.empty()) {
                argv_storage.push_back(current_arg);
                current_arg.clear();
            }
        } else {
            current_arg += c;
        }
    }
    
    // Add final argument if any
    if (!current_arg.empty()) {
        argv_storage.push_back(current_arg);
    }
    
    // If no arguments parsed, add default program name
    if (argv_storage.empty()) {
        argv_storage.push_back("riscv_program");
    }
    
    argc_value = static_cast<int>(argv_storage.size());

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Parsed " << argc_value << " arguments:" << std::endl;

    for (int i = 0; i < argc_value; i++) {
        LOG(LOG_DEBUG) << "[DEBUG][TLM-PK]   argv[" << i << "] = \"" << argv_storage[i] << "\"" << std::endl;
    }
    
    return argc_value;
}

// New system call implementations

long TLMProxyKernel::sys_exit_group(long code) {
    LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Process group exit with code: " << code << std::endl;
    // exit_group is similar to exit but terminates all threads in the process group
    // For simulation, we'll treat it the same as exit
    sc_core::sc_stop();
    return code;
}

long TLMProxyKernel::sys_lseek(long fd, long offset, long whence) {
    if (!is_valid_fd(fd) || !fd_table[fd].is_open) {
        return -EBADF;
    }
    
    // Standard file descriptors don't support seeking
    if (fd <= STDERR_FILENO) {
        return -ESPIPE;  // Illegal seek
    }
    
    long new_position = 0;
    
    switch (whence) {
        case SEEK_SET:
            new_position = offset;
            break;
        case SEEK_CUR:
            new_position = static_cast<long>(fd_table[fd].position) + offset;
            break;
        case SEEK_END:
            // For simulation, assume file size is 0 for simplicity
            new_position = 0 + offset;
            break;
        default:
            return -EINVAL;
    }
    
    if (new_position < 0) {
        return -EINVAL;
    }
    
    fd_table[fd].position = static_cast<uint64_t>(new_position);
    LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] lseek fd=" << fd << " to position " << new_position 
              << " (whence=" << whence << ")" << std::endl;
    return new_position;
}

long TLMProxyKernel::sys_mprotect(long addr, long length, long prot) {
    // Memory protection flags (from mman.h)
    // PROT_READ = 1, PROT_WRITE = 2, PROT_EXEC = 4
    
    if (addr == 0 || length <= 0) {
        return -EINVAL;
    }
    
    // Align address to page boundary
    uint64_t aligned_addr = static_cast<uint64_t>(addr) & ~0xFFF;
    uint64_t aligned_length = (static_cast<uint64_t>(length) + 0xFFF) & ~0xFFF;
    
    // Check if the memory region is valid (in our memory map)
    bool found = false;
    for (const auto& region : memory_map) {
        if (region.allocated && 
            aligned_addr >= region.start && 
            aligned_addr + aligned_length <= region.start + region.size) {
            found = true;
            break;
        }
    }
    
    if (!found) {
        return -ENOMEM;  // Invalid memory region
    }

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] mprotect addr=0x" << std::hex << aligned_addr 
              << " length=0x" << aligned_length << " prot=" << std::dec << prot << std::endl;

    // For simulation, we'll just accept the protection change
    return 0;
}

long TLMProxyKernel::sys_clock_gettime(long clk_id, void* tp) {
    if (!tp) {
        return -EFAULT;
    }
    
    // timespec structure: { time_t tv_sec; long tv_nsec; }
    struct timespec {
        long tv_sec;
        long tv_nsec;
    };
    
    timespec time_data = {};
    
    // Get simulation time in nanoseconds
    sc_core::sc_time current_time = sc_core::sc_time_stamp();
    uint64_t time_ns = static_cast<uint64_t>(current_time.to_double() * 1e9);
    
    switch (clk_id) {
        case CLOCK_REALTIME:
        case CLOCK_MONOTONIC:
            time_data.tv_sec = time_ns / 1000000000ULL;
            time_data.tv_nsec = time_ns % 1000000000ULL;
            break;
            
        case CLOCK_PROCESS_CPUTIME_ID:
        case CLOCK_THREAD_CPUTIME_ID:
            // For simulation, use the same time as real/monotonic
            time_data.tv_sec = time_ns / 1000000000ULL;
            time_data.tv_nsec = time_ns % 1000000000ULL;
            break;
            
        default:
            return -EINVAL;
    }
    
    // Write timespec data to user buffer using memory interface
    if (mem_intf) {
        uint64_t tp_addr = reinterpret_cast<uint64_t>(tp);
        uint8_t* time_bytes = reinterpret_cast<uint8_t*>(&time_data);
        
        for (size_t i = 0; i < sizeof(time_data); i++) {
            mem_intf->writeDataMem(tp_addr + i, time_bytes[i], 1);
        }
    } else {

        LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Warning: No memory interface for clock_gettime" << std::endl;

        return -1;
    }

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] clock_gettime clk_id=" << clk_id 
              << " sec=" << time_data.tv_sec 
              << " nsec=" << time_data.tv_nsec << std::endl;

    return 0;
}

// Additional file system and process management calls

long TLMProxyKernel::sys_stat(const char* pathname, void* statbuf) {
    if (!pathname || !statbuf) {
        return -EFAULT;
    }
    
    // Read pathname from memory
    std::string path_str;
    if (mem_intf) {
        uint64_t path_addr = reinterpret_cast<uint64_t>(pathname);
        
        // Read string until null terminator (max 256 chars for safety)
        for (int i = 0; i < 256; i++) {
            uint8_t ch = mem_intf->readDataMem(path_addr + i, 1);
            if (ch == 0) break;
            path_str += static_cast<char>(ch);
        }
    } else {

        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Warning: No memory interface for stat path reading" << std::endl;

        return -1;
    }
    
    // Simple stat structure simulation (fields from Linux stat)
    struct stat_sim {
        uint64_t st_dev;       // Device ID
        uint64_t st_ino;       // Inode number
        uint32_t st_mode;      // File type and mode
        uint32_t st_nlink;     // Number of hard links
        uint32_t st_uid;       // User ID
        uint32_t st_gid;       // Group ID
        uint64_t st_rdev;      // Device ID (if special file)
        uint64_t st_size;      // Total size in bytes
        uint64_t st_blksize;   // Block size for filesystem I/O
        uint64_t st_blocks;    // Number of 512B blocks allocated
        uint64_t tlm_st_atime; // Time of last access
        uint64_t tlm_st_mtime; // Time of last modification
        uint64_t tlm_st_ctime; // Time of last status change
    };
    
    stat_sim stat_data = {};
    stat_data.st_dev = 1;
    stat_data.st_ino = 1000;  // Simulate inode
    stat_data.st_mode = 0x8000 | 0644;  // Regular file, rw-r--r--
    stat_data.st_nlink = 1;
    stat_data.st_uid = 1000;
    stat_data.st_gid = 1000;
    stat_data.st_size = 0;  // Empty file simulation
    stat_data.st_blksize = 4096;
    stat_data.st_blocks = 0;
    
    // Get current simulation time
    sc_core::sc_time current_time = sc_core::sc_time_stamp();
    uint64_t time_sec = static_cast<uint64_t>(current_time.to_double());
    stat_data.tlm_st_atime = time_sec;
    stat_data.tlm_st_mtime = time_sec;
    stat_data.tlm_st_ctime = time_sec;
    
    // Write stat data to user buffer using memory interface
    if (mem_intf) {
        uint64_t buf_addr = reinterpret_cast<uint64_t>(statbuf);
        uint8_t* stat_bytes = reinterpret_cast<uint8_t*>(&stat_data);
        
        for (size_t i = 0; i < sizeof(stat_data); i++) {
            mem_intf->writeDataMem(buf_addr + i, stat_bytes[i], 1);
        }
    }

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] stat(\"" << path_str << "\") simulated" << std::endl;

    return 0;
}

// ============================================================================
// PCI Communication Proxy - Socket System Calls Implementation
// ============================================================================

void TLMProxyKernel::init_pci_proxy() {
    // Initialize PCI proxy connections
    for (int i = 0; i < MAX_PCI_CONNECTIONS; i++) {
        pci_connections[i].active = false;
        pci_connections[i].proxy_fd = -1;
        pci_connections[i].target_fd = -1;
        pci_connections[i].target_device = "";
        pci_connections[i].pci_address = 0;
    }
    
    //LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] PCI Communication Proxy initialized" << std::endl;

}

int TLMProxyKernel::allocate_socket_fd() {
    for (int i = 3; i < MAX_FDS; i++) {  // Skip stdin, stdout, stderr
        if (!fd_table[i].is_open) {
            fd_table[i].is_open = true;
            fd_table[i].is_socket = true;
            fd_table[i].position = 0;
            fd_table[i].filename = "";
            fd_table[i].socket_type = 0;
            fd_table[i].socket_domain = 0;
            fd_table[i].is_listening = false;
            fd_table[i].bound_port = 0;
            fd_table[i].peer_address = "";
            fd_table[i].is_real_socket = false;
            fd_table[i].host_sockfd = -1;
            fd_table[i].is_unix = false;
            fd_table[i].is_unix_server = false;
            fd_table[i].unix_path.clear();
            fd_table[i].peer_fd = -1;
            fd_table[i].unix_msg_queue.clear();
            fd_table[i].non_blocking = false;
            if (!fd_table[i].conn_event) fd_table[i].conn_event = new sc_core::sc_event();
            if (!fd_table[i].data_event) fd_table[i].data_event = new sc_core::sc_event();
            return i;
        }
    }
    return -1;  // No free descriptors
}

// Sanitize AF_UNIX path (basic security: prevent traversal, enforce namespace & length)
std::string TLMProxyKernel::sanitize_unix_path(const std::string& raw, int& errcode) {
    errcode = 0;
    if (raw.empty()) { errcode = -EINVAL; return {}; }
    if (raw.find("..") != std::string::npos) { errcode = -EACCES; return {}; }
    std::string p = raw;
    if (p[0] != '/') p = std::string("/tmp/tlm_pk_") + p; // sandbox namespace
    if (p.size() > 100) { errcode = -ENAMETOOLONG; return {}; }
    for (char c: p) {
        if (!(std::isalnum((unsigned char)c) || c=='/' || c=='_' || c=='-' || c=='.')) { errcode = -EINVAL; return {}; }
    }
    return p;
}

bool TLMProxyKernel::is_socket_fd(int fd) {
    if (!is_valid_fd(fd)) return false;
    return fd_table[fd].is_socket;
}

void TLMProxyKernel::setup_pci_proxy_connection(int proxy_fd, const std::string& target_device, uint32_t pci_addr) {
    for (int i = 0; i < MAX_PCI_CONNECTIONS; i++) {
        if (!pci_connections[i].active) {
            pci_connections[i].active = true;
            pci_connections[i].proxy_fd = proxy_fd;
            pci_connections[i].target_device = target_device;
            pci_connections[i].pci_address = pci_addr;
 
            LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] PCI proxy connection setup: fd=" << proxy_fd 
                      << " device=" << target_device << " addr=0x" << std::hex << pci_addr << std::dec << std::endl;

            break;
        }
    }
}

long TLMProxyKernel::sys_socket(long domain, long type, long protocol) {

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] socket(domain=" << domain << ", type=" << type << ", protocol=" << protocol << ")" << std::endl;

    // Allocate new socket file descriptor
    int sockfd = allocate_socket_fd();
    if (sockfd < 0) {
        return -EMFILE;  // Too many open files
    }
    
    // Configure socket
    fd_table[sockfd].socket_domain = domain;
    fd_table[sockfd].socket_type = type;
    
    if (domain == AF_UNIX && (type == SOCK_STREAM || type == SOCK_DGRAM)) {
        fd_table[sockfd].is_unix = true;

            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Created AF_UNIX " << (type==SOCK_STREAM?"STREAM":"DGRAM") << " socket fd=" << sockfd << std::endl;

        return sockfd;
    }
    // For PCI proxy, we primarily support AF_INET sockets
    if (domain == AF_INET && (type == SOCK_STREAM || type == SOCK_DGRAM)) {

        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Created AF_INET " << (type==SOCK_STREAM?"STREAM":"DGRAM") << " socket fd=" << sockfd << std::endl;

        return sockfd;
    }
    fd_table[sockfd].is_open = false;
    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Unsupported socket type domain="<<domain<<" type="<<type<< std::endl;

    return -EAFNOSUPPORT;
}

long TLMProxyKernel::sys_bind(long sockfd, const void* addr, long addrlen) {

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] bind(sockfd=" << sockfd << ", addrlen=" << addrlen << ")" << std::endl;

    if (!is_socket_fd(sockfd)) {
        return -EBADF;
    }
    // AF_UNIX path
    if (fd_table[sockfd].is_unix) {
        if (!addr) return -EFAULT;
        uint16_t fam=0;
        if (mem_intf) {
            uint64_t base = reinterpret_cast<uint64_t>(addr);
            fam = mem_intf->readDataMem(base,1) | (mem_intf->readDataMem(base+1,1)<<8);
        } else {
            const struct sockaddr_un* su = reinterpret_cast<const struct sockaddr_un*>(addr);
            fam = su->sun_family;
        }
        if (fam != AF_UNIX) return -EINVAL;
        const int MAX_PATH=108; std::string raw;
        if (mem_intf) {
            uint64_t base = reinterpret_cast<uint64_t>(addr);
            for (int i=0;i<MAX_PATH && (2+i)<addrlen;i++) { char c=(char)mem_intf->readDataMem(base+2+i,1); if(c==0) break; raw.push_back(c);}        
        } else {
            const struct sockaddr_un* su = reinterpret_cast<const struct sockaddr_un*>(addr);
            raw = su->sun_path; // truncated at null
        }
        int ec=0; std::string path = sanitize_unix_path(raw, ec); if (ec) return ec;
        if (unix_socket_registry.count(path)) return -EADDRINUSE;
        fd_table[sockfd].unix_path = path;
        unix_socket_registry[path] = sockfd;
        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] AF_UNIX bound path '"<<path<<"' fd="<<sockfd<< std::endl;
        return 0;
    }
    // AF_INET
    if (mem_intf && addr && addrlen >= sizeof(sockaddr_in_tlm)) {
        sockaddr_in_tlm sock_addr;
        uint8_t* addr_bytes = reinterpret_cast<uint8_t*>(&sock_addr);
        uint64_t addr_ptr = reinterpret_cast<uint64_t>(addr);
        
        for (size_t i = 0; i < sizeof(sockaddr_in_tlm); i++) {
            addr_bytes[i] = mem_intf->readDataMem(addr_ptr + i, 1);
        }
        
        // Convert from network byte order
        uint16_t port = ((sock_addr.sin_port & 0xFF) << 8) | ((sock_addr.sin_port >> 8) & 0xFF);
        
        // For PCI proxy, bind to base port + offset
        if (port == 0) {
            port = PCI_PROXY_BASE_PORT + (sockfd - 3);  // Auto-assign port
        }
        
        fd_table[sockfd].bound_port = port;   
        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Socket bound to port " << port << " for PCI proxy" << std::endl;
        return 0;
    }
    
    return -EFAULT;
}

long TLMProxyKernel::sys_listen(long sockfd, long backlog) {
    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] listen(sockfd=" << sockfd << ", backlog=" << backlog << ")" << std::endl;
    if (!is_socket_fd(sockfd)) {
        return -EBADF;
    }
    
    if (fd_table[sockfd].socket_type != SOCK_STREAM) {
        return -EOPNOTSUPP;
    }
    
    fd_table[sockfd].is_listening = true;
    if (fd_table[sockfd].is_unix) {
        fd_table[sockfd].is_unix_server = true;
        unix_pending_connections[sockfd];
        // Create real host unix domain socket and bind if not yet
        if (fd_table[sockfd].host_sockfd < 0) {
            int h = ::socket(AF_UNIX, SOCK_STREAM, 0);
            if (h >= 0) {
                struct sockaddr_un su; memset(&su,0,sizeof(su)); su.sun_family=AF_UNIX;
                std::snprintf(su.sun_path,sizeof(su.sun_path),"%s", fd_table[sockfd].unix_path.c_str());
                ::unlink(su.sun_path); // ensure free
                if (::bind(h,(struct sockaddr*)&su,sizeof(su))==0 && ::listen(h, (int)backlog)>=0) {
                    fd_table[sockfd].host_sockfd = h;
                    fd_table[sockfd].is_real_socket = true; // reuse field
                } else {
                    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Host unix listen failed: "<<strerror(errno)<< std::endl;
                    ::close(h);
                }
            }
        }
    }

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Socket " << sockfd << (fd_table[sockfd].is_unix?" (AF_UNIX)":"") << " listening" << std::endl;

    return 0;
}

long TLMProxyKernel::sys_accept(long sockfd, void* addr, void* addrlen) {

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] accept(sockfd=" << sockfd << ")" << std::endl;

    if (!is_socket_fd(sockfd) || !fd_table[sockfd].is_listening) {
        return -EBADF;
    }
    
    if (fd_table[sockfd].is_unix_server) {
        // First try pending simulated queue
        auto &q = unix_pending_connections[sockfd];
        if (!q.empty()) {
            int client_fd = q.front(); q.erase(q.begin());
            int accepted_fd = allocate_socket_fd();
            if (accepted_fd < 0) return -EMFILE;
            fd_table[accepted_fd].socket_domain=AF_UNIX; fd_table[accepted_fd].socket_type=SOCK_STREAM;
            fd_table[accepted_fd].is_unix=true; fd_table[accepted_fd].is_unix_server=false;
            fd_table[accepted_fd].unix_path = fd_table[sockfd].unix_path;
            fd_table[accepted_fd].peer_fd = client_fd; fd_table[client_fd].peer_fd = accepted_fd;
            LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Accepted simulated AF_UNIX queued fd="<<accepted_fd<<" client="<<client_fd<< std::endl;
            return accepted_fd;
        }
        // Try real host accept if host socket active
        if (fd_table[sockfd].host_sockfd >=0) {
            int h = ::accept(fd_table[sockfd].host_sockfd,nullptr,nullptr);
            if (h>=0) {
                int accepted_fd = allocate_socket_fd();
                if (accepted_fd<0) { ::close(h); return -EMFILE; }
                fd_table[accepted_fd].socket_domain=AF_UNIX; fd_table[accepted_fd].socket_type=SOCK_STREAM;
                fd_table[accepted_fd].is_unix=true; fd_table[accepted_fd].unix_path=fd_table[sockfd].unix_path;
                fd_table[accepted_fd].is_real_socket=true; fd_table[accepted_fd].host_sockfd=h;
                LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Accepted real host AF_UNIX connection fd="<<accepted_fd<< std::endl;
                return accepted_fd;
            }
            if (fd_table[sockfd].non_blocking) return -EAGAIN; // immediate
            // blocking: spin-wait minimal simulation time slice until something arrives (simplified)
            wait(sc_core::sc_time(10, sc_core::SC_US));
            return -EAGAIN;
        }
        return fd_table[sockfd].non_blocking ? -EAGAIN : -EAGAIN; // placeholder (would wait event)
    }
    // Original PCI path
    int connfd = allocate_socket_fd();
    if (connfd < 0) return -EMFILE;
    fd_table[connfd].socket_domain = fd_table[sockfd].socket_domain;
    fd_table[connfd].socket_type = fd_table[sockfd].socket_type;
    fd_table[connfd].peer_address = "pci_device";
    std::string target_device = "pci_device_" + std::to_string(connfd);
    uint32_t pci_addr = 0x1000 + (connfd << 8);
    setup_pci_proxy_connection(connfd, target_device, pci_addr);
    LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Accepted PCI proxy connection, new fd="<<connfd<< std::endl;
    return connfd;
}

long TLMProxyKernel::sys_connect(long sockfd, const void* addr, long addrlen) {
    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] connect(sockfd=" << sockfd << ", addrlen=" << addrlen << ")" << std::endl;

    if (!is_socket_fd(sockfd)) {
        return -EBADF;
    }
    
    if (fd_table[sockfd].is_unix) {
        if (!addr) return -EFAULT;
        uint16_t fam=0;
        if (mem_intf) {
            uint64_t base = reinterpret_cast<uint64_t>(addr);
            fam = mem_intf->readDataMem(base,1) | (mem_intf->readDataMem(base+1,1)<<8);
        } else {
            fam = reinterpret_cast<const struct sockaddr_un*>(addr)->sun_family;
        }

        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK][DBG] AF_UNIX connect: raw_family=" << fam << " expected=" << AF_UNIX << " addrlen=" << addrlen << std::endl;

        if (fam != AF_UNIX) return -EINVAL;
        std::string path; const int MAX_PATH=108;
        if (mem_intf) {
            uint64_t base = reinterpret_cast<uint64_t>(addr);
            for (int i=0;i<MAX_PATH && (2+i)<addrlen;i++) { char c=(char)mem_intf->readDataMem(base+2+i,1); if(c==0)break; path.push_back(c);}        
        } else {
            path = reinterpret_cast<const struct sockaddr_un*>(addr)->sun_path;
        }
        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK][DBG] AF_UNIX connect path='" << path << "'" << std::endl;

        auto it = unix_socket_registry.find(path);
        if (it==unix_socket_registry.end()) {
            // Try real host path (connect to existing file)
            int h = ::socket(AF_UNIX, SOCK_STREAM,0);
            if (h<0) return -ECONNREFUSED;
            struct sockaddr_un su; memset(&su,0,sizeof(su)); su.sun_family=AF_UNIX; std::snprintf(su.sun_path,sizeof(su.sun_path),"%s", path.c_str());
            if (::connect(h,(struct sockaddr*)&su,sizeof(su))<0) { ::close(h); return fd_table[sockfd].non_blocking ? -EAGAIN : -ECONNREFUSED; }
            fd_table[sockfd].host_sockfd = h; fd_table[sockfd].is_real_socket=true; fd_table[sockfd].unix_path=path; return 0;
        }
        int server_fd = it->second;
        if (!fd_table[server_fd].is_unix_server) return -ECONNREFUSED;
        unix_pending_connections[server_fd].push_back(sockfd);
        fd_table[sockfd].unix_path = path;
        return 0;
    }
    // Read target address
    if (mem_intf && addr && addrlen >= sizeof(sockaddr_in_tlm)) {
        sockaddr_in_tlm sock_addr;
        uint8_t* addr_bytes = reinterpret_cast<uint8_t*>(&sock_addr);
        uint64_t addr_ptr = reinterpret_cast<uint64_t>(addr);
        
        for (size_t i = 0; i < sizeof(sockaddr_in_tlm); i++) {
            addr_bytes[i] = mem_intf->readDataMem(addr_ptr + i, 1);
        }
        
        uint16_t port = ((sock_addr.sin_port & 0xFF) << 8) | ((sock_addr.sin_port >> 8) & 0xFF);
        uint32_t ip_addr = sock_addr.sin_addr;
        
        // Convert IP address to string
        std::string ip_str = std::to_string((ip_addr >> 24) & 0xFF) + "." +
                            std::to_string((ip_addr >> 16) & 0xFF) + "." +
                            std::to_string((ip_addr >> 8) & 0xFF) + "." +
                            std::to_string(ip_addr & 0xFF);
        
         LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Attempting real socket connection to " << ip_str << ":" << port << std::endl;

        
        // Create real socket connection for specific ports (like 8080)
        if (port == 8080 && ip_addr == 0x7F000001) { // localhost:8080
            // Create actual TCP socket
            int host_sockfd = socket(AF_INET, SOCK_STREAM, 0);
            if (host_sockfd < 0) {
 
                LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Failed to create host socket: " << strerror(errno) << std::endl;

                return -ECONNREFUSED;
            }
            
            // Set up host address
            struct sockaddr_in host_addr;
            memset(&host_addr, 0, sizeof(host_addr));
            host_addr.sin_family = AF_INET;
            host_addr.sin_port = htons(port);
            host_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            
            // Connect to real host
            int connect_result = connect(host_sockfd, (struct sockaddr*)&host_addr, sizeof(host_addr));
            if (connect_result < 0) {
                close(host_sockfd);

                LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Failed to connect to host server: " << strerror(errno) << std::endl;

                return -ECONNREFUSED;
            }
            
            // Store host socket fd
            fd_table[sockfd].host_sockfd = host_sockfd;
            fd_table[sockfd].peer_address = "real_host_" + ip_str + ":" + std::to_string(port);
            fd_table[sockfd].is_real_socket = true;
            
            LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Successfully connected to real host server at " << ip_str << ":" << port << std::endl;
            return 0;
        } else {
            // For other addresses, use virtual PCI simulation
            std::string target_device = "pci_target_" + std::to_string(port);
            uint32_t pci_addr = 0x2000 + port;
            
            setup_pci_proxy_connection(sockfd, target_device, pci_addr);
            fd_table[sockfd].peer_address = target_device;
            fd_table[sockfd].is_real_socket = false;

            LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Connected to PCI device via proxy, port=" << port << std::endl;

            return 0;
        }
    }
    
    return -EFAULT;
}

long TLMProxyKernel::sys_send(long sockfd, const void* buf, long len, long flags) {

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] send(sockfd=" << sockfd << ", len=" << len << ", flags=" << flags << ")" << std::endl;

    if (!is_socket_fd(sockfd)) {
        return -EBADF;
    }
    
    // AF_UNIX real or simulated
    if (fd_table[sockfd].is_unix) {
        if (fd_table[sockfd].is_real_socket && fd_table[sockfd].host_sockfd>=0) {
            std::vector<uint8_t> data(len); uint64_t p=reinterpret_cast<uint64_t>(buf);
            if (mem_intf && buf) {
                for (long i=0;i<len;i++) data[i]=mem_intf->readDataMem(p+i,1);
            } else if (buf) {
                std::memcpy(data.data(), buf, len);
            }
            int fl = flags;
            if (fd_table[sockfd].non_blocking) fl |= MSG_DONTWAIT;
            ssize_t s = ::send(fd_table[sockfd].host_sockfd,data.data(),len,fl);
            if (s<0) return (errno==EWOULDBLOCK || errno==EAGAIN)? -EAGAIN : -errno; return s;
        }
        // simulated path
        if (fd_table[sockfd].socket_type==SOCK_STREAM) {
            int peer = fd_table[sockfd].peer_fd; if (peer<0) return -ENOTCONN;
            std::string payload; payload.resize(len); uint64_t p=reinterpret_cast<uint64_t>(buf);
            if (mem_intf && buf) {
                for (long i=0;i<len;i++) payload[i]=(char)mem_intf->readDataMem(p+i,1);
            } else if (buf) {
                std::memcpy(payload.data(), buf, len);
            }
            fd_table[peer].unix_msg_queue.push_back(std::move(payload)); return len;
        } else { // DGRAM
            // require peer set (connect semantics) else error
            int peer = fd_table[sockfd].peer_fd; if (peer<0) return -ENOTCONN;
            std::string payload; payload.resize(len); uint64_t p=reinterpret_cast<uint64_t>(buf);
            if (mem_intf && buf) {
                for (long i=0;i<len;i++) payload[i]=(char)mem_intf->readDataMem(p+i,1);
            } else if (buf) {
                std::memcpy(payload.data(), buf, len);
            }
            fd_table[peer].unix_msg_queue.push_back(std::move(payload)); return len;
        }
    }
    // Check if this is a real socket connection
    if (fd_table[sockfd].is_real_socket && fd_table[sockfd].host_sockfd >= 0) {
        // Read data from RISC-V memory
        std::vector<uint8_t> send_data(len);
        if (mem_intf && buf) {
            uint64_t buf_ptr = reinterpret_cast<uint64_t>(buf);
            for (long i = 0; i < len; i++) send_data[i] = mem_intf->readDataMem(buf_ptr + i, 1);
        } else if (buf) {
            std::memcpy(send_data.data(), buf, len);
        }
        
        // Send to real host socket
        ssize_t sent_bytes = send(fd_table[sockfd].host_sockfd, send_data.data(), len, flags);
        if (sent_bytes < 0) {

            LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Real socket send failed: " << strerror(errno) << std::endl;

            return -errno;
        }
        

        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Sent " << sent_bytes << " bytes to real host socket" << std::endl;

        return sent_bytes;
    }
    
    // Fall back to PCI proxy simulation
    for (int i = 0; i < MAX_PCI_CONNECTIONS; i++) {
        if (pci_connections[i].active && pci_connections[i].proxy_fd == sockfd) {
            // Simulate sending data to PCI device
            LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Sending " << len << " bytes to PCI device " 
                      << pci_connections[i].target_device << " (addr=0x" << std::hex 
                      << pci_connections[i].pci_address << std::dec << ")" << std::endl;

            return len;
        }
    }

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] No PCI proxy connection found for fd=" << sockfd << std::endl;

    return -ENOTCONN;
}

long TLMProxyKernel::sys_recv(long sockfd, void* buf, long len, long flags) {

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] recv(sockfd=" << sockfd << ", len=" << len << ", flags=" << flags << ")" << std::endl;

    if (!is_socket_fd(sockfd)) {
        return -EBADF;
    }
    
    if (fd_table[sockfd].is_unix) {
        if (fd_table[sockfd].is_real_socket && fd_table[sockfd].host_sockfd>=0) {
            std::vector<uint8_t> data(len);
            int fl = flags;
            if (fd_table[sockfd].non_blocking) fl |= MSG_DONTWAIT;
            ssize_t r = ::recv(fd_table[sockfd].host_sockfd,data.data(),len,fl);
            if (r<0) return (errno==EWOULDBLOCK || errno==EAGAIN)? -EAGAIN : -errno;
            if (r>0 && buf) {
                if (mem_intf) { uint64_t p=reinterpret_cast<uint64_t>(buf); for (ssize_t i=0;i<r;i++) mem_intf->writeDataMem(p+i,data[i],1);} else { std::memcpy(buf,data.data(),r); }
            }
            return r;
        }
        if (fd_table[sockfd].unix_msg_queue.empty()) return fd_table[sockfd].non_blocking ? -EAGAIN : -EAGAIN;
        std::string payload = std::move(fd_table[sockfd].unix_msg_queue.front());
        fd_table[sockfd].unix_msg_queue.erase(fd_table[sockfd].unix_msg_queue.begin());
        long copy_len = std::min<long>(len, payload.size());
        if (buf) {
            if (mem_intf) { uint64_t p=reinterpret_cast<uint64_t>(buf); for (long i=0;i<copy_len;i++) mem_intf->writeDataMem(p+i,(uint8_t)payload[i],1);} else { std::memcpy(buf,payload.data(),copy_len); }
        }
        return copy_len;
    }
    // Check if this is a real socket connection
    if (fd_table[sockfd].is_real_socket && fd_table[sockfd].host_sockfd >= 0) {
        // Receive from real host socket
        std::vector<uint8_t> recv_data(len);
        ssize_t received_bytes = recv(fd_table[sockfd].host_sockfd, recv_data.data(), len, flags);
        
        if (received_bytes < 0) {

            LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Real socket recv failed: " << strerror(errno) << std::endl;

            return -errno;
        }
        
        // Write received data to RISC-V memory
        if (mem_intf && buf && received_bytes > 0) {
            uint64_t buf_ptr = reinterpret_cast<uint64_t>(buf);
            for (ssize_t i = 0; i < received_bytes; i++) {
                mem_intf->writeDataMem(buf_ptr + i, recv_data[i], 1);
            }
        }
        

        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Received " << received_bytes << " bytes from real host socket" << std::endl;

        return received_bytes;
    }
    
    // Fall back to PCI proxy simulation
    for (int i = 0; i < MAX_PCI_CONNECTIONS; i++) {
        if (pci_connections[i].active && pci_connections[i].proxy_fd == sockfd) {
            // Simulate receiving data from PCI device
            LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Receiving from PCI device " 
                      << pci_connections[i].target_device << " (addr=0x" << std::hex 
                      << pci_connections[i].pci_address << std::dec << ")" << std::endl;

            // Simulate received data (in real implementation, this would come from PCI device)
            if (mem_intf && buf && len > 0) {
                std::string sim_data = "PCI_DATA_" + std::to_string(i);
                size_t data_len = std::min((size_t)len, sim_data.length());
                
                uint64_t buf_addr = reinterpret_cast<uint64_t>(buf);
                for (size_t j = 0; j < data_len; j++) {
                    mem_intf->writeDataMem(buf_addr + j, sim_data[j], 1);
                }
                
                return data_len;
            }
            
            return 0;  // No data available
        }
    }
    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] No PCI proxy connection found for fd=" << sockfd << std::endl;

    return -ENOTCONN;
}

long TLMProxyKernel::sys_sendto(long sockfd, const void* buf, long len, long flags, const void* dest_addr, long addrlen) {

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] sendto(sockfd=" << sockfd << ", len=" << len << ")" << std::endl;

    if (!is_socket_fd(sockfd)) return -EBADF;
    // AF_UNIX datagram
    if (fd_table[sockfd].is_unix && fd_table[sockfd].socket_type==SOCK_DGRAM) {
        if (dest_addr && dest_addr!=nullptr && addrlen>2 && mem_intf) {
            uint64_t base = reinterpret_cast<uint64_t>(dest_addr);
            uint16_t fam = mem_intf->readDataMem(base,1) | (mem_intf->readDataMem(base+1,1)<<8);
            if (fam==AF_UNIX) {
                std::string path; const int MAX_PATH=108;
                for (int i=0;i<MAX_PATH && (2+i)<addrlen;i++){ char c=(char)mem_intf->readDataMem(base+2+i,1); if(c==0) break; path.push_back(c);}                
                auto it=unix_socket_registry.find(path);
                if (it!=unix_socket_registry.end()) {
                    int target_fd = it->second;
                    std::string payload; payload.resize(len); uint64_t p=reinterpret_cast<uint64_t>(buf);
                    if (mem_intf && buf) for (long i=0;i<len;i++) payload[i]=(char)mem_intf->readDataMem(p+i,1);
                    fd_table[target_fd].unix_msg_queue.push_back(std::move(payload));
                    return len;
                }
                return -ENOENT;
            }
        }
        // fallback to connected peer
        return sys_send(sockfd, buf, len, flags);
    }
    if (fd_table[sockfd].socket_type == SOCK_DGRAM) {
        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] UDP send to PCI device via proxy" << std::endl;
        return len;
    }
    return sys_send(sockfd, buf, len, flags);
}

long TLMProxyKernel::sys_recvfrom(long sockfd, void* buf, long len, long flags, void* src_addr, void* addrlen) {

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] recvfrom(sockfd=" << sockfd << ", len=" << len << ")" << std::endl;

    if (!is_socket_fd(sockfd)) return -EBADF;
    if (fd_table[sockfd].is_unix && fd_table[sockfd].socket_type==SOCK_DGRAM) {
        if (fd_table[sockfd].unix_msg_queue.empty()) return -EAGAIN;
        std::string payload = std::move(fd_table[sockfd].unix_msg_queue.front());
        fd_table[sockfd].unix_msg_queue.erase(fd_table[sockfd].unix_msg_queue.begin());
        long copy_len = std::min<long>(len, payload.size());
        if (mem_intf && buf) { uint64_t p=reinterpret_cast<uint64_t>(buf); for(long i=0;i<copy_len;i++) mem_intf->writeDataMem(p+i,(uint8_t)payload[i],1);}        
        return copy_len;
    }
    if (fd_table[sockfd].socket_type == SOCK_DGRAM) {
        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] UDP receive from PCI device via proxy" << std::endl;
        if (mem_intf && buf && len>0) {
            std::string sim_data="UDP_PCI_DATA"; size_t data_len=std::min((size_t)len, sim_data.length());
            uint64_t p=reinterpret_cast<uint64_t>(buf); for(size_t j=0;j<data_len;j++) mem_intf->writeDataMem(p+j, sim_data[j],1);
            return data_len;
        }
        return 0;
    }
    return sys_recv(sockfd, buf, len, flags);
}

long TLMProxyKernel::sys_shutdown(long sockfd, long how) {

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] shutdown(sockfd=" << sockfd << ", how=" << how << ")" << std::endl;

    if (!is_socket_fd(sockfd)) {
        return -EBADF;
    }
    
    // Clean up PCI proxy connection
    for (int i = 0; i < MAX_PCI_CONNECTIONS; i++) {
        if (pci_connections[i].active && pci_connections[i].proxy_fd == sockfd) {

            LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Shutting down PCI proxy connection for " 
                      << pci_connections[i].target_device << std::endl;

            pci_connections[i].active = false;
            break;
        }
    }
    
    return 0;
}

long TLMProxyKernel::sys_setsockopt(long sockfd, long level, long optname, const void* optval, long optlen) {

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] setsockopt(sockfd=" << sockfd << ", level=" << level 
              << ", optname=" << optname << ", optlen=" << optlen << ")" << std::endl;

    if (!is_socket_fd(sockfd)) {
        return -EBADF;
    }
    
    // For PCI proxy, we support basic socket options
    if (level == SOL_SOCKET) {
        switch (optname) {
            case SO_REUSEADDR:
 
                LOG(LOG_DEBUG) << "[DEBUG][TLM-PK] Set SO_REUSEADDR for PCI proxy socket" << std::endl;

                return 0;
            case SO_KEEPALIVE:
                LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Set SO_KEEPALIVE for PCI proxy socket" << std::endl;
                return 0;
            case SO_NONBLOCK:
                if (optval && optlen>=4 && mem_intf) {
                    uint64_t a=reinterpret_cast<uint64_t>(optval);
                    int v=0; for(int i=0;i<4;i++) v |= mem_intf->readDataMem(a+i,1) << (8*i);
                    fd_table[sockfd].non_blocking = (v!=0);
                    if (fd_table[sockfd].is_real_socket && fd_table[sockfd].host_sockfd>=0) {
                        int flags = ::fcntl(fd_table[sockfd].host_sockfd, F_GETFL, 0);
                        if (flags>=0) {
                            if (fd_table[sockfd].non_blocking) ::fcntl(fd_table[sockfd].host_sockfd, F_SETFL, flags | O_NONBLOCK);
                            else ::fcntl(fd_table[sockfd].host_sockfd, F_SETFL, flags & ~O_NONBLOCK);
                        }
                    }
                    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] SO_NONBLOCK set to "<<fd_table[sockfd].non_blocking<<" fd="<<sockfd<< std::endl;
                    return 0;
                }
                return -EFAULT;
            default:
                LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Unsupported socket option: " << optname << std::endl;
                return -ENOPROTOOPT;
        }
    }
    
    return -ENOPROTOOPT;
}

long TLMProxyKernel::sys_getsockopt(long sockfd, long level, long optname, void* optval, void* optlen) {

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] getsockopt(sockfd=" << sockfd << ", level=" << level 
              << ", optname=" << optname << ")" << std::endl;

    if (!is_socket_fd(sockfd)) {
        return -EBADF;
    }
    
    // Return default values for supported options
    if (level == SOL_SOCKET && mem_intf && optval && optlen) {
        uint64_t optlen_addr = reinterpret_cast<uint64_t>(optlen);
        uint32_t len = mem_intf->readDataMem(optlen_addr, 4);
        
        if (len >= 4) {
            uint64_t optval_addr = reinterpret_cast<uint64_t>(optval);
            uint32_t value = 0;
            switch(optname) {
                case SO_REUSEADDR: value = 1; break;
                case SO_KEEPALIVE: value = 1; break;
                case SO_NONBLOCK: value = fd_table[sockfd].non_blocking ? 1 : 0; break;
                default: return -ENOPROTOOPT;
            }
            
            for (int i = 0; i < 4; i++) {
                mem_intf->writeDataMem(optval_addr + i, (value >> (i * 8)) & 0xFF, 1);
            }
            
            return 0;
        }
    }
    
    return -EFAULT;
}

long TLMProxyKernel::sys_fstat(long fd, void* statbuf) {
    if (!is_valid_fd(fd) || !fd_table[fd].is_open || !statbuf) {
        return -EBADF;
    }
    
    // Use same logic as stat but for file descriptor
    return sys_stat(fd_table[fd].filename.c_str(), statbuf);
}

long TLMProxyKernel::sys_unlink(const char* pathname) {
    if (!pathname) {
        return -EFAULT;
    }
    
    // Read pathname from memory
    std::string path_str;
    if (mem_intf) {
        uint64_t path_addr = reinterpret_cast<uint64_t>(pathname);
        
        // Read string until null terminator (max 256 chars for safety)
        for (int i = 0; i < 256; i++) {
            uint8_t ch = mem_intf->readDataMem(path_addr + i, 1);
            if (ch == 0) break;
            path_str += static_cast<char>(ch);
        }
    } else {
        LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Warning: No memory interface for unlink path reading" << std::endl;

        return -1;
    }

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] unlink(\"" << path_str << "\") simulated - file deleted" << std::endl;

    // For simulation, always succeed
    return 0;
}

long TLMProxyKernel::sys_getpid() {
    // Return current process PID
     LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] getpid() = " << current_process_pid << std::endl;
    return current_process_pid;
}

long TLMProxyKernel::sys_getppid() {
    // Find current process and return its parent PID
    process_info* current_proc = find_process(current_process_pid);
    long ppid = (current_proc != nullptr) ? current_proc->ppid : 1;  // Default to init
    

    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] getppid() = " << ppid << " (parent of PID " << current_process_pid << ")" << std::endl;

    return ppid;
}

// External interface for setting program arguments from simulator
void TLMProxyKernel::set_program_arguments(int argc, char* argv[], const std::string& program_name) {
    argv_storage.clear();
    
    // First argument is always the program name
    argv_storage.push_back(program_name);
    
    // Add additional arguments from command line
    for (int i = 0; i < argc; i++) {
        if (argv[i]) {
            argv_storage.push_back(std::string(argv[i]));
        }
    }
    
    argc_value = static_cast<int>(argv_storage.size());
    
    // Build command line string
    cmdline_buffer.clear();
    for (size_t i = 0; i < argv_storage.size(); i++) {
        if (i > 0) cmdline_buffer += " ";
        
        // Add quotes if argument contains spaces
        if (argv_storage[i].find(' ') != std::string::npos) {
            cmdline_buffer += "\"" + argv_storage[i] + "\"";
        } else {
            cmdline_buffer += argv_storage[i];
        }
    }
    

        //LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Program arguments set from simulator:" << std::endl;
        //LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK]   Command line: " << cmdline_buffer << std::endl;
        //LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK]   argc = " << argc_value << std::endl;
        //for (int i = 0; i < argc_value; i++) {
        //    LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK]   argv[" << i << "] = \"" << argv_storage[i] << "\"" << std::endl;
        //}
}

// Process management implementation
void TLMProxyKernel::init_process_table() {
    for (int i = 0; i < MAX_PROCESSES; i++) {
        process_table[i].pid = 0;
        process_table[i].ppid = 0;
        process_table[i].start_time = 0;
        process_table[i].command = "";
        process_table[i].active = false;
    }
    
    // Create init process (PID 1)
    process_table[0].pid = 1;
    process_table[0].ppid = 0;  // init has no parent
    process_table[0].start_time = 0;
    process_table[0].command = "init";
    process_table[0].active = true;
    

    //LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Process table initialized with init process (PID 1)" << std::endl;

}

long TLMProxyKernel::create_process(const std::string& command, long parent_pid) {
    // Find free slot in process table
    for (int i = 0; i < MAX_PROCESSES; i++) {
        if (!process_table[i].active) {
            // Generate more random PID using current simulation time
            uint64_t current_time = static_cast<uint64_t>(sc_core::sc_time_stamp().to_double() * 1e9);
            long random_offset = (current_time % 500) + 1;  // Add 1-500 to base PID
            long new_pid = next_pid_counter + random_offset;
            next_pid_counter = new_pid + 1;  // Update counter for next process
            
            process_table[i].pid = new_pid;
            process_table[i].ppid = parent_pid;
            process_table[i].start_time = current_time;
            process_table[i].command = command;
            process_table[i].active = true;
            

            //LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Created process: PID=" << new_pid 
            //              << ", PPID=" << parent_pid 
            //              << ", Command=" << command 
            //              << ", StartTime=" << current_time << "ns" << std::endl;

            
            return new_pid;
        }
    }
    
    // No free slots
    //LOG(LOG_ERROR) <<  "[DEBUG][TLM-PK] Error: Process table full, cannot create new process" << std::endl;
    return -1;
}

void TLMProxyKernel::cleanup_process(long pid) {
    for (int i = 0; i < MAX_PROCESSES; i++) {
        if (process_table[i].active && process_table[i].pid == pid) {
            process_table[i].active = false;
            process_table[i].pid = 0;
            process_table[i].ppid = 0;
            process_table[i].command = "";
            

            LOG(LOG_DEBUG) <<  "[DEBUG][TLM-PK] Cleaned up process PID=" << pid << std::endl;

            break;
        }
    }
}

TLMProxyKernel::process_info* TLMProxyKernel::find_process(long pid) {
    for (int i = 0; i < MAX_PROCESSES; i++) {
        if (process_table[i].active && process_table[i].pid == pid) {
            return &process_table[i];
        }
    }
    return nullptr;
}

// =============================================================
// Readiness helpers (very simple heuristics for now)
// =============================================================
bool TLMProxyKernel::fd_read_ready(int fd) {
    if (!is_valid_fd(fd) || !fd_table[fd].is_open) return false;
    if (fd_table[fd].is_unix) {
        if (fd_table[fd].is_real_socket && fd_table[fd].host_sockfd>=0) {
            // poll host socket (non-blocking peek)
            uint8_t tmp; ssize_t r = ::recv(fd_table[fd].host_sockfd,&tmp,1,MSG_PEEK|MSG_DONTWAIT);
            return r>0;
        }
        return !fd_table[fd].unix_msg_queue.empty();
    }
    if (fd_table[fd].is_real_socket && fd_table[fd].host_sockfd>=0) {
        uint8_t tmp; ssize_t r = ::recv(fd_table[fd].host_sockfd,&tmp,1,MSG_PEEK|MSG_DONTWAIT); return r>0;
    }
    return false; // PCI path: could model differently
}

bool TLMProxyKernel::fd_write_ready(int fd) {
    if (!is_valid_fd(fd) || !fd_table[fd].is_open) return false;
    if (fd_table[fd].is_unix) {
        // Always writable for simulation (could limit queue length)
        return true;
    }
    if (fd_table[fd].is_real_socket && fd_table[fd].host_sockfd>=0) {
        // Assume writable (simplified); could use select/poll
        return true;
    }
    return true;
}

// =============================================================
// fcntl (subset)
// =============================================================
long TLMProxyKernel::sys_fcntl(long fd, long cmd, long arg) {
    if (!is_valid_fd(fd) || !fd_table[fd].is_open) return -EBADF;
    switch (cmd) {
        case F_GETFL:
            return fd_table[fd].non_blocking ? O_NONBLOCK : 0;
        case F_SETFL:
            if (arg & O_NONBLOCK) fd_table[fd].non_blocking = true; else fd_table[fd].non_blocking = false; return 0;
        default:
            return -EOPNOTSUPP;
    }
}

// =============================================================
// select (virtual)
// Memory layout: readfds / writefds / exceptfds treated as 64-bit mask (uint64_t)
// =============================================================
long TLMProxyKernel::sys_tlm_select(long nfds, void* readfds, void* writefds, void* exceptfds, void* timeout) {
    if (nfds > 64) nfds = 64; if (nfds < 0) return -EINVAL;
    uint64_t rmask=0, wmask=0, xmask=0;
    if (mem_intf) {
        if (readfds) { uint64_t a=reinterpret_cast<uint64_t>(readfds); for(int i=0;i<8;i++) rmask |= (uint64_t)mem_intf->readDataMem(a+i,1) << (i*8); }
        if (writefds){ uint64_t a=reinterpret_cast<uint64_t>(writefds); for(int i=0;i<8;i++) wmask |= (uint64_t)mem_intf->readDataMem(a+i,1) << (i*8); }
        if (exceptfds){ uint64_t a=reinterpret_cast<uint64_t>(exceptfds); for(int i=0;i<8;i++) xmask |= (uint64_t)mem_intf->readDataMem(a+i,1) << (i*8); }
    }
    uint64_t rready=0,wready=0,xready=0; int ready_count=0;
    for (int fd=0; fd<nfds; fd++) {
        uint64_t bit = 1ULL<<fd; if (rmask & bit) { if (fd_read_ready(fd)) { rready |= bit; ready_count++; } }
        if (wmask & bit) { if (fd_write_ready(fd)) { wready |= bit; ready_count++; } }
    }
    // Write back masks
    if (mem_intf) {
        auto wrmask=[&](uint64_t mask, void* ptr){ if(!ptr) return; uint64_t a=reinterpret_cast<uint64_t>(ptr); for(int i=0;i<8;i++) mem_intf->writeDataMem(a+i, (mask>>(i*8)) & 0xFF,1); };
        wrmask(rready, readfds); wrmask(wready, writefds); wrmask(xready, exceptfds);
    }
    return ready_count;
}

// =============================================================
// poll (virtual)
// struct pollfd { int fd; short events; short revents; } assumed packed 8 bytes
// =============================================================
long TLMProxyKernel::sys_tlm_poll(void* fds, long nfds, long timeout) {
    if (nfds<0 || nfds>64) return -EINVAL; if (!fds) return -EFAULT; if (!mem_intf) return -EFAULT;
    int ready=0; uint64_t base=reinterpret_cast<uint64_t>(fds);
    for (long i=0;i<nfds;i++) {
        uint64_t entry = base + i*8; // simplistic layout
        int fd=0; for(int b=0;b<4;b++) fd |= mem_intf->readDataMem(entry+b,1) << (8*b);
        int16_t events = mem_intf->readDataMem(entry+4,1) | (mem_intf->readDataMem(entry+5,1)<<8);
        int16_t revents = 0;
        if (events & POLLIN) { if (fd_read_ready(fd)) revents |= POLLIN; }
        if (events & POLLOUT){ if (fd_write_ready(fd)) revents |= POLLOUT; }
        if (revents) ready++;
        // write revents back
        mem_intf->writeDataMem(entry+6, revents & 0xFF,1); mem_intf->writeDataMem(entry+7, (revents>>8)&0xFF,1);
    }
    return ready;
}

// =============================================================
// epoll (virtual)
// struct epoll_event { uint32_t events; uint64_t data; }
// =============================================================
long TLMProxyKernel::sys_epoll_create(long size) {
    (void)size; int fd = allocate_socket_fd(); if (fd<0) return -EMFILE; fd_table[fd].is_epoll=true; return fd;
}

long TLMProxyKernel::sys_epoll_ctl(long epfd, long op, long fd, void* event) {
    if (!is_valid_fd(epfd) || !fd_table[epfd].is_open || !fd_table[epfd].is_epoll) return -EBADF;
    if (!is_valid_fd(fd) || !fd_table[fd].is_open) return -EBADF;
    uint32_t events=0; if (event && mem_intf) { uint64_t a=reinterpret_cast<uint64_t>(event); for(int i=0;i<4;i++) events |= (uint32_t)mem_intf->readDataMem(a+i,1) << (8*i); }
    auto &targets = fd_table[epfd].epoll_targets; auto &masks = fd_table[epfd].epoll_target_events;
    auto it = std::find(targets.begin(), targets.end(), fd);
    switch (op) {
        case EPOLL_CTL_ADD:
            if (it!=targets.end()) return -EEXIST; // need EEXIST constant? reuse EINVAL if not defined
            targets.push_back(fd); masks.push_back(events); return 0;
        case EPOLL_CTL_MOD:
            if (it==targets.end()) return -ENOENT; else { size_t idx=it-targets.begin(); masks[idx]=events; return 0; }
        case EPOLL_CTL_DEL:
            if (it==targets.end()) return -ENOENT; else { size_t idx=it-targets.begin(); targets.erase(it); masks.erase(masks.begin()+idx); return 0; }
        default:
            return -EINVAL;
    }
}

long TLMProxyKernel::sys_epoll_wait(long epfd, void* events, long maxevents, long timeout) {
    if (!is_valid_fd(epfd) || !fd_table[epfd].is_open || !fd_table[epfd].is_epoll) return -EBADF;
    if (maxevents<=0) return -EINVAL; if (!events) return -EFAULT; if (!mem_intf) return -EFAULT;
    auto &targets = fd_table[epfd].epoll_targets; auto &masks = fd_table[epfd].epoll_target_events;
    int emitted=0; uint64_t base=reinterpret_cast<uint64_t>(events);
    for (size_t i=0;i<targets.size() && emitted<maxevents;i++) {
        int tfd = targets[i]; uint32_t mask = masks[i]; uint32_t re=0;
        if (mask & EPOLLIN) { if (fd_read_ready(tfd)) re |= EPOLLIN; }
        if (mask & EPOLLOUT){ if (fd_write_ready(tfd)) re |= EPOLLOUT; }
        if (re) {
            uint64_t off = base + emitted * (4 + 8); // events(4)+data(8)
            for(int b=0;b<4;b++) mem_intf->writeDataMem(off+b, (re>>(8*b))&0xFF,1);
            // data: store fd as 64-bit
            uint64_t fdv = (uint64_t)tfd;
            for(int b=0;b<8;b++) mem_intf->writeDataMem(off+4+b, (fdv>>(8*b))&0xFF,1);
            emitted++;
        }
    }
    return emitted;
}

} // namespace riscv_tlm
