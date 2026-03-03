

#ifndef TLM_PK_H
#define TLM_PK_H

// System headers first to avoid conflicts
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/un.h>

#include "systemc"
#include "tlm.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"

#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <vector>
#include <string>
#include <map>

// TLM Proxy Kernel for RISC-V GTX Simulation
// Based on riscv-pk but adapted for TLM environment

namespace riscv_tlm {

// System call numbers (matching Linux RISC-V ABI)
#define SYS_exit                93
#define SYS_exit_group          94
#define SYS_read                63
#define SYS_write               64
#define SYS_open                56
#define SYS_close               57
#define SYS_lseek               62
#define SYS_brk                 214
#define SYS_mmap                223
#define SYS_munmap              215
#define SYS_mprotect            226
#define SYS_gettimeofday        169
#define SYS_clock_gettime       113

// Additional file system calls
#define SYS_stat                106
#define SYS_fstat               80
#define SYS_unlink              87
#define SYS_getpid              172
#define SYS_getppid             173

// Socket system calls for PCI communication proxy
#define SYS_socket              198
#define SYS_bind                200
#define SYS_listen              201
#define SYS_accept              202
#define SYS_connect             203
#define SYS_send                204
#define SYS_recv                205
#define SYS_sendto              206
#define SYS_recvfrom            207
#define SYS_shutdown            208
#define SYS_setsockopt          209
#define SYS_getsockopt          210

// Custom TLM-PK system calls
#define SYS_get_argc            220
#define SYS_get_argv            221
#define SYS_set_cmdline         222
#define SYS_list_processes      224  // New: list all active processes (changed from 223 to avoid conflict with SYS_mmap)
#define SYS_fcntl               25   // subset (F_GETFL/F_SETFL only)
#define SYS_tlm_select          230  // virtual select (bitmask interface, up to 64 fds)
#define SYS_tlm_poll            231  // virtual poll
#define SYS_epoll_create        232  // virtual epoll create
#define SYS_epoll_ctl           233  // virtual epoll ctl
#define SYS_epoll_wait          234  // virtual epoll wait

#define SYS_open_file           1024

// File descriptors
#define STDIN_FILENO            0
#define STDOUT_FILENO           1
#define STDERR_FILENO           2

// Socket constants (avoid conflicts with system headers)
#ifndef AF_INET
#define AF_INET                 2
#endif
#ifndef AF_UNIX
#define AF_UNIX                 1
#endif
#ifndef SOCK_STREAM
#define SOCK_STREAM             1
#endif
#ifndef SOCK_DGRAM
#define SOCK_DGRAM              2
#endif
#ifndef IPPROTO_TCP
#define IPPROTO_TCP             6
#endif
#ifndef IPPROTO_UDP
#define IPPROTO_UDP             17
#endif

// Socket options
#ifndef SOL_SOCKET
#define SOL_SOCKET              1
#endif
#ifndef SO_REUSEADDR
#define SO_REUSEADDR            2
#endif
#ifndef SO_KEEPALIVE
#define SO_KEEPALIVE            9
#endif
#ifndef SO_BROADCAST
#define SO_BROADCAST            6
#endif
#ifndef SO_NONBLOCK
#define SO_NONBLOCK             0x1001
#endif

// PCI communication proxy specific ports
#define PCI_PROXY_BASE_PORT     7000
#define PCI_MAX_CONNECTIONS     16

// lseek whence values
#define SEEK_SET                0
#define SEEK_CUR                1
#define SEEK_END                2

// Clock types for clock_gettime
#define CLOCK_REALTIME          0
#define CLOCK_MONOTONIC         1
#define CLOCK_PROCESS_CPUTIME_ID 2
#define CLOCK_THREAD_CPUTIME_ID 3

// Error codes
#define EBADF                   9
#define EINVAL                  22
#define ENOMEM                  12
#define ENOSYS                  38
#define EFAULT                  14
#define EIO                     5    // I/O error
#define EACCES                  13   // Permission denied
#define EMFILE                  24   // Too many open files
#define ENOTCONN                107  // Transport endpoint is not connected
#define EAFNOSUPPORT            97   // Address family not supported
#define EOPNOTSUPP              95   // Operation not supported
#define ENOPROTOOPT             92   // Protocol not available
#define ESPIPE                  29   // Illegal seek
#define EAGAIN                  11
#define ECONNREFUSED            111
#define EADDRINUSE              98
#define ENOENT                  2
#define ENAMETOOLONG            36

// fcntl constants (subset)
#define F_GETFL                 3
#define F_SETFL                 4
// open flags subset
#define O_NONBLOCK              0x00000800

// poll/epoll event flags (subset)
#define POLLIN                  0x001
#define POLLOUT                 0x004
#define EPOLLIN                 0x001
#define EPOLLOUT                0x004
#define EPOLL_CTL_ADD           1
#define EPOLL_CTL_MOD           2
#define EPOLL_CTL_DEL           3

// TLM Proxy Kernel class
class TLMProxyKernel : public sc_core::sc_module {
public:
    // Constructor
    TLMProxyKernel(sc_core::sc_module_name name, uint64_t heap_start);
    
    // Destructor
    ~TLMProxyKernel() = default;
    
    // TLM socket for communication with CPU
    tlm_utils::simple_target_socket<TLMProxyKernel> syscall_socket;
    
    // Memory interface pointer (set by CPU)
    class MemoryInterface* mem_intf;
    
    // System call handler
    void handle_syscall(tlm::tlm_generic_payload& trans, sc_core::sc_time& delay);
    
    // Individual system call implementations
    long sys_exit(long code);
    long sys_exit_group(long code);
    long sys_write(long fd, const char* buf, long count);
    long sys_read(long fd, char* buf, long count);
    long sys_open(const char* filename, long flags, long mode);
    long sys_close(long fd);
    long sys_lseek(long fd, long offset, long whence);
    long sys_brk(long addr);
    long sys_mmap(long addr, long length, long prot, long flags, long fd, long offset);
    long sys_munmap(long addr, long length);
    long sys_mprotect(long addr, long length, long prot);
    long sys_gettimeofday(void* tv, void* tz);
    long sys_clock_gettime(long clk_id, void* tp);
    
    // Additional file system calls
    long sys_stat(const char* pathname, void* statbuf);
    long sys_fstat(long fd, void* statbuf);
    long sys_unlink(const char* pathname);
    long sys_getpid();
    long sys_getppid();
    long sys_fcntl(long fd, long cmd, long arg);
    long sys_tlm_select(long nfds, void* readfds, void* writefds, void* exceptfds, void* timeout);
    long sys_tlm_poll(void* fds, long nfds, long timeout);
    long sys_epoll_create(long size);
    long sys_epoll_ctl(long epfd, long op, long fd, void* event); // event -> struct epoll_event
    long sys_epoll_wait(long epfd, void* events, long maxevents, long timeout);
    
    // Socket system calls for PCI communication proxy
    long sys_socket(long domain, long type, long protocol);
    long sys_bind(long sockfd, const void* addr, long addrlen);
    long sys_listen(long sockfd, long backlog);
    long sys_accept(long sockfd, void* addr, void* addrlen);
    long sys_connect(long sockfd, const void* addr, long addrlen);
    long sys_send(long sockfd, const void* buf, long len, long flags);
    long sys_recv(long sockfd, void* buf, long len, long flags);
    long sys_sendto(long sockfd, const void* buf, long len, long flags, const void* dest_addr, long addrlen);
    long sys_recvfrom(long sockfd, void* buf, long len, long flags, void* src_addr, void* addrlen);
    long sys_shutdown(long sockfd, long how);
    long sys_setsockopt(long sockfd, long level, long optname, const void* optval, long optlen);
    long sys_getsockopt(long sockfd, long level, long optname, void* optval, void* optlen);
    
    // Command line argument system calls
    long sys_get_argc();
    long sys_get_argv(char* buf, long buf_size);
    long sys_set_cmdline(const char* cmdline);
    long sys_list_processes(void* buf, long buf_size);  // New: list processes
    
    // External interface for setting command line arguments
    void set_program_arguments(int argc, char* argv[], const std::string& program_name = "riscv_program");
    
    // Memory interface setter
    void set_memory_interface(MemoryInterface* intf);
    
    // Memory access helpers
    bool read_memory(uint64_t addr, void* data, size_t size);
    bool write_memory(uint64_t addr, const void* data, size_t size);    // TLM transport interface
    void b_transport(tlm::tlm_generic_payload& trans, sc_core::sc_time& delay);
    
    // Utility functions
    void print_syscall_info(long syscall_num, long arg0, long arg1, long arg2);
    bool is_valid_fd(long fd);
    
private:
    bool verbose = false; // Verbose output flag
    // Internal state
    uint64_t heap_ptr;
    uint64_t heap_max;
    sc_core::sc_time default_delay;
    
    // File system simulation
    struct file_descriptor {
        bool is_open;
        uint64_t position;
        std::string filename;
        int open_flags = 0;          // POSIX-like open flags (O_RDONLY/O_WRONLY/O_RDWR/O_CREAT/O_TRUNC/O_APPEND)
        bool is_socket;  // Flag to distinguish sockets from files
        int socket_type; // SOCK_STREAM or SOCK_DGRAM
        int socket_domain; // AF_INET, AF_UNIX, etc.
        bool is_listening; // For server sockets
        uint16_t bound_port; // For bound sockets
        std::string peer_address; // For connected sockets
        std::string file_data; // Store file contents in memory
        // Real socket support
        bool is_real_socket; // Flag for actual network sockets
        int host_sockfd; // Host system socket file descriptor
    // AF_UNIX (Unix domain) support
        bool is_unix = false;            // true if AF_UNIX
        bool is_unix_server = false;     // listening unix domain server
        std::string unix_path;           // bound path
        int peer_fd = -1;                // connected peer fd (stream)
        std::vector<std::string> unix_msg_queue; // queued datagram messages (for SOCK_DGRAM) or stream buffering slices
        // Blocking/async support
        bool non_blocking = false;       // set via (custom) SO_NONBLOCK
        sc_core::sc_event* conn_event = nullptr; // connection arrival
        sc_core::sc_event* data_event = nullptr; // data arrival
        // Epoll instance
        bool is_epoll = false;                       // descriptor is an epoll instance
        std::vector<int> epoll_targets;              // watched fds
        std::vector<uint32_t> epoll_target_events;   // events mask per target
    };
    
    static const int MAX_FDS = 32;
    file_descriptor fd_table[MAX_FDS];
    
    // Simple file system - persistent file storage
    std::map<std::string, std::string> file_system;
    
    // PCI communication proxy state
    struct pci_proxy_connection {
        int proxy_fd;
        int target_fd;
        bool active;
        std::string target_device;
        uint32_t pci_address;
    };
    
    static const int MAX_PCI_CONNECTIONS = 16;
    pci_proxy_connection pci_connections[MAX_PCI_CONNECTIONS];
    
    // Socket address structures (simplified)
    struct sockaddr_in_tlm {
        uint16_t sin_family;
        uint16_t sin_port;
        uint32_t sin_addr;
        uint8_t sin_zero[8];
    };
    
    // Memory management
    struct memory_region {
        uint64_t start;
        uint64_t size;
        bool allocated;
    };
    
    std::vector<memory_region> memory_map;

    // Unix domain socket registry: path -> server fd
    std::map<std::string,int> unix_socket_registry;

    // Pending connection queue for AF_UNIX stream servers: server fd -> list of client fds waiting accept
    std::map<int, std::vector<int>> unix_pending_connections;

    // Readiness helpers
    bool fd_read_ready(int fd);
    bool fd_write_ready(int fd);

    // Helper utilities
    std::string sanitize_unix_path(const std::string& raw, int& errcode);
    
    // Command line arguments management
    std::vector<std::string> argv_storage;
    int argc_value;
    std::string cmdline_buffer;
    
    // Process management simulation
    struct process_info {
        long pid;
        long ppid;
        uint64_t start_time;  // Simulation time when process started
        std::string command;
        bool active;
    };
    
    static const int MAX_PROCESSES = 64;
    process_info process_table[MAX_PROCESSES];
    long next_pid_counter;
    long current_process_pid;
    
    
    // Initialize file descriptor table
    void init_fd_table();
    void init_pci_proxy();
    void init_process_table();
    
    // Process management helpers
    long create_process(const std::string& command, long parent_pid);
    void cleanup_process(long pid);
    process_info* find_process(long pid);
    
    // Socket management helpers
    int allocate_socket_fd();
    bool is_socket_fd(int fd);
    void setup_pci_proxy_connection(int proxy_fd, const std::string& target_device, uint32_t pci_addr);
    
    // Memory management helpers
    uint64_t allocate_memory(uint64_t size);
    bool deallocate_memory(uint64_t addr, uint64_t size);
};

} // namespace riscv_tlm

#endif // TLM_PK_H
