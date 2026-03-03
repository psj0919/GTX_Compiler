

#ifndef __PCIE_H__
#define __PCIE_H__

#include <iostream>
#include <fstream>
#include <cstdint>
#include <poll.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <string.h>
#define SC_INCLUDE_DYNAMIC_PROCESSES

#include "systemc"
#include "config.h"
#include "MemoryInterface.h"
#include "Memory.h"
#include "tlm.h"
#include "tlm_utils/simple_target_socket.h"

#include "BusCtrl.h"

#ifdef USE_VFIO_USER
#include "VfioUserAdapter.h"
#endif

namespace riscv_tlm::peripherals {
/**
 * @brief GTX NPU PCIe Controller (Simple Version)
 *
 * Architecture:
 *   Host (ggml-gtx.cpp) <-> Unix Socket <-> PCIE module <-> TLM Memory <-> RISC-V CPU
 *
 * This module provides:
 *   - Memory read/write via TLM for RISC-V access
 *   - Unix socket communication with Host
 *   - Simple interrupt flag registers (cfg_intr_flag/mask/status)
 *   - IRQ line to RISC-V Hart (legacy external interrupt)
 *   - DBI doorbell detection for Device->Host notification
 */
    class PCIE : sc_core::sc_module {
    public:
        // TLM-2 socket, defaults to 32-bits wide, base protocol
        tlm_utils::simple_target_socket<PCIE> socket;

        tlm_utils::simple_initiator_socket<PCIE> irq_line;

        MemoryInterface *mem_intf;

        sc_core::sc_event ev_cmd3;
        int sock_fd ;
        int conn_fd;

        /**
         *
         * @brief Constructor
         * @param name module name
         */
        explicit PCIE(sc_core::sc_module_name const &name, Memory *main_mem);
        bool recv_all_nb(int fd, void* buf, size_t len) {
            // non-blocking 환경에서 len만큼 "완전히" 받는 간단 버전
            // (poll로 readable 확인 후 호출한다고 가정)
            char* p = (char*)buf;
            size_t got = 0;
            while (got < len) {
                ssize_t n = ::recv(fd, p + got, len - got, 0);
                if (n == 0) return false; // peer closed
                if (n < 0) {
                    if (errno == EINTR) continue;
                    if (errno == EAGAIN || errno == EWOULDBLOCK) return false; // 아직 다 안 옴
                    return false;
                }
                got += (size_t)n;
            }
            return true;
        }
        static bool recv_exact(int fd, void* buf, size_t len) {
            char* p = (char*)buf;
            size_t got = 0;
            while (got < len) {
                ssize_t n = ::recv(fd, p + got, len - got, 0);
                if (n == 0) return false;          // peer closed
                if (n < 0) { if (errno==EINTR) continue; return false; }
                got += (size_t)n;
            }
            return true;
        }
        void connect_uds(const char* path) {

            sock_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
            if (sock_fd < 0) {
                LOG(LOG_WARNING) << "[WARNING] PCIE: socket() failed, PCIE disabled\n";
                sock_fd = -1;
                return;
            }
            ::unlink(path);

            sockaddr_un addr{};
            addr.sun_family = AF_UNIX;
            std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path);

            // 2) bind -> 여기서 path에 소켓 파일이 생성됨
            if (::bind(sock_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
                LOG(LOG_WARNING) << "[WARNING] PCIE: bind() failed, PCIE disabled\n";
                ::close(sock_fd);
                sock_fd = -1;
                return;
            }
            if (::listen(sock_fd, 8) < 0) {
                LOG(LOG_WARNING) << "[WARNING] PCIE: listen() failed, PCIE disabled\n";
                ::close(sock_fd);
                sock_fd = -1;
                return;
            }

             // listen_fd를 non-blocking으로
            int flags = ::fcntl(sock_fd, F_GETFL, 0);
            if (flags < 0) {
                LOG(LOG_WARNING) << "[WARNING] PCIE: fcntl(F_GETFL) failed\n";
            } else if (::fcntl(sock_fd, F_SETFL, flags | O_NONBLOCK) < 0) {
                LOG(LOG_WARNING) << "[WARNING] PCIE: fcntl(F_SETFL O_NONBLOCK) failed\n";
            }
        }
        /**
         * @brief Waits for event timer_event and triggers an IRQ
         *
         * Waits for event timer_event and triggers an IRQ (if it is not already
         * triggered).
         * After that, it posts the timer_event to 20 ns in the future to clear the IRQ
         * line.
         *
         */
        [[noreturn]] void run();

        /**
         *
         * @brief TLM-2.0 socket implementation
         * @param trans TLM-2.0 transaction
         * @param delay transaction delay time
         */
        virtual void b_transport(tlm::tlm_generic_payload &trans,
                                 sc_core::sc_time &delay);

        Memory *mainmem;

        /* Binary protocol handling (unused legacy) */
        void process_binary_protocol();
        void handle_write_msg(const uint8_t* data, uint16_t len);
        void handle_read_msg();
        void handle_doorbell();
        void handle_status();
        void send_response(uint8_t cmd, uint8_t status, const void* data, uint16_t len);

        /* MSI handling - send TX queue response to Host */
        void check_and_send_msi_response();
        uint32_t last_tx_head;  /* Track TX queue changes */

        /*=================================================================
         * vfio-user protocol types (used in both legacy & vfio modes)
         *================================================================*/

        /* vfio-user message header (packed, matches libvfio-user wire format) */
        struct __attribute__((packed)) vfio_msg_hdr {
            uint16_t msg_id;
            uint16_t cmd;
            uint32_t msg_size;
            uint32_t flags;
            uint32_t error_no;
        };

        /* vfio-user region access (packed) */
        struct __attribute__((packed)) vfio_region_access {
            uint64_t offset;
            uint32_t region;
            uint32_t count;
            /* data follows */
        };

        /* vfio-user commands */
        enum VfioCmd : uint16_t {
            VFIO_USER_VERSION               = 1,
            VFIO_USER_DEVICE_GET_INFO       = 4,
            VFIO_USER_DEVICE_GET_REGION_INFO = 5,
            VFIO_USER_DEVICE_GET_IRQ_INFO   = 7,
            VFIO_USER_DEVICE_SET_IRQS       = 8,
            VFIO_USER_REGION_READ           = 9,
            VFIO_USER_REGION_WRITE          = 10,
            VFIO_USER_DEVICE_RESET          = 13,
        };

        static constexpr uint32_t VFIO_FLAG_COMMAND = 0x0;
        static constexpr uint32_t VFIO_FLAG_REPLY   = 0x1;

        /* MSI eventfds received from host (legacy mode) */
        static constexpr int MAX_MSI_FDS = 32;
        int msi_fds[MAX_MSI_FDS];
        int msi_fd_count;

        /* Handle a complete vfio-user message from connected client */
        void handle_vfio_message(const vfio_msg_hdr &hdr,
                                 const uint8_t *payload, size_t payload_len,
                                 const int *fds, int fd_count);

        /* Send vfio-user reply to client */
        void send_vfio_reply(uint16_t msg_id, uint16_t cmd,
                             const void *data, size_t data_len);

        /* Trigger MSI to host via eventfd (legacy mode) */
        void legacy_trigger_msi(int vector);

    private:

        sc_dt::sc_uint<32> cfg_intr_flag;
        sc_dt::sc_uint<32> cfg_intr_mask;
        sc_dt::sc_uint<32> cfg_intr_status;
        sc_dt::sc_uint<32> dbi_msi;

        /**
         * @brief Legacy socket command processing state machine
         */
        enum CmdState {
            CMD_IDLE            = 0,   /* Waiting for mode selection (1-4) */
            CMD_FILE_LOAD       = 1,   /* Waiting for file name input */
            CMD_INTERRUPT       = 2,   /* Waiting for interrupt number input */
            CMD_WRITE_ADDR      = 3,   /* Waiting for write address input */
            CMD_WRITE_DATA      = 31,  /* Waiting for write data input */
            CMD_READ_ADDR       = 4,   /* Waiting for read address input */
        };

        CmdState cmd_state;
        int pkt_addr;
        int pkt_data;
        int intr_clear;
        std::string file_path;

#ifdef USE_VFIO_USER
        /* vfio-user adapter for PCIe emulation */
        VfioUserAdapter *vfio_adapter;

        /* vfio-user BAR access callbacks */
        ssize_t vfio_bar0_access(uint64_t offset, void* data, size_t len, bool write);
        ssize_t vfio_bar2_access(uint64_t offset, void* data, size_t len, bool write);
        void vfio_irq_inject(uint32_t irq_bits);

        /* vfio-user run thread */
        void run_vfio();
#endif

    };
}
#endif
