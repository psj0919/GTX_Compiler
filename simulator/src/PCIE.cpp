// SPDX-License-Identifier: GPL-3.0-or-later

#include "PCIE.h"
#include <cstdint>

namespace riscv_tlm::peripherals {

    SC_HAS_PROCESS(PCIE);

    PCIE::PCIE(sc_core::sc_module_name const &name, Memory *main_mem) :
            sc_module(name), socket("pcie_socket") {
        cfg_intr_flag   = 0;
        cfg_intr_mask   = 0;
        cfg_intr_status = 0;
        sock_fd = -1;
        conn_fd = -1;
        cmd_state = CMD_IDLE;
        intr_clear =false;
        msi_fd_count = 0;
        for (int i = 0; i < MAX_MSI_FDS; i++) msi_fds[i] = -1;
        mem_intf = new MemoryInterface((std::string(name)+ "_mem").c_str());
        mainmem = main_mem;

        socket.register_b_transport(this, &PCIE::b_transport);

#ifdef USE_VFIO_USER
        /* Initialize vfio-user adapter */
        vfio_adapter = new VfioUserAdapter("/tmp/gtx-vfio.sock");

        /* Setup BAR callbacks */
        vfio_adapter->setBar0Callback([this](uint64_t offset, void* data, size_t len, bool write) {
            return this->vfio_bar0_access(offset, data, len, write);
        });

        vfio_adapter->setBar2Callback([this](uint64_t offset, void* data, size_t len, bool write) {
            return this->vfio_bar2_access(offset, data, len, write);
        });

        vfio_adapter->setIrqInjectCallback([this](uint32_t irq_bits) {
            this->vfio_irq_inject(irq_bits);
        });

        SC_THREAD(run_vfio);
#else
        SC_THREAD(run);
#endif
    }

#ifdef USE_VFIO_USER
    /*=========================================================================
     * vfio-user implementation
     *========================================================================*/

    ssize_t PCIE::vfio_bar0_access(uint64_t offset, void* data, size_t len, bool write)
    {
        /* BAR0 is registers/config space */
        if (write) {
            /* Handle special register writes */
            if (offset == 0x0E40 && len == 4) {
                /* ELBI doorbell - inject interrupt to RISC-V */
                uint32_t doorbell = *static_cast<uint32_t*>(data);
                LOG(LOG_INFO) << "[VFIO] BAR0 Doorbell write: 0x" << std::hex << doorbell << "\n";
                mem_intf->writeDataMem(PCIE_INTERRUPT, doorbell, 4);
                /* Also set cfg_intr_status so run_vfio loop triggers IRQ */
                cfg_intr_status |= doorbell;
            } else {
                /* Write to memory-mapped region */
                mem_intf->writeDataMem(PCIE_CFG_BASE + offset, *static_cast<uint32_t*>(data), len);
            }
        } else {
            /* Read from memory-mapped region */
            uint32_t val = mem_intf->readDataMem(PCIE_CFG_BASE + offset, len);
            memcpy(data, &val, len);
        }
        return len;
    }

    ssize_t PCIE::vfio_bar2_access(uint64_t offset, void* data, size_t len, bool write)
    {
        /* BAR2 is shared memory (slave region) */
        uint64_t addr = 0x60000000 + offset;  /* PCIE slave base */
        uint8_t* buf = static_cast<uint8_t*>(data);

        /* Handle bulk transfers by breaking into 4-byte chunks */
        size_t remaining = len;
        size_t pos = 0;

        while (remaining > 0) {
            size_t chunk = (remaining >= 4) ? 4 : remaining;

            if (write) {
                uint32_t val = 0;
                memcpy(&val, buf + pos, chunk);
                mem_intf->writeDataMem(addr + pos, val, chunk);
            } else {
                uint32_t val = mem_intf->readDataMem(addr + pos, chunk);
                memcpy(buf + pos, &val, chunk);
            }

            pos += chunk;
            remaining -= chunk;
        }

        return len;
    }

    void PCIE::vfio_irq_inject(uint32_t irq_bits)
    {
        LOG(LOG_INFO) << "[VFIO] IRQ inject: 0x" << std::hex << irq_bits << "\n";
        mem_intf->writeDataMem(PCIE_INTERRUPT, irq_bits, 4);
    }

    void PCIE::run_vfio()
    {
        LOG(LOG_INFO) << "[PCIE] Starting vfio-user mode\n";
        LOG(LOG_INFO) << "[PCIE] PCIe Device Info:\n";
        LOG(LOG_INFO) << "[PCIE]   Vendor:Device = 1234:5678\n";
        LOG(LOG_INFO) << "[PCIE]   Class Code    = 120000 (Processing Accelerators)\n";
        LOG(LOG_INFO) << "[PCIE]   BAR0          = 1MB (Registers)\n";
        LOG(LOG_INFO) << "[PCIE]   BAR2          = 64MB (Shared Memory)\n";
        LOG(LOG_INFO) << "[PCIE]   Socket        = /tmp/gtx-vfio.sock\n";

        /* Start vfio-user adapter */
        if (vfio_adapter->start() < 0) {
            LOG(LOG_ERROR) << "[PCIE] Failed to start vfio-user adapter\n";
            return;
        }

        auto *irq_trans = new tlm::tlm_generic_payload;
        sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
        std::uint64_t cause = ((uint64_t)1 << 63) | 0x0B;  // Machine external interrupt
        irq_trans->set_command(tlm::TLM_WRITE_COMMAND);
        irq_trans->set_data_ptr(reinterpret_cast<unsigned char *>(&cause));
        irq_trans->set_data_length(8);
        irq_trans->set_streaming_width(8);
        irq_trans->set_byte_enable_ptr(nullptr);
        irq_trans->set_dmi_allowed(false);
        irq_trans->set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
        irq_trans->set_address(0);

        /* Main loop - handle interrupts */
        while (vfio_adapter->isRunning()) {
            /* Check for pending interrupts from host doorbell */
            if (cfg_intr_status) {
                uint32_t status = cfg_intr_status;
                cfg_intr_status = 0;  /* Clear before triggering */
                std::cout << "[PCIE] Triggering IRQ to RISC-V, status=0x"
                          << std::hex << status << std::dec << std::endl;
                irq_line->b_transport(*irq_trans, delay);
            }

            /* Handle interrupt clear */
            if (intr_clear) {
                intr_clear = false;
                irq_trans->set_command(tlm::TLM_READ_COMMAND);
                irq_line->b_transport(*irq_trans, delay);
                irq_trans->set_command(tlm::TLM_WRITE_COMMAND);
            }

            /* Check for MSI from device - send all pending vectors (LSB = highest priority) */
            if (dbi_msi != 0) {
                uint32_t pending = dbi_msi;
                dbi_msi = 0;
                while (pending != 0) {
                    int vector = __builtin_ctz(pending);
                    vfio_adapter->triggerMSI(vector);
                    pending &= ~(1U << vector);
                }
            }

            sc_core::wait(sc_core::sc_time(100, sc_core::SC_NS));
        }

        delete irq_trans;
    }
#else
    /*=========================================================================
     * Legacy Unix socket implementation (when USE_VFIO_USER is not defined)
     *
     * Supports two sub-protocols auto-detected per connection:
     *   1) vfio-user binary protocol (from gtx-irq-test / pcie_irq_host.cpp)
     *   2) Text-based interactive protocol (from manual/telnet usage)
     *========================================================================*/

    /*-----------------------------------------------------------------
     * vfio-user binary protocol handler for legacy mode
     *-----------------------------------------------------------------*/

    void PCIE::send_vfio_reply(uint16_t msg_id, uint16_t cmd,
                                const void *data, size_t data_len)
    {
        vfio_msg_hdr reply_hdr;
        reply_hdr.msg_id   = msg_id;
        reply_hdr.cmd      = cmd;
        reply_hdr.msg_size = (uint32_t)(sizeof(vfio_msg_hdr) + data_len);
        reply_hdr.flags    = VFIO_FLAG_REPLY;
        reply_hdr.error_no = 0;

        struct iovec iov[2];
        int iovcnt = 1;
        iov[0].iov_base = &reply_hdr;
        iov[0].iov_len  = sizeof(reply_hdr);
        if (data && data_len > 0) {
            iov[1].iov_base = const_cast<void*>(data);
            iov[1].iov_len  = data_len;
            iovcnt = 2;
        }

        struct msghdr msg{};
        msg.msg_iov    = iov;
        msg.msg_iovlen = iovcnt;

        ::sendmsg(conn_fd, &msg, MSG_NOSIGNAL);
    }

    void PCIE::handle_vfio_message(const vfio_msg_hdr &hdr,
                                    const uint8_t *payload, size_t payload_len,
                                    const int *fds, int fd_count)
    {
        switch (hdr.cmd) {
        case VFIO_USER_VERSION: {
            /* Reply with our version */
            struct __attribute__((packed)) {
                uint16_t major;
                uint16_t minor;
            } ver = {0, 1};
            const char *caps = "{\"capabilities\":{\"max_msg_fds\":8,\"max_data_xfer_size\":1048576}}";
            size_t caps_len = strlen(caps) + 1;
            size_t total = sizeof(ver) + caps_len;
            uint8_t buf[256];
            memcpy(buf, &ver, sizeof(ver));
            memcpy(buf + sizeof(ver), caps, caps_len);
            send_vfio_reply(hdr.msg_id, hdr.cmd, buf, total);
            LOG(LOG_INFO) << "[PCIE-VFIO] Version negotiated\n";
            break;
        }

        case VFIO_USER_DEVICE_GET_INFO: {
            struct __attribute__((packed)) {
                uint32_t argsz;
                uint32_t flags;
                uint32_t num_regions;
                uint32_t num_irqs;
            } info;
            info.argsz = sizeof(info);
            info.flags = 0;
            info.num_regions = 9;
            info.num_irqs = 3;   /* INTX, MSI, MSIX */
            send_vfio_reply(hdr.msg_id, hdr.cmd, &info, sizeof(info));
            LOG(LOG_INFO) << "[PCIE-VFIO] Device info sent\n";
            break;
        }

        case VFIO_USER_DEVICE_GET_REGION_INFO: {
            struct __attribute__((packed)) {
                uint32_t argsz;
                uint32_t flags;
                uint32_t index;
                uint32_t cap_offset;
                uint64_t size;
                uint64_t offset;
            } region_info{};
            region_info.argsz = sizeof(region_info);

            uint32_t region_idx = 0;
            if (payload_len >= 12) {
                memcpy(&region_idx, payload + 8, 4);
            }
            region_info.index = region_idx;

            switch (region_idx) {
            case 0: /* BAR0 */
                region_info.flags = 0x3; /* RW */
                region_info.size = 0x100000; /* 1MB */
                break;
            case 2: /* BAR2 */
                region_info.flags = 0x3;
                region_info.size = 0x4000000; /* 64MB */
                break;
            default:
                region_info.flags = 0;
                region_info.size = 0;
                break;
            }
            send_vfio_reply(hdr.msg_id, hdr.cmd, &region_info, sizeof(region_info));
            break;
        }

        case VFIO_USER_DEVICE_GET_IRQ_INFO: {
            struct __attribute__((packed)) {
                uint32_t argsz;
                uint32_t flags;
                uint32_t index;
                uint32_t count;
            } irq_info{};
            irq_info.argsz = sizeof(irq_info);
            if (payload_len >= 12) {
                memcpy(&irq_info.index, payload + 8, 4);
            }
            if (irq_info.index == 1) { /* MSI */
                irq_info.flags = 0x1; /* VFIO_IRQ_INFO_EVENTFD */
                irq_info.count = 32;
            }
            send_vfio_reply(hdr.msg_id, hdr.cmd, &irq_info, sizeof(irq_info));
            break;
        }

        case VFIO_USER_DEVICE_SET_IRQS: {
            /* Host sends eventfds for MSI vectors via SCM_RIGHTS */
            if (fds && fd_count > 0) {
                for (int i = 0; i < fd_count && i < MAX_MSI_FDS; i++) {
                    msi_fds[i] = fds[i];
                }
                msi_fd_count = (fd_count < MAX_MSI_FDS) ? fd_count : MAX_MSI_FDS;
                LOG(LOG_INFO) << "[PCIE-VFIO] Received " << msi_fd_count << " MSI eventfds\n";
            }
            send_vfio_reply(hdr.msg_id, hdr.cmd, nullptr, 0);
            break;
        }

        case VFIO_USER_REGION_WRITE: {
            if (payload_len < sizeof(vfio_region_access)) break;
            const vfio_region_access *acc = reinterpret_cast<const vfio_region_access*>(payload);
            const uint8_t *write_data = payload + sizeof(vfio_region_access);
            size_t write_len = payload_len - sizeof(vfio_region_access);

            if (acc->region == 0) {
                /* BAR0 write */
                if (acc->offset == 0x0E40 && write_len >= 4) {
                    uint32_t doorbell = 0;
                    memcpy(&doorbell, write_data, 4);
                    LOG(LOG_INFO) << "[PCIE-VFIO] BAR0 Doorbell write: 0x"
                                  << std::hex << doorbell << std::dec << "\n";
                    mem_intf->writeDataMem(PCIE_INTERRUPT, doorbell, 4);
                    cfg_intr_flag |= doorbell;
                    cfg_intr_status = ~cfg_intr_mask & cfg_intr_flag;
                } else {
                    for (size_t i = 0; i + 3 < write_len; i += 4) {
                        uint32_t val = 0;
                        memcpy(&val, write_data + i, 4);
                        mem_intf->writeDataMem(PCIE_CFG_BASE + acc->offset + i, val, 4);
                    }
                }
            } else if (acc->region == 2) {
                uint64_t addr = 0x60000000 + acc->offset;
                for (size_t i = 0; i < write_len; i += 4) {
                    size_t chunk = (write_len - i >= 4) ? 4 : (write_len - i);
                    uint32_t val = 0;
                    memcpy(&val, write_data + i, chunk);
                    mem_intf->writeDataMem(addr + i, val, chunk);
                }
            }

            vfio_region_access reply_acc;
            reply_acc.offset = acc->offset;
            reply_acc.region = acc->region;
            reply_acc.count  = acc->count;
            send_vfio_reply(hdr.msg_id, hdr.cmd, &reply_acc, sizeof(reply_acc));
            break;
        }

        case VFIO_USER_REGION_READ: {
            if (payload_len < sizeof(vfio_region_access)) break;
            const vfio_region_access *acc = reinterpret_cast<const vfio_region_access*>(payload);

            size_t reply_size = sizeof(vfio_region_access) + acc->count;
            uint8_t *reply_buf = new uint8_t[reply_size];
            memset(reply_buf, 0, reply_size);

            vfio_region_access *reply_acc = reinterpret_cast<vfio_region_access*>(reply_buf);
            reply_acc->offset = acc->offset;
            reply_acc->region = acc->region;
            reply_acc->count  = acc->count;
            uint8_t *read_data = reply_buf + sizeof(vfio_region_access);

            if (acc->region == 0) {
                for (size_t i = 0; i + 3 < acc->count; i += 4) {
                    uint32_t val = mem_intf->readDataMem(PCIE_CFG_BASE + acc->offset + i, 4);
                    memcpy(read_data + i, &val, 4);
                }
            } else if (acc->region == 2) {
                uint64_t addr = 0x60000000 + acc->offset;
                for (size_t i = 0; i < acc->count; i += 4) {
                    size_t chunk = (acc->count - i >= 4) ? 4 : (acc->count - i);
                    uint32_t val = mem_intf->readDataMem(addr + i, chunk);
                    memcpy(read_data + i, &val, chunk);
                }
            }

            send_vfio_reply(hdr.msg_id, hdr.cmd, reply_buf, reply_size);
            delete[] reply_buf;
            break;
        }

        case VFIO_USER_DEVICE_RESET:
            cfg_intr_flag   = 0;
            cfg_intr_mask   = 0;
            cfg_intr_status = 0;
            send_vfio_reply(hdr.msg_id, hdr.cmd, nullptr, 0);
            LOG(LOG_INFO) << "[PCIE-VFIO] Device reset\n";
            break;

        default:
            LOG(LOG_WARNING) << "[PCIE-VFIO] Unknown command: " << hdr.cmd << "\n";
            send_vfio_reply(hdr.msg_id, hdr.cmd, nullptr, 0);
            break;
        }
    }

    void PCIE::legacy_trigger_msi(int vector)
    {
        if (vector < 0 || vector >= MAX_MSI_FDS) return;

        if (vector < msi_fd_count && msi_fds[vector] >= 0) {
            /* Write to eventfd to notify host */
            uint64_t val = 1;
            ssize_t ret = ::write(msi_fds[vector], &val, sizeof(val));
            if (ret < 0) {
                LOG(LOG_WARNING) << "[PCIE-VFIO] MSI eventfd write failed for vector "
                                 << vector << ": " << strerror(errno) << "\n";
            } else {
                LOG(LOG_INFO) << "[PCIE-VFIO] MSI vector " << vector
                              << " triggered via eventfd\n";
            }
        } else {
            /* Fallback: send text message */
            if (conn_fd >= 0) {
                char msi_msg[64];
                int msi_len = snprintf(msi_msg, sizeof(msi_msg), "MSI:%d\n", vector);
                ::send(conn_fd, msi_msg, msi_len, MSG_NOSIGNAL);
                LOG(LOG_INFO) << "[PCIE] MSI vector " << vector
                              << " triggered via text msg\n";
            }
        }
    }

    /*-----------------------------------------------------------------
     * Main legacy run() loop with auto-detect vfio-user binary protocol
     *-----------------------------------------------------------------*/

    [[noreturn]] void PCIE::run() {
        connect_uds("/tmp/gtx-vfio.sock");

        auto *irq_trans = new tlm::tlm_generic_payload;
        sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
        std::uint64_t cause = ((uint64_t)1 << 63) | 0x0B;     // Machine external interrupt
        irq_trans->set_command(tlm::TLM_WRITE_COMMAND);
        irq_trans->set_data_ptr(reinterpret_cast<unsigned char *>(&cause));
        irq_trans->set_data_length(8);
        irq_trans->set_streaming_width(8);
        irq_trans->set_byte_enable_ptr(nullptr);
        irq_trans->set_dmi_allowed(false);
        irq_trans->set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
        irq_trans->set_address(0);

        std::string acc;
        acc.reserve(4096);

        /* Per-connection flag: is this a vfio-user binary client? */
        bool is_vfio_client = false;

        while (true) {
            // 1) Accept new connection
            if (conn_fd < 0) {

                sockaddr_un caddr{};
                socklen_t clen = sizeof(caddr);
                int fd = ::accept(sock_fd, (sockaddr*)&caddr, &clen);
                if (fd >= 0) {
                    conn_fd = fd;
                    is_vfio_client = false;
                    cmd_state = CMD_IDLE;

                    int fl = ::fcntl(conn_fd, F_GETFL, 0);
                    if (fl >= 0) ::fcntl(conn_fd, F_SETFL, fl | O_NONBLOCK);

                    /* Reset MSI eventfds from previous connection */
                    for (int i = 0; i < msi_fd_count; i++) {
                        if (msi_fds[i] >= 0) {
                            ::close(msi_fds[i]);
                            msi_fds[i] = -1;
                        }
                    }
                    msi_fd_count = 0;

                    LOG(LOG_INFO) << "[INFO] pcie - accepted client, conn_fd=" << conn_fd << "\n";
                }
            }

            // 2) Process socket data
            if (conn_fd >= 0) {

                pollfd pfd{};
                pfd.fd = conn_fd;
                pfd.events = POLLIN;
                int pr = ::poll(&pfd, 1, 0);

                if (pr > 0) {
                    if (pfd.revents & (POLLHUP | POLLERR | POLLNVAL)) {
                        LOG(LOG_INFO) << "[INFO] pcie - client dead, revents=0x"
                                      << std::hex << pfd.revents << std::dec << "\n";
                        ::close(conn_fd);
                        conn_fd = -1;
                        is_vfio_client = false;
                    } else if (pfd.revents & POLLIN) {

                        /*==============================================
                         * Receive data with possible SCM_RIGHTS fds
                         *=============================================*/
                        char buf[4096];
                        struct iovec iov;
                        iov.iov_base = buf;
                        iov.iov_len  = sizeof(buf);

                        char cmsg_buf[CMSG_SPACE(sizeof(int) * MAX_MSI_FDS)];
                        struct msghdr msg{};
                        msg.msg_iov        = &iov;
                        msg.msg_iovlen     = 1;
                        msg.msg_control    = cmsg_buf;
                        msg.msg_controllen = sizeof(cmsg_buf);

                        ssize_t n = ::recvmsg(conn_fd, &msg, 0);

                        /* Extract file descriptors if any */
                        int recv_fds[MAX_MSI_FDS];
                        int recv_fd_count = 0;
                        for (struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
                             cmsg != NULL;
                             cmsg = CMSG_NXTHDR(&msg, cmsg)) {
                            if (cmsg->cmsg_level == SOL_SOCKET &&
                                cmsg->cmsg_type == SCM_RIGHTS) {
                                size_t fd_size = cmsg->cmsg_len - CMSG_LEN(0);
                                recv_fd_count = fd_size / sizeof(int);
                                if (recv_fd_count > MAX_MSI_FDS) 
                                    recv_fd_count = MAX_MSI_FDS;
                                memcpy(recv_fds, CMSG_DATA(cmsg),
                                       recv_fd_count * sizeof(int));
                            }
                        }

                        if (n <= 0) {
                            LOG(LOG_INFO) << "[INFO] pcie - client closed or recv error n=" << n
                                      << " errno=" << errno << " (" << strerror(errno) << ")\n";
                            ::close(conn_fd);
                            conn_fd = -1;
                            is_vfio_client = false;
                        } else {

                            /*==============================================
                             * Auto-detect protocol on first message:
                             * vfio-user header = 16 bytes, cmd = 1..13,
                             * flags[3:0] = 0 (command).
                             * Text protocol starts with ASCII '1'..'4'.
                             *=============================================*/
                            if (!is_vfio_client && n >= (ssize_t)sizeof(vfio_msg_hdr)) {
                                const vfio_msg_hdr *probe =
                                    reinterpret_cast<const vfio_msg_hdr*>(buf);
                                if (probe->cmd >= VFIO_USER_VERSION &&
                                    probe->cmd <= VFIO_USER_DEVICE_RESET &&
                                    probe->msg_size >= sizeof(vfio_msg_hdr) &&
                                    probe->msg_size <= 1048576 &&
                                    (probe->flags & 0xF) == VFIO_FLAG_COMMAND) {
                                    is_vfio_client = true;
                                    LOG(LOG_INFO) << "[PCIE] Detected vfio-user binary client\n";
                                }
                            }

                            if (is_vfio_client) {
                                /*--- vfio-user binary protocol path ---*/
                                size_t offset = 0;
                                while (offset + sizeof(vfio_msg_hdr) <= (size_t)n) {
                                    const vfio_msg_hdr *hdr =
                                        reinterpret_cast<const vfio_msg_hdr*>(buf + offset);

                                    size_t payload_len = 0;
                                    if (hdr->msg_size > sizeof(vfio_msg_hdr)) {
                                        payload_len = hdr->msg_size - sizeof(vfio_msg_hdr);
                                    }

                                    /* If entire message not in buffer, read remaining */
                                    if (offset + hdr->msg_size > (size_t)n) {
                                        size_t remaining = (offset + hdr->msg_size) - n;
                                        char extra[4096];
                                        size_t got = 0;
                                        while (got < remaining) {
                                            ssize_t r = ::recv(conn_fd, extra + got,
                                                               remaining - got, 0);
                                            if (r <= 0) break;
                                            got += r;
                                        }
                                        size_t avail_in_buf = n - offset - sizeof(vfio_msg_hdr);
                                        uint8_t *combined = new uint8_t[payload_len];
                                        if (avail_in_buf > 0)
                                            memcpy(combined,
                                                   buf + offset + sizeof(vfio_msg_hdr),
                                                   avail_in_buf);
                                        if (got > 0)
                                            memcpy(combined + avail_in_buf, extra, got);
                                        handle_vfio_message(*hdr, combined, payload_len,
                                                           recv_fds, recv_fd_count);
                                        delete[] combined;
                                        break;
                                    }

                                    const uint8_t *payload =
                                        reinterpret_cast<const uint8_t*>(
                                            buf + offset + sizeof(vfio_msg_hdr));

                                    handle_vfio_message(*hdr, payload, payload_len,
                                                       recv_fds, recv_fd_count);
                                    recv_fd_count = 0; /* Only pass fds on first msg */

                                    offset += hdr->msg_size;
                                }

                            } else {
                                /*--- Legacy text protocol path ---*/
                                buf[n] = '\0';
                                LOG(LOG_INFO) << "[INFO] pcie -  RX " << n
                                              << " bytes / BUF - " << buf << std::endl;

                                int value;
                                long unsigned int p;
                                std::string s;

                                switch(cmd_state){
                                    case CMD_IDLE:
                                        if (buf[0] >= '1' && buf[0] <= '4') {
                                            value = buf[0] - '0';
                                            cmd_state = static_cast<CmdState>(value);
                                            LOG(LOG_INFO) << "[INFO] (" << value << ") selected\n";
                                        }else{
                                            LOG(LOG_WARNING) << "[WARNING] not a int value enter again\n";
                                            break;
                                        }
                                        switch(cmd_state){
                                            case CMD_FILE_LOAD:
                                                LOG(LOG_INFO) << "[INFO] file load option - enter file name\n";
                                                break;
                                            case CMD_INTERRUPT:
                                                LOG(LOG_INFO) << "[INFO] interrupt generate option - enter interrupt id \n";
                                                break;
                                            case CMD_WRITE_ADDR:
                                                LOG(LOG_INFO) << "[INFO] data packet option - enter address \n";
                                                break;
                                            case CMD_READ_ADDR:
                                                LOG(LOG_INFO) << "[INFO] memory read option - enter address \n";
                                                break;
                                            default:
                                                LOG(LOG_WARNING) << "[WARNING] unknown command -pcie\n";
                                                break;
                                        }
                                        break;
                                    case CMD_FILE_LOAD:
                                        file_path = std::string(buf,n);
                                        p = file_path.find_first_of("\r\n");
                                        if (p != std::string::npos)
                                            file_path.resize(p);
                                        mainmem->readDataFile(file_path.c_str(), 0);
                                        cmd_state = CMD_IDLE;
                                        LOG(LOG_INFO) << "[INFO] select mode\n";
                                        break;
                                    case CMD_INTERRUPT:
                                        try{
                                            value = std::stoi(buf);
                                        }catch(const std::invalid_argument&){
                                            LOG(LOG_WARNING) << "[WARNING] not a int value enter interrupt number again\n";
                                            break;
                                        }
                                        mem_intf->writeDataMem(PCIE_INTERRUPT, value, 4);
                                        cmd_state = CMD_IDLE;
                                        LOG(LOG_INFO) << "[INFO] select mode\n";
                                        break;
                                    case CMD_WRITE_ADDR:
                                        try{
                                            s = std::string(buf,n);
                                            pkt_addr = (uint32_t)std::stoul(s, nullptr, 16);
                                            cmd_state = CMD_WRITE_DATA;
                                        }catch(const std::invalid_argument&){
                                            LOG(LOG_WARNING) << "[WARNING] not a int value enter address again\n";
                                            break;
                                        }
                                        LOG(LOG_INFO) << "[INFO] data packet option - enter data\n";
                                        break;
                                    case CMD_WRITE_DATA:
                                        try{
                                            s = std::string(buf,n);
                                            pkt_data = (uint32_t)std::stoul(s, nullptr, 16);
                                            mem_intf->writeDataMem(pkt_addr, pkt_data, 4);
                                            printf("pkt_addr = %x, pkt_data = %x\n",pkt_addr,pkt_data);
                                            cmd_state = CMD_IDLE;
                                        }catch(const std::invalid_argument&){
                                            LOG(LOG_WARNING) << "[WARNING] not a int value enter data again\n";
                                            break;
                                        }
                                        LOG(LOG_INFO) << "[INFO] select mode\n";
                                        break;
                                    case CMD_READ_ADDR:
                                        try{
                                            s = std::string(buf,n);
                                            pkt_addr = (uint32_t)std::stoul(s, nullptr, 16);
                                            uint32_t read_val = mem_intf->readDataMem(pkt_addr, 4);
                                            char response[64];
                                            snprintf(response, sizeof(response), "%08x\n", read_val);
                                            ::send(conn_fd, response, strlen(response), 0);
                                            LOG(LOG_INFO) << "[INFO] read addr=0x" << std::hex << pkt_addr
                                                          << " value=0x" << read_val << std::dec << "\n";
                                            cmd_state = CMD_IDLE;
                                        }catch(const std::invalid_argument&){
                                            LOG(LOG_WARNING) << "[WARNING] not a int value enter address again\n";
                                            break;
                                        }
                                        LOG(LOG_INFO) << "[INFO] select mode\n";
                                        break;
                                }
                            }
                        }
                    }
                }
            }

            // 3) Handle device interrupts (Host -> Device via ELBI doorbell)
            if (cfg_intr_status) {
                uint32_t pending_irq = cfg_intr_status;
                cfg_intr_status = 0;  /* Clear before triggering to avoid repeated IRQ */
                LOG(LOG_INFO) << "[PCIE] Triggering IRQ to RISC-V, status=0x"
                              << std::hex << pending_irq << std::dec << "\n";
                irq_line->b_transport(*irq_trans, delay);
            }else if(intr_clear){
                intr_clear = false;
                irq_trans->set_command(tlm::TLM_READ_COMMAND);
                irq_line->b_transport(*irq_trans, delay);
                irq_trans->set_command(tlm::TLM_WRITE_COMMAND);
            }

            // 4) Handle MSI from device (Device -> Host) - send all pending vectors (LSB = highest priority)
            if (dbi_msi != 0) {
                uint32_t pending = dbi_msi;
                dbi_msi = 0;
                while (pending != 0) {
                    int vector = __builtin_ctz(pending);
                    LOG(LOG_INFO) << "[PCIE] MSI vector " << vector << " triggered (Device -> Host)\n";
                    legacy_trigger_msi(vector);
                    pending &= ~(1U << vector);
                }
            }

            sc_core::wait(sc_core::sc_time(100, sc_core::SC_NS));
        }

    }
#endif /* USE_VFIO_USER */

    void PCIE::b_transport(tlm::tlm_generic_payload &trans,
                            sc_core::sc_time &delay) {

        tlm::tlm_command cmd = trans.get_command();
        sc_dt::uint64 addr = trans.get_address();
        unsigned char *ptr = trans.get_data_ptr();
        unsigned int len = trans.get_data_length();
        delay = sc_core::SC_ZERO_TIME;

        std::uint32_t aux_value = 0;


        if (cmd == tlm::TLM_WRITE_COMMAND) {
            memcpy(&aux_value, ptr, len);
            switch (addr) {
                case PCIE_DBI_MSI: //riscv -> host interrupt
                    this->dbi_msi = aux_value;
                    break;
                case PCIE_CFG_INTERRUPT_FLAG: // interrupt flag clear(write)
                    cfg_intr_flag &= ~aux_value;
                    cfg_intr_status = ~cfg_intr_mask & cfg_intr_flag;
                    if(cfg_intr_status == 0x0) intr_clear = true;
                    break;
                case PCIE_CFG_INTERRUPT_MASK: // interrupt mask
                    cfg_intr_mask = aux_value;
                    cfg_intr_status = ~cfg_intr_mask & cfg_intr_flag;
                    break;
                case PCIE_CFG_MASKED_STATUS:
                    LOG(LOG_WARNING) << "[WARNING] MAKSED STATUS - READONLY REGISTER!!\n";
                    break;
                case PCIE_INTERRUPT:
                    cfg_intr_flag |= aux_value;
                    cfg_intr_status = ~cfg_intr_mask & cfg_intr_flag;
                    break;
                default:
                    trans.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
                    return;
            }
        } else { // TLM_READ_COMMAND
            switch (addr) {
                case PCIE_DBI_MSI:
                    LOG(LOG_WARNING) << "[WARNING] DBI-MSI -  WRITEONLY REGISTER!!\n";
                    break;
                case PCIE_CFG_INTERRUPT_FLAG:
                    aux_value = cfg_intr_flag;
                    break;
                case PCIE_CFG_INTERRUPT_MASK:
                    aux_value = cfg_intr_mask;
                    break;
                case PCIE_CFG_MASKED_STATUS:
                    aux_value = cfg_intr_status;
                    break;
                case PCIE_INTERRUPT://host -> riscv interrupt
                    LOG(LOG_WARNING) << "[WARNING] PCIE interrupt - WRITEONLY!!\n";
                    break;
                default:
                    return;
            }
            memcpy(ptr, &aux_value, len);
        }

        trans.set_response_status(tlm::TLM_OK_RESPONSE);
    }
}
