/**
 * @file VfioUserAdapter.cpp
 * @brief libvfio-user adapter implementation
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifdef USE_VFIO_USER

/*
 * IMPORTANT: Include order matters here!
 * config.h defines LogLevel enum with LOG_FATAL, LOG_ERROR, etc.
 * libvfio-user.h includes syslog.h which redefines these as macros.
 * We include config.h first and save our log level values.
 */
#include "config.h"

/* Save our log level enum values before syslog.h redefines them */
static constexpr int VFIO_LOG_FATAL   = LOG_FATAL;
static constexpr int VFIO_LOG_ERROR   = LOG_ERROR;
static constexpr int VFIO_LOG_WARNING = LOG_WARNING;
static constexpr int VFIO_LOG_INFO    = LOG_INFO;
static constexpr int VFIO_LOG_DEBUG   = LOG_DEBUG;

/* Now undefine the LOG macro temporarily so it doesn't conflict */
#undef LOG

#include "VfioUserAdapter.h"

#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <iostream>

/* C++ compatibility for C11 _Static_assert */
#ifndef _Static_assert
#define _Static_assert(expr, msg) static_assert(expr, msg)
#endif

/* loff_t may not be defined on all systems */
#ifndef loff_t
typedef off_t loff_t;
#endif

/* Include libvfio-user header with C linkage - this pulls in syslog.h */
extern "C" {
#include <libvfio-user.h>
}

/* Redefine our LOG macro using saved values */
#define VFIO_LOG(level) if (level <= log_level) std::cout

namespace riscv_tlm::peripherals {

/*=============================================================================
 * Static Callbacks for libvfio-user
 *============================================================================*/

static ssize_t bar0AccessCb(vfu_ctx_t* ctx, char* buf, size_t count,
                            loff_t offset, bool is_write)
{
    VfioUserAdapter* adapter = static_cast<VfioUserAdapter*>(vfu_get_private(ctx));
    if (!adapter) return -1;

    /* Check bounds */
    if (static_cast<size_t>(offset) + count > VfioUserAdapter::BAR0_SIZE) {
        errno = EINVAL;
        return -1;
    }

    /* Use callback if registered (connects to simulator's PCIe device) */
    if (adapter->hasBar0Callback()) {
        if (is_write && offset == 0x0E40) {
            VFIO_LOG(VFIO_LOG_INFO) << "[VFIO] BAR0 Doorbell via callback: 0x"
                << std::hex << *reinterpret_cast<uint32_t*>(buf) << "\n";
        }
        return adapter->invokeBar0Callback(offset, buf, count, is_write);
    } else {
        VFIO_LOG(VFIO_LOG_INFO) << "[VFIO] WARNING: BAR0 no callback registered!\n";
    }

    /* Fallback: Direct buffer access */
    char* bar = static_cast<char*>(adapter->getBar0Buffer());
    if (is_write) {
        memcpy(bar + offset, buf, count);

        /* Check for doorbell write (ELBI) at offset 0x0E40 */
        if (offset == 0x0E40 && count == 4) {
            uint32_t doorbell = *reinterpret_cast<uint32_t*>(buf);
            VFIO_LOG(VFIO_LOG_DEBUG) << "[VFIO] Doorbell write: 0x" << std::hex << doorbell << "\n";
        }
    } else {
        memcpy(buf, bar + offset, count);
    }

    return count;
}

static ssize_t bar2AccessCb(vfu_ctx_t* ctx, char* buf, size_t count,
                            loff_t offset, bool is_write)
{
    VfioUserAdapter* adapter = static_cast<VfioUserAdapter*>(vfu_get_private(ctx));
    if (!adapter) return -1;

    /* Check bounds */
    if (static_cast<size_t>(offset) + count > VfioUserAdapter::BAR2_SIZE) {
        errno = EINVAL;
        return -1;
    }

    /* Use callback if registered (connects to simulator's PCIe device) */
    if (adapter->hasBar2Callback()) {
        return adapter->invokeBar2Callback(offset, buf, count, is_write);
    }

    /* Fallback: Direct buffer access */
    char* bar = static_cast<char*>(adapter->getBar2Buffer());
    if (is_write) {
        memcpy(bar + offset, buf, count);
    } else {
        memcpy(buf, bar + offset, count);
    }

    return count;
}

static int deviceResetCb(vfu_ctx_t* ctx, vfu_reset_type_t type)
{
    (void)type;
    VfioUserAdapter* adapter = static_cast<VfioUserAdapter*>(vfu_get_private(ctx));
    if (!adapter) return 0;

    VFIO_LOG(VFIO_LOG_INFO) << "[VFIO] Device reset\n";

    /* Reinitialize BAR0 */
    void* bar0 = adapter->getBar0Buffer();
    if (bar0) {
        memset(bar0, 0, VfioUserAdapter::BAR0_SIZE);
        uint32_t* regs = static_cast<uint32_t*>(bar0);
        regs[0] = 0x47545800;  /* "GTX\0" */
        regs[1] = 0x00010000;  /* Version 1.0.0 */
    }

    return 0;
}

/*=============================================================================
 * Constructor / Destructor
 *============================================================================*/

VfioUserAdapter::VfioUserAdapter(const char* socket_path)
    : m_ctx(nullptr)
    , m_socket_path(socket_path)
    , m_bar0_buffer(nullptr)
    , m_bar2_buffer(nullptr)
    , m_running(false)
{
    /* Allocate BAR buffers */
    m_bar0_buffer = mmap(NULL, BAR0_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    m_bar2_buffer = mmap(NULL, BAR2_SIZE, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (m_bar0_buffer == MAP_FAILED || m_bar2_buffer == MAP_FAILED) {
        VFIO_LOG(VFIO_LOG_ERROR) << "[VFIO] Failed to allocate BAR buffers\n";
    }

    /* Initialize BAR0 with device ID */
    if (m_bar0_buffer && m_bar0_buffer != MAP_FAILED) {
        uint32_t* regs = static_cast<uint32_t*>(m_bar0_buffer);
        regs[0] = 0x47545800;  /* "GTX\0" */
        regs[1] = 0x00010000;  /* Version 1.0.0 */
    }
}

VfioUserAdapter::~VfioUserAdapter()
{
    stop();

    if (m_bar0_buffer && m_bar0_buffer != MAP_FAILED) {
        munmap(m_bar0_buffer, BAR0_SIZE);
    }
    if (m_bar2_buffer && m_bar2_buffer != MAP_FAILED) {
        munmap(m_bar2_buffer, BAR2_SIZE);
    }
}

/*=============================================================================
 * Device Setup
 *============================================================================*/

int VfioUserAdapter::setupDevice()
{
    int ret;

    /* Create vfio-user context */
    m_ctx = vfu_create_ctx(VFU_TRANS_SOCK, m_socket_path.c_str(), 0, this,
                           VFU_DEV_TYPE_PCI);
    if (!m_ctx) {
        VFIO_LOG(VFIO_LOG_ERROR) << "[VFIO] Failed to create context: " << strerror(errno) << "\n";
        return -1;
    }

    /* Setup logging - use syslog levels here since libvfio-user expects them */
    vfu_setup_log(m_ctx, [](vfu_ctx_t*, int level, const char* msg) {
        /* level here is syslog level (LOG_ERR=3, LOG_WARNING=4, LOG_INFO=6, LOG_DEBUG=7) */
        if (level <= 4) {  /* LOG_WARNING or more severe */
            VFIO_LOG(VFIO_LOG_WARNING) << "[VFIO-LIB] " << msg << "\n";
        } else {
            VFIO_LOG(VFIO_LOG_DEBUG) << "[VFIO-LIB] " << msg << "\n";
        }
    }, 7);  /* LOG_DEBUG = 7 in syslog.h */

    /* Initialize PCI configuration space - MUST be called before vfu_pci_set_id() */
    ret = vfu_pci_init(m_ctx, VFU_PCI_TYPE_CONVENTIONAL,
                       PCI_HEADER_TYPE_NORMAL, 0);
    if (ret < 0) {
        VFIO_LOG(VFIO_LOG_ERROR) << "[VFIO] Failed to init PCI: " << strerror(errno) << "\n";
        vfu_destroy_ctx(m_ctx);
        m_ctx = nullptr;
        return -1;
    }

    /* Setup PCI config */
    vfu_pci_set_id(m_ctx, VENDOR_ID, DEVICE_ID, VENDOR_ID, 0x0001);
    vfu_pci_set_class(m_ctx, CLASS_CODE >> 16, (CLASS_CODE >> 8) & 0xFF, CLASS_CODE & 0xFF);

    /* Setup device reset callback */
    vfu_setup_device_reset_cb(m_ctx, deviceResetCb);

    /* Setup BAR0 - Registers */
    ret = vfu_setup_region(m_ctx, VFU_PCI_DEV_BAR0_REGION_IDX,
                           BAR0_SIZE, bar0AccessCb,
                           VFU_REGION_FLAG_RW, NULL, 0, -1, 0);
    if (ret < 0) {
        VFIO_LOG(VFIO_LOG_ERROR) << "[VFIO] Failed to setup BAR0: " << strerror(errno) << "\n";
        vfu_destroy_ctx(m_ctx);
        m_ctx = nullptr;
        return -1;
    }

    /* Setup BAR2 - Shared memory (mappable for DMA) */
    ret = vfu_setup_region(m_ctx, VFU_PCI_DEV_BAR2_REGION_IDX,
                           BAR2_SIZE, bar2AccessCb,
                           VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM,
                           NULL, 0, -1, 0);
    if (ret < 0) {
        VFIO_LOG(VFIO_LOG_ERROR) << "[VFIO] Failed to setup BAR2: " << strerror(errno) << "\n";
        vfu_destroy_ctx(m_ctx);
        m_ctx = nullptr;
        return -1;
    }

    /* Setup MSI (32 vectors as per CLAUDE.md specification) */
    ret = vfu_setup_device_nr_irqs(m_ctx, VFU_DEV_MSI_IRQ, 32);
    if (ret < 0) {
        VFIO_LOG(VFIO_LOG_ERROR) << "[VFIO] Failed to setup MSI: " << strerror(errno) << "\n";
        vfu_destroy_ctx(m_ctx);
        m_ctx = nullptr;
        return -1;
    }

    /* Realize the device */
    ret = vfu_realize_ctx(m_ctx);
    if (ret < 0) {
        VFIO_LOG(VFIO_LOG_ERROR) << "[VFIO] Failed to realize device: " << strerror(errno) << "\n";
        vfu_destroy_ctx(m_ctx);
        m_ctx = nullptr;
        return -1;
    }

    VFIO_LOG(VFIO_LOG_INFO) << "[VFIO] Device setup complete\n";
    VFIO_LOG(VFIO_LOG_INFO) << "[VFIO] Vendor:Device = " << std::hex << VENDOR_ID << ":" << DEVICE_ID << "\n";
    VFIO_LOG(VFIO_LOG_INFO) << "[VFIO] BAR0: " << std::dec << (BAR0_SIZE / 1024) << " KB\n";
    VFIO_LOG(VFIO_LOG_INFO) << "[VFIO] BAR2: " << (BAR2_SIZE / (1024*1024)) << " MB\n";

    return 0;
}

/*=============================================================================
 * Run Loop
 *============================================================================*/

void VfioUserAdapter::runLoop()
{
    VFIO_LOG(VFIO_LOG_INFO) << "[VFIO] Waiting for connection on " << m_socket_path << "...\n";

    /* Attach and wait for client */
    int ret = vfu_attach_ctx(m_ctx);
    if (ret < 0) {
        VFIO_LOG(VFIO_LOG_ERROR) << "[VFIO] Failed to attach: " << strerror(errno) << "\n";
        return;
    }

    VFIO_LOG(VFIO_LOG_INFO) << "[VFIO] Client connected!\n";

    /* Event loop */
    while (m_running) {
        ret = vfu_run_ctx(m_ctx);
        if (ret < 0) {
            if (errno == EAGAIN) continue;
            if (errno == ENOTCONN) {
                VFIO_LOG(VFIO_LOG_INFO) << "[VFIO] Client disconnected\n";
                break;
            }
            VFIO_LOG(VFIO_LOG_ERROR) << "[VFIO] vfu_run_ctx error: " << strerror(errno) << "\n";
            break;
        }
    }
}

/*=============================================================================
 * Public Methods
 *============================================================================*/

int VfioUserAdapter::start()
{
    if (m_running) return 0;

    /* Remove existing socket */
    unlink(m_socket_path.c_str());

    /* Setup device */
    if (setupDevice() < 0) {
        return -1;
    }

    m_running = true;

    /* Start thread */
    m_thread = std::thread(&VfioUserAdapter::runLoop, this);

    return 0;
}

void VfioUserAdapter::stop()
{
    if (!m_running) return;

    m_running = false;

    if (m_thread.joinable()) {
        m_thread.join();
    }

    if (m_ctx) {
        vfu_destroy_ctx(m_ctx);
        m_ctx = nullptr;
    }

    unlink(m_socket_path.c_str());
}

void VfioUserAdapter::triggerMSI(int vector)
{
    if (!m_ctx || !m_running) return;

    std::lock_guard<std::mutex> lock(m_mutex);
    int ret = vfu_irq_trigger(m_ctx, vector);
    if (ret < 0) {
        VFIO_LOG(VFIO_LOG_WARNING) << "[VFIO] Failed to trigger MSI vector " << vector << "\n";
    } else {
        VFIO_LOG(VFIO_LOG_DEBUG) << "[VFIO] Triggered MSI vector " << vector << "\n";
    }
}

} /* namespace riscv_tlm::peripherals */

#endif /* USE_VFIO_USER */
