/**
 * @file VfioUserAdapter.h
 * @brief libvfio-user adapter for GTX simulator
 *
 * Provides PCIe device emulation via vfio-user protocol.
 * This allows guests to see the GTX device via lspci.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef __VFIO_USER_ADAPTER_H__
#define __VFIO_USER_ADAPTER_H__

#ifdef USE_VFIO_USER

#include <cstdint>
#include <functional>
#include <thread>
#include <atomic>
#include <mutex>
#include <string>
#include <sys/types.h>  /* for off_t, ssize_t */

/* Forward declaration to avoid including libvfio-user.h here.
 * This prevents syslog.h macro conflicts from spreading to other files. */
struct vfu_ctx;
typedef struct vfu_ctx vfu_ctx_t;

namespace riscv_tlm::peripherals {

/**
 * @brief Callback type for BAR access
 *
 * @param offset Offset within BAR
 * @param data   Data pointer
 * @param len    Access length
 * @param write  true for write, false for read
 * @return bytes transferred, or -1 on error
 */
using BarAccessCallback = std::function<ssize_t(uint64_t offset, void* data, size_t len, bool write)>;

/**
 * @brief Callback type for interrupt injection
 *
 * @param irq_bits Interrupt bits to set
 */
using IrqInjectCallback = std::function<void(uint32_t irq_bits)>;

/**
 * @brief VfioUserAdapter - bridges vfio-user protocol to simulator
 */
class VfioUserAdapter {
public:
    /* GTX Device configuration */
    static constexpr uint16_t VENDOR_ID = 0x1234;
    static constexpr uint16_t DEVICE_ID = 0x5678;
    static constexpr uint32_t CLASS_CODE = 0x120000;  /* Processing accelerators */

    /* BAR sizes */
    static constexpr size_t BAR0_SIZE = 1 << 20;      /* 1MB - registers */
    static constexpr size_t BAR2_SIZE = 64 << 20;     /* 64MB - shared memory */

    /**
     * @brief Constructor
     * @param socket_path Path for vfio-user socket (e.g., /tmp/gtx-vfio.sock)
     */
    explicit VfioUserAdapter(const char* socket_path);

    /**
     * @brief Destructor
     */
    ~VfioUserAdapter();

    /**
     * @brief Start vfio-user server
     * @return 0 on success, -1 on error
     */
    int start();

    /**
     * @brief Stop vfio-user server
     */
    void stop();

    /**
     * @brief Check if running
     */
    bool isRunning() const { return m_running; }

    /**
     * @brief Set callback for BAR0 access
     */
    void setBar0Callback(BarAccessCallback cb) { m_bar0_cb = std::move(cb); }

    /**
     * @brief Set callback for BAR2 access
     */
    void setBar2Callback(BarAccessCallback cb) { m_bar2_cb = std::move(cb); }

    /**
     * @brief Set callback for interrupt injection from host
     */
    void setIrqInjectCallback(IrqInjectCallback cb) { m_irq_cb = std::move(cb); }

    /**
     * @brief Trigger MSI to host
     * @param vector MSI vector number (0-7)
     */
    void triggerMSI(int vector);

    /**
     * @brief Get BAR0 local buffer (for direct access)
     */
    void* getBar0Buffer() { return m_bar0_buffer; }

    /**
     * @brief Get BAR2 local buffer (for direct access)
     */
    void* getBar2Buffer() { return m_bar2_buffer; }

    /**
     * @brief Check if BAR0 callback is registered
     */
    bool hasBar0Callback() const { return static_cast<bool>(m_bar0_cb); }

    /**
     * @brief Check if BAR2 callback is registered
     */
    bool hasBar2Callback() const { return static_cast<bool>(m_bar2_cb); }

    /**
     * @brief Invoke BAR0 callback
     */
    ssize_t invokeBar0Callback(uint64_t offset, void* data, size_t len, bool write) {
        if (m_bar0_cb) return m_bar0_cb(offset, data, len, write);
        return -1;
    }

    /**
     * @brief Invoke BAR2 callback
     */
    ssize_t invokeBar2Callback(uint64_t offset, void* data, size_t len, bool write) {
        if (m_bar2_cb) return m_bar2_cb(offset, data, len, write);
        return -1;
    }

private:
    /* vfio-user context (opaque pointer) */
    vfu_ctx_t* m_ctx;
    std::string m_socket_path;

    /* BAR buffers */
    void* m_bar0_buffer;
    void* m_bar2_buffer;

    /* Callbacks */
    BarAccessCallback m_bar0_cb;
    BarAccessCallback m_bar2_cb;
    IrqInjectCallback m_irq_cb;

    /* Thread management */
    std::thread m_thread;
    std::atomic<bool> m_running;
    std::mutex m_mutex;

    /* Internal methods */
    int setupDevice();
    void runLoop();
};

} /* namespace riscv_tlm::peripherals */

#endif /* USE_VFIO_USER */

#endif /* __VFIO_USER_ADAPTER_H__ */
