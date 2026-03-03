
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Timer.h"
#include <cstdint>

namespace riscv_tlm::peripherals {

    SC_HAS_PROCESS(Timer);

    Timer::Timer(sc_core::sc_module_name const &name) :
            sc_module(name), socket("timer_socket"), m_mtime(0), m_mtimecmp(0) {
        m_mctrl = 0;
        m_mdiv  = 0;
        m_count = 0;
        cstart = false;
        timer_enable = false;
        notify_time = 0;
        socket.register_b_transport(this, &Timer::b_transport);

        SC_THREAD(run);
    }

    [[noreturn]] void Timer::run() {

        auto *irq_trans = new tlm::tlm_generic_payload;
        sc_core::sc_time delay = sc_core::SC_ZERO_TIME;
        std::uint64_t cause = ((uint64_t)1 << 63) | 0x07;     // Machine timer interrupt
        irq_trans->set_command(tlm::TLM_WRITE_COMMAND);
        irq_trans->set_data_ptr(reinterpret_cast<unsigned char *>(&cause));
        irq_trans->set_data_length(8);
        irq_trans->set_streaming_width(8);
        irq_trans->set_byte_enable_ptr(nullptr);
        irq_trans->set_dmi_allowed(false);
        irq_trans->set_response_status(tlm::TLM_INCOMPLETE_RESPONSE);
        irq_trans->set_address(0);

        bool interrupt_pending = false;

        while (true) {
            if(!cstart){
                wait(timer_event);
            }

            // Always increment mtime when timer is enabled
            if(timer_enable){
                m_mtime += 1;
            }

            // Generate interrupt when mtime >= mtimecmp
            // Only fire ONE interrupt, then wait for software to clear the condition
            if((m_mtime >= m_mtimecmp) && timer_enable){
                if (!interrupt_pending) {
                    irq_line->b_transport(*irq_trans, delay);
                    interrupt_pending = true;
                }
            } else {
                // Condition cleared (software updated mtimecmp or reset mtime)
                interrupt_pending = false;
            }

            sc_core::wait(sc_core::sc_time(10,sc_core::SC_NS));
        }
    }

    void Timer::b_transport(tlm::tlm_generic_payload &trans,
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
                case TIMERCTRL_MEMORY_ADDRESS:
                    m_mctrl.range(1,0) = aux_value;
                    if(aux_value & 0x1 == 0x1) {
                        timer_enable = true;
                        printf("timer enabled!\n");
                    }else{
                        timer_enable = false;
                    }
                    break;
                case TIMERDIV_MEMORY_ADDRESS:
                    m_mdiv.range(9,0) = aux_value;
                    break;
                case TIMER_MEMORY_ADDRESS_LO:
                    m_mtime.range(31, 0) = aux_value;
                    break;
                case TIMER_MEMORY_ADDRESS_HI:
                    m_mtime.range(63, 32) = aux_value;
                    break;
                case TIMERCMP_MEMORY_ADDRESS_LO:
                    m_mtimecmp.range(31, 0) = aux_value;
                    break;
                case TIMERCMP_MEMORY_ADDRESS_HI:
                    m_mtimecmp.range(63, 32) = aux_value;
                    notify_time = (m_mtimecmp - m_mtime)*(m_mdiv + 1);

                    //std::uint64_t notify_time;
                    // notify needs relative time, mtimecmp works in absolute time
                    //notify_time = (m_mtimecmp - m_mtime)*(m_mdiv+1);
                    //timer_event.notify(sc_core::sc_time(notify_time, sc_core::SC_NS));
                    //timer_event.notify(sc_core::sc_time::from_value(notify_time));
                    cstart = true;
                    timer_event.notify(sc_core::sc_time::from_value(1));
                    

                    break;
                default:
                    trans.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
                    return;
            }
        } else { // TLM_READ_COMMAND
            switch (addr) {
                case TIMERCTRL_MEMORY_ADDRESS:
                    aux_value = m_mctrl.range(1,0);
                    break;
                case TIMERDIV_MEMORY_ADDRESS:
                    aux_value = m_mdiv.range(9,0);
                    break;
                case TIMER_MEMORY_ADDRESS_LO:
                    aux_value = m_mtime.range(31, 0);
                    break;
                case TIMER_MEMORY_ADDRESS_HI:
                    aux_value = m_mtime.range(63, 32);
                    break;
                case TIMERCMP_MEMORY_ADDRESS_LO:
                    aux_value = m_mtimecmp.range(31, 0);
                    break;
                case TIMERCMP_MEMORY_ADDRESS_HI:
                    aux_value = m_mtimecmp.range(63, 32);
                    break;
                default:
                    trans.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
                    return;
            }
            memcpy(ptr, &aux_value, len);
        }

        trans.set_response_status(tlm::TLM_OK_RESPONSE);
    }
}