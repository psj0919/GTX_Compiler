
#ifndef __BUSCTRL_H__
#define __BUSCTRL_H__

#include <iostream>
#include <fstream>

#define SC_INCLUDE_DYNAMIC_PROCESSES

#include "systemc"

#include "tlm.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"

namespace riscv_tlm {

/**
 * Memory mapped Trace peripheral address
 */
//#define TRACE_MEMORY_ADDRESS 0x40000000

#define TIMERCTRL_MEMORY_ADDRESS   0x600000
#define TIMERDIV_MEMORY_ADDRESS    0x600004
#define TIMER_MEMORY_ADDRESS_LO    0x600008
#define TIMER_MEMORY_ADDRESS_HI    0x60000c
#define TIMERCMP_MEMORY_ADDRESS_LO 0x600010
#define TIMERCMP_MEMORY_ADDRESS_HI 0x600014


#define VIRTUAL_DEVICE_COMMAND_LO  0x40000000
#define VIRTUAL_DEVICE_COMMAND_HI  0x40000004
#define VIRTUAL_DEVICE_FILEDESC_LO 0x40000008
#define VIRTUAL_DEVICE_FILEDESC_HI 0x4000000C
#define VIRTUAL_DEVICE_OPENFLAG_LO 0x40000010
#define VIRTUAL_DEVICE_OPENFLAG_HI 0x40000014
#define VIRTUAL_DEVICE_ADDRESS_LO  0x40000018
#define VIRTUAL_DEVICE_ADDRESS_HI  0x4000001C
#define VIRTUAL_DEVICE_LENGTH_LO   0x40000020
#define VIRTUAL_DEVICE_LENGTH_HI   0x40000024
#define VIRTUAL_DEVICE_PARAM0_LO   0x40000028
#define VIRTUAL_DEVICE_PARAM0_HI   0x4000002C
#define VIRTUAL_DEVICE_PARAM1_LO   0x40000030
#define VIRTUAL_DEVICE_PARAM1_HI   0x40000034
#define VIRTUAL_DEVICE_RESULT_LO   0x40000038
#define VIRTUAL_DEVICE_RESULT_HI   0x4000003C
#define VIRTUAL_DEVICE_RESULT1_LO  0x40000040
#define VIRTUAL_DEVICE_RESULT1_HI  0x40000044
#define VIRTUAL_DEVICE_LAST_LO     0x40000048
#define VIRTUAL_DEVICE_LAST_HI     0x4000004C

#define PCIE_DBI_BASE              0x40000000
#define PCIE_DBI_MSI               0x40200d70

#define PCIE_CFG_BASE              0x40A00000
#define PCIE_CFG_INTERRUPT_FLAG    0x40A00330
#define PCIE_CFG_INTERRUPT_MASK    0x40A0033c
#define PCIE_CFG_MASKED_STATUS     0x40A00340
#define PCIE_INTERRUPT             0x40A00e40



/**
 * @brief Simple bus controller
 *
 * This module manages instructon & data bus. It has 2 target ports,
 * cpu_instr_socket and cpu_data_socket that receives accesses from CPU and
 * has 2 initiator ports to access main Memory and Trace module.
 * It will be expanded with more ports when required (for DMA,
 * other peripherals, etc.)
 */
    class BusCtrl : sc_core::sc_module {
    public:
        /**
         * @brief TLM target socket CPU instruction memory bus
         */
        tlm_utils::simple_target_socket<BusCtrl> cpu_instr_socket;

        /**
         * @brief TLM target socket CPU data memory bus
         */
        tlm_utils::simple_target_socket<BusCtrl> cpu_data_socket;

        /**
         * @brief TLM initiator socket Main memory bus
         */
        tlm_utils::simple_initiator_socket<BusCtrl> memory_socket;

        tlm_utils::simple_target_socket<BusCtrl> PCIE_data_socket;

        /**
         * @brief TLM initiator socket Trace module
         */
        //tlm_utils::simple_initiator_socket<BusCtrl> trace_socket;

        /**
         * @brief TLM initiator socket Trace module
         */
        tlm_utils::simple_initiator_socket<BusCtrl> timer_socket;

        tlm_utils::simple_initiator_socket<BusCtrl> PCIE_socket;


        tlm_utils::simple_initiator_socket<BusCtrl> vm_socket;

        /**
         * @brief constructor
         * @param name module's name
         */
        explicit BusCtrl(sc_core::sc_module_name const &name);

        /**
         * @brief TLM-2 blocking mechanism
         * @param trans transtractino to perform
         * @param delay delay associated to this transaction
         */
        virtual void b_transport(tlm::tlm_generic_payload &trans,
                                 sc_core::sc_time &delay);


        bool is_vmenable;

    private:
        bool instr_direct_mem_ptr(tlm::tlm_generic_payload &,
                                  tlm::tlm_dmi &dmi_data);

        void invalidate_direct_mem_ptr(sc_dt::uint64 start, sc_dt::uint64 end);
    };
}
#endif
