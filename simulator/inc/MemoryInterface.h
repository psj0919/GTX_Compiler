

#ifndef INC_MEMORYINTERFACE_H_
#define INC_MEMORYINTERFACE_H_

#include <cstdint>
#include "systemc"

#include "tlm.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/tlm_quantumkeeper.h"

#include "memory.h"
#include <cstdint>

namespace riscv_tlm {

/**
 * @brief Memory Interface
 */
    class MemoryInterface {
    
    public:

        tlm_utils::simple_initiator_socket<MemoryInterface> data_bus;

        MemoryInterface(sc_core::sc_module_name const &name);

        std::uint32_t readDataMem(std::uint64_t addr, int size);

        void writeDataMem(std::uint64_t addr, std::uint32_t data, int size);

    protected:
        bool is_failed = false;
        
    };

    class CPUMemintf: public MemoryInterface{
        struct WriteIntf{
            uint64_t addr;
            uint32_t data;
            int size;

            bool operator==(const WriteIntf& other) const {
                return addr == other.addr && data == other.data && size == other.size;
            }
        };
        public:
            CPUMemintf(sc_core::sc_module_name const &name);

            void writeDataMem(std::uint64_t addr, std::uint32_t data, int size);

            bool WriteMonitor();

            void reset_vector();

            bool monitoring;

            bool fst_routine;

            uint64_t high_stack;
            
            uint64_t low_stack;

        private:
            std::vector<WriteIntf> write_vect;
            std::vector<WriteIntf> write_ref;

    };
}
#endif /* INC_MEMORYINTERFACE_H_ */
