

#ifndef __MEMORY_H__
#define __MEMORY_H__

#include <iostream>
#include <fstream>
#include <elfio/elfio.hpp>

#define SC_INCLUDE_DYNAMIC_PROCESSES
#include "systemc"
#include "Performance.h"

#include "tlm.h"
#include "tlm_utils/simple_target_socket.h"
#include "config.h"


namespace riscv_tlm {
/**
 * @brief Basic TLM-2 memory
 */
    class Memory : sc_core::sc_module {

    struct Section {
        uint64_t size;
        uint64_t type_size;      // SHT_PROGBITS, SHT_NOBITS, ...
        uint64_t address;        // (VMA)
        uint64_t offset_align;   // (alignment)
    };
    public:
        // TLM-2 socket, defaults to 32-bits wide, base protocol
        tlm_utils::simple_target_socket<Memory> socket;
        tlm_utils::simple_target_socket<Memory> socket1;
        tlm_utils::simple_target_socket<Memory> socket2[TMU_NUM];
        tlm_utils::simple_target_socket<Memory> socket3;//for virtual machine
        
        enum {
            SIZE = MAIN_SIZE
        };
        const sc_core::sc_time LATENCY;

        Memory(sc_core::sc_module_name const &name, std::string const &filename);

        explicit Memory(const sc_core::sc_module_name &name);

        ~Memory() override;

        std::string get_file_extension(const std::string& filename);

        /**
         * @brief Returns Program Counter read from hexfile
         * @return Initial PC
         */
        virtual std::uint64_t getPCfromHEX();

        // TLM-2 blocking transport methoda
        virtual void b_transport(tlm::tlm_generic_payload &trans,
                                 sc_core::sc_time &delay);

        // *********************************************
        // TLM-2 forward DMI method
        // *********************************************
        virtual bool get_direct_mem_ptr(tlm::tlm_generic_payload &trans,
                                        tlm::tlm_dmi &dmi_data);

        // *********************************************
        // TLM-2 debug transport method
        // *********************************************
        virtual unsigned int transport_dbg(tlm::tlm_generic_payload &trans);

        uint64_t heap_start;

        // custom
        void dump_memtofile(std::uint64_t dump_st_addr, std::uint64_t dump_size, bool addr_dump);

        void readDataFile(std::string const &filename, std::uint64_t address);

        bool is_os;

    private:

        /**
         * @brief Memory array in bytes
         */
        std::array<uint8_t, MAIN_SIZE> mem{};

        /*
            elf reader
        */
        ELFIO::elfio elf_reader;

        bool is_fail = false;

        /**
         * @brief Program counter (PC) read from hex file
         */
        std::uint64_t program_counter;

        bool dmi_allowed;

        Performance *perf;

        /**
         * @brief Read Intel hex file
         * @param filename file name to read
         */
        
        void readHexFile(const std::string &filename);

        void readElfFile(const std::string &filename);
    };

    class L1MEMORY : sc_core::sc_module {
    public:
        // TLM-2 socket, defaults to 32-bits wide, base protocol
        tlm_utils::simple_target_socket<L1MEMORY> socket_s;
        tlm_utils::simple_target_socket<L1MEMORY> socket_g;

        /*  MBytes */
        enum {
            SIZE = L1_SIZE
        };
        const sc_core::sc_time LATENCY;

        L1MEMORY(sc_core::sc_module_name const &name, std::string const &filename);

        explicit L1MEMORY(const sc_core::sc_module_name &name);

        ~L1MEMORY() override;

        // TLM-2 blocking transport method
        virtual void b_transport(tlm::tlm_generic_payload &trans,
                                 sc_core::sc_time &delay);

        // *********************************************
        // TLM-2 forward DMI method
        // *********************************************
        virtual bool get_direct_mem_ptr(tlm::tlm_generic_payload &trans,
                                        tlm::tlm_dmi &dmi_data);

        // *********************************************
        // TLM-2 debug transport method
        // *********************************************
        virtual unsigned int transport_dbg(tlm::tlm_generic_payload &trans);


        void readDataFile(std::string const &filename, int address);


        // custom
        void dump_memtofile(bool addr_dump, uint64_t addr, uint64_t dump_size);

    private:

        std::array<uint8_t, L1MEMORY::SIZE> mem{};

        bool is_fail = false;

        bool dmi_allowed;

        Performance *perf;


    };
    class L2MEMORY : sc_core::sc_module {
    public:
        // TLM-2 socket, defaults to 32-bits wide, base protocol
        tlm_utils::simple_target_socket<L2MEMORY> socket_nsu;
        tlm_utils::simple_target_socket<L2MEMORY> socket_tmu;
        tlm_utils::simple_target_socket<L2MEMORY> socket_spu[SPU_NUM/TMU_NUM];

        enum {
            SIZE = L2_SIZE
        };
        const sc_core::sc_time LATENCY;

        L2MEMORY(sc_core::sc_module_name const &name, std::string const &filename);

        explicit L2MEMORY(const sc_core::sc_module_name &name);

        ~L2MEMORY() override;



        // TLM-2 blocking transport method
        virtual void b_transport(tlm::tlm_generic_payload &trans,
                                 sc_core::sc_time &delay);

        // *********************************************
        // TLM-2 forward DMI method
        // *********************************************
        virtual bool get_direct_mem_ptr(tlm::tlm_generic_payload &trans,
                                        tlm::tlm_dmi &dmi_data);

        // *********************************************
        // TLM-2 debug transport method
        // *********************************************
        virtual unsigned int transport_dbg(tlm::tlm_generic_payload &trans);


        void readDataFile(std::string const &filename, int address);

        // custom
        void dump_memtofile(bool addr_dump, uint64_t addr, uint64_t dump_size);

    private:

        std::array<uint8_t, L2MEMORY::SIZE> mem{};

        bool is_fail = false;

        bool dmi_allowed;

        Performance *perf;

    };

    class L0MEMORY : sc_core::sc_module {
    public:
        // TLM-2 socket, defaults to 32-bits wide, base protocol
        tlm_utils::simple_target_socket<L0MEMORY> socket_s;

        /*  MBytes */
        enum {
            SIZE = L0_SIZE
        };
        const sc_core::sc_time LATENCY;

        L0MEMORY(sc_core::sc_module_name const &name, std::string const &filename);

        explicit L0MEMORY(const sc_core::sc_module_name &name);

        ~L0MEMORY() override;

        // TLM-2 blocking transport method
        virtual void b_transport(tlm::tlm_generic_payload &trans,
                                 sc_core::sc_time &delay);

        // *********************************************
        // TLM-2 forward DMI method
        // *********************************************
        virtual bool get_direct_mem_ptr(tlm::tlm_generic_payload &trans,
                                        tlm::tlm_dmi &dmi_data);

        // *********************************************
        // TLM-2 debug transport method
        // *********************************************
        virtual unsigned int transport_dbg(tlm::tlm_generic_payload &trans);


        // custom
        void dump_memtofile();

    private:

        std::array<uint8_t, L0MEMORY::SIZE> mem{};

        bool is_fail = false;

        bool dmi_allowed;

        Performance *perf;


    };
}
#endif 
