
#include "Memory.h"
#include "config.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

namespace riscv_tlm {

    SC_HAS_PROCESS(Memory);

    Memory::Memory(sc_core::sc_module_name const &name, std::string const &filename) :
            sc_module(name), socket((std::string(name) + "_socket").c_str()), LATENCY(sc_core::SC_ZERO_TIME) {
        // Register callbacks for incoming interface method calls
        socket.register_b_transport(this, &Memory::b_transport);
        socket.register_get_direct_mem_ptr(this, &Memory::get_direct_mem_ptr);
        socket.register_transport_dbg(this, &Memory::transport_dbg);

        socket1.register_b_transport(this, &Memory::b_transport);
        socket1.register_get_direct_mem_ptr(this, &Memory::get_direct_mem_ptr);
        socket1.register_transport_dbg(this, &Memory::transport_dbg);

        for(int i = 0; i < TMU_NUM; i++){
            socket2[i].register_b_transport(this, &Memory::b_transport);
            socket2[i].register_get_direct_mem_ptr(this, &Memory::get_direct_mem_ptr);
            socket2[i].register_transport_dbg(this, &Memory::transport_dbg);
        }

        socket3.register_b_transport(this, &Memory::b_transport);
        socket3.register_get_direct_mem_ptr(this, &Memory::get_direct_mem_ptr);
        socket3.register_transport_dbg(this, &Memory::transport_dbg);



        dmi_allowed = false;
        perf = Performance::getInstance();
        program_counter = 0;
        if((get_file_extension(filename) == "elf") || (get_file_extension(filename) == "exe") || (get_file_extension(filename) == "o")){
            readElfFile(filename);
            //LOG(LOG_INFO) << "[INFO] Simple kernel = ON\n";
            //is_os = true;
            is_os = false;
        }else {
            readHexFile(filename);
            is_os = false;
        }
    }

    Memory::Memory(sc_core::sc_module_name const &name) :
            sc_module(name), socket((std::string(name) + "_socket").c_str()), LATENCY(sc_core::SC_ZERO_TIME) {
        socket.register_b_transport(this, &Memory::b_transport);
        socket.register_get_direct_mem_ptr(this, &Memory::get_direct_mem_ptr);
        socket.register_transport_dbg(this, &Memory::transport_dbg);

        socket1.register_b_transport(this, &Memory::b_transport);
        socket1.register_get_direct_mem_ptr(this, &Memory::get_direct_mem_ptr);
        socket1.register_transport_dbg(this, &Memory::transport_dbg);

        for(int i = 0; i < TMU_NUM; i++){
            socket2[i].register_b_transport(this, &Memory::b_transport);
            socket2[i].register_get_direct_mem_ptr(this, &Memory::get_direct_mem_ptr);
            socket2[i].register_transport_dbg(this, &Memory::transport_dbg);
        }

        socket3.register_b_transport(this, &Memory::b_transport);
        socket3.register_get_direct_mem_ptr(this, &Memory::get_direct_mem_ptr);
        socket3.register_transport_dbg(this, &Memory::transport_dbg);
        


	    dmi_allowed = false;
        perf = Performance::getInstance();
        program_counter = 0;

    }


    Memory::~Memory() = default;

    std::string Memory::get_file_extension(const std::string& filename) {
        auto pos = filename.find_last_of('.');
        if (pos == std::string::npos) return "";
    
        std::string ext = filename.substr(pos + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower); 
        return ext;
    }

    std::uint64_t Memory::getPCfromHEX() {
        return program_counter;

    }

    void Memory::dump_memtofile(std::uint64_t dump_st_addr, std::uint64_t dump_size, bool addr_dump){
        std::ofstream ofs(main_dfname);
        if(!ofs){
            LOG(LOG_ERROR) << "[ERROR] MEMORY DUMP FILE OPEN ERROR \n";
        }
        
        dump_st_addr = dump_st_addr & (~0x1F);
        uint64_t max_size = dump_st_addr+dump_size;
        max_size = (max_size > mem.size())? mem.size() : max_size;
        for (size_t base = dump_st_addr; base < max_size; base += 32) {
        if(addr_dump){
            ofs <<"ADDR:"<< std::hex << std::setw(8) << std::setfill('0') << base << " || ";
        }
        size_t end = std::min(base + 32, max_size);
        for (size_t i = 0; i < end - base; i++) {
            size_t j = end -1 -i;
            ofs << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(mem[j]);
        }
        ofs << "\n";
    }
        ofs.close();
    }

    void Memory::b_transport(tlm::tlm_generic_payload &trans,
                             sc_core::sc_time &delay) {
        tlm::tlm_command cmd = trans.get_command();
        sc_dt::uint64 adr = trans.get_address();
        unsigned char *ptr = trans.get_data_ptr();
        unsigned int len = trans.get_data_length();
        unsigned char *byt = trans.get_byte_enable_ptr();
        unsigned int wid = trans.get_streaming_width();

        // *********************************************
        // Generate the appropriate error response
        // *********************************************
        
        if (adr >= sc_dt::uint64(Memory::SIZE)) {
            trans.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
            if(!is_fail) {
                LOG(LOG_ERROR) << "[ERROR] Memory Access out of Main Memory range\n";
                is_fail = true;
            }
            return;
        }
        if (byt != nullptr) {
            trans.set_response_status(tlm::TLM_BYTE_ENABLE_ERROR_RESPONSE);
            if(!is_fail) {
                LOG(LOG_ERROR) << "[ERROR] Byte Error at Main Memory Access\n";
                is_fail = true;
            }
            return;
        }
        if (len > 4 || wid < len) {
            trans.set_response_status(tlm::TLM_BURST_ERROR_RESPONSE);
            if(!is_fail) {
                LOG(LOG_ERROR) << "[ERROR] Burst Error occured at Main Memory Access\n";
                is_fail = true;
            }
            return;
        }
        

        // Obliged to implement read and write commands
        if (cmd == tlm::TLM_READ_COMMAND) {
            std::copy_n(mem.cbegin() + adr, len, ptr);
            perf->dataMemoryRbyte(len);
        } else if (cmd == tlm::TLM_WRITE_COMMAND) {
            std::copy_n(ptr, len, mem.begin() + adr);
            perf->dataMemoryWbyte(len);
        }

        // Illustrates that b_transport may block
        //sc_core::wait(delay);

        // Reset timing annotation after waiting
        delay = sc_core::SC_ZERO_TIME;

        // *********************************************
        // Set DMI hint to indicated that DMI is supported
        // *********************************************
        trans.set_dmi_allowed(dmi_allowed);

        // Obliged to set response status to indicate successful completion
        trans.set_response_status(tlm::TLM_OK_RESPONSE);
    }

    bool Memory::get_direct_mem_ptr(tlm::tlm_generic_payload &trans,
                                    tlm::tlm_dmi &dmi_data) {

        (void) trans;

        if (!dmi_allowed) {
            return false;
        }

        // Permit read and write access
        dmi_data.allow_read_write();

        // Set other details of DMI region
        dmi_data.set_dmi_ptr(reinterpret_cast<unsigned char *>(&mem[0]));
        dmi_data.set_start_address(0);
        dmi_data.set_end_address(Memory::SIZE * 4 - 1);
        dmi_data.set_read_latency(LATENCY);
        dmi_data.set_write_latency(LATENCY);

        return true;
    }

    unsigned int Memory::transport_dbg(tlm::tlm_generic_payload &trans) {
        tlm::tlm_command cmd = trans.get_command();
        sc_dt::uint64 adr = trans.get_address();
        unsigned char *ptr = trans.get_data_ptr();
        unsigned int len = trans.get_data_length();

        if (adr >= sc_dt::uint64(Memory::SIZE)) {
            trans.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
            return 0;
        }

        // Calculate the number of bytes to be actually copied
        unsigned int num_bytes = (len < (Memory::SIZE - adr) * 4) ? len : (Memory::SIZE - adr) * 4;

        if (cmd == tlm::TLM_READ_COMMAND) {
            std::copy_n(mem.cbegin() + adr, len, ptr);
        } else if (cmd == tlm::TLM_WRITE_COMMAND) {
            std::copy_n(ptr, len, mem.begin() + adr);
        }

        return num_bytes;
    }
    void Memory::readDataFile(std::string const &filename, uint64_t address) {
        std::ifstream hexfile;
        std::string line;
        std::uint32_t memory_offset = 0;

        hexfile.open(filename);
        int t =0;;
        if (hexfile.is_open()) {

            while (getline(hexfile, line)) {
                if (line.empty()) continue;
                
                if (line[0] == '@') {
                    dfile_addr_set = true;
                    std::string addr_str = line.substr(1);
                    addr_str.erase(std::remove(addr_str.begin(), addr_str.end(), '_'), addr_str.end());
                    address = std::stoull(addr_str, nullptr, 16);
                    t = 0;
                    continue;
                }
                if(!dfile_addr_set){
                        LOG(LOG_WARNING) << "[WARNING] The load address for the data file has not been specified yet.(DATA LOAD FAILED) \n";
                        return;
                }
                for(int i = line.length()-2 ; i >= 0 ; i-=2 ){
                    mem[address + t] = static_cast<char>(std::stoi(line.substr(i,2),nullptr, 16));
                    t++;
                }
            }
            hexfile.close();

            if (memory_offset != 0) {
                dmi_allowed = false;
            } else {
                dmi_allowed = true;
            }

        } else {
            LOG(LOG_WARNING) <<"[WARNING] MEM DATA"<< " File open error\n";
        }
    }
    void Memory::readElfFile(std::string const &filename){
        if (!elf_reader.load(filename)) {
            LOG(LOG_ERROR) << "[ERROR] FAILED to open Program File. (plz check file path!!)\n";
            is_boot_success = false;
        return ;
        }
        for (const std::unique_ptr<ELFIO::section>& sec_ptr : elf_reader.sections) {
            const ELFIO::section* sec = sec_ptr.get();
        if (sec->get_size() == 0 || sec->get_address() == 0) continue; // bss
        if(sec->get_name() == ".bss"){
            heap_start = (sec->get_address() + sec->get_size());
        }
        }
        for (int i = 0; i < elf_reader.segments.size(); ++i) {
        const ELFIO::segment* seg = elf_reader.segments[i];
        if (seg->get_type() != ELFIO::PT_LOAD) continue;  // Only loadable segments

        uint64_t address = seg->get_virtual_address();
        uint64_t size = seg->get_file_size();

        if (address + size <= Memory::SIZE) {
            const char* data = seg->get_data();
            if (data != nullptr) {
                std::copy(data, data + size, mem.begin() + address);
            }
        } else {
            LOG(LOG_WARNING) << "[WARNING] DATA LOADED OVER MEMORY AREA!\nSegment " << i << "\n";
            return;
        }
    }

    

        program_counter = elf_reader.get_entry();
        LOG(LOG_INFO) << "[INFO] PC set : 0x" << std::hex
                                  << program_counter << std::dec << std::endl;

    

    return ;

    }

    void Memory::readHexFile(std::string const &filename) {
        std::ifstream hexfile;
        std::string line;
        std::uint32_t memory_offset = 0;
        bool is_entry = false;

        hexfile.open(filename);

        if (hexfile.is_open()) {
            std::uint32_t extended_address = 0;

            while (getline(hexfile, line)) {
                if (line[0] == ':') {
                    if (line.substr(7, 2) == "00") {
                        /* Data */
                        int byte_count;
                        std::uint32_t address;
                        byte_count = std::stoi(line.substr(1, 2), nullptr, 16);
                        address = std::stoi(line.substr(3, 4), nullptr, 16);
                        address = address + extended_address + memory_offset;

                        for (int i = 0; i < byte_count; i++) {
                            mem[address + i] = stol(line.substr(9 + (i * 2), 2),
                                                    nullptr, 16);
                        }
                    } else if (line.substr(7, 2) == "02") {
                        /* Extended segment address */
                        extended_address = stol(line.substr(9, 4), nullptr, 16)
                                           * 16;
                        //LOG(LOG_INFO) << "[INFO] 02 extended address 0x" << std::hex
                        //          << extended_address << std::dec << std::endl;
                    } else if (line.substr(7, 2) == "03") {
                        /* Start segment address */
                        std::uint32_t code_segment;
                        code_segment = stol(line.substr(9, 4), nullptr, 16) * 16; /* ? */
                        program_counter = stol(line.substr(13, 4), nullptr, 16);
                        program_counter = program_counter + code_segment;
                        LOG(LOG_INFO) << "[INFO] PC set : 0x" << std::hex
                                  << program_counter << std::dec << std::endl;
                        is_entry = true;
                    } else if (line.substr(7, 2) == "04") {
                        /* Start segment address */
                        memory_offset = stol(line.substr(9, 4), nullptr, 16) << 16;
                        extended_address = 0;
                        //LOG(LOG_INFO) << "[INFO] address set to 0x" << std::hex
                        //          << extended_address << std::dec << std::endl;
                        //LOG(LOG_INFO) << "[INFO] offset set to 0x" << std::hex
                        //          << memory_offset << std::dec << std::endl;
                    } else if (line.substr(7, 2) == "05") {
                        program_counter = stol(line.substr(9, 8), nullptr, 16);
                        LOG(LOG_INFO) << "[INFO] 05 PC set : 0x" << std::hex
                                  << program_counter << std::dec << std::endl;
                        is_entry = true;
                    }
                }
            }
            hexfile.close();
            if(!is_entry){
                LOG(LOG_ERROR) << "[ERROR] FAILED to set Entry point.... please check elf file \n"
                          << "[ERROR] END Simulation....\n\n";
                exit(1);
            }

            if (memory_offset != 0) {
                dmi_allowed = false;
            } else {
                dmi_allowed = true;
            }

        } else {
            LOG(LOG_ERROR) << "[ERROR] FAILED to open File!!\n";
            if (sc_core::sc_is_running() && !sc_core::sc_end_of_simulation_invoked()) {
                sc_core::sc_stop();
            }
        }
    }
/////////L2MEMORY


    L2MEMORY::L2MEMORY(sc_core::sc_module_name const &name, std::string const &filename) :
            sc_module(name), socket_nsu((std::string(name) + "_socket").c_str()), LATENCY(sc_core::SC_ZERO_TIME) {
        // Register callbacks for incoming interface method calls
        socket_nsu.register_b_transport(this, &L2MEMORY::b_transport);
        socket_nsu.register_get_direct_mem_ptr(this, &L2MEMORY::get_direct_mem_ptr);
        socket_nsu.register_transport_dbg(this, &L2MEMORY::transport_dbg);

        socket_tmu.register_b_transport(this, &L2MEMORY::b_transport);
        socket_tmu.register_get_direct_mem_ptr(this, &L2MEMORY::get_direct_mem_ptr);
        socket_tmu.register_transport_dbg(this, &L2MEMORY::transport_dbg);

        for(int i = 0 ; i < SPU_NUM  ;i++){
        socket_spu[i].register_b_transport(this, &L2MEMORY::b_transport);
        socket_spu[i].register_get_direct_mem_ptr(this, &L2MEMORY::get_direct_mem_ptr);
        socket_spu[i].register_transport_dbg(this, &L2MEMORY::transport_dbg);
        }
        dmi_allowed = false;
        perf = Performance::getInstance();
    }

    L2MEMORY::L2MEMORY(sc_core::sc_module_name const &name) :
            sc_module(name), socket_nsu((std::string(name) + "_socket").c_str()), LATENCY(sc_core::SC_ZERO_TIME) {
                
        socket_nsu.register_b_transport(this, &L2MEMORY::b_transport);
        socket_nsu.register_get_direct_mem_ptr(this, &L2MEMORY::get_direct_mem_ptr);
        socket_nsu.register_transport_dbg(this, &L2MEMORY::transport_dbg);

        socket_tmu.register_b_transport(this, &L2MEMORY::b_transport);
        socket_tmu.register_get_direct_mem_ptr(this, &L2MEMORY::get_direct_mem_ptr);
        socket_tmu.register_transport_dbg(this, &L2MEMORY::transport_dbg);

        for(int i = 0 ; i < (SPU_NUM/TMU_NUM)  ;i++){
        socket_spu[i].register_b_transport(this, &L2MEMORY::b_transport);
        socket_spu[i].register_get_direct_mem_ptr(this, &L2MEMORY::get_direct_mem_ptr);
        socket_spu[i].register_transport_dbg(this, &L2MEMORY::transport_dbg);
        }
	    dmi_allowed = false;
        perf = Performance::getInstance();
        
    }
    

    L2MEMORY::~L2MEMORY() = default;


    void L2MEMORY::dump_memtofile(bool addr_dump, uint64_t addr, uint64_t dump_size){
        std::ofstream ofs(std::string(this->basename())+"_DUMP.hex");
        if(!ofs){
            LOG(LOG_WARNING) << "[WARNING] MEMORY DUMP FILE OPEN ERR \n";
        }
        
        addr = addr & (~0x1F);
        uint64_t max_size = addr+dump_size;
        max_size = (max_size > mem.size())? mem.size() : max_size;
        for (size_t base = addr; base < max_size; base += 32) {
        if(addr_dump){
            ofs <<"ADDR:"<< std::hex << std::setw(8) << std::setfill('0') << base << " || ";
        }
        size_t end = std::min(base + 32, max_size);
        for (size_t i = 0; i < end - base; i++) {
            size_t j = end -1 -i;
            ofs << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(mem[j]);
        }
        ofs << "\n";
    }
        ofs.close();
    }

    void L2MEMORY::b_transport(tlm::tlm_generic_payload &trans,
                             sc_core::sc_time &delay) {
        tlm::tlm_command cmd = trans.get_command();
        sc_dt::uint64 adr = trans.get_address();
        unsigned char *ptr = trans.get_data_ptr();
        unsigned int len = trans.get_data_length();
        unsigned char *byt = trans.get_byte_enable_ptr();
        unsigned int wid = trans.get_streaming_width();

        // *********************************************
        // Generate the appropriate error response
        // *********************************************
        if (adr >= sc_dt::uint64(L2MEMORY::SIZE)) {
            trans.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
            if(!is_fail) {
                LOG(LOG_ERROR) << "[ERROR] Memory Access out of L2 Memory range\n";
                is_fail = true;
            }
            return;
        }
        if (byt != nullptr) {
            trans.set_response_status(tlm::TLM_BYTE_ENABLE_ERROR_RESPONSE);
            if(!is_fail) {
                LOG(LOG_ERROR) << "[ERROR] Byte Error at L2 Memory Access\n";
                is_fail = true;
            }
            return;
        }
        if (len > 4 || wid < len) {
            trans.set_response_status(tlm::TLM_BURST_ERROR_RESPONSE);
            if(!is_fail) {
                LOG(LOG_ERROR) << "[ERROR] Burst Error occured at L2 Memory Access\n";
                is_fail = true;
            }
            return;
        }

        // Obliged to implement read and write commands
        if (cmd == tlm::TLM_READ_COMMAND) {
            std::copy_n(mem.cbegin() + adr, len, ptr);
            perf->dataMemoryRbyte(len);
        } else if (cmd == tlm::TLM_WRITE_COMMAND) {
            std::copy_n(ptr, len, mem.begin() + adr);
            perf->dataMemoryWbyte(len);
        }

        // Illustrates that b_transport may block
        //sc_core::wait(delay);

        // Reset timing annotation after waiting
        delay = sc_core::SC_ZERO_TIME;

        // *********************************************
        // Set DMI hint to indicated that DMI is supported
        // *********************************************
        trans.set_dmi_allowed(dmi_allowed);

        // Obliged to set response status to indicate successful completion
        trans.set_response_status(tlm::TLM_OK_RESPONSE);
    }

    bool L2MEMORY::get_direct_mem_ptr(tlm::tlm_generic_payload &trans,
                                    tlm::tlm_dmi &dmi_data) {

        (void) trans;

        if (!dmi_allowed) {
            return false;
        }

        // Permit read and write access
        dmi_data.allow_read_write();

        // Set other details of DMI region
        dmi_data.set_dmi_ptr(reinterpret_cast<unsigned char *>(&mem[0]));
        dmi_data.set_start_address(0);
        dmi_data.set_end_address(L2MEMORY::SIZE * 4 - 1);
        dmi_data.set_read_latency(LATENCY);
        dmi_data.set_write_latency(LATENCY);

        return true;
    }

    unsigned int L2MEMORY::transport_dbg(tlm::tlm_generic_payload &trans) {
        tlm::tlm_command cmd = trans.get_command();
        sc_dt::uint64 adr = trans.get_address();
        unsigned char *ptr = trans.get_data_ptr();
        unsigned int len = trans.get_data_length();

        if (adr >= sc_dt::uint64(L2MEMORY::SIZE)) {
            trans.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
            return 0;
        }

        // Calculate the number of bytes to be actually copied
        unsigned int num_bytes = (len < (L2MEMORY::SIZE - adr) * 4) ? len : (L2MEMORY::SIZE - adr) * 4;

        if (cmd == tlm::TLM_READ_COMMAND) {
            std::copy_n(mem.cbegin() + adr, len, ptr);
        } else if (cmd == tlm::TLM_WRITE_COMMAND) {
            std::copy_n(ptr, len, mem.begin() + adr);
        }

        return num_bytes;
    }
    void L2MEMORY::readDataFile(std::string const &filename, int address) {
        std::ifstream hexfile;
        std::string line;
        std::uint32_t memory_offset = 0;

        hexfile.open(filename);
        int t =0;
        if (hexfile.is_open()) {
            std::uint32_t extended_address = 0;

            while (getline(hexfile, line)) {
                if (line.empty()) continue;
                
                if (line[0] == '@') {
                    std::string addr_str = line.substr(1);
                    addr_str.erase(std::remove(addr_str.begin(), addr_str.end(), '_'), addr_str.end());
                    address = std::stoi(addr_str, nullptr, 16);
                    t = 0;
                    continue;
                }

                for(int i = line.length()-2 ; i >= 0 ; i-=2 ){
                    mem[address + t] = static_cast<char>(std::stoi(line.substr(i,2),nullptr, 16));
                    t++;
                }
            }
            hexfile.close();

            if (memory_offset != 0) {
                dmi_allowed = false;
            } else {
                dmi_allowed = true;
            }

        } else {
            LOG(LOG_WARNING) <<"[WARNING] L2 DATA"<< " File open error\n";
        }
    }
    

    /////////L1MEMORY


    L1MEMORY::L1MEMORY(sc_core::sc_module_name const &name, std::string const &filename) :
            sc_module(name), socket_s((std::string(name) + "_socket").c_str()), socket_g((std::string(name) + "_socketg").c_str()), LATENCY(sc_core::SC_ZERO_TIME) {
        // Register callbacks for incoming interface method calls
        socket_s.register_b_transport(this, &L1MEMORY::b_transport);
        socket_s.register_get_direct_mem_ptr(this, &L1MEMORY::get_direct_mem_ptr);
        socket_s.register_transport_dbg(this, &L1MEMORY::transport_dbg);

        socket_g.register_b_transport(this, &L1MEMORY::b_transport);
        socket_g.register_get_direct_mem_ptr(this, &L1MEMORY::get_direct_mem_ptr);
        socket_g.register_transport_dbg(this, &L1MEMORY::transport_dbg);
        perf = Performance::getInstance();
        dmi_allowed = false;
    }

    L1MEMORY::L1MEMORY(sc_core::sc_module_name const &name) :
            sc_module(name), socket_s((std::string(name) + "_sockets").c_str()), socket_g((std::string(name) + "_socketg").c_str()), LATENCY(sc_core::SC_ZERO_TIME) {
        socket_s.register_b_transport(this, &L1MEMORY::b_transport);
        socket_s.register_get_direct_mem_ptr(this, &L1MEMORY::get_direct_mem_ptr);
        socket_s.register_transport_dbg(this, &L1MEMORY::transport_dbg);

        socket_g.register_b_transport(this, &L1MEMORY::b_transport);
        socket_g.register_get_direct_mem_ptr(this, &L1MEMORY::get_direct_mem_ptr);
        socket_g.register_transport_dbg(this, &L1MEMORY::transport_dbg);
        perf = Performance::getInstance();
	    dmi_allowed = false;


    }

    L1MEMORY::~L1MEMORY() = default;


    void L1MEMORY::dump_memtofile(bool addr_dump, uint64_t addr, uint64_t dump_size){
        std::ofstream ofs(std::string(this->basename())+"_DUMP.hex");
        if(!ofs){
            LOG(LOG_WARNING) << "[WARNING] MEMORY DUMP FILE OPEN ERR \n";
        }
        
        addr = addr & (~0x1F);
        uint64_t max_size = addr+dump_size;
        max_size = (max_size > mem.size())? mem.size() : max_size;
        for (size_t base = addr; base < max_size; base += 32) {
        if(addr_dump){
            ofs <<"ADDR:"<< std::hex << std::setw(8) << std::setfill('0') << base << " || ";
        }
        size_t end = std::min(base + 32, max_size);
        for (size_t i = 0; i < end - base; i++) {
            size_t j = end -1 -i;
            ofs << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(mem[j]);
        }
        ofs << "\n";
    }
        ofs.close();
    }

    void L1MEMORY::b_transport(tlm::tlm_generic_payload &trans,
                             sc_core::sc_time &delay) {
        tlm::tlm_command cmd = trans.get_command();
        sc_dt::uint64 adr = trans.get_address();
        unsigned char *ptr = trans.get_data_ptr();
        unsigned int len = trans.get_data_length();
        unsigned char *byt = trans.get_byte_enable_ptr();
        unsigned int wid = trans.get_streaming_width();

        // *********************************************
        // Generate the appropriate error response
        // *********************************************
        if (adr >= sc_dt::uint64(L1MEMORY::SIZE)) {
            trans.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
            if(!is_fail){
                LOG(LOG_ERROR) << "[ERROR] Memory Access out of L1 Memory range\n";
                is_fail = true;
            }
            return;
        }
        if (byt != nullptr) {
            trans.set_response_status(tlm::TLM_BYTE_ENABLE_ERROR_RESPONSE);
            if(!is_fail) {
                LOG(LOG_ERROR) << "[ERROR] Byte Error at L1 Memory Access\n";
                is_fail = true;
            }
            return;
        }
        if (len > 4 || wid < len) {
            trans.set_response_status(tlm::TLM_BURST_ERROR_RESPONSE);
            if(!is_fail) {
                LOG(LOG_ERROR) << "[ERROR] Burst Error occured at L1 Memory Access\n";
                is_fail = true;
            }
            return;
        }

        // Obliged to implement read and write commands
        if (cmd == tlm::TLM_READ_COMMAND) {
            std::copy_n(mem.cbegin() + adr, len, ptr);
            perf->dataMemoryRbyte(len);
        } else if (cmd == tlm::TLM_WRITE_COMMAND) {
            std::copy_n(ptr, len, mem.begin() + adr);
            perf->dataMemoryWbyte(len);
        }

        // Illustrates that b_transport may block
        //sc_core::wait(delay);

        // Reset timing annotation after waiting
        delay = sc_core::SC_ZERO_TIME;

        // *********************************************
        // Set DMI hint to indicated that DMI is supported
        // *********************************************
        trans.set_dmi_allowed(dmi_allowed);

        // Obliged to set response status to indicate successful completion
        trans.set_response_status(tlm::TLM_OK_RESPONSE);
    }

    bool L1MEMORY::get_direct_mem_ptr(tlm::tlm_generic_payload &trans,
                                    tlm::tlm_dmi &dmi_data) {

        (void) trans;

        if (!dmi_allowed) {
            return false;
        }

        // Permit read and write access
        dmi_data.allow_read_write();

        // Set other details of DMI region
        dmi_data.set_dmi_ptr(reinterpret_cast<unsigned char *>(&mem[0]));
        dmi_data.set_start_address(0);
        dmi_data.set_end_address(L1MEMORY::SIZE * 4 - 1);
        dmi_data.set_read_latency(LATENCY);
        dmi_data.set_write_latency(LATENCY);

        return true;
    }

    unsigned int L1MEMORY::transport_dbg(tlm::tlm_generic_payload &trans) {
        tlm::tlm_command cmd = trans.get_command();
        sc_dt::uint64 adr = trans.get_address();
        unsigned char *ptr = trans.get_data_ptr();
        unsigned int len = trans.get_data_length();

        if (adr >= sc_dt::uint64(L1MEMORY::SIZE)) {
            trans.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
            return 0;
        }

        // Calculate the number of bytes to be actually copied
        unsigned int num_bytes = (len < (L1MEMORY::SIZE - adr) * 4) ? len : (L1MEMORY::SIZE - adr) * 4;

        if (cmd == tlm::TLM_READ_COMMAND) {
            std::copy_n(mem.cbegin() + adr, len, ptr);
        } else if (cmd == tlm::TLM_WRITE_COMMAND) {
            std::copy_n(ptr, len, mem.begin() + adr);
        }

        return num_bytes;
    }


    void L1MEMORY::readDataFile(std::string const &filename, int address) {
        std::ifstream hexfile;
        std::string line;
        std::uint32_t memory_offset = 0;

        hexfile.open(filename);
        int t =0;;
        if (hexfile.is_open()) {
            std::uint32_t extended_address = 0;

            while (getline(hexfile, line)) {
                if (line.empty()) continue;
                
                if (line[0] == '@') {
                    std::string addr_str = line.substr(1);
                    addr_str.erase(std::remove(addr_str.begin(), addr_str.end(), '_'), addr_str.end());
                    address = std::stoi(addr_str, nullptr, 16);
                    t = 0;
                    continue;
                }
                
                for(int i = line.length()-2 ; i >= 0 ; i-=2  ){
                    mem[address + t] = static_cast<char>(std::stoi(line.substr(i,2),nullptr, 16));
                    t++;
                }
            }
            hexfile.close();

            if (memory_offset != 0) {
                dmi_allowed = false;
            } else {
                dmi_allowed = true;
            }

        } else {
            LOG(LOG_WARNING) <<"[WARNING] L1 DATA"<< " File open error\n";
        }
    }

    L0MEMORY::L0MEMORY(sc_core::sc_module_name const &name, std::string const &filename) :
            sc_module(name), socket_s((std::string(name) + "_socket").c_str()),  LATENCY(sc_core::SC_ZERO_TIME) {
        // Register callbacks for incoming interface method calls
        socket_s.register_b_transport(this, &L0MEMORY::b_transport);
        socket_s.register_get_direct_mem_ptr(this, &L0MEMORY::get_direct_mem_ptr);
        socket_s.register_transport_dbg(this, &L0MEMORY::transport_dbg);

        perf = Performance::getInstance();
        dmi_allowed = false;
    }

    L0MEMORY::L0MEMORY(sc_core::sc_module_name const &name) :
            sc_module(name), socket_s((std::string(name) + "_sockets").c_str()), LATENCY(sc_core::SC_ZERO_TIME) {
        socket_s.register_b_transport(this, &L0MEMORY::b_transport);
        socket_s.register_get_direct_mem_ptr(this, &L0MEMORY::get_direct_mem_ptr);
        socket_s.register_transport_dbg(this, &L0MEMORY::transport_dbg);
    
        perf = Performance::getInstance();
	    dmi_allowed = false;


    }

    L0MEMORY::~L0MEMORY() = default;


    void L0MEMORY::dump_memtofile(){
        std::ofstream ofs(std::string(this->basename())+"_DUMP.hex");
        if(!ofs){
            LOG(LOG_WARNING) << "[WARNING] MEMORY DUMP FILE OPEN ERR \n";
        }
        
        for (size_t base = 0; base < mem.size(); base += 32) {

        ofs <<"SVRADDR:"<< std::hex << std::setw(2) << std::setfill('0') << base/32 << " || ";

        size_t end = std::min(base + 32, mem.size());
        for (size_t i = 0; i < end - base; i++) {
            size_t j = end -1 -i;
            ofs << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(mem[j]);
        }
        ofs << "\n";
    }
        ofs.close();
    }

    void L0MEMORY::b_transport(tlm::tlm_generic_payload &trans,
                             sc_core::sc_time &delay) {
        tlm::tlm_command cmd = trans.get_command();
        sc_dt::uint64 adr = trans.get_address();
        unsigned char *ptr = trans.get_data_ptr();
        unsigned int len = trans.get_data_length();
        unsigned char *byt = trans.get_byte_enable_ptr();
        unsigned int wid = trans.get_streaming_width();

        // *********************************************
        // Generate the appropriate error response
        // *********************************************
        if (adr >= sc_dt::uint64(L0MEMORY::SIZE)) {
            trans.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
            if(!is_fail){
                LOG(LOG_ERROR) << "[ERROR] Memory Access out of L0 Memory range\n";
                is_fail = true;
            }
            return;
        }
        if (byt != nullptr) {
            trans.set_response_status(tlm::TLM_BYTE_ENABLE_ERROR_RESPONSE);
            if(!is_fail) {
                LOG(LOG_ERROR) << "[ERROR] Byte Error at L0 Memory Access\n";
                is_fail = true;
            }
            return;
        }
        if (len > 4 || wid < len) {
            trans.set_response_status(tlm::TLM_BURST_ERROR_RESPONSE);
            if(!is_fail) {
                LOG(LOG_ERROR) << "[ERROR] Burst Error occured at L0 Memory Access\n";
                is_fail = true;
            }
            return;
        }

        // Obliged to implement read and write commands
        if (cmd == tlm::TLM_READ_COMMAND) {
            std::copy_n(mem.cbegin() + adr, len, ptr);
            perf->dataMemoryRbyte(len);
        } else if (cmd == tlm::TLM_WRITE_COMMAND) {
            std::copy_n(ptr, len, mem.begin() + adr);
            perf->dataMemoryWbyte(len);
        }

        // Illustrates that b_transport may block
        //sc_core::wait(delay);

        // Reset timing annotation after waiting
        delay = sc_core::SC_ZERO_TIME;

        // *********************************************
        // Set DMI hint to indicated that DMI is supported
        // *********************************************
        trans.set_dmi_allowed(dmi_allowed);

        // Obliged to set response status to indicate successful completion
        trans.set_response_status(tlm::TLM_OK_RESPONSE);
    }

    bool L0MEMORY::get_direct_mem_ptr(tlm::tlm_generic_payload &trans,
                                    tlm::tlm_dmi &dmi_data) {

        (void) trans;

        if (!dmi_allowed) {
            return false;
        }

        // Permit read and write access
        dmi_data.allow_read_write();

        // Set other details of DMI region
        dmi_data.set_dmi_ptr(reinterpret_cast<unsigned char *>(&mem[0]));
        dmi_data.set_start_address(0);
        dmi_data.set_end_address(L1MEMORY::SIZE * 4 - 1);
        dmi_data.set_read_latency(LATENCY);
        dmi_data.set_write_latency(LATENCY);

        return true;
    }

    unsigned int L0MEMORY::transport_dbg(tlm::tlm_generic_payload &trans) {
        tlm::tlm_command cmd = trans.get_command();
        sc_dt::uint64 adr = trans.get_address();
        unsigned char *ptr = trans.get_data_ptr();
        unsigned int len = trans.get_data_length();

        if (adr >= sc_dt::uint64(L0MEMORY::SIZE)) {
            trans.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
            return 0;
        }

        // Calculate the number of bytes to be actually copied
        unsigned int num_bytes = (len < (L0MEMORY::SIZE - adr) * 4) ? len : (L0MEMORY::SIZE - adr) * 4;

        if (cmd == tlm::TLM_READ_COMMAND) {
            std::copy_n(mem.cbegin() + adr, len, ptr);
        } else if (cmd == tlm::TLM_WRITE_COMMAND) {
            std::copy_n(ptr, len, mem.begin() + adr);
        }

        return num_bytes;
    }

}
