/*!
 \file Simulator.cpp
 */

#define SC_INCLUDE_DYNAMIC_PROCESSES
#define GTX_ISS_VERSION "v1.1.4"
#include "systemc"
#include <cblas.h>
#include <omp.h>

#include <csignal>
#include <unistd.h>
#include <chrono>
#include <cstdint>
#include <string>
#include <sstream>


#include "CPU.h"
#include "BusCtrl.h"
#include "config.h"
//#include "Trace.h"
#include "PCIE.h"
#include "Timer.h"
#include "NSU.h"
#include "TMU.h"
#include "vert_dev.h"

#include "tlm_pk.h"
#include "tlm_syscall_interface.h"


std::string filename, temp_str;
std::unordered_map<std::string,std::uint64_t> dfilename;
std::string l1_fname;
std::string l2_fname;
std::string main_dfname = "MEM_DUMP.hex";
std::string reg_dfname  = "REG_DUMP.txt";
std::string cmd_dfname  = "COMMAND_DUMP.hex";
bool debug_session = false;
bool x_point = false;
bool mem_dump = false;
bool addr_dump = false;
bool warp_monitor = false;
bool check_perf = false;
bool reg_dump = false;
bool vm_enable = true;
bool gtx_command_dump = false;
bool dfile_addr_set = false;
bool is_boot_success = true;
bool L1_dump_mode = false;
bool is_L1_dump = false;
bool gtx_tracker = false;
bool hw_mode = false;
bool is_monitoring = false;
int exitcode = 0;
int spu_exec_time = 1;
int L1_mode_id = 0;
uint32_t warp_time = 0;
uint64_t dump_addr_st = 0;
uint64_t dump_addr_size = 0x100000;
uint64_t dump_l1_size = 0x100000;
uint64_t dump_l2_size = 0x100000;
uint64_t reg_dump_addr = 0;
uint64_t load_addr = 0x100000;
uint32_t temp_addr;
LogLevel log_level = LOG_INFO;
int kernel_switch = 0;

#ifdef _WIN32
  #include <direct.h>   // _mkdir
  inline void make_dir(const char* p){ _mkdir(p); }    
#else
  #include <sys/stat.h> // mkdir
  #include <errno.h>
  inline void make_dir(const char* p){ if (mkdir(p, 0777) && errno != EEXIST) {} }
#endif


std::vector<std::string> program_args;
std::string program_name;
 
riscv_tlm::cpu_types_t cpu_type_opt = riscv_tlm::RV64;

/**
 * @class Simulator
 * This class instantiates all necessary modules, connects its ports and starts
 * the simulation.
 *
 * @brief Top simulation entity
 */
class Simulator : sc_core::sc_module {
public:
    riscv_tlm::CPU *cpu;
	riscv_tlm::Memory *MainMemory;
    riscv_tlm::L0MEMORY *L0_Memory[SPU_NUM];
    riscv_tlm::L1MEMORY *L1_Memory[SPU_NUM];
    riscv_tlm::L2MEMORY *L2_Memory[TMU_NUM];
    riscv_tlm::BusCtrl *Bus;

    riscv_tlm::NSU *nsu;
    riscv_tlm::TMU *tmu[TMU_NUM];
    riscv_tlm::SPU* spu[SPU_NUM];
    riscv_tlm::peripherals::Timer *timer;
    riscv_tlm::peripherals::PCIE *PCIE;//PCIE model which communicate with host program
    riscv_tlm::peripherals::VMachine *vm;

    // TLM Proxy Kernel components
    riscv_tlm::TLMProxyKernel *proxy_kernel;
    riscv_tlm::TLMSyscallInterface *syscall_interface;



	explicit Simulator(sc_core::sc_module_name const &name, riscv_tlm::cpu_types_t cpu_type_m): sc_module(name) {
		std::uint32_t start_PC;
        LOG(LOG_INFO) << "---------------------------------------------\n";
        LOG(LOG_INFO) << "[INFO] System MAIN MEMORY SIZE : 0x"<< std::uppercase << std::hex<<MAIN_SIZE << std::endl;
        LOG(LOG_INFO) << "---------------------------------------------\n";
        LOG(LOG_INFO) << "[INFO] System L2 MEMORY COUNT  : "<<std::dec <<TMU_NUM << std::endl;
        LOG(LOG_INFO) << "[INFO] System L2 MEMORY SIZE   : 0x"<< std::uppercase << std::hex<<L2_SIZE << std::endl;
        LOG(LOG_INFO) << "---------------------------------------------\n";
        LOG(LOG_INFO) << "[INFO] System L1 MEMORY COUNT  : "<<std::dec <<SPU_NUM << std::endl;
        LOG(LOG_INFO) << "[INFO] System L1 MEMORY SIZE   : 0x"<< std::uppercase << std::hex<<L1_SIZE << std::endl;
        LOG(LOG_INFO) << "---------------------------------------------\n";
        
        
		MainMemory = new riscv_tlm::Memory("Main_Memory", filename);
        for(const auto& pair : dfilename){
            temp_str  = pair.first;
            temp_addr = pair.second;
            MainMemory -> readDataFile(temp_str, temp_addr);
        }
        for(int j =0; j < TMU_NUM ; j++){
            L2_Memory[j]  = new riscv_tlm::L2MEMORY(("L2_Memory"+std::to_string(j)).c_str());
        }
        for(int i =0; i < SPU_NUM ; i++){
            spu[i] = new riscv_tlm::SPU(("SPU"+std::to_string(i)).c_str(), i);
            L1_Memory[i] = new riscv_tlm::L1MEMORY(("L1_Memory"+std::to_string(i)).c_str());
            L0_Memory[i] = new riscv_tlm::L0MEMORY(("L0_Memory"+std::to_string(i)).c_str());
        }
        for(int j =0; j < TMU_NUM ; j++){
            tmu[j]  = new riscv_tlm::TMU(("TMU"+std::to_string(j)).c_str(),&spu[j*16],j);
        }
           

		start_PC = MainMemory->getPCfromHEX();
        vm = new riscv_tlm::peripherals::VMachine("VMachine");
        cpu_type = cpu_type_m;
        vm -> heap_end = MainMemory -> heap_start;
        vm -> is_os = MainMemory -> is_os;

        nsu = new riscv_tlm::NSU("NSU", spu, tmu);

        cpu = new riscv_tlm::CPURV64("cpu", start_PC, debug_session, nsu, warp_monitor, warp_time,vm);
        
        // Initialize TLM Proxy Kernel
        proxy_kernel = new riscv_tlm::TLMProxyKernel("proxy_kernel", MainMemory -> heap_start);
        syscall_interface = new riscv_tlm::TLMSyscallInterface();
        // Connect TLM Proxy Kernel to CPU
        cpu->set_syscall_interface(syscall_interface);
        syscall_interface->pk_socket.bind(proxy_kernel->syscall_socket);
        // Set memory interface for TLM Proxy Kernel
        proxy_kernel->set_memory_interface(cpu->mem_intf);

        // Set program arguments from command line
         if (!program_args.empty()) {
             std::vector<char*> argv_ptrs;
             for (const auto& arg : program_args) {
                 argv_ptrs.push_back(const_cast<char*>(arg.c_str()));
             }
             proxy_kernel->set_program_arguments(program_args.size(), argv_ptrs.data(), program_name);
         } else {
             // No additional arguments, just set program name
             proxy_kernel->set_program_arguments(0, nullptr, program_name);
         }


		Bus = new riscv_tlm::BusCtrl("BusCtrl");

		timer = new riscv_tlm::peripherals::Timer("Timer");

        PCIE = new riscv_tlm::peripherals::PCIE("PCIE", MainMemory);

        


		cpu->instr_bus.bind(Bus->cpu_instr_socket);
		cpu->mem_intf->data_bus.bind(Bus->cpu_data_socket);

		Bus->memory_socket.bind(MainMemory->socket);

		Bus->timer_socket.bind(timer->socket);

        Bus->PCIE_socket.bind(PCIE->socket);

        Bus->vm_socket.bind(vm->socket);

        Bus->is_vmenable = vm_enable;

        vm->mem_intf->data_bus.bind(MainMemory->socket3);

        nsu->mem_intf->data_bus.bind(MainMemory->socket1);

        PCIE->mem_intf->data_bus.bind(Bus->PCIE_data_socket);

        for(int i =0; i <TMU_NUM ;i++){
            nsu->l2_intf[i]->data_bus.bind(L2_Memory[i]->socket_nsu);
            tmu[i]->mem_intf->data_bus.bind(MainMemory->socket2[i]);
            tmu[i]->l2_intf-> data_bus.bind(L2_Memory[i] -> socket_tmu);

        }

        
        
        for(int i =0 ; i < SPU_NUM ; i++){
            spu[i] -> l2_intf-> data_bus.bind(L2_Memory[i/(SPU_NUM/TMU_NUM)] -> socket_spu[i%(SPU_NUM/TMU_NUM)]);
            tmu[i/(SPU_NUM/TMU_NUM)] -> l1_intf[i%(SPU_NUM/TMU_NUM)]-> data_bus.bind(L1_Memory[i] -> socket_g);
            spu[i] -> l1_intf-> data_bus.bind(L1_Memory[i] -> socket_s);
            spu[i] -> l0_intf-> data_bus.bind(L0_Memory[i] -> socket_s);
        }
    
        

        if(check_perf) {cpu -> set_perfcheck();}
        if(reg_dump)   {cpu -> is_regdump = true; cpu -> reg_dump_addr = reg_dump_addr; cpu->is_regaddr = addr_dump;}

		timer->irq_line.bind(cpu->irq_line_socket);
        PCIE->irq_line.bind(cpu->irq_line_socket_1);
        


	}
    

	~Simulator() override {
        if(is_L1_dump ){
            for(int i = 0 ; i < SPU_NUM ; i++){
                if(i == L1_mode_id){
                    L1_Memory[L1_mode_id] -> dump_memtofile(addr_dump, 0x0, L1_SIZE);
                }
            }
        }
	    if (mem_dump) {
            MainMemory -> dump_memtofile(dump_addr_st, dump_addr_size,addr_dump);
        }
        for(auto ptr : L1_Memory){
            delete ptr;
        }
        for(auto ptr : spu){
            delete ptr;
        }
        for(auto ptr : tmu){
            delete ptr;
        }
        for(auto ptr : L2_Memory){
            delete ptr;
        }
        
        delete nsu;
    
		delete MainMemory;
		delete cpu;
		delete Bus;
        delete proxy_kernel;
        delete syscall_interface;
		delete timer;
        delete PCIE;
        delete vm;
	}


private:
    riscv_tlm::cpu_types_t cpu_type;
};

Simulator *top;

void end_of_microcode(){
    if(gtx_command_dump){
        std::string fp;
        fp = cmd_dfname;
        std::ofstream ofs(fp,std::ios::out | std::ios::app);
        if(!ofs) {
            LOG(LOG_WARNING) << "[WARNING] FAILED TO OPEN COMMAND DUMPFILE\n";
            return ;
        }
        uint64_t rs4;
        rs4 = 0x3b8;

        ofs << std::uppercase <<std::hex << std::setfill('0') 
            << std::setw(16) << rs4
            << std::setw(16) << 0x0 
            << std::setw(16) << 0x0 
            << std::setw(16) << 0x0 << std::endl;
    }
    return ;
}
void print_banner() {
    LOG(LOG_INFO) << "=======================================\n";
    LOG(LOG_INFO) << "[INFO] < GTX_ISS Simulator -- " << GTX_ISS_VERSION << " >\n";
    LOG(LOG_INFO) << "=======================================\n\n";
}
void intHandler(int dummy) {
	delete top;
	(void) dummy;
	sc_core::sc_stop();
    LOG(LOG_INFO) << "\n[INFO] Simulation Terminated by user (ctrl+c)...\n"<<std::endl;
	exit(-1);
}
void clear_terminal() {
	std::cout << "\033[2J\033[H" << std::flush;
}

void debug_help()
{
    std::cout << "[DBG-UI] \n";
    std::cout << " ======================================\n";
    std::cout << "          GTX_ISS DEBUGGER MENU        \n";
    std::cout << " ======================================\n";
    std::cout << std::endl;
    std::cout << " --- Execution Control ---\n";
    std::cout << " [c] - excute insructions {N} (defualt N = 1)(0:continue..)" << std::endl;
    std::cout << " [x] - exit debug mode" << std::endl;
    std::cout << std::endl;

    std::cout << " --- Register & Memory ---\n";
    std::cout << " [r] - print RISCV CPU registers" << std::endl;
    std::cout << " [f] - print RISCV CPU FP registers" << std::endl;
    std::cout << " [m] - memory load/dump menu" << std::endl;
    std::cout << " [s] - memory dump size setting menu"<< std::endl;
    std::cout << " [l] - 2: tmu(l2_memory)  1: spu(l1_memory,svr) 0: nsu {dump,load menu}" << std::endl;
    std::cout << " [a] - address dump in dumpfile - toggle\n";
    std::cout << " [C] - Check CSR value (ex> C 0x001) " << std::endl;
    std::cout << std::endl;

    std::cout << " --- Breakpoint Mangement ---\n";
    std::cout << " [b] - set break point at {0x???}              (ex> b 0x10004)" << std::endl;
    std::cout << " [b] - print break point *type b only          (ex> b        )" << std::endl;
    std::cout << " [d] - delete breakpoint at   {0x???}          (ex> d 0x10004)" << std::endl;
    std::cout << " [d] - delete all the breakpoints *type d only (ex> d        )" << std::endl;
    std::cout << std::endl;

    
    std::cout << " --- Help ---\n";
    std::cout << " [h] - print this message" << std::endl <<std::endl;
}

void debug_handler()
{
    std::string cmd;
    LOG(LOG_INFO) << "[INFO] Entering debug session..\n";
    std::cout << "[DBG-UI] Type h to see commands.\n";
    while(debug_session)
    {
        
        std::cout << "[DBG-UI] cmd > ";
        getline(std::cin,cmd);
        std::istringstream iss(cmd);
        std::string argv, filename;        
        iss >> argv;
        int cycle, id, address;
        std::stringstream ss;
        std::uint64_t addr;
        int csr_addr;

        switch(cmd[0]){
            case 'c' : 
                if( iss >> cycle){
                    if(cycle !=0){
                        std::cout << "[DBG-UI] Run - " << cycle << " instructions!!\n";
                    }else {
                        cycle = -1;
                        std::cout << "[DBG-UI] continue..." << std::endl;
                    }
                    top -> cpu -> set_cycle(cycle);
                }
                sc_core::sc_start();
                break;
            case 'r' :
                std::cout << "[DBG-UI] \n";
                top -> cpu -> dump_reg(); break;
            case 'f' :
                std::cout << "[DBG-UI] \n";
                top -> cpu -> dump_fp_reg(); break;
            case 'C' :
                if( iss >> std::hex >> csr_addr){
                    std::cout << "[DBG-UI] \n";
                    top -> cpu -> dump_csr_reg(csr_addr);
                }
                break;
            case 'a' :
                addr_dump = ~addr_dump;
                if(addr_dump){
                    std::cout << "[DBG-UI] address dump - on\n";
                }else{
                    std::cout << "[DBG-UI] address dump - off\n";
                }
                break;
            case 'h' :
                debug_help(); break;
            case 'm' :
                std::cout<< "[DBG-UI][m] \n";
                std::cout<< " --------------------------------------------------------------\n";
                std::cout<< " MAIN MEMORY DEBUG MENU\n";
                std::cout<< " --------------------------------------------------------------\n";
                std::cout<< " [d] - dump main memory with start address\n";
                std::cout<< " [l] - load datafile with start address [ex) l data.txt 0x1000]"<<std::endl;
                std::cout<< " --------------------------------------------------------------\n";
                std::cout<< "[DBG-UI][m] cmd > ";
                getline(std::cin, cmd);
                iss.clear();
                iss.str(cmd);
                iss >> argv;
                if(argv[0] == 'd'){
                    iss >> argv;
                    std::cout << "[DBG-UI][m] d - selected\n";
                    top-> MainMemory->dump_memtofile(strtoull(argv.c_str(),nullptr, 16), dump_addr_size,addr_dump);
                }
                else if(argv[0] == 'l'){
                    std::cout << "[DBG-UI][m] l - selected\n";
                    iss >> filename >> std::hex >> address;
                    top -> MainMemory -> readDataFile(filename, address);
                }else{
                    std::cout << "[DBG-UI][m] UNkNOWN COMMAND\n";
                }
                break;
            case 'l' :
                std::cout << "[DBG-UI][l] \n";
                std::cout << " -----------------------\n";
                std::cout << " L1/L2 MEMORY DEBUG MENU\n";
                std::cout << " -----------------------\n";
                std::cout << " [1] - SPU | L1 MEMORY\n";
                std::cout << " [2] - TMU | L2_MEMORY\n";
                std::cout << " [3] - NSU\n";
                std::cout << " -----------------------\n";
                std::cout << "[DBG-UI][l] cmd > ";
                if(std::cin >> id){
                    std::cin.ignore();
                    if(id == 2){
                        std::cout << "[DBG-UI][l-2] INPUT L2 Memory(TMU) ID(0~"<< (TMU_NUM-1)<<")" << std::endl;
                        std::cout << "[DBG-UI][l-2] id > ";
                        if(std::cin >> id){
                            if(id>=0 && id<TMU_NUM){
                                std::cout << "[DBG-UI][l-2-"<< id <<"] NEST : " << id <<"\n";
                                std::cin.ignore();
                                std::cout<< "[DBG-UI][l-2-"<< id <<"] \n";
                                std::cout<< " --------------------------------------------------------------\n";
                                std::cout<< " L2 MEMORY(TMU) DEBUG MENU\n";
                                std::cout<< " --------------------------------------------------------------\n";
                                std::cout<< " [d] - dump l2 memory with start address\n";
                                std::cout<< " [r] - dump NSPR  \n";
                                std::cout<< " [l] - load datafile with start address [ex) l data.txt 0x1000]"<<std::endl;
                                std::cout<< " --------------------------------------------------------------\n";
                                std::cout<< "[DBG-UI][l-2-"<< id <<"] cmd > ";
                                getline(std::cin, cmd);
                                iss.clear();
                                iss.str(cmd);
                                iss >> argv;
                                if(argv[0] == 'd'){
                                    iss >> argv;
                                    std::cout << "[DBG-UI][l-2-"<< id <<"] d - selected\n";
                                    top -> L2_Memory[id] -> dump_memtofile(addr_dump,strtoull(argv.c_str(),nullptr, 16), dump_l2_size);
                                } else if(argv[0] == 'l'){
                                    std::cout << "[DBG-UI][l-2-"<< id <<"] l - selected\n";
                                    iss >> filename >> std::hex >> address;
                                    top -> L2_Memory[id] -> readDataFile(filename, address);
                                } else if (argv[0] == 'r'){
                                    std::cout << "[DBG-UI][l-2-"<< id <<"] r - selected\n"; 
                                    std::cout << "[DBG-UI][l-2-"<< id <<"] \n";
                                    top -> tmu[id] -> dump_NSPR();
                                } else{
                                    std::cout << "[DBG-UI][l-2-"<< id <<"] UNkNOWN COMMAND\n";
                                }
                            }else {         
                                std::cin.clear();                   
                                std::cout << "[DBG-UI][l-2] NO ID(l failed)" << std::endl;
                            }
                        }
                    }else if(id == 1){
                        std::cout << "[DBG-UI][l-1] INPUT L1 Memory(SPU) ID(0~"<<std::dec<<(SPU_NUM-1)<<")" << std::endl;
                        std::cout << "[DBG-UI][l-1] id > ";
                        if(std::cin >> id){
                            if(id>=0 && id<SPU_NUM){
                                std::cout << "[DBG-UI][l-1-"<< id <<"] NEST ID : " << ((id*TMU_NUM)/SPU_NUM) << " -- SPU ID : " << (id %(SPU_NUM/TMU_NUM))  <<"\n";
                                std::cin.ignore();
                                std::cout<< "[DBG-UI][l-1-"<< id <<"] \n";
                                std::cout<< " --------------------------------------------------------------\n";
                                std::cout<< " L1 MEMORY(SPU) DEBUG MENU\n";
                                std::cout<< " --------------------------------------------------------------\n";
                                std::cout<< " [d] - dump l1 memory with start address\n";
                                std::cout<< " [r] - dump LSPR  \n";
                                std::cout<< " [v] - dump SVR    \n";
                                std::cout<< " [l] - load datafile with start address [ex) l data.txt 0x1000]"<<std::endl;
                                std::cout<< " --------------------------------------------------------------\n";
                                std::cout<< "[DBG-UI][l-1-"<< id <<"] cmd > ";
                                getline(std::cin, cmd);
                                iss.clear();
                                iss.str(cmd);
                                iss >> argv;
                                if(argv[0] == 'd'){
                                    iss >> argv;
                                    std::cout << "[DBG-UI][l-1-"<< id <<"] d - selected\n";
                                    top -> L1_Memory[id] -> dump_memtofile(addr_dump,strtoull(argv.c_str(),nullptr, 16), dump_l1_size);
                                } else if(argv[0] == 'l'){
                                    std::cout << "[DBG-UI][l-1-"<< id <<"] l - selected\n";
                                    iss >> filename >> std::hex >> address;
                                    top -> L1_Memory[id] -> readDataFile(filename, address);
                                } else if (argv[0] == 'r'){
                                    std::cout << "[DBG-UI][l-1-"<< id <<"] r - selected\n";
                                    std::cout << "[DBG-UI][l-1-"<< id <<"] \n";
                                    top -> spu[id] -> dump_LSPR();
                                } else if (argv[0] == 'v'){
                                    std::cout << "[DBG-UI][l-1-"<< id <<"] v - selected\n";
                                    top -> L0_Memory[id] -> dump_memtofile();
                                } else{
                                    std::cout << "[DBG-UI][l-1-"<< id <<"] UNkNOWN COMMAND\n";
                                }
                            }else {         
                                std::cin.clear();                   
                                std::cout << "[DBG-UI][l-1] NO ID(l failed)" << std::endl;
                            }
                        }
                    }else if(id == 3){
                        std::cout<< "[DBG-UI][l-3] \n";
                        std::cout<< " ---------------\n";
                        std::cout<< " NSU DEBUG MENU\n";
                        std::cout<< " ---------------\n";
                        std::cout<< " [r] - dump GSPR "<<std::endl;
                        std::cout<< " ---------------\n";
                        std::cout<< "[DBG-UI][l-3] cmd > ";
                        getline(std::cin, cmd);
                        iss.clear();
                        iss.str(cmd);
                        iss >> argv;
                        if (argv[0] == 'r'){
                            std::cout << "[DBG-UI][l-3] r - selected\n";
                            std::cout << "[DBG-UI][l-3] \n";
                            top -> nsu -> dump_GSPR();
                        }else{
                            std::cout << "[DBG-UI][l-3] UNkNOWN COMMAND\n";
                        }
                    }
                    else{
                        std::cout << "[DBG-UI][l] UNKNOWN COMMAND (l failed)" << std::endl;
                        std::cin.clear();
                    }
               }else{
                    std::cin.clear();
                    std::cout << "[DBG-UI][l] UNKNOWN COMMAND"<<std::endl;
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
                iss.clear();
                break;
            case 'b' :
                if(iss >> std::hex >> addr){
                    top -> cpu -> set_breakpoint(addr);
                }else{
                    top -> cpu -> print_breakpoint();
                }
                break;
            case 's' :
                    std::cout<< "[DBG-UI][s] \n";
                    std::cout<< " --------------------------------------------------------------\n";
                    std::cout<< " DUMP MEMORY SIZE SETTING MENU\n";
                    std::cout<< " --------------------------------------------------------------\n";
                    std::cout<< " [m] - MAIN MEMORY dump size [ex) m 0x10000]\n";
                    std::cout<< " [t] - L2 MEMORY dump size   [ex) t 0x10000]\n";
                    std::cout<< " [s] - L1 MEMORY dump size   [ex) s 0x10000]"<<std::endl;
                    std::cout<< " --------------------------------------------------------------\n";
                    std::cout<< "[DBG-UI][s] cmd > ";
                    getline(std::cin, cmd);
                    iss.clear();
                    iss.str(cmd);
                    iss >> argv;
                    if(argv[0] == 'm'){
                        if(iss >>std::hex>> dump_addr_size){ 
                            std::cout << "[DBG-UI][s] current MAIN dump size = 0x" << std::hex << dump_addr_size << std::endl;
                        }else{
                            std::cin.clear();
                            std::cout << "[DBG-UI][s] failed to set dump size\n";
                        }
                    }else if(argv[0] == 't'){
                        if(iss >>std::hex>>dump_l2_size){ 
                            std::cout << "[DBG-UI][s] current L2 dump size = 0x" << std::hex << dump_l2_size << std::endl;
                        }else{
                            std::cin.clear();
                            std::cout << "[DBG-UI][s] failed to set dump size\n";
                        }
                    }else if(argv[0] == 's'){
                        if(iss >>std::hex>> dump_l1_size){ 
                            std::cout << "[DBG-UI][s] current L1 dump size = 0x" << std::hex << dump_l1_size << std::endl;
                        }else{
                            std::cin.clear();
                            std::cout << "[DBG-UI][s] failed to set dump size\n";
                        }
                    }else{
                        std::cin.clear();
                        std::cout << "[DBG-UI][s] size set failed..\n";
                    }
                break;
            case 'd' :
                if(iss >> std::hex >> addr){
                    top -> cpu -> delete_breakpoint(addr);
                }else{
                    top -> cpu -> delete_breakpoint_all();
                }
                break;
            case 'X' :
                clear_terminal(); break;
            case 'x' :
                std::cout << "[DBG-UI] Finish debugging Mode\n";
                debug_session = false;
                top -> cpu -> off_debug();
                sc_core::sc_start();
                break;
                return;
            default :
                if(std::cin.eof()) std::cout << std::endl;
                std::cout << "[DBG-UI] UNkNOWN COMMAND - Try 'h' for see commands\n";
                if (std::cin.rdbuf()->in_avail() > 0) std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                std::cin.clear();
                break;
        }
    }
}
static int is_option(const char *s) {
    return s && s[0] == '-' && s[1] != '\0' && !isdigit((unsigned char)s[1]);
}
void process_arguments(int argc, char *argv[]) {

	int c;
    long l;
	long int debug_level;
    int taken = 1;

    

	debug_session = false;
    cpu_type_opt = riscv_tlm::RV64;
    opterr = 0;

	while ((c = getopt(argc, argv, "DdTCGPVHMAI:E:l:B:L:W:f:R:X:L:S:U:K:a:h")) != -1) {
		switch (c) {
        case 'd':
		case 'D':
			debug_session = true;
			break;
        case 'C':
            if(warp_monitor){
                LOG(LOG_WARNING) << "[WARNING] -C Command dump option ignored due to -M(monitoring) option\n";
            }else{
                gtx_command_dump = true;
            }
            if(optind < argc && argv[optind][0] != '-'){
                cmd_dfname = argv[optind];
            }
            break;
        case 'T':
            mem_dump = true;
            if(optind < argc && argv[optind][0] != '-'){
                main_dfname = argv[optind];
            }
            break;
        case 'G':
            gtx_tracker = true;
            make_dir("./tracker"); 
            break;
        case 'A':
            addr_dump = true;
            break;
        case 'B':
            dump_addr_st = std::strtoul (optarg, nullptr, 16);
            break;
        case 'K':
            kernel_switch = std::strtoul (optarg, nullptr, 16);
            break;
        case 'E':
            dump_addr_size = std::strtoul(optarg, nullptr, 16);
            dump_l2_size = dump_addr_size;
            dump_l1_size = dump_addr_size;
            break;
		case 'f':
			filename = std::string(optarg);
			break;
        case 'L':
            dfilename[std::string(optarg)] = load_addr;
            break;
        case 'S':
            dfile_addr_set = true;
            load_addr = std::strtoul(optarg, nullptr, 16);
            break;
        case 'R':
            char* endptr;
            uint64_t rtemp;
            reg_dump = true;
            rtemp = std::strtoul(optarg, &endptr, 16);
            if(*endptr != '\0'){
                reg_dfname = optarg;
            }else{
                reg_dump_addr = rtemp;
            }
            break;
        case 'W':
            warp_time = (std::uint32_t)std::strtoul(optarg, nullptr,16);
            break;
        case 'M':
            if(gtx_command_dump){
                LOG(LOG_WARNING) << "[WARNING] -M Command dump option ignored due to -C(command dump) option \n";
            }else{
                warp_monitor = true;
            }
            
            break;
        case 'V':
            vm_enable = false;
            break;
        case 'P':
            check_perf = true;
            break;
        case 'X':
            L1_dump_mode = true;
            L1_mode_id = (int)strtol(optarg, NULL, 0);
            if (optind < argc && !is_option(argv[optind])) {
                spu_exec_time = (int)strtol(argv[optind++], NULL, 0);
                taken++;
            }
            if (taken != 2) {
                LOG(LOG_WARNING) << "[WARNING] X option ignored\n";
                L1_dump_mode = false;
            }
            break;
        case 'l':
            l = std::strtol(optarg,nullptr,10);
            switch(l){
                case 0: log_level = LOG_FATAL; break;
                case 1: log_level = LOG_ERROR; break;
                case 2: log_level = LOG_WARNING; break;
                case 3: log_level = LOG_INFO; break;
                case 4: log_level = LOG_DEBUG; break;
                default:
                    LOG(LOG_WARNING) << "[WARNING] wrong level input\n";
                    break;
            }
            break;
        case 'H':
            hw_mode = true;
            break;
		case 'h':
			std::cout << "EX) ./GTX_ISS -I program.elf\n"
                      << " -I ${filepath}: Load program(essential)\n"
                      << " -D            : execute with debug mode\n"
                      << " -T            : dump memory data when simulation end (MEM_DUMP.hex)\n"
                      << " -T ${filename}: change dump memory file name\n"
                      << " -B ${address} : memory dump start address (MEM_DUMP.hex)\n"
                      << " -E ${addrsize}: memory dump size (MEM_DUMP.hex)\n"
                      << " -R ${filename}: change register dump file name\n"
                      << " -R ${address} : Register dump at PC {address}(without -R option dump last PC register)\n"
                      << " -L ${filepath}: load memory data file (set -S option first)\n"
                      << " -S ${address} : memory data load with address(main memory) (default = 0x100000)\n"
                      << " -A            : add address part in dump file (without -A only dump data)\n"
                      << " -W (warptime) : GTX split join warp time (default 0)\n"
                      << " -M            : GTX split join warp monitoring ON *(Notice that this mode only verifies the integrity of the code, the simulator’s result data may be corrupted or invalid)\n"
                      << " -P            : Measure CPU cycle counts per instruction for performance analysis.\n"
                      << " -V            : Virtual device disable\n"
                      << " -l (level)    : set log level |[0]FATAL|[1]ERROR|[2]WARNING|[3]INFO(default)|[4]DEBUG|\n"
                      << " -C            : GTX Command dump on\n"
                      << " -C ${filename}: change command dump file name\n"
                      << " -H            : Hardware Mode enable\n"
                      << " -X {id} {time}: SPU L1 dump mode on , {id} = SPU_ID , {time} = execute spu command number\n"
                      << " -K (level)    : Kernel switch [0]-no kenrel(default), [1]-ISS basic kernel , [2]-Proxy kernel(pre)\n"
					  << std::endl;
            break;
        case 'a':{
            program_args.clear();
            std::string args_str = optarg;
            if (args_str.empty()) {
                LOG(LOG_WARNING) << "[WARNING] Empty argument string provided(-a option)" << std::endl;
                break;
            }
            std::stringstream ss(args_str);
            std::string arg;
            while (std::getline(ss, arg, ',')) {
                size_t start = arg.find_first_not_of(" \t\r\n");
                size_t end = arg.find_last_not_of(" \t\r\n");
                
                if (start != std::string::npos && end != std::string::npos) {
                    std::string trimmed = arg.substr(start, end - start + 1);
                    if (!trimmed.empty()) {
                        program_args.push_back(trimmed);
                    }
                }
            }
            break;
            }
        case 'I':
            filename = optarg;
            break;
		default:
			LOG(LOG_WARNING) << "[WARNING] unknown command exist " << std::endl;
            break;
		}
	}
	if (filename.empty()) {
        if(mem_dump = true){
            main_dfname = "MEM_DUMP.hex";
        }
        if(optind < argc || argv[optind] != nullptr){
		    filename = std::string(argv[optind]);
        }else{
           LOG(LOG_ERROR)<<"[ERROR] No file to open (type GTX_ISS -h to see commands)\n";
           exit(1);
        }
	}
    print_banner();
	LOG(LOG_INFO) << "\n[INFO] File : " << filename << '\n';
}




int sc_main(int argc, char *argv[]) {
    omp_set_num_threads(MAX_THREAD);
    openblas_set_num_threads(MAX_THREAD);
    Performance *perf = Performance::getInstance();

	/* Capture Ctrl+C and finish the simulation */
	signal(SIGINT, intHandler);

	/* SystemC time resolution set to 1 ns*/
	sc_core::sc_set_time_resolution(1, sc_core::SC_NS);
	/* Parse and process program arguments. -f is mandatory */

	process_arguments(argc, argv);

    //if(log_level < LOG_INFO){
    sc_core::sc_report_handler::set_actions("/OSCI/SystemC",sc_core::SC_DO_NOTHING);
    //}

	top = new Simulator("top", cpu_type_opt);
    if(is_boot_success){
        LOG(LOG_INFO) << "\n============================================="<<std::endl;
        LOG(LOG_INFO) << "[INFO] Simulation initialized. Executing.. \n";
        LOG(LOG_INFO) << "=============================================\n"<<std::endl;
	    auto start = std::chrono::steady_clock::now();
	    sc_core::sc_start();
        if(debug_session){
            debug_handler();
        }
	    auto end = std::chrono::steady_clock::now();

	    std::chrono::duration<double> elapsed_seconds = end - start;
        long instructions = static_cast<int>(perf->getInstructions()) / elapsed_seconds.count();
        long cycles = static_cast<long>(perf->getcycles());
        long mdata_read = static_cast<long>(perf->getMemoryRead());
        long mdata_write = static_cast<long>(perf->getMemoryWrite());
        long mdata_rbyte = static_cast<long>(perf->getRbyte());
        long mdata_wbyte = static_cast<long>(perf->getWbyte());

        LOG(LOG_INFO) << "[INFO] Total Memory read  : " <<std::dec<< mdata_read << std::endl;
        LOG(LOG_INFO) << "[INFO] Total Memory rbyte : " << mdata_rbyte << std::endl;
        LOG(LOG_INFO) << "[INFO] Total Memory write : " << mdata_write << std::endl;
        LOG(LOG_INFO) << "[INFO] Total Memory wbyte : " << mdata_wbyte << std::endl;
	    LOG(LOG_INFO) << "[INFO] Total elapsed time : " << elapsed_seconds.count() << "s" << std::endl;
        LOG(LOG_INFO) << "[INFO] Total CPU cycles   : " << cycles  << std::endl;
	    LOG(LOG_INFO) << "[INFO] Simulated " << int(std::round(instructions)) << " instr/sec" << std::endl;
    }
    end_of_microcode();
	delete top;

	return exitcode;
}
