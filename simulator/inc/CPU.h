
#ifndef CPU_BASE_H
#define CPU_BASE_H

#define SC_INCLUDE_DYNAMIC_PROCESSES

#include "systemc"
#include "tlm.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"

#include "BASE_ISA.h"
#include "NSU.h"
#include "C_extension.h"
#include "M_extension.h"
#include "A_extension.h"
#include "F_extension.h"
#include "GTX_extension.h"
#include "MemoryInterface.h"
#include "Performance.h"
#include "Registers.h"
#include "vert_dev.h"
// Forward declaration for TLM Proxy Kernel
namespace riscv_tlm {
    class TLMSyscallInterface;
}
namespace riscv_tlm {

    typedef enum {RV64} cpu_types_t;


    class CPU : sc_core::sc_module  {
    public:

        /* Constructors */
        explicit CPU(sc_core::sc_module_name const &name, bool debug, riscv_tlm::peripherals::VMachine *vm);

        CPU() noexcept = delete;
        CPU(const CPU& other) noexcept = delete;
        CPU(CPU && other) noexcept = delete;
        CPU& operator=(const CPU& other) noexcept = delete;
        CPU& operator=(CPU&& other) noexcept = delete;

        /* Destructors */
        ~CPU() override = default;

        /**
         * @brief Perform one instruction step
         * @return Breackpoint found (TBD, always true)
         */
        virtual bool CPU_step() = 0;
        
        /**
         * @brief Instruction Memory bus socket
         * @param trans transction to perfoem
         * @param delay time to annotate
         */
        tlm_utils::simple_initiator_socket<CPU> instr_bus;

        /**
         * @brief IRQ line socket
         * @param trans transction to perform (empty)
         * @param delay time to annotate
         */
        tlm_utils::simple_target_socket<CPU> irq_line_socket;

        tlm_utils::simple_target_socket<CPU> irq_line_socket_1;

        /**
        * @brief DMI pointer is not longer valid
        * @param start memory address region start
        * @param end memory address region end
        */
        void invalidate_direct_mem_ptr(sc_dt::uint64 start, sc_dt::uint64 end);


        /**
        * @brief CPU main thread
        */
        [[noreturn]] void CPU_thread();


        /**
         * @brief Process and triggers IRQ if all conditions met
         * @return true if IRQ is triggered, false otherwise
         */
        virtual bool cpu_process_IRQ() = 0;

        /**
         * @brief callback for IRQ simple socket
         * @param trans transaction to perform (empty)
         * @param delay time to annotate
         *
         * it triggers an IRQ when called
         */
        virtual void call_interrupt(tlm::tlm_generic_payload &trans,
                            sc_core::sc_time &delay) = 0;

        virtual std::uint64_t getStartDumpAddress() = 0;
        virtual std::uint64_t getEndDumpAddress() = 0;

        virtual void dump_reg() = 0;
        virtual void dump_fp_reg() = 0;
        virtual void dump_csr_reg(int csr_addr) = 0;
        virtual bool check_breakpoint() = 0;
        virtual void set_breakpoint(std::uint64_t addr) =0;
        virtual void delete_breakpoint(std::uint64_t addr) =0;
        virtual void delete_breakpoint_all() =0;
        virtual void print_breakpoint() =0;
        virtual void dump_reg_file(bool is_addr) = 0;
        virtual void set_syscall_interface(TLMSyscallInterface * interface)  =0;

        void set_perfcheck();

        void off_debug();
        void set_cycle(int cycles);

        



    public:
        CPUMemintf *mem_intf;
        riscv_tlm::peripherals::VMachine *vert_m;
        bool is_regdump;
        bool is_regaddr;
        uint64_t reg_dump_addr;

    protected:
        
        Performance *perf;
        tlm_utils::tlm_quantumkeeper *m_qk;
        Instruction inst;
        bool interrupt;
        bool irq_already_down;
        sc_core::sc_time default_time;
        bool dmi_ptr_valid;
        tlm::tlm_generic_payload trans;
        unsigned char *dmi_ptr = nullptr;
        bool vts_debug ;
        bool is_breakpoint;
        int cycle;
        
        // TLM Proxy Kernel interface

    };

 
    /**
     * @brief RISC_V CPU 64 bits model
     * @param name name of the module
     */
    class CPURV64 : public CPU {
    public:
        using BaseType = std::uint64_t;

        /**
         * @brief Constructor
         * @param name Module name
         * @param PC   Program Counter initialize value
         * @param debug To start debugging
         */
        CPURV64(sc_core::sc_module_name const &name, BaseType PC, bool debug,NSU *nsu_b, bool enable, uint32_t warptime, riscv_tlm::peripherals::VMachine *vm);
        void dump_reg();
        void dump_fp_reg();
        void dump_csr_reg(int csr_addr);
        bool check_breakpoint();
        void set_breakpoint(std::uint64_t pc_point);
        void delete_breakpoint(std::uint64_t pc_point);
        void print_breakpoint();
        void delete_breakpoint_all();
        
        void dump_reg_file(bool is_addr);

        void set_syscall_interface(TLMSyscallInterface* interface){
            base_inst->set_syscall_interface(interface);
        }

        /**
         * @brief Destructor
         */
        ~CPURV64() override;

        bool CPU_step() override;
        Registers<BaseType> *getRegisterBank() { return register_bank; }



    private:
        Registers<BaseType> *register_bank;
        NSU *nsu;
        SPU *spu[64];
        C_extension<BaseType> *c_inst;
        M_extension<BaseType> *m_inst;
        A_extension<BaseType> *a_inst;
        F_extension<BaseType> *f_inst;
        GTX_extension<BaseType> *gtx_inst;
        BASE_ISA<BaseType> *base_inst;
        BaseType int_cause;
        BaseType INSTR;

        std::unordered_set<std::uint64_t> breakpoints;

        /**
         *
         * @brief Process and triggers IRQ if all conditions met
         * @return true if IRQ is triggered, false otherwise
         */
        bool cpu_process_IRQ() override;

        /**
         * @brief callback for IRQ simple socket
         * @param trans transaction to perform (empty)
         * @param delay time to annotate
         *
         * it triggers an IRQ when called
         */
        void call_interrupt(tlm::tlm_generic_payload &trans,
                            sc_core::sc_time &delay) override;

        std::uint64_t getStartDumpAddress() override;
        std::uint64_t getEndDumpAddress() override;
    }; // RV64 class

}
#endif
