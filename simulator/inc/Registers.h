
#ifndef REGISTERS_H
#define REGISTERS_H

#define SC_INCLUDE_DYNAMIC_PROCESSES

#include <iomanip>
#include <unordered_map>

#include "systemc"
#include "tlm.h"

#include "Performance.h"
#include "Memory.h"
#include "config.h"

namespace riscv_tlm {

#define MISA_A_EXTENSION (1 << 0)
#define MISA_B_EXTENSION (1 << 1)
#define MISA_C_EXTENSION (1 << 2)
#define MISA_I_BASE (1 << 8)
#define MISA_M_EXTENSION (1 << 12)
#define MISA_F_EXTENSION (1 << 5)
#define MISA_MXL (1 << 30)

#define CSR_MVENDORID (0xF11)
#define CSR_MARCHID (0xF12)
#define CSR_MIMPID (0xF13)
#define CSR_MHARTID (0xF14)

#define CSR_USTATUS (0x000)
#define CSR_SSTATUS (0x100)
#define CSR_SEDELEG (0x102)

#define CSR_MSTATUS (0x300)
#define CSR_MISA (0x301)
#define CSR_MEDELEG (0x302)
#define CSR_MIDELEG (0x303)
#define CSR_MIE (0x304)
#define CSR_MTVEC (0x305)
#define CSR_MCOUNTEREN (0x306)
#define CSR_MSTATUSH (0x310)

#define CSR_MSCRATCH (0x340)
#define CSR_MEPC (0x341)
#define CSR_MCAUSE (0x342)
#define CSR_MTVAL (0x343)
#define CSR_MIP (0x344)

#define CSR_SSCRATCH (0x140)
#define CSR_SEPC (0x141)
#define CSR_SCAUSE (0x142)
#define CSR_STVAL (0x143)
#define CSR_SIP (0x144)

#define CSR_MCYCLE (0xB00)
#define CSR_MINSTRET (0xB02)
#define CSR_MCYCLEH (0xB80)
#define CSR_MINSTRETH (0xB82)

#define CSR_CYCLE (0xC00)
#define CSR_TIME (0xC01)
#define CSR_INSTRET (0xC02)

#define CSR_CYCLEH (0xC80)
#define CSR_TIMEH (0xC81)
#define CSR_INSTRETH (0xC82)

#define CSR_STVEC (0x105)

#define CSR_ADDR_FFLAGS (0x001)
#define CSR_ADDR_FRM    (0x002)
#define CSR_ADDR_FCSR   (0x003)

#define MSTATUS_UIE (1 << 0)
#define MSTATUS_SIE (1 << 1)
#define MSTATUS_MIE (1 << 3)
#define MSTATUS_UPIE (1 << 4)
#define MSTATUS_SPIE (1 << 5)
#define MSTATUS_MPIE (1 << 7)
#define MSTATUS_SPP (1 << 8)
#define MSTATUS_MPP (1 << 11)
#define MSTATUS_FS  (1 << 13)
#define MSTATUS_XS    (1 << 15)
#define MSTATUS_MPRV (1 << 17)
#define MSTATUS_SUM (1 << 18)
#define MSTATUS_MXR (1 << 19)
#define MSTATUS_TVM (1 << 20)
#define MSTATUS_TW (1 << 21)
#define MSTATUS_TSR (1 << 22)

#define MIP_USIP (1 << 0)
#define MIP_SSIP (1 << 1)
#define MIP_MSIP (1 << 3)
#define MIP_UTIP (1 << 4)
#define MIP_STIP (1 << 5)
#define MIP_MTIP (1 << 7)
#define MIP_UEIP (1 << 8)
#define MIP_SEIP (1 << 9)
#define MIP_MEIP (1 << 11)

#define MIE_USIE (1 << 0)
#define MIE_SSIE (1 << 1)
#define MIE_MSIE (1 << 3)
#define MIE_UTIE (1 << 4)
#define MIE_STIE (1 << 5)
#define MIE_MTIE (1 << 7)
#define MIE_UEIE (1 << 8)
#define MIE_SEIE (1 << 9)
#define MIE_MEIE (1 << 11)

#define MEDELEG_UE (1 << 8)
#define MEDELEG_SE (1 << 9)
#define MEDELEG_HE (1 << 10)
#define MEDELEG_ME (1 << 11)

/* 1 ns tick in CYCLE & TIME counters */
#define TICKS_PER_SECOND (1000000)

/**
 * @brief Register file implementation
 */
    template<typename T>
    class Registers {
    public:
        enum {
            x0 = 0,
            x1 = 1,
            x2,
            x3,
            x4,
            x5,
            x6,
            x7,
            x8,
            x9,
            x10,
            x11,
            x12,
            x13,
            x14,
            x15,
            x16,
            x17,
            x18,
            x19,
            x20,
            x21,
            x22,
            x23,
            x24,
            x25,
            x26,
            x27,
            x28,
            x29,
            x30,
            x31,
            zero = x0,
            ra = x1,
            sp = x2,
            gp = x3,
            tp = x4,
            t0 = x5,
            t1 = x6,
            t2 = x7,
            s0 = x8,
            fp = x8,
            s1 = x9,
            a0 = x10,
            a1 = x11,
            a2 = x12,
            a3 = x13,
            a4 = x14,
            a5 = x15,
            a6 = x16,
            a7 = x17,
            s2 = x18,
            s3 = x19,
            s4 = x20,
            s5 = x21,
            s6 = x22,
            s7 = x23,
            s8 = x24,
            s9 = x25,
            s10 = x26,
            s11 = x27,
            t3 = x28,
            t4 = x29,
            t5 = x30,
            t6 = x31,
            ft0 = x0,
            ft1 = x1,
            ft2 = x2,
            ft3 = x3,
            ft4 = x4,
            ft5 = x5,
            ft6 = x6,
            ft7 = x7,
            fs0 = x8,
            fs1 = x9,
            fa0 = x10,
            fa1 = x11,
            fa2 = x12,
            fa3 = x13,
            fa4 = x14,
            fa5 = x15,
            fa6 = x16,
            fa7 = x17,
            fs2 = x18,
            fs3 = x19,
            fs4 = x20,
            fs5 = x21,
            fs6 = x22,
            fs7 = x23,
            fs8 = x24,
            fs9 = x25,
            fs10 = x26,
            fs11 = x27,
            ft8 = x28,
            ft9 = x29,
            ft10 = x30,
            ft11 = x31

        };

        /**
         * Default constructor
         */
        Registers() {
            perf = Performance::getInstance();

            initCSR();
            register_bank[sp] = Memory::SIZE - 4; // default stack at the end of the memory
            register_PC = 0x80000000;       // default _start address
        };

        /**
         * Set value for a register
         * @param reg_num register number
         * @param value   register value
         */
        void setValue(unsigned int reg_num, T value) {
            if ((reg_num != 0) && (reg_num < 32)) {
                register_bank[reg_num] = value;
                perf->registerWrite();
            }
        }
        void setfValue(unsigned int reg_num, T value) {
            if ((reg_num < 32)) {
                register_fbank[reg_num] = value;
                perf->registerWrite();
            }
        }
        /**
         * Returns register value
         * @param  reg_num register number
         * @return         register value
         */
        T getValue(unsigned int reg_num) const {
            if (reg_num < 32) {
                perf->registerRead();
                return register_bank[reg_num];
            } else {
                /* Extend sign for any possible T type */
                return static_cast<T>(std::numeric_limits<T>::max());
            }
        }
        T getfValue(unsigned int reg_num) const {
            if (reg_num < 32) {
                perf->registerRead();
                return register_fbank[reg_num];
            } else {
                /* Extend sign for any possible T type */
                return static_cast<T>(std::numeric_limits<T>::max());
            }
        }

        /**
         * Returns PC value
         * @return PC value
         */
        T getPC() const {
            return register_PC;
        }

        /**
         * Sets arbitrary value to PC
         * @param new_pc new address to PC
         */
        void setPC(T new_pc) {
            register_PC = new_pc;
        }

        /**
         * Increments PC counter to next address
         */
        inline void incPC() {
            register_PC += 4;
        }

        inline void incPCby2() {
            register_PC += 2;
        }

        /**
         * @brief Get CSR value
         * @param csr CSR number to access
         * @return CSR value
         */
        T getCSR(int csr) {
            T ret_value;

            switch (csr) {
                case CSR_CYCLE:
                case CSR_MCYCLE:
                    ret_value = static_cast<std::uint64_t>(sc_core::sc_time(
                            sc_core::sc_time_stamp()
                            - sc_core::sc_time(sc_core::SC_ZERO_TIME)).to_double())
                                & 0x00000000FFFFFFFF;
                    break;
                case CSR_CYCLEH:
                case CSR_MCYCLEH:
                    ret_value = static_cast<std::uint32_t>((std::uint64_t) (sc_core::sc_time(
                            sc_core::sc_time_stamp()
                            - sc_core::sc_time(sc_core::SC_ZERO_TIME)).to_double())
                                                                   >> 32 & 0x00000000FFFFFFFF);
                    break;
                case CSR_TIME:
                    ret_value = static_cast<std::uint64_t>(sc_core::sc_time(
                            sc_core::sc_time_stamp()
                            - sc_core::sc_time(sc_core::SC_ZERO_TIME)).to_double())
                                & 0x00000000FFFFFFFF;
                    break;
                case CSR_TIMEH:
                    ret_value = static_cast<std::uint32_t>((std::uint64_t) (sc_core::sc_time(
                            sc_core::sc_time_stamp()
                            - sc_core::sc_time(sc_core::SC_ZERO_TIME)).to_double())
                                                                   >> 32 & 0x00000000FFFFFFFF);
                    break;
                case CSR_ADDR_FCSR:     // FPU CSR
                    ret_value = ( CSR[CSR_ADDR_FRM] << 5 ) | CSR[CSR_ADDR_FFLAGS];;
                    break;
                [[likely]] default:
                    ret_value = CSR[csr];
                    break;
            }
            return ret_value;
        }

        /**
         * @brief Set CSR value
         * @param csr   CSR number to access
         * @param value new value to register
         */
        void setCSR(int csr, T value) {
            /* @FIXME: rv32mi-p-ma_fetch tests doesn't allow MISA to be writable,
             * but Volume II: Privileged Architecture v1.10 says MISA is writable (?)
             */
            if (csr != CSR_MISA) {
                switch ( csr ){
                    // FPU CSR
                    case CSR_ADDR_FFLAGS:
                        CSR[CSR_ADDR_FFLAGS] = value & 0x1f;
                        CSR[CSR_ADDR_FCSR]   = (CSR[CSR_ADDR_FRM] << 5) | CSR[CSR_ADDR_FFLAGS];
                        break;
                    case CSR_ADDR_FRM:
                        CSR[CSR_ADDR_FRM]    = (value & 0x7);
                        CSR[CSR_ADDR_FCSR]   = (CSR[CSR_ADDR_FRM] << 5) | CSR[CSR_ADDR_FFLAGS];
                        break;
                    case CSR_ADDR_FCSR:
                        CSR[CSR_ADDR_FFLAGS] = value & 0x1f;
                        CSR[CSR_ADDR_FRM]    = (value & 0xe0) >> 5;
                        CSR[CSR_ADDR_FCSR]   = (CSR[CSR_ADDR_FRM] << 5) | CSR[CSR_ADDR_FFLAGS];
                        break;
                    default: 
                        CSR[csr] = value;
                }
            }
        }

        /**
         * Dump register data to console
         */
        void dump() const;
        void fp_dump() const;
        void csr_dump(int addr);
        void reg_copy(){
            register_copy  = register_bank;
            register_fcopy = register_fbank;
        }
        void reg_restore(){
            register_bank  = register_copy;
            register_fbank = register_fcopy;
        }

        void dump_regtofile(bool is_addr){
            std::ofstream ofs(reg_dfname);
            if(!ofs){
                LOG(LOG_WARNING) << "[WARNING] REG Dump failed (File open error!!)\n";
                return;
            }
            if(is_addr){
                ofs << "PC" <<"  || "<<"0x";
            }
            ofs << std::setfill('0')<< std::setw(16)<< std::hex<<register_PC<<std::endl;
            
            for(int i = 0; i< 32; i++){
                if(is_addr){
                    ofs << "x" <<std::setfill('0')<< std::setw(2)<< i << " || "<<"0x";
                }
                ofs << std::setfill('0')<< std::setw(16)<<std::hex << register_bank[i]<<std::endl;
            }
            for(int i = 0; i< 32; i++){
                if(is_addr){
                    ofs << "f" <<std::setfill('0')<< std::setw(2)<< i << " || "<<"0x";
                }
                ofs << std::setfill('0')<< std::setw(16)<< std::hex << register_fbank[i]<<std::endl;
            }
            
            ofs.close();

        }


    private:
        /**
         * bank of registers (32 regs)
         */
        std::array<T, 32> register_bank = {{0}};

        std::array<T, 32> register_copy = {{0}};

        /**
         * bank of f_registers (32 regs)
         */
        std::array<T, 32> register_fbank = {{0}};

        std::array<T, 32> register_fcopy = {{0}};

        /**
         * Program counter (32 bits width)
         */
        T register_PC;

        /**
         * CSR registers (4096 maximum)
         */
        std::unordered_map<unsigned int, T> CSR;

        Performance *perf;

        void initCSR();
    };
}
#endif
