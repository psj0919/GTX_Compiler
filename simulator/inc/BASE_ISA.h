
#ifndef Execute_H
#define Execute_H

#define SC_INCLUDE_DYNAMIC_PROCESSES
#include <type_traits>
#include <limits>
#include "systemc"
#include "vert_dev.h"
#include "config.h"

#include "BASE_ISA.h"
#include "extension_base.h"
#include "Instrcycle_info.h"
#include "tlm_syscall_interface.h"

extern bool debug_session;

namespace riscv_tlm {

    typedef enum {
        OP_LUI,
        OP_AUIPC,
        OP_JAL,
        OP_JALR,

        OP_BEQ,
        OP_BNE,
        OP_BLT,
        OP_BGE,
        OP_BLTU,
        OP_BGEU,

        OP_LB,
        OP_LH,
        OP_LW,
        OP_LBU,
        OP_LHU,

        OP_SB,
        OP_SH,
        OP_SW,

        OP_ADDI,
        OP_SLTI,
        OP_SLTIU,
        OP_XORI,
        OP_ORI,
        OP_ANDI,
        OP_SLLI,
        OP_SRLI,
        OP_SRAI,

        OP_ADD,
        OP_SUB,
        OP_SLL,
        OP_SLT,
        OP_SLTU,
        OP_XOR,
        OP_SRL,
        OP_SRA,
        OP_OR,
        OP_AND,

        OP_FENCE,
        OP_ECALL,
        OP_EBREAK,

        OP_CSRRW,
        OP_CSRRS,
        OP_CSRRC,
        OP_CSRRWI,
        OP_CSRRSI,
        OP_CSRRCI,

        OP_URET,
        OP_SRET,
        OP_MRET,
        OP_WFI,
        OP_SFENCE,

         /* RV64 */
        OP_LWU,
        OP_LD,
        OP_SD,
        OP_ADDIW,
        OP_SLLIW,
        OP_SRLIW,
        OP_SRAIW,
        OP_ADDW,
        OP_SUBW,
        OP_SLLW,
        OP_SRLW,
        OP_SRAW,

        OP_ERROR
    } opCodes;

    enum Codes {
        LUI = 0b0110111,
        AUIPC = 0b0010111,
        JAL = 0b1101111,
        JALR = 0b1100111,

        BEQ = 0b1100011,
        BEQ_F = 0b000,
        BNE_F = 0b001,
        BLT_F = 0b100,
        BGE_F = 0b101,
        BLTU_F = 0b110,
        BGEU_F = 0b111,

        LB = 0b0000011,
        LB_F = 0b000,
        LH_F = 0b001,
        LW_F = 0b010,
        LBU_F = 0b100,
        LHU_F = 0b101,
        LWU_F = 0b110,
        LD_F = 0b011,

        SB = 0b0100011,
        SB_F = 0b000,
        SH_F = 0b001,
        SW_F = 0b010,
        SD_F = 0b011,

        ADDI = 0b0010011,
        ADDI_F = 0b000,
        SLTI_F = 0b010,
        SLTIU_F = 0b011,
        XORI_F = 0b100,
        ORI_F = 0b110,
        ANDI_F = 0b111,
        SLLI_F = 0b001,
        SRLI_F = 0b101,
        SRLI_F7 = 0b000000,
        SRAI_F7 = 0b010000,

        ADD = 0b0110011,
        ADD_F = 0b000,
        SUB_F = 0b000,
        ADD_F7 = 0b0000000,
        SUB_F7 = 0b0100000,

        SLL_F = 0b001,
        SLT_F = 0b010,
        SLTU_F = 0b011,
        XOR_F = 0b100,
        SRL_F = 0b101,
        SRA_F = 0b101,
        SRL_F7 = 0b0000000,
        SRA_F7 = 0b0100000,
        OR_F = 0b110,
        AND_F = 0b111,

        FENCE = 0b0001111,
        ECALL = 0b1110011,
        ECALL_F = 0b000000000000,
        EBREAK_F = 0b000000000001,
        URET_F = 0b000000000010,
        SRET_F = 0b000100000010,
        MRET_F = 0b001100000010,
        WFI_F = 0b000100000101,
        SFENCE_F = 0b0001001,

        ECALL_F3 = 0b000,
        CSRRW = 0b001,
        CSRRS = 0b010,
        CSRRC = 0b011,
        CSRRWI = 0b101,
        CSRRSI = 0b110,
        CSRRCI = 0b111,

        ADDIW = 0b0011011,
        ADDIW_F = 0b000,
        SLLIW_F = 0b001,
        SRLIW_F = 0b101,
        SRLIW_F7 = 0b000000,
        SRAIW_F7 = 0b010000,
        ADDW = 0b0111011,
        ADDW_F = 0b000,
        ADDW_F7 = 0b000000,
        SUBW_F7 = 0b010000,
        SLLW = 0b001,
        SRLW = 0b101,
        SRLW_F7 = 0b000000,
        SRAW_F7 = 0b010000,
    };

/**
 * @brief Risc_V execute module
 */
    template<typename T>
    class BASE_ISA : public extension_base<T> {
    public:
        //virtual machine 
        riscv_tlm::peripherals::VMachine *vm;
        
        bool wait_intr;
        bool stop_wait;
        
        //CPU handle for kernel

        TLMSyscallInterface* syscall_interface;

        void set_syscall_interface(TLMSyscallInterface* interface) {
         this->syscall_interface = interface;
         }
     
        TLMSyscallInterface* get_syscall_interface() {
            return this->syscall_interface;
        }
     

        /**
         * @brief Constructor, same as base class
         */
        using extension_base<T>::extension_base;

        /**
         * @brief Deduce signed type for T type
         */
        using signed_T = typename std::make_signed<T>::type;

        /**
         * @brief Deduce unsigned type for T type
         */
        using unsigned_T = typename std::make_unsigned<T>::type;

        /**
         * @brief Access to funct7 field
         * @return funct7 field
         */
        inline std::uint32_t get_funct7() const {
            return this->m_instr.range(31, 25);
        }

        /**
         * @brief Access to funct7 field in case of RV64 SLLL, SRLI, SRAI is different
         * @return funct7 field-
         */
        std::uint32_t get_funct7b() const;

        /**
         * @brief Gets immediate field value for I-type
         * @return immediate_I field
         * @note Specialized
         */
        signed_T get_imm_I() const;

        /**
         * @brief Gets immediate field value for S-type
         * @return immediate_S field
         * @note Specialized
         */
         signed_T get_imm_S() const;

        /**
         * @brief Gets immediate field value for U-type
         * @return immediate_U field
         */
        inline unsigned_T get_imm_U() const {
            return this->m_instr.range(31, 12);
        }

        /**
         * @brief Gets immediate field value for B-type
         * @return immediate_B field
         * @note Specialized
         */
        signed_T get_imm_B() const;

        /**
         * @brief Gets immediate field value for J-type
         * @return immediate_J field
         * @note Specialized
         */
        signed_T get_imm_J() const;

        /**
         * @brief Returns shamt field for Shifts instructions
         * @return value corresponding to inst(25:20)
         */
        inline unsigned_T get_shamt() const {
            return this->m_instr.range(25, 20);
        }

        /**
         * @brief Returns CSR field for CSR instructions
         * @return value corresponding to instr(31:20)
         */
        inline unsigned_T get_csr() const {
            return this->m_instr.range(31, 20);
        }



        /**
         * @brief Access to opcode field
         * @return return opcode field
         */
        inline unsigned_T opcode() const override {
            return this->m_instr.range(6, 0);
        }

        bool Exec_LUI() const {
            unsigned int rd;
            unsigned_T imm;

            rd = this->get_rd();
            imm = static_cast<std::int32_t>(get_imm_U() << 12);
            this->regs->setValue(rd, imm);
            this->perf->cycleInc(LUI_CYC);
            return true;
        }

        bool Exec_AUIPC() const {
            unsigned int rd;
            unsigned_T imm;
            unsigned_T new_pc;

            rd = this->get_rd();
            imm = static_cast<std::int32_t>(get_imm_U() << 12);
            new_pc = this->regs->getPC() + imm;

            this->regs->setValue(rd, new_pc);

            this->perf->cycleInc(AUIPC_CYC);
            return true;
        }

        bool Exec_JAL() const {
            unsigned int rd;
            signed_T mem_addr;
            unsigned_T new_pc, old_pc;

            rd = this->get_rd();
            mem_addr = get_imm_J();
            old_pc = this->regs->getPC();
            new_pc = old_pc + mem_addr;

            this->regs->setPC(new_pc);

            old_pc = old_pc + 4;
            this->regs->setValue(rd, old_pc);

            if(mem_addr == 0){ 
                LOG(LOG_INFO) << "\n=============================================";
                LOG(LOG_INFO) << "\n[INFO] Program Finished at PC(self jump) : 0x"<< std::hex << std::uppercase << new_pc << "\nexiting...";
                LOG(LOG_INFO) << "\n=============================================\n\n";
                debug_session = false;
                exitcode = this->regs->getValue(10);
                if (sc_core::sc_is_running() && !sc_core::sc_end_of_simulation_invoked()) {
                    if(!is_monitoring) sc_core::sc_stop();
                }
            }
            this->perf->cycleInc(JAL_CYC);

            return true;
        }

        bool Exec_JALR() const {
            signed_T offset;
            unsigned int rd, rs1;
            unsigned_T new_pc, old_pc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            offset = get_imm_I();
            old_pc = this->regs->getPC();

            new_pc = static_cast<unsigned_T>((this->regs->getValue(rs1) + offset) & ~1);
            this->regs->setValue(rd, old_pc + 4);
            this->regs->setPC(new_pc);

            this->perf->cycleInc(JALR_CYC);
            return true;
        }

        bool Exec_BEQ() const {
            unsigned int rs1, rs2;
            unsigned_T val1, val2;

            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            val1 = this->regs->getValue(rs1);
            val2 = this->regs->getValue(rs2);

            if (val1 == val2) {
                unsigned_T new_pc;
                new_pc = static_cast<unsigned_T>(this->regs->getPC() + get_imm_B());
                this->regs->setPC(new_pc);
            } else {
                this->regs->incPC();
            }

            this->perf->cycleInc(BEQ_CYC);

            return true;
        }

        bool Exec_BNE() const {
            unsigned int rs1, rs2;
            unsigned_T val1, val2;
            unsigned_T old_pc;

            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            val1 = this->regs->getValue(rs1);
            val2 = this->regs->getValue(rs2);
            old_pc = static_cast<unsigned_T>(this->regs->getPC());

            if (val1 != val2) {
                unsigned_T new_pc;
                new_pc = static_cast<unsigned_T>(this->regs->getPC() + get_imm_B());
                this->regs->setPC(new_pc);
            } else {
                this->regs->incPC();
            }

            this->perf->cycleInc(BNE_CYC);
            return true;
        }

        bool Exec_BLT() const {
            unsigned int rs1, rs2;
            signed_T val1, val2;
            unsigned_T old_pc;

            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            val1 = static_cast<signed_T>(this->regs->getValue(rs1));
            val2 = static_cast<signed_T>(this->regs->getValue(rs2));
            old_pc = static_cast<unsigned_T>(this->regs->getPC());

            if (val1 < val2) {
                unsigned_T new_pc;
                new_pc = static_cast<unsigned_T>(this->regs->getPC() + get_imm_B());
                this->regs->setPC(new_pc);
            } else {
                this->regs->incPC();
            }

            this->perf->cycleInc(BLT_CYC);
            return true;
        }

        bool Exec_BGE() const {
            unsigned int rs1, rs2;
            signed_T val1, val2;
            unsigned_T old_pc;

            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            val1 = static_cast<signed_T>(this->regs->getValue(rs1));
            val2 = static_cast<signed_T>(this->regs->getValue(rs2));
            old_pc = static_cast<unsigned_T>(this->regs->getPC());

            if (val1 >= val2) {
                unsigned_T new_pc;
                new_pc = static_cast<unsigned_T>(this->regs->getPC() + get_imm_B());
                this->regs->setPC(new_pc);
            } else {
                this->regs->incPC();
            }

            this->perf->cycleInc(BGE_CYC);
            return true;
        }

        bool Exec_BLTU() const {
            unsigned int rs1, rs2;
            unsigned_T val1, val2;
            unsigned_T old_pc;

            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            val1 = this->regs->getValue(rs1);
            val2 = this->regs->getValue(rs2);
            old_pc = static_cast<unsigned_T>(this->regs->getPC());

            if (val1 < val2) {
                unsigned_T new_pc;
                new_pc = static_cast<unsigned_T>(old_pc + get_imm_B());
                this->regs->setPC(new_pc);
            } else {
                this->regs->incPC();
            }
            this->perf->cycleInc(BLTU_CYC);
            return true;
        }

        bool Exec_BGEU() const {
            unsigned int rs1, rs2;
            unsigned_T val1, val2;
            unsigned_T old_pc;

            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            val1 = this->regs->getValue(rs1);
            val2 = this->regs->getValue(rs2);
            old_pc = static_cast<unsigned_T>(this->regs->getPC());

            if (val1 >= val2) {
                unsigned_T new_pc;
                new_pc = static_cast<unsigned_T>(this->regs->getPC() + get_imm_B());
                this->regs->setPC(new_pc);
            } else {
                this->regs->incPC();
            }
            this->perf->cycleInc(BGEU_CYC);
            return true;
        }

        bool Exec_LB() const {
            unsigned_T mem_addr;
            unsigned int rd, rs1;
            signed_T imm;
            std::int8_t data;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            imm = get_imm_I();

            mem_addr = imm + this->regs->getValue(rs1);
            data = static_cast<std::int8_t>(this->mem_intf->readDataMem(mem_addr, 1));
            this->perf->dataMemoryRead();
            this->regs->setValue(rd, data);
            this->perf->cycleInc(LB_CYC);
            return true;
        }

        bool Exec_LH() const {
            unsigned_T mem_addr;
            unsigned int rd, rs1;
            signed_T imm;
            std::int16_t data;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            imm = get_imm_I();

            mem_addr = imm + this->regs->getValue(rs1);
            data = static_cast<std::int16_t>(this->mem_intf->readDataMem(mem_addr, 2));
            this->perf->dataMemoryRead();
            this->regs->setValue(rd, data);
            this->perf->cycleInc(LH_CYC);
            return true;
        }

        bool Exec_LW() const {
            unsigned_T mem_addr;
            unsigned int rd, rs1;
            signed_T imm;
            std::int32_t data;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            imm = get_imm_I();

            mem_addr = imm + this->regs->getValue(rs1);
            data = static_cast<std::int32_t>(this->mem_intf->readDataMem(mem_addr, 4));

            this->perf->dataMemoryRead();
            this->regs->setValue(rd, data);
            this->perf->cycleInc(LW_CYC);

            return true;
        }

        bool Exec_LBU() const {
            unsigned_T mem_addr;
            unsigned int rd, rs1;
            signed_T imm;
            std::uint8_t data;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            imm = get_imm_I();

            mem_addr = imm + this->regs->getValue(rs1);
            data =  static_cast<std::uint8_t>(this->mem_intf->readDataMem(mem_addr, 1));

            this->perf->dataMemoryRead();
            this->regs->setValue(rd, data);
            this->perf->cycleInc(LBU_CYC);
            return true;
        }

        bool Exec_LHU() const {
            unsigned_T mem_addr;
            unsigned int rd, rs1;
            signed_T imm;
            std::uint16_t data;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            imm = get_imm_I();

            mem_addr = imm + this->regs->getValue(rs1);
            data =  static_cast<std::uint16_t>(this->mem_intf->readDataMem(mem_addr, 2));

            this->perf->dataMemoryRead();
            this->regs->setValue(rd, data);

            this->perf->cycleInc(LHU_CYC);
            return true;
        }

        bool Exec_LWU() const {
            unsigned_T mem_addr;
            unsigned int rd, rs1;
            signed_T imm;
            std::uint32_t data;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            imm = get_imm_I();

            mem_addr = imm + this->regs->getValue(rs1);
            data = this->mem_intf->readDataMem(mem_addr, 4);

            this->perf->dataMemoryRead();
            this->regs->setValue(rd, data);
            this->perf->cycleInc(LWU_CYC);
            return true;
        }

        bool Exec_LD() const {
            unsigned_T mem_addr;
            unsigned int rd, rs1;
            signed_T imm;
            signed_T offset;

            std::uint64_t data;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            imm = get_imm_I();

            offset = this->regs->getValue(rs1);
            mem_addr = imm + offset;

            data = static_cast<std::uint32_t>(this->mem_intf->readDataMem(mem_addr, 4));
            std::uint64_t aux = static_cast<std::uint32_t>(this->mem_intf->readDataMem(mem_addr + 4, 4));
            data |= aux << 32;

            this->perf->dataMemoryRead();
            this->regs->setValue(rd, data);
            this->perf->cycleInc(LD_CYC);
            return true;
        }


        bool Exec_SD() const {
            unsigned_T mem_addr;
            unsigned int rs1, rs2;
            signed_T imm;
            std::uint64_t data;

            rs1 = this->get_rs1();
            rs2 = this->get_rs2();
            imm = get_imm_S();

            mem_addr = imm + this->regs->getValue(rs1);
            data = this->regs->getValue(rs2);

            this->mem_intf->writeDataMem(mem_addr, data & 0xFFFFFFFF, 4);
            this->mem_intf->writeDataMem(mem_addr + 4, data >> 32, 4);
            this->perf->dataMemoryWrite();
            this->perf->cycleInc(SD_CYC);
            return true;
        }


        bool Exec_SB() const {
            unsigned_T mem_addr;
            unsigned int rs1, rs2;
            signed_T imm;
            std::uint32_t data;

            rs1 = this->get_rs1();
            rs2 = this->get_rs2();
            imm = get_imm_S();

            mem_addr = imm + this->regs->getValue(rs1);
            data = this->regs->getValue(rs2);

            this->mem_intf->writeDataMem(mem_addr, data, 1);
            this->perf->dataMemoryWrite();
            this->perf->cycleInc(SB_CYC);

            return true;
        }


        bool Exec_SH() const {
            unsigned_T mem_addr;
            unsigned int rs1, rs2;
            signed_T imm;
            std::uint32_t data;

            rs1 = this->get_rs1();
            rs2 = this->get_rs2();
            imm = get_imm_S();

            mem_addr = imm + this->regs->getValue(rs1);
            data = this->regs->getValue(rs2);

            this->mem_intf->writeDataMem(mem_addr, data, 2);
            this->perf->dataMemoryWrite();
            this->perf->cycleInc(SH_CYC);
            return true;
        }


        bool Exec_SW() const {
            unsigned_T mem_addr;
            unsigned int rs1, rs2;
            signed_T imm;
            std::uint32_t data;

            rs1 = this->get_rs1();
            rs2 = this->get_rs2();
            imm = get_imm_S();

            mem_addr = imm + this->regs->getValue(rs1);
            data = this->regs->getValue(rs2);

            this->mem_intf->writeDataMem(mem_addr, data, 4);
            this->perf->dataMemoryWrite();
            this->perf->cycleInc(SW_CYC);
            return true;
        }

        bool Exec_ADDI() const {
            unsigned int rd, rs1;
            signed_T imm;
            unsigned_T calc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            imm = get_imm_I();

            calc = this->regs->getValue(rs1) + imm;

            this->regs->setValue(rd, calc);
            this->perf->cycleInc(ADDI_CYC);

            return true;
        }

        bool Exec_ADDIW() const {
            unsigned int rd, rs1;
            std::int32_t imm;
            std::int32_t aux;
            std::int64_t calc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            imm = get_imm_I();

            aux = static_cast<std::int32_t>(this->regs->getValue(rs1) & 0xFFFFFFFF);
            aux = static_cast<std::int32_t>(aux + imm);
            calc = static_cast<std::int32_t>(aux);

            this->regs->setValue(rd, calc);
            this->perf->cycleInc(ADDIW_CYC);
            return true;
        }

        bool Exec_SLTI() const {
            unsigned int rd, rs1;
            signed_T imm;
            signed_T val1;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            imm = get_imm_I();

            val1 = static_cast<signed_T>(this->regs->getValue(rs1));

            if (val1 < imm) {
                this->regs->setValue(rd, 1);
            } else {
                this->regs->setValue(rd, 0);
            }
            this->perf->cycleInc(SLTI_CYC);
            return true;
        }

        bool Exec_SLTIU() const {
            unsigned int rd, rs1;
            unsigned_T imm;
            unsigned_T val1;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            imm = static_cast<unsigned_T>(get_imm_I());
            val1 = static_cast<unsigned_T>(this->regs->getValue(rs1));

            if (val1 < imm) {
                this->regs->setValue(rd, 1);

            } else {
                this->regs->setValue(rd, 0);

            }
            this->perf->cycleInc(SLTIU_CYC);
            return true;
        }

        bool Exec_XORI() const {
            unsigned int rd, rs1;
            signed_T imm;
            unsigned_T calc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            imm = get_imm_I();

            calc = this->regs->getValue(rs1) ^ imm;
            this->regs->setValue(rd, calc);

            this->perf->cycleInc(XORI_CYC);
            return true;
        }

        bool Exec_ORI() const {
            unsigned int rd, rs1;
            signed_T imm;
            unsigned_T calc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            imm = get_imm_I();

            calc = this->regs->getValue(rs1) | imm;
            this->regs->setValue(rd, calc);
            this->perf->cycleInc(ORI_CYC);
            return true;
        }

        bool Exec_ANDI() const {
            unsigned int rd, rs1;
            signed_T imm;
            unsigned_T calc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            imm = get_imm_I();

            calc = this->regs->getValue(rs1) & imm;
            this->regs->setValue(rd, calc);
            this->perf->cycleInc(ANDI_CYC);
            return true;
        }


        bool Exec_SLLI();


        bool Exec_SRLI() const;


        bool Exec_SRAI() const;


        bool Exec_SLLIW() {
            unsigned int rd, rs1, rs2;
            unsigned_T shift;
            unsigned_T calc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = get_shamt();

            if (rs2 >= 0x20) {
                LOG(LOG_ERROR) << "[ERROR] ILLEGAL INSTRUCTION, shamt[5] != 0" << "\n";
                if(!is_monitoring){
                    sc_core::sc_stop();
                    this->RaiseException(Exception_cause::ILLEGAL_INSTRUCTION, this->m_instr);
                }

                return false;
            }

            shift = rs2 & 0x1F;

            calc = static_cast<std::int32_t>(static_cast<std::uint32_t>(this->regs->getValue(rs1)) << shift);
            this->regs->setValue(rd, calc);

            this->perf->cycleInc(SLLIW_CYC);
            return true;
        }


        bool Exec_SRLIW() {
            unsigned int rd, rs1, rs2;
            unsigned_T shift;
            unsigned_T calc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = get_shamt();

            if (rs2 >= 0x20) {
                LOG(LOG_ERROR) << "[ERROR] ILLEGAL INSTRUCTION, shamt[5] != 0" << "\n";
                if(!is_monitoring){
                sc_core::sc_stop();
                this->RaiseException(Exception_cause::ILLEGAL_INSTRUCTION, this->m_instr);
                }
                this->perf->cycleInc(SRLIW_CYC);
                return false;
            }

            shift = rs2 & 0x1F;

            calc = static_cast<std::int32_t>(static_cast<std::uint32_t>(this->regs->getValue(rs1) & 0xFFFFFFFF) >> shift);
            this->regs->setValue(rd, calc);
            this->perf->cycleInc(SRLIW_CYC);
            return true;
        }


        bool Exec_SRAIW() const {
            unsigned int rd, rs1, rs2;
            unsigned_T shift;
            signed_T calc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_shamt_slli();

            shift = rs2 & 0x1F;

            calc = static_cast<std::int32_t>(static_cast<std::int32_t>(this->regs->getValue(rs1)) >> shift);
            this->regs->setValue(rd, calc);
            this->perf->cycleInc(SRAIW_CYC);
            return true;
        }

        bool Exec_ADDW() const {
            unsigned int rd, rs1, rs2;
            unsigned_T calc;
            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            calc = static_cast<std::int32_t>((this->regs->getValue(rs1) + this->regs->getValue(rs2)) & 0xFFFFFFFF);

            this->regs->setValue(rd, calc);
            this->perf->cycleInc(ADDW_CYC);

            return true;
        }

        bool Exec_SUBW() const {
            unsigned int rd, rs1, rs2;
            unsigned_T calc;
            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            calc =  static_cast<std::int32_t>((this->regs->getValue(rs1) - this->regs->getValue(rs2)) & 0xFFFFFFFF);

            this->regs->setValue(rd, calc);
            this->perf->cycleInc(SUBW_CYC);

            return true;
        }


        bool Exec_SLLW() const {
            unsigned int rd, rs1, rs2;
            unsigned_T shift;
            unsigned_T calc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            shift = this->regs->getValue(rs2) & 0x1F;
            calc = static_cast<std::int32_t>(this->regs->getValue(rs1) & 0xFFFFFFFF) << shift;

            this->regs->setValue(rd, calc);
            this->perf->cycleInc(SLLW_CYC);

            return true;
        }


        bool Exec_SRLW() const {
            unsigned int rd, rs1, rs2;
            unsigned_T shift;
            unsigned_T calc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            shift = this->regs->getValue(rs2) & 0x1F;
            calc = static_cast<std::int32_t>(static_cast<std::uint32_t>(this->regs->getValue(rs1) & 0xFFFFFFFF) >> shift);

            this->regs->setValue(rd, calc);
            this->perf->cycleInc(SRLW_CYC);
            return true;
        }


        bool Exec_SRAW() const {
            unsigned int rd, rs1, rs2;
            unsigned_T shift;
            signed_T calc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            shift = this->regs->getValue(rs2) & 0x1F;
            calc = static_cast<std::int32_t>(this->regs->getValue(rs1) & 0xFFFFFFFF) >> shift;

            this->regs->setValue(rd, calc);
            this->perf->cycleInc(SRAW_CYC);
            return true;
        }

        bool Exec_ADD() const {
            unsigned int rd, rs1, rs2;
            signed_T calc;
            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            calc = static_cast<signed_T>(this->regs->getValue(rs1)) + static_cast<signed_T>(this->regs->getValue(rs2));

            this->regs->setValue(rd, calc);

            this->perf->cycleInc(ADD_CYC);
            return true;
        }

        bool Exec_SUB() const {
            unsigned int rd, rs1, rs2;
            signed_T calc;
            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            calc = static_cast<signed_T>(this->regs->getValue(rs1)) - static_cast<signed_T>(this->regs->getValue(rs2));
            this->regs->setValue(rd, calc);

            this->perf->cycleInc(SUB_CYC);
            return true;
        }


        bool Exec_SLL() const;

        bool Exec_SLT() const {
            unsigned int rd, rs1, rs2;
            signed_T val1, val2;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            val1 = static_cast<signed_T>(this->regs->getValue(rs1));
            val2 = static_cast<signed_T>(this->regs->getValue(rs2));

            if (val1 < val2) {
                this->regs->setValue(rd, 1);
            } else {
                this->regs->setValue(rd, 0);

            }
            this->perf->cycleInc(SLT_CYC);
            return true;
        }

        bool Exec_SLTU() const {
            unsigned int rd, rs1, rs2;
            unsigned_T val1, val2;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            val1 = this->regs->getValue(rs1);
            val2 = this->regs->getValue(rs2);

            if (val1 < val2) {
                this->regs->setValue(rd, 1);

            } else {
                this->regs->setValue(rd, 0);

            }
            this->perf->cycleInc(SLTU_CYC);
            return true;
        }

        bool Exec_XOR() const {
            unsigned int rd, rs1, rs2;
            unsigned_T calc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            calc = this->regs->getValue(rs1) ^ this->regs->getValue(rs2);
            this->regs->setValue(rd, calc);
            this->perf->cycleInc(XOR_CYC);
            return true;
        }


        bool Exec_SRL() const;


        bool Exec_SRA() const;

        bool Exec_OR() const {
            unsigned int rd, rs1, rs2;
            unsigned_T calc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            calc = this->regs->getValue(rs1) | this->regs->getValue(rs2);
            this->regs->setValue(rd, calc);
            this->perf->cycleInc(OR_CYC);
            return true;
        }

        bool Exec_AND() const {
            unsigned int rd, rs1, rs2;
            unsigned_T calc;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            calc = this->regs->getValue(rs1) & this->regs->getValue(rs2);
            this->regs->setValue(rd, calc);
            this->perf->cycleInc(AND_CYC);

            return true;
        }

        bool Exec_FENCE() const {
            
            // Check if next instruction is a FENCE, if so, stop simulation
            uint32_t ant_pc = this->regs->getPC();
            ant_pc = ant_pc + 4;
            uint32_t ant_inst = this->mem_intf->readDataMem(ant_pc, 4);

            if (ant_inst == 0x00000073) {
                if(!is_monitoring) sc_core::sc_stop();
            }
            this->perf->cycleInc(FENCE_CYC);
            return true;
        }

        bool Exec_ECALL() {
            uint64_t medeleg = this->regs->getCSR(CSR_MEDELEG);
            if((medeleg & 0x200)){
            if((kernel_switch ==1) && check_syscall()){
                this->regs->setCSR(CSR_MTVEC, (this->regs->getPC() + 4));
                this->RaiseException(Exception_cause::CALL_FROM_M_MODE, this->m_instr);
                return false;
            }
            else if((kernel_switch == 2) && exec_syscall()){
                this->regs->setCSR(CSR_MTVEC, (this->regs->getPC() + 4));
                this->RaiseException(Exception_cause::CALL_FROM_M_MODE, this->m_instr);
                return false;
            }else{
                LOG(LOG_WARNING) << "[WARNING] ecall failed (only support M mode)!!\n"  << std::endl;
                LOG(LOG_INFO) << "============================================="<<std::endl;
                LOG(LOG_INFO) << "[INFO] End simulation(ECALL failed)..\n";
                LOG(LOG_INFO) << "============================================="<<std::endl << std::endl;
                //this->perf->dump();
                if(!is_monitoring) sc_core::sc_stop();
            }
            }else{
            this->RaiseException(Exception_cause::CALL_FROM_M_MODE, this->m_instr);
            this->perf->cycleInc(ECALL_CYC);
            if(this->regs->getPC() == 0){
                LOG(LOG_WARNING) << "[WARNING] End Simulation because ECALL executed without a valid trap handler!!\n"  << std::endl; 
                LOG(LOG_INFO) << "============================================="<<std::endl;
                LOG(LOG_INFO) << "[INFO] End simulation(ECALL failed)..\n";
                LOG(LOG_INFO) << "============================================="<<std::endl << std::endl;
                //this->regs->dump();
                //LOG(LOG_INFO) << "[INFO] Simulation time " << sc_core::sc_time_stamp() << "\n";
                //this->perf->dump();
                if(!is_monitoring) sc_core::sc_stop();
            }
            }
            return false;//PC effected
        }
    
        bool exec_syscall(){
            this->perf->cycleInc(ECALL_CYC);
         
            // Handle system call via TLM Proxy Kernel
            // Get system call number from a7 register (x17)
            typename BASE_ISA<T>::unsigned_T syscall_num = this->regs->getValue(17);

            // Get arguments from registers a0-a5 (x10-x15)
            typename BASE_ISA<T>::unsigned_T arg0 = this->regs->getValue(10);
            typename BASE_ISA<T>::unsigned_T arg1 = this->regs->getValue(11);
            typename BASE_ISA<T>::unsigned_T arg2 = this->regs->getValue(12);
            typename BASE_ISA<T>::unsigned_T arg3 = this->regs->getValue(13);
            typename BASE_ISA<T>::unsigned_T arg4 = this->regs->getValue(14);
            typename BASE_ISA<T>::unsigned_T arg5 = this->regs->getValue(15);

            typename BASE_ISA<T>::signed_T result = 0;

            // Try to use TLM Proxy Kernel if available
            if (syscall_interface != nullptr) {
                result = syscall_interface->execute_syscall(
                    syscall_num, arg0, arg1, arg2, arg3, arg4, arg5, 0);
            } else {
                // Fallback to basic implementation
                switch (syscall_num) {
                    case 93: // sys_exit
                        LOG(LOG_INFO) << "\n=============================================";
                        LOG(LOG_INFO) << "\n[INFO] Program exit by systemcall(Exit_code:"<<arg0 <<")"<< std::endl;
                        LOG(LOG_INFO) << "=============================================\n\n";
                        sc_core::sc_stop();
                        break;
                    case 64: // sys_write 
                     if (arg0 == 1) { // stdout
                        //  std::cout << "[STDOUT] Hello World output detected" << std::endl;
                         result = arg2; // return number of bytes written
                     }
                    break;
                    default:
                        LOG(LOG_WARNING) << "[WARNING] System call " << syscall_num << " not implemented" << std::endl;
                        result = -1; // ENOSYS
                        break;
                }
            }

            // Set return value in a0 register
            this->regs->setValue(10, result);

            return true; 
        }
        bool check_syscall(){
            int num;
            char c;
            bool pass = true;
            uint64_t a7, a0, a1, a2, a3, a4, a5;
            
            a7 = this->regs->getValue(17);
            a0 = this->regs->getValue(10);
            a1 = this->regs->getValue(11);
            a2 = this->regs->getValue(12);
            
            switch(a7){
                case 62://seek
                    if(vm->cmd_seek(a0,a1,a2)){
                        this->regs->setValue(10,0);
                    }else{
                        this->regs->setValue(10,-1);
                        LOG(LOG_DEBUG) << "[DEBUG][SK] seek failed in ISS\n";
                    }
                    break;
                case 63://read
                    if((a0 == 1)||(a0 == 2)){
                        this->regs->setValue(10,-1);
                    }else{
                        if(vm->cmd_read(a0,a1,a2,true)){
                            this->regs->setValue(10,a2);
                        }else{
                            this->regs->setValue(10,-1);
                            LOG(LOG_DEBUG) << "[DEBUG][SK] fread failed in ISS\n";
                        }
                    }
                    break;
                case 64://write
                    if(a0  == 1){
                        vm ->cmd_write(0,a1,a2);
                        this->regs->setValue(10,a2);
                    }
                    else if(a0 == 0){
                        this->regs->setValue(10,-1);
                        LOG(LOG_DEBUG) << "[DEBUG][SK] write failed in ISS\n";
                    }
                    else{
                        if(vm->cmd_write(a0,a1,a2)){
                            this->regs->setValue(10,a2);
                        }else{
                            this->regs->setValue(10,-1);
                            LOG(LOG_DEBUG) << "[DEBUG][SK] write failed in ISS\n";
                        }
                    }
                    break;
                case 57://close
                    if((a0 == 1) || (a0 == 0) || (a0 == 2)){
                        this->regs->setValue(10,0);
                    }else{
                        if(vm ->cmd_close(a0)){
                            this->regs->setValue(10,0);
                        }{
                            this->regs->setValue(10,-1);
                            LOG(LOG_DEBUG) << "[DEBUG][SK] close failed in ISS\n";
                        }
                        
                    }
                    break;
                case 80://check file discriptor
                    if(a0 == 1 ){
                        this->regs->setValue(10,0);
                    }else if(vm->is_fdt_exist(a0)){
                        this->regs->setValue(10,0);
                    }else{
                        pass = false;
                    }
                    break;
                case 93:// [sys exit] a0:err code
                    LOG(LOG_INFO) << "\n=============================================";
                    LOG(LOG_INFO) << "\n[INFO] Program exit by systemcall(Exit_code:"<<a0 <<")"<< std::endl;
                    LOG(LOG_INFO) << "=============================================\n\n";
                    exitcode = a0;
                    sc_core::sc_stop();
                    return true;
                    break;
                case 214:
                    // heap pointer ++
                    if (a0 !=0){
                        vm->heap_end = a0;
                    }
                    this->regs->setValue(10,vm->heap_end); 
                    
                    break;
                case 56:
                case 1024://fopen
                    num = 0;
                    c = ' ';
                    while(c != '\0'){
                        c = this->mem_intf->readDataMem(a0 + num, 1);
                        num++;
                    }
                    if(a1 == 0x601){//simple write
                        if(vm -> cmd_open(a0, num, OF_W , 0)){
                            this->regs->setValue(10,vm->recent_fdt);
                        }else{
                            this->regs->setValue(10,-1);
                            LOG(LOG_DEBUG) << "[DEBUG][SK] open failed in ISS\n";
                        }
                        
                    }else if(a1 == 0x0){//simple read
                        if(vm -> cmd_open(a0,num,OF_R,0)){
                            this->regs->setValue(10,vm->recent_fdt);
                        }else{
                            this->regs->setValue(10,-1);
                            LOG(LOG_DEBUG) << "[DEBUG][SK] open failed in ISS\n";
                        }
                    }else{
                        pass = false;
                    }
                    

                    break;
                default:
                    pass = false;
                    break;
                
                
            }
            if(pass){
                LOG(LOG_DEBUG) << "[DEBUG][SK] ECALL invoked but handled by ISS kernel. / syscall num - " <<a7<<std::endl;
                return true;
            }else {
                return false;
            }
        }

        bool Exec_EBREAK() {

            LOG(LOG_INFO) << std::endl << "[INFO] EBREAK  Instruction called, dumping information"
                      << std::endl;
            this->regs->dump();
            LOG(LOG_INFO) << "[INFO] Simulation time " << sc_core::sc_time_stamp() << "\n";
            //this->perf->dump();

            this->RaiseException(Exception_cause::BREAK, this->m_instr);
            this->perf->cycleInc(EBREAK_CYC);
            return false;
        }

        bool Exec_CSRRW() const {
            unsigned int rd, rs1;
            int csr;
            unsigned_T aux, aux2;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            csr = get_csr();

            aux = this->regs->getCSR(csr);
            aux2 = this->regs->getValue(rs1);

            if (rd != 0) {
                this->regs->setValue(rd, aux);
            }

            this->regs->setCSR(csr, aux2);

            this->perf->cycleInc(CSRRW_CYC);
            return true;
        }

        bool Exec_CSRRS() const {
            unsigned int rd, rs1;
            int csr;
            unsigned_T bitmask, aux, aux2;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            csr = get_csr();

            

            /* These operations must be atomical */
            aux = this->regs->getCSR(csr);
            bitmask = this->regs->getValue(rs1);
            if (rd != 0){ 
                this->regs->setValue(rd, aux);
            }
            aux2 = aux | bitmask;
            this->regs->setCSR(csr, aux2);

            this->perf->cycleInc(CSRRS_CYC);
            return true;
        }

        bool Exec_CSRRC() const {
            unsigned int rd, rs1;
            int csr;
            unsigned_T bitmask, aux, aux2;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            csr = get_csr();

            

            /* These operations must be atomical */
            aux = this->regs->getCSR(csr);
            bitmask = this->regs->getValue(rs1);
            if (rd != 0) {
                this->regs->setValue(rd, aux);
            }

            aux2 = aux & ~bitmask;
            this->regs->setCSR(csr, aux2);

            this->perf->cycleInc(CSRRC_CYC);
            return true;
        }

        bool Exec_CSRRWI() const {
            unsigned int rd, rs1;
            int csr;
            unsigned_T aux;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            csr = get_csr();

            /* These operations must be atomical */
            if (rd != 0) {
                aux = this->regs->getCSR(csr);
                this->regs->setValue(rd, aux);
            }
            aux = rs1;
            this->regs->setCSR(csr, aux);
            this->perf->cycleInc(CSRRWI_CYC);
            return true;
        }

        bool Exec_CSRRSI() const {
            unsigned int rd, rs1;
            int csr;
            unsigned_T bitmask, aux;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            csr = get_csr();

            if (rs1 == 0) {
                return true;
            }

            /* These operations must be atomical */
            aux = this->regs->getCSR(csr);
            this->regs->setValue(rd, aux);

            bitmask = rs1;
            aux = aux | bitmask;
            this->regs->setCSR(csr, aux);

            this->perf->cycleInc(CSRRSI_CYC);
            return true;
        }

        bool Exec_CSRRCI() const {
            unsigned rd, rs1;
            int csr;
            unsigned_T bitmask, aux;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            csr = get_csr();

            if (rs1 == 0) {
                return true;
            }

            /* These operations must be atomical */
            aux = this->regs->getCSR(csr);
            this->regs->setValue(rd, aux);

            bitmask = rs1;
            aux = aux & ~bitmask;
            this->regs->setCSR(csr, aux);
            this->perf->cycleInc(CSRRCI_CYC);
            return true;
        }

/*********************** Privileged Instructions ******************************/

        bool Exec_MRET() const {
            unsigned_T new_pc = 0;

            new_pc = this->regs->getCSR(CSR_MEPC);
            this->regs->setPC(new_pc);

            // update mstatus
            unsigned_T csr_temp;
            csr_temp = this->regs->getCSR(CSR_MSTATUS);
            if (csr_temp & MSTATUS_MPIE) {
                csr_temp |= MSTATUS_MIE;
            }
            csr_temp |= MSTATUS_MPIE;
            this->regs->setCSR(CSR_MSTATUS, csr_temp);
            this->perf->cycleInc(MRET_CYC);
            return true;        
        }

        bool Exec_SRET() const {
            unsigned_T new_pc = 0;

            new_pc = this->regs->getCSR(CSR_SEPC);

            this->regs->setPC(new_pc);
            this->perf->cycleInc(SRET_CYC);
            return true;
        }

        
        bool Exec_WFI() const {
            this->perf->cycleInc(WFI_CYC);
            return true;
        }

        bool Exec_SFENCE() const {
            this->perf->cycleInc(SFENCE_CYC);
            return true;
        }

        /**
         * @brief Executes default ISA instruction
         * @param  inst instruction to execute
         * @return  true if PC is affected by instruction
         */
        bool exec_instruction(Instruction &inst, bool *breakpoint, opCodes code) {
            bool PC_not_affected = true;

            *breakpoint = false;
            this->setInstr(inst.getInstr());

            switch (code) {
                case OP_LUI:
                    Exec_LUI();
                    break;
                case OP_AUIPC:
                    Exec_AUIPC();
                    break;
                case OP_JAL:
                    Exec_JAL();
                    PC_not_affected = false;
                    break;
                case OP_JALR:
                    Exec_JALR();
                    PC_not_affected = false;
                    break;
                case OP_BEQ:
                    Exec_BEQ();
                    PC_not_affected = false;
                    break;
                case OP_BNE:
                    Exec_BNE();
                    PC_not_affected = false;
                    break;
                case OP_BLT:
                    Exec_BLT();
                    PC_not_affected = false;
                    break;
                case OP_BGE:
                    Exec_BGE();
                    PC_not_affected = false;
                    break;
                case OP_BLTU:
                    Exec_BLTU();
                    PC_not_affected = false;
                    break;
                case OP_BGEU:
                    Exec_BGEU();
                    PC_not_affected = false;
                    break;
                case OP_LB:
                    Exec_LB();
                    break;
                case OP_LH:
                    Exec_LH();
                    break;
                case OP_LW:
                    Exec_LW();
                    break;
                case OP_LBU:
                    Exec_LBU();
                    break;
                case OP_LHU:
                    Exec_LHU();
                    break;
                case OP_LWU:
                    Exec_LWU();
                    break;
                case OP_LD:
                    Exec_LD();
                    break;
                case OP_SB:
                    Exec_SB();
                    break;
                case OP_SH:
                    Exec_SH();
                    break;
                case OP_SW:
                    Exec_SW();
                    break;
                case OP_SD:
                    Exec_SD();
                    break;
                case OP_ADDI:
                    Exec_ADDI();
                    break;
                case OP_SLTI:
                    Exec_SLTI();
                    break;
                case OP_SLTIU:
                    Exec_SLTIU();
                    break;
                case OP_XORI:
                    Exec_XORI();
                    break;
                case OP_ORI:
                    Exec_ORI();
                    break;
                case OP_ANDI:
                    Exec_ANDI();
                    break;
                case OP_SLLI:
                    PC_not_affected = Exec_SLLI();
                    break;
                case OP_SRLI:
                    Exec_SRLI();
                    break;
                case OP_SRAI:
                    Exec_SRAI();
                    break;
                case OP_ADD:
                    Exec_ADD();
                    break;
                case OP_SUB:
                    Exec_SUB();
                    break;
                case OP_SLL:
                    Exec_SLL();
                    break;
                case OP_SLT:
                    Exec_SLT();
                    break;
                case OP_SLTU:
                    Exec_SLTU();
                    break;
                case OP_XOR:
                    Exec_XOR();
                    break;
                case OP_SRL:
                    Exec_SRL();
                    break;
                case OP_SRA:
                    Exec_SRA();
                    break;
                case OP_OR:
                    Exec_OR();
                    break;
                case OP_AND:
                    Exec_AND();
                    break;
                case OP_FENCE:
                    Exec_FENCE();
                    break;
                case OP_ECALL:
                    PC_not_affected = Exec_ECALL();
                    break;
                case OP_EBREAK:
                    PC_not_affected = Exec_EBREAK();
//		*breakpoint = true;
                    break;
                case OP_CSRRW:
                    Exec_CSRRW();
                    break;
                case OP_CSRRS:
                    Exec_CSRRS();
                    break;
                case OP_CSRRC:
                    Exec_CSRRC();
                    break;
                case OP_CSRRWI:
                    Exec_CSRRWI();
                    break;
                case OP_CSRRSI:
                    Exec_CSRRSI();
                    break;
                case OP_CSRRCI:
                    Exec_CSRRCI();
                    break;
                case OP_MRET:
                    Exec_MRET();
                    PC_not_affected = false;
                    break;
                case OP_SRET:
                    Exec_SRET();
                    PC_not_affected = false;
                    break;
                case OP_WFI:
                    Exec_WFI();
                    wait_intr = true;
                    if(stop_wait){
                        PC_not_affected = true;
                        wait_intr = false;
                        stop_wait = false;
                    }else{
                        PC_not_affected = false;
                    }
                    break;
                case OP_SFENCE:
                    Exec_SFENCE();
                    break;
                case OP_ADDIW:
                    Exec_ADDIW();
                    break;
                case OP_SLLIW:
                    Exec_SLLIW();
                    break;
                case OP_SRLIW:
                    Exec_SRLIW();
                    break;
                case OP_SRAIW:
                    Exec_SRAIW();
                    break;
                case OP_ADDW:
                    Exec_ADDW();
                    break;
                case OP_SUBW:
                    Exec_SUBW();
                    break;
                case OP_SLLW:
                    Exec_SLLW();
                    break;
                case OP_SRLW:
                    Exec_SRLW();
                    break;
                case OP_SRAW:
                    Exec_SRAW();
                    break;
                [[unlikely]] default:
                    LOG(LOG_WARNING) << "[WARNING] Wrong instruction" << "\n";
                    inst.dump();
                    this->NOP();
                    break;
            }

            return PC_not_affected;
        }

        /**
         * @brief Decodes opcode of instruction
         * @return opcode of instruction
         */
        opCodes decode() const {
            switch (opcode()) {
                case LUI:
                    return OP_LUI;
                case AUIPC:
                    return OP_AUIPC;
                case JAL:
                    return OP_JAL;
                case JALR:
                    return OP_JALR;
                case BEQ:
                    switch (this->get_funct3()) {
                        case BEQ_F:
                            return OP_BEQ;
                        case BNE_F:
                            return OP_BNE;
                        case BLT_F:
                            return OP_BLT;
                        case BGE_F:
                            return OP_BGE;
                        case BLTU_F:
                            return OP_BLTU;
                        case BGEU_F:
                            return OP_BGEU;
                    }
                    return OP_ERROR;
                case LB:
                    switch (this->get_funct3()) {
                        case LB_F:
                            return OP_LB;
                        case LH_F:
                            return OP_LH;
                        case LW_F:
                            return OP_LW;
                        case LBU_F:
                            return OP_LBU;
                        case LHU_F:
                            return OP_LHU;
                        case LWU_F:
                            return OP_LWU;
                        case LD_F:
                            return OP_LD;
                    }
                    return OP_ERROR;
                case SB:
                    switch (this->get_funct3()) {
                        case SB_F:
                            return OP_SB;
                        case SH_F:
                            return OP_SH;
                        case SW_F:
                            return OP_SW;
                        case SD_F:
                            return OP_SD;
                    }
                    return OP_ERROR;
                case ADDI:
                    switch (this->get_funct3()) {
                        case ADDI_F:
                            return OP_ADDI;
                        case SLTI_F:
                            return OP_SLTI;
                        case SLTIU_F:
                            return OP_SLTIU;
                        case XORI_F:
                            return OP_XORI;
                        case ORI_F:
                            return OP_ORI;
                        case ANDI_F:
                            return OP_ANDI;
                        case SLLI_F:
                            return OP_SLLI;
                        case SRLI_F:
                            // TODO: Why funct7b is not working?
                            //switch (this->get_funct7b()) {
                            switch(this->m_instr.to_uint() >> 26) {
                                case SRLI_F7:
                                    return OP_SRLI;
                                case SRAI_F7:
                                    return OP_SRAI;
                            }
                            return OP_ERROR;
                    }
                    return OP_ERROR;
                case ADD: {
                    if ( (this->get_funct7() != 0) && (this->get_funct7() != 0b0100000) ) {
                        return OP_ERROR;
                    }
                    switch (this->get_funct3()) {
                        case ADD_F:
                            switch (this->get_funct7()) {
                                case ADD_F7:
                                    return OP_ADD;
                                case SUB_F7:
                                    return OP_SUB;
                                default:
                                    return OP_ERROR;
                            }
                            break;
                        case SLL_F:
                            return OP_SLL;
                        case SLT_F:
                            return OP_SLT;
                        case SLTU_F:
                            return OP_SLTU;
                        case XOR_F:
                            return OP_XOR;
                        case SRL_F:
                            switch (this->get_funct7()) {
                                case SRL_F7:
                                    return OP_SRL;
                                case SRA_F7:
                                    return OP_SRA;
                                default:
                                    return OP_ERROR;
                            }
                        case OR_F:
                            return OP_OR;
                        case AND_F:
                            return OP_AND;
                        default:
                            return OP_ERROR;
                    }
                } /* ADD */
                case FENCE:
                    return OP_FENCE;
                case ECALL: {
                    switch (this->get_funct3()) {
                        case ECALL_F3:
                            switch (this->get_csr()) {
                                case ECALL_F:
                                    return OP_ECALL;
                                case EBREAK_F:
                                    return OP_EBREAK;
                                case URET_F:
                                    return OP_URET;
                                case SRET_F:
                                    return OP_SRET;
                                case MRET_F:
                                    return OP_MRET;
                                case WFI_F:
                                    return OP_WFI;
                                case SFENCE_F:
                                    return OP_SFENCE;
                            }
                            if (this->m_instr.range(31, 25) == 0b0001001) {
                                return OP_SFENCE;
                            }
                            break;
                        case CSRRW:
                            return OP_CSRRW;
                            break;
                        case CSRRS:
                            return OP_CSRRS;
                            break;
                        case CSRRC:
                            return OP_CSRRC;
                            break;
                        case CSRRWI:
                            return OP_CSRRWI;
                            break;
                        case CSRRSI:
                            return OP_CSRRSI;
                            break;
                        case CSRRCI:
                            return OP_CSRRCI;
                            break;
                    }
                }
                    break;
                case ADDIW:
                    switch (this->get_funct3()) {
                        case ADDIW_F:
                            return OP_ADDIW;
                            break;
                        case SLLIW_F:
                            return OP_SLLIW;
                            break;
                        case SRLIW_F:
                            switch (this->m_instr.to_uint() >> 26) {
                                case SRLIW_F7:
                                    return OP_SRLIW;
                                    break;
                                case SRAIW_F7:
                                    return OP_SRAIW;
                                    break;
                                default:
                                    return OP_ERROR;
                            }
                            break;
                        default:
                            return OP_ERROR;
                    }
                    break;
                case ADDW:
                    if ( (this->get_funct7() != 0) && (this->get_funct7() != 0b0100000) ) {
                        return OP_ERROR;
                    }
                    switch (this->get_funct3()) {
                        case ADDW_F:
                        switch (this->m_instr.to_uint() >> 26) {
                            case ADDW_F7:
                                return OP_ADDW;
                                break;
                            case SUBW_F7:
                                return OP_SUBW;
                                break;
                            default:
                                return OP_ERROR;
                                break;
                        }
                        break;
                        case SLLW:
                            return OP_SLLW;
                            break;
                        case SRLW:
                            switch (this->m_instr.to_uint() >> 26) {
                                case SRLW_F7:
                                    return OP_SRLW;
                                    break;
                                case SRAW_F7:
                                    return OP_SRAW;
                                    break;
                                default:
                                    return OP_ERROR;
                                    break;
                            }
                        default:
                            return OP_ERROR;
                    }

                default:
                    return OP_ERROR;
            }

            return OP_ERROR;
        }
    };
}
#endif
