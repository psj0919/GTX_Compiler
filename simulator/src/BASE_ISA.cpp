/*!
 \file BASE_ISA.cpp

 */

#include "BASE_ISA.h"
#include "Instrcycle_info.h"

namespace riscv_tlm { 

    ///// RV32 Specialization
    template<>
    std::int32_t BASE_ISA<std::uint32_t>::get_imm_I() const {
        std::uint32_t aux = 0;

        aux = this->m_instr.range(31, 20);

        /* sign extension (optimize) */
        if (this->m_instr[31] == 1) {
            aux |= (0b11111111111111111111) << 12;
        }
        
        return static_cast<std::int32_t>(aux);
    }

    template<>
    std::int32_t BASE_ISA<std::uint32_t>::get_imm_S() const {
        std::uint32_t aux = 0;

        aux = this->m_instr.range(31, 25) << 5;
        aux |= this->m_instr.range(11, 7);

        if (this->m_instr[31] == 1) {
            aux |= (0b11111111111111111111) << 12;
        }

        return static_cast<std::int32_t>(aux);
    }

    template<>
    std::int32_t BASE_ISA<std::uint32_t>::get_imm_B() const {
        std::uint32_t aux = 0;

        aux |= this->m_instr[7] << 11;
        aux |= this->m_instr.range(30, 25) << 5;
        aux |= this->m_instr[31] << 12;
        aux |= this->m_instr.range(11, 8) << 1;

        if (this->m_instr[31] == 1) {
            aux |= (0b11111111111111111111) << 12;
        }

        return static_cast<std::int32_t>(aux);
    }

    template<>
    std::int32_t BASE_ISA<std::uint32_t>::get_imm_J() const {
        std::uint32_t aux = 0;

        aux = this->m_instr[31] << 20;
        aux |= this->m_instr.range(19, 12) << 12;
        aux |= this->m_instr[20] << 11;
        aux |= this->m_instr.range(30, 21) << 1;

        /* bit extension (better way to do that?) */
        if (this->m_instr[31] == 1) {
            aux |= (0b111111111111) << 20;
        }

        return static_cast<std::int32_t>(aux);
    }

    template<>
    std::uint32_t BASE_ISA<std::uint32_t>::get_funct7b() const {
        return this->m_instr.range(31, 25);
    }

    // SLLI, SRLI and SRAI get different shamt field depending on RV32 or RV64
    template<>
    bool BASE_ISA<std::uint32_t>::Exec_SLLI() {
        unsigned int rd, rs1, rs2;
        unsigned_T shift;
        unsigned_T calc;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = get_shamt();
        //if(this->perf->is_perfcheck)
        this->perf->cycleInc(SLLI_CYC);

        if (rs2 >= 0x20) {
            LOG(LOG_ERROR) << "[ERROR] ILLEGAL INSTRUCTION, shamt[5] != 0" << "\n";
            if(!is_monitoring){
            sc_core::sc_stop();
            this->RaiseException(Exception_cause::ILLEGAL_INSTRUCTION, this->m_instr);
            }

            return false;
        }

        shift = rs2 & 0x1F;

        calc = static_cast<unsigned_T>(this->regs->getValue(rs1)) << shift;
        this->regs->setValue(rd, calc);


        return true;
    }

    template<>
    bool BASE_ISA<std::uint32_t>::Exec_SRLI() const {
        unsigned int rd, rs1, rs2;
        unsigned_T shift;
        unsigned_T calc;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_rs2();

        shift = rs2 & 0x1F;

        calc = static_cast<unsigned_T>(this->regs->getValue(rs1)) >> shift;
        this->regs->setValue(rd, calc);
        this->perf->cycleInc(SRLI_CYC);

        return true;
    }

    template<>
    bool BASE_ISA<std::uint32_t>::Exec_SRAI() const {
        unsigned int rd, rs1, rs2;
        unsigned_T shift;
        signed_T calc;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_rs2();

        shift = rs2 & 0x1F;

        calc = static_cast<signed_T>(this->regs->getValue(rs1)) >> shift;
        this->regs->setValue(rd, calc);
        this->perf->cycleInc(SRAI_CYC);

        return true;
    }

    template<>
    bool BASE_ISA<std::uint32_t>::Exec_SRL() const {
        unsigned int rd, rs1, rs2;
        unsigned_T shift;
        unsigned_T calc;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_rs2();

        shift = this->regs->getValue(rs2) & 0x1F;
        calc = this->regs->getValue(rs1) >> shift;

        this->regs->setValue(rd, calc);
        this->perf->cycleInc(SRL_CYC);

        return true;
    }

    template<>
    bool BASE_ISA<std::uint32_t>::Exec_SRA() const {
        unsigned int rd, rs1, rs2;
        unsigned_T shift;
        signed_T calc;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_rs2();

        shift = this->regs->getValue(rs2) & 0x1F;
        calc = static_cast<signed_T>(this->regs->getValue(rs1)) >> shift;
        this->regs->setValue(rd, calc);

        this->perf->cycleInc(SRA_CYC);
        return true;
    }

    template<>
    bool BASE_ISA<std::uint32_t>::Exec_SLL() const {
        unsigned int rd, rs1, rs2;
        unsigned_T shift;
        unsigned_T calc;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_rs2();

        shift = this->regs->getValue(rs2) & 0x1F;

        calc = this->regs->getValue(rs1) << shift;
        this->regs->setValue(rd, calc);
        this->perf->cycleInc(SLL_CYC);


        return true;
    }

    ///// RV64 Specialization
    template<>
    std::int64_t BASE_ISA<std::uint64_t>::get_imm_I() const {
        std::uint64_t aux = 0;

        aux = this->m_instr.range(31, 20);

        /* sign extension (optimize) */
        if (this->m_instr[31] == 1) {
            aux |= (0b11111111111111111111) << 12;
            aux |= 0xFFFFFFFFULL << 32;
        }

        return static_cast<std::int64_t>(aux);
    }

    template<>
    std::int64_t BASE_ISA<std::uint64_t>::get_imm_S() const {
        std::uint64_t aux = 0;

        aux = this->m_instr.range(31, 25) << 5;
        aux |= this->m_instr.range(11, 7);

        if (this->m_instr[31] == 1) {
            aux |= (0b11111111111111111111) << 12;
            aux |= 0xFFFFFFFFULL << 32;
        }

        return static_cast<std::int64_t>(aux);
    }

    template<>
    std::int64_t BASE_ISA<std::uint64_t>::get_imm_B() const {
        std::uint64_t aux = 0;

        aux |= this->m_instr[7] << 11;
        aux |= this->m_instr.range(30, 25) << 5;
        aux |= this->m_instr[31] << 12;
        aux |= this->m_instr.range(11, 8) << 1;

        if (this->m_instr[31] == 1) {
            aux |= (0b11111111111111111111) << 12;
            aux |= 0xFFFFFFFFULL << 32;
        }

        return static_cast<std::int64_t>(aux);
    }

    template<>
    std::int64_t BASE_ISA<std::uint64_t>::get_imm_J() const {
        std::uint64_t aux = 0;

        aux = this->m_instr[31] << 20;
        aux |= this->m_instr.range(19, 12) << 12;
        aux |= this->m_instr[20] << 11;
        aux |= this->m_instr.range(30, 21) << 1;

        /* bit extension (better way to do that?) */
        if (this->m_instr[31] == 1) {
            aux |= (0b111111111111) << 20;
        }

        return static_cast<std::int64_t>(aux);
    }

    template<>
    std::uint32_t BASE_ISA<std::uint64_t>::get_funct7b() const {
        return this->m_instr.range(31, 26);
    }

    template<>
    bool BASE_ISA<std::uint64_t>::Exec_SLLI() {
        unsigned int rd, rs1, rs2;
        unsigned_T shift;
        unsigned_T calc;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_shamt_slli();

        if (rs2 >= 0x40) {
            LOG(LOG_ERROR) << "[ERROR] ILLEGAL INSTRUCTION, shamt[5] > 0x40" << "\n";
            if(!is_monitoring){
            sc_core::sc_stop();
            this->RaiseException(Exception_cause::ILLEGAL_INSTRUCTION, this->m_instr);
            }
            this->perf->cycleInc(SLLI_CYC);
            return false;
        }

        shift = rs2 & 0x3F;
        calc = this->regs->getValue(rs1) << shift;

        this->regs->setValue(rd, calc);

        this->perf->cycleInc(SLLI_CYC);
        return true;
    }

    template<>
    bool BASE_ISA<std::uint64_t>::Exec_SRLI() const {
        unsigned int rd, rs1, rs2;
        unsigned_T shift;
        unsigned_T calc;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_shamt_slli();

        shift = rs2 & 0x3F;

        calc = static_cast<unsigned_T>(this->regs->getValue(rs1)) >> shift;
        this->regs->setValue(rd, calc);
        this->perf->cycleInc(SRLI_CYC);
        return true;
    }

    template<>
    bool BASE_ISA<std::uint64_t>::Exec_SRAI() const {
        unsigned int rd, rs1, rs2;
        unsigned_T shift;
        signed_T calc;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_shamt_slli();

        shift = rs2 & 0x3F;

        calc = static_cast<signed_T>(static_cast<signed_T>(this->regs->getValue(rs1)) >> shift);
        this->regs->setValue(rd, calc);
        this->perf->cycleInc(SRAI_CYC);
        return true;
    }

    template<>
    bool BASE_ISA<std::uint64_t>::Exec_SRL() const {
        unsigned int rd, rs1, rs2;
        unsigned_T shift;
        unsigned_T calc;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_rs2();

        shift = this->regs->getValue(rs2) & 0x3F;
        calc = this->regs->getValue(rs1) >> shift;

        this->regs->setValue(rd, calc);
        this->perf->cycleInc(SRL_CYC);

        return true;
    }

    template<>
    bool BASE_ISA<std::uint64_t>::Exec_SRA() const {
        unsigned int rd, rs1, rs2;
        unsigned_T shift;
        signed_T calc;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_rs2();

        shift = this->regs->getValue(rs2) & 0x3F;
        calc = static_cast<signed_T>(this->regs->getValue(rs1)) >> shift;
        this->regs->setValue(rd, calc);

        this->perf->cycleInc(SRA_CYC);
        return true;
    }

    template<>
    bool BASE_ISA<std::uint64_t>::Exec_SLL() const {
        unsigned int rd, rs1, rs2;
        unsigned_T shift;
        unsigned_T calc;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_rs2();

        shift = this->regs->getValue(rs2) & 0x3F;

        calc = this->regs->getValue(rs1) << shift;
        this->regs->setValue(rd, calc);
        this->perf->cycleInc(SLL_CYC);
        return true;
    }
}

