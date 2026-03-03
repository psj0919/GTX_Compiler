

#include "M_extension.h"
#include "Instrcycle_info.h"
namespace riscv_tlm {
    // RV32
    template<>
    void M_extension<std::uint32_t>::Exec_M_MULH() const {
        unsigned int rd, rs1, rs2;
        signed_T multiplier, multiplicand;
        std::int64_t result;
        signed_T ret_value;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_rs2();

        multiplier = this->regs->getValue(rs1);
        multiplicand = this->regs->getValue(rs2);


        result = static_cast<std::int64_t>(multiplier) * static_cast<std::int64_t>(multiplicand);
        ret_value = static_cast<std::int32_t>((result >> 32) & 0x00000000FFFFFFFF);

        this->regs->setValue(rd, ret_value);
        this->perf->cycleInc(MULH_CYC);
    }

    template<>
    void M_extension<std::uint32_t>::Exec_M_MULHSU() const {
        unsigned int rd, rs1, rs2;
        std::int32_t multiplier;
        std::uint32_t multiplicand;
        std::int64_t result;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_rs2();

        multiplier = static_cast<std::int32_t>(this->regs->getValue(rs1));
        multiplicand = this->regs->getValue(rs2);

        result = static_cast<std::int64_t>(multiplier * static_cast<std::uint64_t>(multiplicand));
        result = (result >> 32) & 0x00000000FFFFFFFF;
        this->regs->setValue(rd, static_cast<std::int32_t>(result));
        this->perf->cycleInc(MULHSU_CYC);

    }

    template<>
    void M_extension<std::uint32_t>::Exec_M_MULHU() const {
        unsigned int rd, rs1, rs2;
        std::uint32_t multiplier, multiplicand;
        std::uint64_t result;
        std::int32_t ret_value;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_rs2();

        multiplier = static_cast<std::int32_t>(this->regs->getValue(rs1));
        multiplicand = static_cast<std::int32_t>(this->regs->getValue(rs2));

        result = static_cast<std::uint64_t>(multiplier) * static_cast<std::uint64_t>(multiplicand);
        ret_value = static_cast<std::int32_t>((result >> 32) & 0x00000000FFFFFFFF);
        this->regs->setValue(rd, ret_value);
        this->perf->cycleInc(MULHU_CYC);
    }


    // RV64
    // I need to use SystemC bigint with 128 bits to perform 64 x 64 bits multiplication and keep the high half
    template<>
    void M_extension<std::uint64_t>::Exec_M_MULH() const {
        unsigned int rd, rs1, rs2;
        signed_T multiplier, multiplicand;
        signed_T result;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_rs2();

        multiplier = this->regs->getValue(rs1);
        multiplicand = this->regs->getValue(rs2);

        sc_dt::sc_bigint<128> mul = multiplier;
        sc_dt::sc_bigint<128> muld = multiplicand;
        sc_dt::sc_bigint<128> res = mul * muld;
        result = res.range(127, 64).to_int64();

        this->regs->setValue(rd, result);
        this->perf->cycleInc(MULH_CYC);
    }

    template<>
    void M_extension<std::uint64_t>::Exec_M_MULHSU() const {
        unsigned int rd, rs1, rs2;
        signed_T multiplier;
        unsigned_T multiplicand;
        signed_T result;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_rs2();

        multiplier = this->regs->getValue(rs1);
        multiplicand = this->regs->getValue(rs2);

        sc_dt::sc_bigint<128> mul = multiplier;
        sc_dt::sc_bigint<128> muld = multiplicand;
        sc_dt::sc_bigint<128> res = mul * muld;
        result = res.range(127, 64).to_int64();

        this->regs->setValue(rd, result);
        this->perf->cycleInc(MULHSU_CYC);
    }

    template<>
    void M_extension<std::uint64_t>::Exec_M_MULHU() const {
        unsigned int rd, rs1, rs2;
        unsigned_T multiplier, multiplicand;
        unsigned_T result;

        rd = this->get_rd();
        rs1 = this->get_rs1();
        rs2 = this->get_rs2();

        multiplier = this->regs->getValue(rs1);
        multiplicand = this->regs->getValue(rs2);

        sc_dt::sc_bigint<128> mul = multiplier;
        sc_dt::sc_bigint<128> muld = multiplicand;
        sc_dt::sc_bigint<128> res = mul * muld;
        result = res.range(127, 64).to_uint64();

        this->regs->setValue(rd, result);
        this->perf->cycleInc(MULHU_CYC);

    }

}