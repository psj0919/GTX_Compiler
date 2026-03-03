
#ifndef A_EXTENSION__H
#define A_EXTENSION__H

#include "systemc"

#include <unordered_set>

#include "config.h"
#include "Registers.h"
#include "MemoryInterface.h"
#include "extension_base.h"
#include "Instrcycle_info.h"

namespace riscv_tlm {

    typedef enum {
        OP_A_LR_W,
        OP_A_SC_W,
        OP_A_AMOSWAP_W,
        OP_A_AMOADD_W,
        OP_A_AMOXOR_W,
        OP_A_AMOAND_W,
        OP_A_AMOOR_W,
        OP_A_AMOMIN_W,
        OP_A_AMOMAX_W,
        OP_A_AMOMINU_W,
        OP_A_AMOMAXU_W,
        OP_A_LR_D,
        OP_A_SC_D,
        OP_A_AMOSWAP_D,
        OP_A_AMOADD_D,
        OP_A_AMOXOR_D,
        OP_A_AMOAND_D,
        OP_A_AMOOR_D,
        OP_A_AMOMIN_D,
        OP_A_AMOMAX_D,
        OP_A_AMOMINU_D,
        OP_A_AMOMAXU_D,

        OP_A_ERROR
    } op_A_Codes;

    typedef enum {
        A_LR = 0b00010,
        A_SC = 0b00011,
        A_AMOSWAP = 0b00001,
        A_AMOADD = 0b00000,
        A_AMOXOR = 0b00100,
        A_AMOAND = 0b01100,
        A_AMOOR = 0b01000,
        A_AMOMIN = 0b10000,
        A_AMOMAX = 0b10100,
        A_AMOMINU = 0b11000,
        A_AMOMAXU = 0b11100,
        A_W_F = 0b010,
        A_D_F = 0b011
    } A_Codes;

/**
 * @brief Instruction decoding and fields access
 */
    template<typename T>
    class A_extension : public extension_base<T> {
    public:

        /**
         * @brief Constructor, same as base class
         */
        using extension_base<T>::extension_base;

        using signed_T = typename std::make_signed<T>::type;
        using unsigned_T = typename std::make_unsigned<T>::type;

        /**
         * @brief Access to opcode field
         * @return return opcode field
         */
        inline unsigned_T opcode() const override {
            return static_cast<unsigned_T>(this->m_instr.range(31, 27));
        }

        /**
         * @brief Decodes opcode of instruction
         * @return opcode of instruction
         */
        op_A_Codes decode() const {

            switch (opcode()) {
                case A_LR:
                    switch(this->get_funct3()){
                        case A_W_F :
                            return OP_A_LR_W;
                            break;
                        case A_D_F :
                            return OP_A_LR_D;
                            break;
                        default : 
                            return OP_A_ERROR;
                            break;
                    }
                    break;
                case A_SC:
                    switch(this->get_funct3()){
                        case A_W_F :
                            return OP_A_SC_W;
                            break;
                        case A_D_F :
                            return OP_A_SC_D;
                            break;
                        default : 
                            return OP_A_ERROR;
                            break;
                    }
                    break;
                case A_AMOSWAP:
                    switch(this->get_funct3()){
                        case A_W_F :
                            return OP_A_AMOSWAP_W;
                            break;
                        case A_D_F :
                            return OP_A_AMOSWAP_D;
                            break;
                        default : 
                            return OP_A_ERROR;
                            break;
                    }
                    break;
                case A_AMOADD:
                    switch(this->get_funct3()){
                        case A_W_F :
                            return OP_A_AMOADD_W;
                            break;
                        case A_D_F :
                            return OP_A_AMOADD_D;
                            break;
                        default : 
                            return OP_A_ERROR;
                            break;
                    }
                    break;
                case A_AMOXOR:
                    switch(this->get_funct3()){
                        case A_W_F :
                            return OP_A_AMOXOR_W;
                            break;
                        case A_D_F :
                            return OP_A_AMOXOR_D;
                            break;
                        default : 
                            return OP_A_ERROR;
                            break;
                    }
                    break;
                case A_AMOAND:
                    switch(this->get_funct3()){
                        case A_W_F :
                            return OP_A_AMOAND_W;
                            break;
                        case A_D_F :
                            return OP_A_AMOAND_D;
                            break;
                        default : 
                            return OP_A_ERROR;
                            break;
                    }
                    break;
                case A_AMOOR:
                    switch(this->get_funct3()){
                        case A_W_F :
                            return OP_A_AMOOR_W;
                            break;
                        case A_D_F :
                            return OP_A_AMOOR_D;
                            break;
                        default : 
                            return OP_A_ERROR;
                            break;
                    }
                    break;
                case A_AMOMIN:
                    switch(this->get_funct3()){
                        case A_W_F :
                            return OP_A_AMOMIN_W;
                            break;
                        case A_D_F :
                            return OP_A_AMOMIN_D;
                            break;
                        default : 
                            return OP_A_ERROR;
                            break;
                    }
                    break;
                case A_AMOMAX:
                    switch(this->get_funct3()){
                        case A_W_F :
                            return OP_A_AMOMAX_W;
                            break;
                        case A_D_F :
                            return OP_A_AMOMAX_D;
                            break;
                        default : 
                            return OP_A_ERROR;
                            break;
                    }
                    break;;
                case A_AMOMINU:
                    switch(this->get_funct3()){
                        case A_W_F :
                            return OP_A_AMOMINU_W;
                            break;
                        case A_D_F :
                            return OP_A_AMOMINU_D;
                            break;
                        default : 
                            return OP_A_ERROR;
                            break;
                    }
                    break;
                case A_AMOMAXU:
                    switch(this->get_funct3()){
                        case A_W_F :
                            return OP_A_AMOMAXU_W;
                            break;
                        case A_D_F :
                            return OP_A_AMOMAXU_D;
                            break;
                        default : 
                            return OP_A_ERROR;
                            break;
                    }
                    break;
                    [[unlikely]] default:
                    return OP_A_ERROR;
                    break;

            }

            return OP_A_ERROR;
        }

        inline void dump() const override {
            LOG(LOG_WARNING) << std::hex << "0x" << this->m_instr << std::dec << std::endl;
        }

        template<typename T1, typename T2> //std::uint32_t, std::uint64_t
        bool Exec_A_LR() {
            T mem_addr = 0;
            T1 rd, rs1, rs2;
            T1 data;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            /*if (rs2 != 0) {
                std::cout << "ILEGAL INSTRUCTION, LR.W: rs2 != 0" << std::endl;
                this->RaiseException(Exception_cause::ILLEGAL_INSTRUCTION, this->m_instr);

                return false;
            }*/

            mem_addr = this->regs->getValue(rs1);
            data = this->mem_intf->readDataMem(mem_addr, 4);
            this->perf->dataMemoryRead();
            this->regs->setValue(rd, static_cast<T1>(data));

            TLB_reserve(mem_addr);

            return true;
        }

        template<typename T1, typename T2>
        bool Exec_A_SC() {
            T mem_addr;
            T1 rd, rs1, rs2;
            T1 data;

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            mem_addr = this->regs->getValue(rs1);
            data = this->regs->getValue(rs2);

            if (TLB_reserved(mem_addr)) {
                this->mem_intf->writeDataMem(mem_addr, data, sizeof(T1));
                this->perf->dataMemoryWrite();
                this->regs->setValue(rd, 0);  // SC writes 0 to rd on success
            } else {
                this->regs->setValue(rd, 1);  // SC writes nonzero on failure
            }


            return true;
        }
        template<typename T1, typename T2>
        bool Exec_A_AMOSWAP() const {
            T mem_addr;
            T1 rd, rs1, rs2;
            T1 data;
            T1 aux;

            /* These instructions must be atomic */

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            mem_addr = this->regs->getValue(rs1);
            data = this->mem_intf->readDataMem(mem_addr, sizeof(T1));
            this->perf->dataMemoryRead();
            this->regs->setValue(rd, static_cast<T1>(data));

            // swap
            aux = this->regs->getValue(rs2);
            this->regs->setValue(rs2, static_cast<T1>(data));

            this->mem_intf->writeDataMem(mem_addr, aux, sizeof(T1));
            this->perf->dataMemoryWrite();


            return true;
        }

        template<typename T1, typename T2>
        bool Exec_A_AMOADD() const {
            T mem_addr;
            T1 rd, rs1, rs2;
            T1 data;

            /* These instructions must be atomic */

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            mem_addr = this->regs->getValue(rs1);
            data = this->mem_intf->readDataMem(mem_addr, sizeof(T1));
            this->perf->dataMemoryRead();

            this->regs->setValue(rd, static_cast<T1>(data));

            // add
            data = data + this->regs->getValue(rs2);

            this->mem_intf->writeDataMem(mem_addr, data, sizeof(T1));
            this->perf->dataMemoryWrite();


            return true;
        }

        template<typename T1, typename T2>
        bool Exec_A_AMOXOR() const {
            T mem_addr;
            T1 rd, rs1, rs2;
            T1 data;

            /* These instructions must be atomic */

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            mem_addr = this->regs->getValue(rs1);
            data = this->mem_intf->readDataMem(mem_addr, sizeof(T1));
            this->perf->dataMemoryRead();

            this->regs->setValue(rd, static_cast<T1>(data));

            // add
            data = data ^ this->regs->getValue(rs2);

            this->mem_intf->writeDataMem(mem_addr, data, sizeof(T1));
            this->perf->dataMemoryWrite();


            return true;
        }

        template<typename T1, typename T2>
        bool Exec_A_AMOAND() const {
            T mem_addr;
            T1 rd, rs1, rs2;
            T1 data;

            /* These instructions must be atomic */

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            mem_addr = this->regs->getValue(rs1);
            data = this->mem_intf->readDataMem(mem_addr, sizeof(T1));
            this->perf->dataMemoryRead();

            this->regs->setValue(rd, static_cast<T1>(data));

            // add
            data = data & this->regs->getValue(rs2);

            this->mem_intf->writeDataMem(mem_addr, data, sizeof(T1));
            this->perf->dataMemoryWrite();


            return true;
        }

        template<typename T1, typename T2>
        bool Exec_A_AMOOR() const {
            T mem_addr;
            T1 rd, rs1, rs2;
            T1 data;

            /* These instructions must be atomic */

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            mem_addr = this->regs->getValue(rs1);
            data = this->mem_intf->readDataMem(mem_addr, sizeof(T1));
            this->perf->dataMemoryRead();

            this->regs->setValue(rd, static_cast<T1>(data));

            // add
            data = data | this->regs->getValue(rs2);

            this->mem_intf->writeDataMem(mem_addr, data, sizeof(T1));
            this->perf->dataMemoryWrite();


            return true;
        }

        template<typename T1, typename T2>
        bool Exec_A_AMOMIN() const {
            T mem_addr;
            T1 rd, rs1, rs2;
            T1 data;
            T1 aux;

            /* These instructions must be atomic */

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            mem_addr = this->regs->getValue(rs1);
            data = this->mem_intf->readDataMem(mem_addr, sizeof(T1));
            this->perf->dataMemoryRead();

            this->regs->setValue(rd, static_cast<T2>(data));

            // min
            aux = this->regs->getValue(rs2);
            if ((T2) data < (T2) aux) {
                aux = data;
            }

            this->mem_intf->writeDataMem(mem_addr, aux, sizeof(T1));
            this->perf->dataMemoryWrite();


            return true;
        }

        template<typename T1, typename T2>
        bool Exec_A_AMOMAX() const {
            T mem_addr;
            T1 rd, rs1, rs2;
            T1 data;
            T1 aux;

            /* These instructions must be atomic */

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            mem_addr = this->regs->getValue(rs1);
            data = this->mem_intf->readDataMem(mem_addr, sizeof(T1));
            this->perf->dataMemoryRead();

            this->regs->setValue(rd, static_cast<T2>(data));

            // >
            aux = this->regs->getValue(rs2);
            if ((T2) data > (T2) aux) {
                aux = data;
            }

            this->mem_intf->writeDataMem(mem_addr, aux, sizeof(T));
            this->perf->dataMemoryWrite();

            return true;
        }
        
        template<typename T1, typename T2>
        bool Exec_A_AMOMINU() const {
            T mem_addr;
            T1 rd, rs1, rs2;
            T1 data;
            T1 aux;

            /* These instructions must be atomic */

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            mem_addr = this->regs->getValue(rs1);
            data = this->mem_intf->readDataMem(mem_addr, sizeof(T1));
            this->perf->dataMemoryRead();

            this->regs->setValue(rd, static_cast<T2>(data));

            // min
            aux = this->regs->getValue(rs2);
            if (data < aux) {
                aux = data;
            }

            this->mem_intf->writeDataMem(mem_addr, aux, sizeof(T1));
            this->perf->dataMemoryWrite();

            return true;
        }
        template<typename T1, typename T2>
        bool Exec_A_AMOMAXU() const {
            T mem_addr;
            T1 rd, rs1, rs2;
            T1 data;
            T1 aux;

            /* These instructions must be atomic */

            rd = this->get_rd();
            rs1 = this->get_rs1();
            rs2 = this->get_rs2();

            mem_addr = this->regs->getValue(rs1);
            data = this->mem_intf->readDataMem(mem_addr, sizeof(T));
            this->perf->dataMemoryRead();

            this->regs->setValue(rd, static_cast<T2>(data));

            // max
            aux = this->regs->getValue(rs2);
            if (data > aux) {
                aux = data;
            }

            this->mem_intf->writeDataMem(mem_addr, aux, sizeof(T));
            this->perf->dataMemoryWrite();


            return true;
        }

        void TLB_reserve(T address) {
            TLB_A_Entries.insert(address);
        }

        bool TLB_reserved(T address) {
            if (TLB_A_Entries.count(address) == 1) {
                TLB_A_Entries.erase(address);
                return true;
            } else {
                return false;
            }
        }

        bool exec_instruction(Instruction &inst, op_A_Codes code) {
            bool PC_not_affected = true;

            this->setInstr(inst.getInstr());

            switch (code) {
                case OP_A_LR_W:
                    Exec_A_LR<std::uint32_t,std::int32_t>();
                    this->perf->cycleInc(LR_W_CYC);
                    break;
                case OP_A_SC_W:
                    Exec_A_SC<std::uint32_t,std::int32_t>();
                    this->perf->cycleInc(SC_W_CYC);
                    break;
                case OP_A_AMOSWAP_W:
                    Exec_A_AMOSWAP<std::uint32_t,std::int32_t>();
                    this->perf->cycleInc(AMOSWAP_W_CYC);
                    break;
                case OP_A_AMOADD_W:
                    Exec_A_AMOADD<std::uint32_t,std::int32_t>();
                    this->perf->cycleInc(AMOADD_W_CYC);
                    break;
                case OP_A_AMOXOR_W:
                    Exec_A_AMOXOR<std::uint32_t,std::int32_t>();
                    this->perf->cycleInc(AMOXOR_W_CYC);
                    break;
                case OP_A_AMOAND_W:
                    Exec_A_AMOAND<std::uint32_t,std::int32_t>();
                    this->perf->cycleInc(AMOAND_W_CYC);
                    break;
                case OP_A_AMOOR_W:
                    Exec_A_AMOOR<std::uint32_t,std::int32_t>();
                    this->perf->cycleInc(AMOOR_W_CYC);
                    break;
                case OP_A_AMOMIN_W:
                    Exec_A_AMOMIN<std::uint32_t,std::int32_t>();
                    this->perf->cycleInc(AMOMIN_W_CYC);
                    break;
                case OP_A_AMOMAX_W:
                    Exec_A_AMOMAX<std::uint32_t,std::int32_t>();
                    this->perf->cycleInc(AMOMAX_W_CYC);
                    break;
                case OP_A_AMOMINU_W:
                    Exec_A_AMOMINU<std::uint32_t,std::int32_t>();
                    this->perf->cycleInc(AMOMINU_W_CYC);
                    break;
                case OP_A_AMOMAXU_W:
                    Exec_A_AMOMAXU<std::uint32_t,std::int32_t>();
                    this->perf->cycleInc(AMOMAXU_W_CYC);
                    break;
                case OP_A_LR_D:
                    Exec_A_LR<std::uint64_t,std::int64_t>();
                    this->perf->cycleInc(LR_D_CYC);
                    break;
                case OP_A_SC_D:
                    Exec_A_SC<std::uint64_t,std::int64_t>();
                    this->perf->cycleInc(SC_D_CYC);
                    break;
                case OP_A_AMOSWAP_D:
                    Exec_A_AMOSWAP<std::uint64_t,std::int64_t>();
                    this->perf->cycleInc(AMOSWAP_D_CYC);
                    break;
                case OP_A_AMOADD_D:
                    Exec_A_AMOADD<std::uint64_t,std::int64_t>();
                    this->perf->cycleInc(AMOADD_D_CYC);
                    break;
                case OP_A_AMOXOR_D:
                    Exec_A_AMOXOR<std::uint64_t,std::int64_t>();
                    this->perf->cycleInc(AMOXOR_D_CYC);
                    break;
                case OP_A_AMOAND_D:
                    Exec_A_AMOAND<std::uint64_t,std::int64_t>();
                    this->perf->cycleInc(AMOAND_D_CYC);
                    break;
                case OP_A_AMOOR_D:
                    Exec_A_AMOOR<std::uint64_t,std::int64_t>();
                    this->perf->cycleInc(AMOOR_D_CYC);
                    break;
                case OP_A_AMOMIN_D:
                    Exec_A_AMOMIN<std::uint64_t,std::int64_t>();
                    this->perf->cycleInc(AMOMIN_D_CYC);
                    break;
                case OP_A_AMOMAX_D:
                    Exec_A_AMOMAX<std::uint64_t,std::int64_t>();
                    this->perf->cycleInc(AMOMAX_D_CYC);
                    break;
                case OP_A_AMOMINU_D:
                    Exec_A_AMOMINU<std::uint64_t,std::int64_t>();
                    this->perf->cycleInc(AMOMINU_D_CYC);
                    break;
                case OP_A_AMOMAXU_D:
                    Exec_A_AMOMAXU<std::uint64_t,std::int64_t>();
                    this->perf->cycleInc(AMOMAXU_D_CYC);
                    break;
                    [[unlikely]] default:
                    LOG(LOG_WARNING) << "[WARNING] A instruction not implemented yet" << std::endl;
                    inst.dump();
                    this->NOP();
                    break;
            }

            return PC_not_affected;
        }

    private:
        std::unordered_set<T> TLB_A_Entries;
    };
}

#endif
