
// SPDX-License-Identifier: GPL-3.0-or-later

#include "Registers.h"

namespace riscv_tlm {
    /* Specialization for each XLEN (RV32, RV64)*/
    template<>
    void Registers<std::uint32_t>::initCSR() {
        CSR[CSR_MISA] = MISA_MXL | MISA_M_EXTENSION | MISA_C_EXTENSION
                        | MISA_A_EXTENSION | MISA_F_EXTENSION | MISA_I_BASE;
        CSR[CSR_MSTATUS] = 0x1800;
        
    }

    

    template<>
    void Registers<std::uint64_t>::initCSR() {
        CSR[CSR_MISA] = (((std::uint64_t) 0x02) << 30) | MISA_M_EXTENSION | MISA_C_EXTENSION
                        | MISA_A_EXTENSION | MISA_F_EXTENSION | MISA_I_BASE;
        CSR[CSR_MSTATUS] = 0x1800;
        if(kernel_switch){
            CSR[CSR_MEDELEG] = MEDELEG_SE;
        }
    }

    template<>
    void Registers<std::uint64_t>::dump() const {
        LOG(LOG_INFO) << " ************************************" << std::endl;
        LOG(LOG_INFO) << "           CPU REGSITERS \n";
        LOG(LOG_INFO) << " ====================================\n";
        LOG(LOG_INFO) << std::setfill('0') << std::uppercase;
        LOG(LOG_INFO) << " x0 (zero):  0x" << std::right << std::setw(16)
                  << std::hex << register_bank[0];
        LOG(LOG_INFO) << " x1  (ra):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[1];
        LOG(LOG_INFO) << " x2  (sp):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[2];
        LOG(LOG_INFO) << " x3  (gp):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[3] << std::endl;

        LOG(LOG_INFO) << " x4  (tp):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[4];
        LOG(LOG_INFO) << " x5  (t0):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[5];
        LOG(LOG_INFO) << " x6  (t1):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[6];
        LOG(LOG_INFO) << " x7  (t2):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[7] << std::endl;

        LOG(LOG_INFO) << " x8 (s0/fp): 0x" << std::right << std::setw(16)
                  << std::hex << register_bank[8];
        LOG(LOG_INFO) << " x9  (s1):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[9];
        LOG(LOG_INFO) << " x10 (a0):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[10];
        LOG(LOG_INFO) << " x11 (a1):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[11] << std::endl;

        LOG(LOG_INFO) << " x12 (a2):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[12];
        LOG(LOG_INFO) << " x13 (a3):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[13];
        LOG(LOG_INFO) << " x14 (a4):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[14];
        LOG(LOG_INFO) << " x15 (a5):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[15] << std::endl;

        LOG(LOG_INFO) << " x16 (a6):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[16];
        LOG(LOG_INFO) << " x17 (a7):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[17];
        LOG(LOG_INFO) << " x18 (s2):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[18];
        LOG(LOG_INFO) << " x19 (s3):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[19] << std::endl;

        LOG(LOG_INFO) << " x20 (s4):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[20];
        LOG(LOG_INFO) << " x21 (s5):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[21];
        LOG(LOG_INFO) << " x22 (s6):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[22];
        LOG(LOG_INFO) << " x23 (s7):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[23] << std::endl;

        LOG(LOG_INFO) << " x24 (s8):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[24];
        LOG(LOG_INFO) << " x25 (s9):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[25];
        LOG(LOG_INFO) << " x26 (s10):  0x" << std::right << std::setw(16)
                  << std::hex << register_bank[26];
        LOG(LOG_INFO) << " x27 (s11):  0x" << std::right << std::setw(16)
                  << std::hex << register_bank[27] << std::endl;

        LOG(LOG_INFO) << " x28 (t3):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[28];
        LOG(LOG_INFO) << " x29 (t4):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[29];
        LOG(LOG_INFO) << " x30 (t5):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[30];
        LOG(LOG_INFO) << " x31 (t6):   0x" << std::right << std::setw(16)
                  << std::hex << register_bank[31] << std::endl;

        LOG(LOG_INFO) << " PC: 0x" << std::setw(16) << std::hex << register_PC << std::dec << std::endl;
        LOG(LOG_INFO) << " ************************************" << std::endl;
    }
    template<>
    void Registers<std::uint64_t>::fp_dump() const {
        LOG(LOG_INFO) << " ************************************" << std::endl;
        LOG(LOG_INFO) << "           CPU FP REGSITERS \n";
        LOG(LOG_INFO) << " ====================================\n";
        LOG(LOG_INFO) << std::setfill('0') << std::uppercase;
        LOG(LOG_INFO) << " f0  (ft0):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[0];
        LOG(LOG_INFO) << " f1  (ft1):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[1];
        LOG(LOG_INFO) << " f2  (ft2):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[2];
        LOG(LOG_INFO) << " f3  (ft3):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[3] << std::endl;

        LOG(LOG_INFO) << " f4  (ft4):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[4];
        LOG(LOG_INFO) << " f5  (ft5):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[5];
        LOG(LOG_INFO) << " f6  (ft6):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[6];
        LOG(LOG_INFO) << " f7  (ft7):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[7] << std::endl;

        LOG(LOG_INFO) << " f8  (fs0):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[8];
        LOG(LOG_INFO) << " f9  (fs1):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[9];
        LOG(LOG_INFO) << " f10 (fa0):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[10];
        LOG(LOG_INFO) << " f11 (fa1):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[11] << std::endl;

        LOG(LOG_INFO) << " f12 (fa2):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[12];
        LOG(LOG_INFO) << " f13 (fa3):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[13];
        LOG(LOG_INFO) << " f14 (fa4):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[14];
        LOG(LOG_INFO) << " f15 (fa5):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[15] << std::endl;

        LOG(LOG_INFO) << " f16 (fa6):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[16];
        LOG(LOG_INFO) << " f17 (fa7):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[17];
        LOG(LOG_INFO) << " f18 (fs2):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[18];
        LOG(LOG_INFO) << " f19 (fs3):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[19] << std::endl;

        LOG(LOG_INFO) << " f20 (fs4):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[20];
        LOG(LOG_INFO) << " f21 (fs5):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[21];
        LOG(LOG_INFO) << " f22 (fs6):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[22];
        LOG(LOG_INFO) << " f23 (fs7):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[23] << std::endl;

        LOG(LOG_INFO) << " f24 (fs8):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[24];
        LOG(LOG_INFO) << " f25 (fs9):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[25];
        LOG(LOG_INFO) << " f26 (fs10):  0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[26];
        LOG(LOG_INFO) << " f27 (fs11):  0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[27] << std::endl;

        LOG(LOG_INFO) << " f28 (ft8):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[28];
        LOG(LOG_INFO) << " f29 (ft9):   0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[29];
        LOG(LOG_INFO) << " f30 (ft10):  0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[30];
        LOG(LOG_INFO) << " f31 (ft11):  0x" << std::right << std::setw(16)
                  << std::hex << register_fbank[31] << std::endl;
        LOG(LOG_INFO) << " ************************************" << std::endl;
    }
    
    template<>
    void Registers<std::uint64_t>::csr_dump(int addr) {
        LOG(LOG_INFO) << " ************************************\n";
        LOG(LOG_INFO) << "          CSR REGSITER\n";
        LOG(LOG_INFO) << " ====================================\n";
        LOG(LOG_INFO) << " [0x"<<std::hex<<addr<<"] : 0x"<<std::setfill('0') << std::uppercase;
        
        uint64_t csr = this -> getCSR(addr);

        LOG(LOG_INFO) << std::hex << std::right << std::setw(16) << csr << std::endl;
        // LOG(LOG_INFO) << "f0 (ft0):  0x" << std::right << std::setw(8)
        //           << std::hex << CSR.find(addr);
        LOG(LOG_INFO) << " ************************************" << std::endl;
    }
}