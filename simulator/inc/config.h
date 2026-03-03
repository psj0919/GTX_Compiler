
#ifndef __CONFIG_H__
#define __CONFIG_H__
#include <iostream>
extern bool dfile_addr_set;

extern int kernel_switch;

extern int exitcode;

extern bool hw_mode;

enum LogLevel {
    LOG_FATAL = 0,
    LOG_ERROR = 1,
    LOG_WARNING = 2,
    LOG_INFO = 3,
    LOG_DEBUG = 4
};

extern LogLevel log_level;

#define LOG(level) if (level <= log_level) std::cout

extern bool gtx_command_dump;

extern bool gtx_tracker;

extern bool L1_dump_mode;

extern int spu_exec_time;

extern int L1_mode_id;

extern bool is_L1_dump;

extern bool is_monitoring;//execute 2nd time loop

extern bool is_boot_success;

extern std::string l1_fname, l1_dfname;
extern std::string l2_fname, l2_dfname;
extern std::string main_dfname, reg_dfname, cmd_dfname;

#endif

#define SC_SIM_OUTPORT 0xfffe0
#define MAX_THREAD  4


#ifndef SPU_NUM
#define SPU_NUM      64
#endif

#ifndef TMU_NUM
#define TMU_NUM      4
#endif

#ifndef GDMAC_NUM
#define GDMAC_NUM    TMU_NUM
#endif

#ifndef MAIN_SIZE
#define MAIN_SIZE       0x100000000
#endif

#ifndef L2_SIZE
#define L2_SIZE         0x1000000
#endif

#ifndef L1_SIZE
#define L1_SIZE         0x60000
#endif

#ifndef L0_SIZE
#define L0_SIZE         0x400
#endif