#ifndef VERT_DEV_H_
#define VERT_DEV_H_

#include <iostream>
#include <cstdint>
#include "systemc"

#include "tlm.h"
#include "tlm_utils/simple_target_socket.h"

#include "BusCtrl.h"
#include "MemoryInterface.h"
//configurable
#define VDEV_MAX_OPEN_FILES			10

#define VIRT_DEV_BASE	0x40000000 // VDEV_REG_CMD


// command
#define CMD_FOPEN			1
#define CMD_OPEN			2
#define CMD_CLOSE			3
#define CMD_WRITE			4
#define CMD_READ			5
#define CMD_READ_LINE		6
#define CMD_SEEK			7
#define CMD_TELL			8
#define CMD_CLOCK_GETTIME	9
#define CMD_TIME			10
#define CMD_FLUSH			11
#define CMD_PRINT           12

// basic file desc ( always open-state )
#define FD_STDOUT		((void*)0)
#define FD_STDIN		((void*)1)
#define FD_STDERR		((void*)2)
// file open flag
// internal use only
#define OF_R			1
#define OF_W			2
#define OF_T			4
#define OF_B			8

#define STDOUT	0
#define STDIN	1
#define STDERR	2
// allowed combination
#define OF_RB			(OF_R|OF_B)
#define OF_RT			(OF_R|OF_B)
#define OF_WB			(OF_W|OF_B)
#define OF_WT			(OF_W|OF_T)

typedef struct __file_desc
{
	bool 	opened;
	FILE*	real_fd;
	int		of;
} file_desc_t;

// virtual device registers
enum __virt_dev_reg
{
	VDEV_REG_CMD  = 	0,		// command
	VDEV_REG_FD,				// file desc
	VDEV_REG_OF,				// open flags
	VDEV_REG_ADDR,				// data pointer
	VDEV_REG_LEN,				// data length
	VDEV_REG_PARAM0,			// seek
	VDEV_REG_PARAM1,			// seek
	VDEV_REG_RESULT,			// result of operation
	VDEV_REG_RESULT1,			// result1 ( get clock time )

	VDEV_REG_LAST				// just indication of "last"
};
namespace riscv_tlm::peripherals{
    

    class VMachine : sc_core::sc_module{
        public:
            VMachine(sc_core::sc_module_name const &name);

            tlm_utils::simple_target_socket<VMachine> socket;

            MemoryInterface *mem_intf;

            int recent_fdt;

            uint64_t heap_end;

            bool is_os;
            
            bool is_fdt_exist(int ndx);

            void set_fdt_entry(int ndx, FILE* real_fd, int of);

            void reset_fdt_entry(int ndx);

            void put_result(uint64_t res);

            void put_result1(uint64_t res);

            bool is_virt_dev(uint64_t addr);

            int find_empty_fdt();

            bool process_cmd(uint64_t cmd);
            
            bool cmd_open(uint64_t addr, uint64_t len , uint64_t of, bool is_fopen);

            bool cmd_close(uint64_t fd);

            bool cmd_write(uint64_t fd, uint64_t addr, uint64_t len);

            bool cmd_read(uint64_t fd, uint64_t addr, uint64_t len, bool read_line);

            bool cmd_seek(uint64_t fd, uint64_t offset, uint64_t origin);

            bool cmd_tell(uint64_t fd);

            bool cmd_clock_gettime(uint64_t type);

            bool cmd_time(uint64_t tm);

            bool cmd_flush(uint64_t fd);

            virtual void b_transport(tlm::tlm_generic_payload &trans,
                                 sc_core::sc_time &delay);

        private:
            
            sc_dt::sc_uint<64> vdev_cmd;
            sc_dt::sc_uint<64> vdev_fd;
            sc_dt::sc_uint<64> vdev_of;
            sc_dt::sc_uint<64> vdev_addr;
            sc_dt::sc_uint<64> vdev_len;
            sc_dt::sc_uint<64> vdev_param0;
            sc_dt::sc_uint<64> vdev_param1; 
            sc_dt::sc_uint<64> vdev_result;
            sc_dt::sc_uint<64> vdev_result1; 
            sc_dt::sc_uint<64> vdev_last;



            file_desc_t file_desc[VDEV_MAX_OPEN_FILES];

    };
}


#endif