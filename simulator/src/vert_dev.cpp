
#include "vert_dev.h"
#include <cstdint>

namespace riscv_tlm::peripherals {

    VMachine::VMachine(sc_core::sc_module_name const &name):sc_module(name),socket("vm_socket"){
        mem_intf = new MemoryInterface(name);

        socket.register_b_transport(this, &VMachine::b_transport);

        heap_end = 0;

        is_os = false;
        
        for(int i = 0 ; i < VDEV_MAX_OPEN_FILES; i++){
            reset_fdt_entry(i);
            set_fdt_entry(STDOUT,stdout,OF_W);
            set_fdt_entry(STDIN,stdin,OF_R);
            set_fdt_entry(STDERR,stderr,OF_W);
        }
        
    }

    void VMachine::b_transport(tlm::tlm_generic_payload &trans, sc_core::sc_time &delay){
        tlm::tlm_command cmd = trans.get_command();
        sc_dt::uint64 addr = trans.get_address();
        unsigned char *ptr = trans.get_data_ptr();
        unsigned int len = trans.get_data_length();
        delay = sc_core::SC_ZERO_TIME;

        std::uint32_t aux_value = 0;


        if (cmd == tlm::TLM_WRITE_COMMAND) {
            memcpy(&aux_value, ptr, len);
            switch (addr) {
                case VIRTUAL_DEVICE_COMMAND_LO:
                    vdev_cmd.range(31,0) = aux_value;
                    process_cmd((uint64_t)aux_value);
                    break;
                case VIRTUAL_DEVICE_COMMAND_HI:
                    vdev_cmd.range(63,32) = aux_value;
                    break;
                case VIRTUAL_DEVICE_FILEDESC_LO:
                    vdev_fd.range(31,0) = aux_value;
                    break;
                case VIRTUAL_DEVICE_FILEDESC_HI:
                    vdev_fd.range(63,32) = aux_value;
                    break;
                case VIRTUAL_DEVICE_OPENFLAG_LO:
                    vdev_of.range(31,0) = aux_value;
                    break;
                case VIRTUAL_DEVICE_OPENFLAG_HI:
                    vdev_of.range(63,32) = aux_value;
                    break;
                case VIRTUAL_DEVICE_ADDRESS_LO :
                    vdev_addr.range(31,0) = aux_value;
                    break;
                case VIRTUAL_DEVICE_ADDRESS_HI :
                    vdev_addr.range(63,32) = aux_value;
                    break;
                case VIRTUAL_DEVICE_LENGTH_LO  :
                    vdev_len.range(31,0) = aux_value;
                    break;
                case VIRTUAL_DEVICE_LENGTH_HI  :
                    vdev_len.range(63,32) = aux_value;
                    break;
                case VIRTUAL_DEVICE_PARAM0_LO  :
                    vdev_param0.range(31,0) = aux_value;
                    break;
                case VIRTUAL_DEVICE_PARAM0_HI  :
                    vdev_param0.range(63,32) = aux_value;
                    break;    
                case VIRTUAL_DEVICE_PARAM1_LO  :
                    vdev_param1.range(31,0) = aux_value;
                    break;
                case VIRTUAL_DEVICE_PARAM1_HI  :
                    vdev_param1.range(63,32) = aux_value;
                    break;
                case VIRTUAL_DEVICE_RESULT_LO  :
                    vdev_result.range(31,0) = aux_value;
                    break;
                case VIRTUAL_DEVICE_RESULT_HI  :
                    vdev_result.range(63,32) = aux_value;
                    break;   
                case VIRTUAL_DEVICE_RESULT1_LO :
                    vdev_result1.range(31,0) = aux_value;
                    break;
                case VIRTUAL_DEVICE_RESULT1_HI :
                    vdev_result1.range(63,32) = aux_value;
                    break;
                case VIRTUAL_DEVICE_LAST_LO    :
                    vdev_last.range(31,0) = aux_value;
                    break;
                case VIRTUAL_DEVICE_LAST_HI    :
                    vdev_last.range(63,32) = aux_value;
                    break;
                default:
                    trans.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
                    return;
            }
        } else { // TLM_READ_COMMAND
            switch (addr) {
                case VIRTUAL_DEVICE_COMMAND_LO:
                    aux_value = vdev_cmd.range(31,0);
                    break;
                case VIRTUAL_DEVICE_COMMAND_HI:
                    aux_value = vdev_cmd.range(63,32);
                    break;
                case VIRTUAL_DEVICE_FILEDESC_LO:
                    aux_value = vdev_fd.range(31,0);
                    break;
                case VIRTUAL_DEVICE_FILEDESC_HI:
                    aux_value = vdev_fd.range(63,32);
                    break;
                case VIRTUAL_DEVICE_OPENFLAG_LO:
                    aux_value = vdev_of.range(31,0);
                    break;
                case VIRTUAL_DEVICE_OPENFLAG_HI:
                    aux_value = vdev_of.range(63,32);
                    break;
                case VIRTUAL_DEVICE_ADDRESS_LO :
                    aux_value = vdev_addr.range(31,0);
                    break;
                case VIRTUAL_DEVICE_ADDRESS_HI :
                    aux_value = vdev_addr.range(63,32);
                    break;
                case VIRTUAL_DEVICE_LENGTH_LO  :
                    aux_value = vdev_len.range(31,0);
                    break;
                case VIRTUAL_DEVICE_LENGTH_HI  :
                    aux_value = vdev_len.range(63,32);
                    break;
                case VIRTUAL_DEVICE_PARAM0_LO  :
                    aux_value = vdev_param0.range(31,0);
                    break;
                case VIRTUAL_DEVICE_PARAM0_HI  :
                    aux_value = vdev_param0.range(63,32);
                    break;    
                case VIRTUAL_DEVICE_PARAM1_LO  :
                    aux_value = vdev_param1.range(31,0);
                    break;
                case VIRTUAL_DEVICE_PARAM1_HI  :
                    aux_value = vdev_param1.range(63,32);
                    break;
                case VIRTUAL_DEVICE_RESULT_LO  :
                    aux_value = vdev_result.range(31,0);
                    break;
                case VIRTUAL_DEVICE_RESULT_HI  :
                    aux_value = vdev_result.range(63,32);
                    break;   
                case VIRTUAL_DEVICE_RESULT1_LO :
                    aux_value = vdev_result1.range(31,0);
                    break;
                case VIRTUAL_DEVICE_RESULT1_HI :
                    aux_value = vdev_result1.range(63,32);
                    break;
                case VIRTUAL_DEVICE_LAST_LO    :
                    aux_value = vdev_last.range(31,0);
                    break;
                case VIRTUAL_DEVICE_LAST_HI    :
                    aux_value = vdev_last.range(63,32);
                    break;
                default:
                    trans.set_response_status(tlm::TLM_ADDRESS_ERROR_RESPONSE);
                    return;
            }
            memcpy(ptr, &aux_value, len);
        }

        trans.set_response_status(tlm::TLM_OK_RESPONSE);
    }

    void VMachine::set_fdt_entry( int ndx, FILE* real_fd, int of )
    {
	    file_desc_t* en = file_desc + ndx;
	    en->real_fd = real_fd;
	    en->of = of;
	    en->opened = true;
    }

    void VMachine::reset_fdt_entry( int ndx )
    {
    	file_desc_t* en = file_desc + ndx;
    	en->real_fd = 0;
    	en->of = 0;
    	en->opened = false;
    }

    void VMachine::put_result( uint64_t res )
    {
    	vdev_result.range(63,0) = res;
    }

    void VMachine::put_result1( uint64_t res )
    {
    	vdev_result1.range(63,0) = res;
    }

    bool VMachine::is_virt_dev( uint64_t addr )
    {
    	if( ( ( addr & 7 ) == 0 ) &&
    		( VIRT_DEV_BASE <= addr && addr <= VIRT_DEV_BASE + (VDEV_REG_LAST-1)*sizeof( uint64_t ) ) )
    		return true;
    	return false;
    }

    int VMachine::find_empty_fdt( )
    {
    	for( int i = 0; i < VDEV_MAX_OPEN_FILES; i++ )
    		if( !file_desc[i].opened )
    			return i;

    	return -1;
    }

    bool VMachine::is_fdt_exist(int ndx){
        if(ndx >= VDEV_MAX_OPEN_FILES){
            return false;
        }
        if(file_desc[ndx].opened == true){
            return true;
        }else{
            return false;
        }
    }



    bool VMachine::process_cmd(uint64_t cmd){
        uint64_t fd = vdev_fd.range(63,0);
	    uint64_t len = vdev_len.range(63,0);
	    uint64_t addr = vdev_addr.range(63,0);
	    uint64_t of = vdev_of.range(63,0);
	    uint64_t param0 = vdev_param0.range(63,0);
	    uint64_t param1 = vdev_param1.range(63,0);

	    bool ret;

	    if( cmd == CMD_FOPEN || cmd == CMD_OPEN )
	    	ret = cmd_open( addr, len, of, cmd == CMD_FOPEN );
	    else if( cmd == CMD_CLOSE )
	    	ret = cmd_close( fd );
	    else if( cmd == CMD_WRITE )
	    	ret = cmd_write( fd, addr, len );
	    else if( cmd == CMD_READ || cmd == CMD_READ_LINE )
	    	ret = cmd_read( fd, addr, len, cmd == CMD_READ_LINE );
	    else if( cmd == CMD_SEEK )
	    	ret = cmd_seek( fd, param0, param1 );
	    else if( cmd == CMD_TELL )
	    	ret = cmd_tell( fd );
	    else if( cmd == CMD_CLOCK_GETTIME )
	    	ret = cmd_clock_gettime( param0 );
	    else if( cmd == CMD_TIME )
	    	ret = cmd_time( param0 );
	    else if( cmd == CMD_FLUSH )
	    	ret = cmd_flush( fd );
	    else
	    	ret = false;

	    vdev_fd.range(63,0) = 0;
	    vdev_len.range(63,0) = 0;
	    vdev_addr.range(63,0) = 0;
	    vdev_of.range(63,0) = 0;
	    vdev_param0.range(63,0) = 0;
	    vdev_param1.range(63,0) = 0;

	return ret;
    }
    bool VMachine::cmd_open( uint64_t addr, uint64_t len, uint64_t of, bool is_fopen )
    {
    	int slot = find_empty_fdt( );
    	uint64_t err_ret = is_fopen ? 0 : -1;

    	if( slot < 0 )
    	{
    		put_result( err_ret );
    		return false;
    	}

    	// file name from riscv memory
    	uint64_t allowed;
    	char* real_addr = new char[len+1]; 
        for(int i= 0; i < len ; i++){
            real_addr[i] = mem_intf->readDataMem(addr+i,1);
        }
        
        
    	// if not enough memory for file name
    	/*if( len > allowed )
    	{
    		put_result( err_ret );
    		return false;
    	}*/

    	const char* mode;
    	if( of == OF_RB )
    		mode = "rb";
    	else if( of == OF_RT )
    		mode = "rt";
    	else if( of == OF_WB )
    		mode = "wb";
    	else if( of == OF_WT )
    		mode = "wt";
        else if( of == OF_W  )
            mode = "w";
        else if( of == OF_R  )
            mode = "r";
    	else
    		mode = "rt";

    	FILE* real_fd = fopen( real_addr, mode );

        delete[] real_addr;
    	if( real_fd == 0 )
    	{
    		// fopen never returns 0
    		put_result( err_ret );
    		//log_write( LT_ERROR, "cannot open file %s (%d)\n", real_addr, errno );
    		return false;
    	}
    	set_fdt_entry( slot, real_fd, of );
        recent_fdt = slot;
    	put_result( slot );
    	return true;
    }

    bool VMachine::cmd_close(uint64_t fd )
    {
    	if( fd <= STDERR || fd >= VDEV_MAX_OPEN_FILES )
    		return false;

    	file_desc_t* desc = file_desc + fd;
    	if( desc->real_fd )
    		fclose( desc->real_fd );
    	reset_fdt_entry( fd );
    	return true;
    }

    bool VMachine::cmd_write(uint64_t fd, uint64_t addr, uint64_t len )
    {
    	if( fd == STDIN || fd >= VDEV_MAX_OPEN_FILES )
    	{
    		put_result( 0 );
    		return false;
    	}
    	file_desc_t* fdesc = &file_desc[fd];
    	if( !fdesc->opened || ( fdesc->of & OF_W ) == 0 )
    	{
    		put_result( 0 );
    		return false;
    	}

    	char* real_addr = new char[len+1];
        for(int i = 0; i < len ; i++){
            real_addr[i] = mem_intf->readDataMem(addr+i,1);
        }

    	if( fd == STDOUT )
    	{
    		real_addr[len] = 0;
    		printf( real_addr );
    	}
    	else if( fd == STDERR )
    	{
    		real_addr[len] = 0;
    		fprintf( stderr, real_addr );
    	}
    	else
    	{
    		int n = fwrite( real_addr, len, 1, fdesc->real_fd );
    		put_result( n * len );
    	}
        delete[] real_addr;
    	return true;
    }

    bool VMachine::cmd_read( uint64_t fd, uint64_t addr, uint64_t len, bool read_line )
    {
    	if( fd <= STDOUT || fd == STDERR || fd >= VDEV_MAX_OPEN_FILES )
    	{
    		put_result( 0 );
    		//log_write( LT_ERROR, "invalid fd : %ld\n", fd );
    		return false;
    	}

    	file_desc_t* fdesc = &file_desc[fd];
    	if( !fdesc->opened || !( fdesc->of & OF_R ) )
    	{
    		//log_write( LT_ERROR, "invalid fd1\n" );
    		put_result( 0 );
    		return false;
    	}

    	uint64_t allowed;
    	char* real_addr = new char[len+1];



    	if( read_line )
    	{
    		char* s = fgets( real_addr, len, fdesc->real_fd );
    		if( s == NULL )
    			put_result( 0 );
    		else
    		{
    			int nr = strlen( s ) + 1;
    			put_result( nr );
    		}
    	}
    	else
    	{
    		int nr = fread( real_addr, len, 1, fdesc->real_fd );
    		// printf( "read : %d %d\n", nr, len );
    		put_result( len * nr );
    	}
        for(int i = 0; i < len ; i++){
            mem_intf->writeDataMem(addr+i,real_addr[i],1);
        }
        delete[] real_addr;
    	return true;
    }

    bool VMachine::cmd_seek( uint64_t fd, uint64_t offset, uint64_t origin )
    {
    	if( fd <= STDERR || fd >= VDEV_MAX_OPEN_FILES )
    	{
    		put_result( 0 );
    		return false;
    	}

    	file_desc_t* fdesc = &file_desc[fd];
    	if( !fdesc->opened || !( fdesc->of & OF_R ) )
    	{
    		put_result( 0 );
    		return false;
    	}

    	int r = fseek( fdesc->real_fd, offset, origin );

    	//log_write( LT_DEBUG, "seek(%ld,%ld) : %d\n", offset, origin, r );

    	put_result( r );
    	if( r != 0 )
    		return false;

    	return true;
    }

    // _ftell
    bool VMachine::cmd_tell(uint64_t fd )
    {
    	if( fd <= STDERR || fd >= VDEV_MAX_OPEN_FILES )
    	{
    		put_result( 0 );
    		return false;
    	}

    	file_desc_t* fdesc = &file_desc[fd];
    	if( !fdesc->opened || !( fdesc->of & OF_R ) )
    	{
    		put_result( 0 );
    		return false;
    	}

    	long int r = ftell( fdesc->real_fd );

    	//log_write( LT_DEBUG, "ftell : %ld\n", r );

    	put_result( r );
    	if( r == -1 )
    		return false;

    	return true;
    }

    bool VMachine::cmd_clock_gettime( uint64_t type )
    {
    	struct timespec ts;
    	clock_gettime( type, &ts );
    	put_result( ts.tv_sec );
    	put_result1( ts.tv_nsec );
    	return true;
    }

    bool VMachine::cmd_time( uint64_t tm )
    {
    	time_t save;
    	time_t t = time( &save );

    	//if( tm )
    	//	mmu->write( tm, sizeof( time_t ), (char*)&save );

    	put_result( t );
    	//log_write( LT_DEBUG, "time : %d\n", t );
    	return true;
    }

    bool VMachine::cmd_flush( uint64_t fd )
    {
    	if( fd == STDIN || fd >= VDEV_MAX_OPEN_FILES )
    	{
    		put_result( EOF );
    		return false;
    	}

    	file_desc_t* fdesc = &file_desc[fd];
    	if( !fdesc->opened || !( fdesc->of & OF_W ) )
    	{
    		put_result( EOF );
    		return false;
    	}

    	put_result( fflush( fdesc->real_fd ) );

    	return true;
    }
    

}