

#include "Performance.h"
#include "config.h"

Performance* Performance::getInstance() {
	if (instance == nullptr) {
		instance = new Performance();
	}

	return instance;
}

Performance::Performance() {
	data_memory_read = 0;
	data_memory_write = 0;
	code_memory_read = 0;
	code_memory_write = 0;
	register_read = 0;
	register_write = 0;
	instructions_executed = 0;
	execution_cycles = 0;
	is_perfcheck = false;
}

void Performance::dump() const {
    LOG(LOG_INFO) << "************************************" << std::endl;
	LOG(LOG_INFO) << " [instructions executed] : " << instructions_executed << std::endl;
	LOG(LOG_INFO) << " [execution cycles     ] : " << execution_cycles << std::endl;
    LOG(LOG_INFO) << "************************************" << std::endl;
}

Performance *Performance::instance = nullptr;
