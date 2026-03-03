
#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#define SC_INCLUDE_DYNAMIC_PROCESSES

#include "systemc"

#include "tlm.h"

/**
 * @brief Performance indicators class
 *
 * Singleton class to be shared among all other classes
 */
class Performance {
public:

	/**
	 * @brief Get an instance of the class
	 * @return pointer to Performance class
	 */
	static Performance* getInstance();

	inline void dataMemoryRbyte(int byte){
		data_memory_rbyte += byte;
	}

	inline void dataMemoryWbyte(int byte){
		data_memory_wbyte += byte;
	}

	inline void dataMemoryRead() {
		data_memory_read++;
	}

	inline void dataMemoryWrite() {
		data_memory_write++;
	}

	/**
	 * @brief Increment code memory read counter
	 */
	inline void codeMemoryRead() {
		code_memory_read++;
	}

	/**
	 * @brief Increment code memory write counter
	 */
	inline void codeMemoryWrite() {
		code_memory_write++;
	}

	/**
	 * @brief Increment register read counter
	 */
	inline void registerRead() {
		register_read++;
	}

	/**
	 * @brief Increment register write counter
	 */
	inline void registerWrite() {
		register_write++;
	}

	/**
	 * @brief Increment instructions executed counter
	 */
	inline void instructionsInc() {
		instructions_executed++;
	}

	inline void cycleInc(uint32_t cycle){
		execution_cycles += cycle;
	}

	/**
	 * @brief Dump counters to cout
	 */
	void dump() const;

	inline uint_fast64_t getInstructions() const {
	  return instructions_executed;
	}
	inline uint_fast64_t getcycles()const{
		return execution_cycles;
	}
	inline uint_fast64_t getMemoryRead()const{
		return data_memory_read;
	}
	inline uint_fast64_t getMemoryWrite()const{
		return data_memory_read;
	}
	inline uint_fast64_t getRbyte()const{
		return data_memory_rbyte;
	}
	inline uint_fast64_t getWbyte()const{
		return data_memory_wbyte;
	}
	bool is_perfcheck;

private:
	static Performance *instance;
	Performance();

	uint_fast64_t data_memory_rbyte;
	uint_fast64_t data_memory_wbyte;
	uint_fast64_t data_memory_read;
	uint_fast64_t data_memory_write;
	uint_fast64_t code_memory_read;
	uint_fast64_t code_memory_write;
	uint_fast64_t register_read;
	uint_fast64_t register_write;
	uint_fast64_t instructions_executed;
	uint_fast64_t execution_cycles;
};

#endif
