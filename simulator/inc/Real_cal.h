#ifndef REAL_CAL_H_
#define REAL_CAL_H_
#include <cstdint>
#include <stdio.h>
#include <iostream>
#include "float_conv.h"

int highest_bit_pos_u16(uint16_t x);

//for softmax
uint16_t HW_SM_LN(uint16_t input, uint16_t max_val);

//for softmax
uint16_t HW_SM_EXP(uint16_t input);

uint16_t HW_SIGM(uint16_t input);

uint16_t HW_TANH(uint16_t input);

uint16_t HW_GELU(uint16_t input);

uint16_t HW_LN(uint16_t input,uint16_t op2_sel);

uint16_t HW_EXP(uint16_t input, uint16_t op2_sel);

uint16_t HW_SQRT(uint16_t input);










#endif