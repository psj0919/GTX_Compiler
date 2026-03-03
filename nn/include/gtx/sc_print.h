/// Copyright by Syntacore LLC © 2016, 2017. See LICENSE for details
/// @file       <sc_print.h>
///
/// Printf implementation for baremetal RISC-V
/// Based on Marco Paland's printf implementation (MIT License)

#ifndef SC_PRINT_H
#define SC_PRINT_H

#include <stddef.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Printf to the simulator output port
 * Supports: %d, %i, %u, %x, %X, %o, %b, %c, %s, %p, %f, %F, %%
 * Flags: 0, -, +, space, #
 * Width: number or *
 * Precision: .number or .*
 * Length: h, hh, l, ll, z, j, t
 */
int sc_printf(const char *format, ...);

/**
 * sprintf - format to buffer (no size limit)
 */
int sc_sprintf(char *buffer, const char *format, ...);

/**
 * snprintf - format to buffer with size limit
 */
int sc_snprintf(char *buffer, size_t count, const char *format, ...);

/**
 * vsnprintf - format to buffer with va_list
 */
int sc_vsnprintf(char *buffer, size_t count, const char *format, va_list va);

/**
 * Low-level putchar
 */
int putchar(int ch);

// Redirect printf to sc_printf for baremetal
#define printf sc_printf
#define sprintf sc_sprintf
#define snprintf sc_snprintf
#define vsnprintf sc_vsnprintf

#ifdef __cplusplus
}
#endif

#endif // SC_PRINT_H
