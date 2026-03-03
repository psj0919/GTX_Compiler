

//GTX instruction performance parameters
#ifndef GTX_PARAM
#define GTX_PARAM
#include <cstdint>
#include <math.h>
///////////
//NSU
///////////
#define NSU_MVMEM_CYCLE(length, height)               ((length/256)*height + 6)
#define NSU_MCAST_S2S_CYCLE(length, height)               ((length/256)*height + 6)
#define NSU_MCAST_G2S_CYCLE(length, height)               ((length/256)*height + 6)

///////////
//TMU
///////////
#define SMU_EXEC_CYCLE(length, height)              ((length/256)*height + height)
#define MCAST_EXEC_CYCLE(length,height)             ((length/256)*height + 6)
#define TRANSPOSE_EXEC_CYCLE(dim0, dim1, dim2)      ((dim2/16)*dim1*dim0)
#define FILL_EXEC_CYCLE(length, height)             ((length/256)*height + 1)
#define CREDIT_EXEC_CYCLE                           3
#define CREDIT_CHECK_CYCLE                          1

///////////
//SPU
///////////
#define MM_EXEC_CYCLE(A_row, A_col, B_col)          (A_col * B_col + A_row)
#define MM_S_EXEC_CYCLE(A_row, A_col, B_col)        (A_col * B_col + A_row)
#define MM_O_EXEC_CYCLE(A_row, A_col, B_col)        (A_col * B_col + A_row)
#define MM_T_EXEC_CYCLE(A_row, A_col, B_col)        (A_col * B_col + A_row)
#define MMC_EXEC_CYCLE(A_row, A_col, B_col)         (A_col * B_col + A_row)
#define MMC_S_EXEC_CYCLE(A_row, A_col, B_col)       (A_col * B_col + A_row)
#define MMC_O_EXEC_CYCLE(A_row, A_col, B_col)       (A_col * B_col + A_row)
#define MMC_T_EXEC_CYCLE(A_row, A_col, B_col)       (A_col * B_col + A_row)

//#define IM2COL_EXEC_CYCLE(ksize, stride, A_row, A_col)        ()
uint64_t IM2COL_EXEC_CYCLE(uint16_t ksize, uint16_t stride, uint16_t A_row, uint16_t A_col);
#define NMCONV_EXEC_CYCLE(ksize, A_row, A_col, stride, nkern, nchan) (A_row * A_col *nkern + nchan + ksize)
#define DWCONV_EXEC_CYCLE(ksize, A_row, A_col, stride, nkern, nchan) (A_row * A_col *nkern + nchan + ksize)
#define DSCONV_EXEC_CYCLE(ksize, A_row, A_col, stride, nkern, nchan) (A_row * A_col *nkern + nchan + ksize)


#define SADD_EXEC_CYCLE(dnum)        (6+(dnum/16))
#define SDIV_EXEC_CYCLE(dnum)        (4 + (dnum/16)*5)
#define SMAX_EXEC_CYCLE(dnum)        (4+(dnum/16))
#define SADD_IMM_EXEC_CYCLE          6
#define SDIV_IMM_EXEC_CYCLE          9
#define SMAX_IMM_EXEC_CYCLE          4
#define VSUM_EXEC_CYCLE(dnum)        (16+ (dnum/16))
#define VEXP_EXEC_CYCLE(dnum)        (6 + (dnum/16)*8)
#define VSQRT_EXEC_CYCLE(dnum)       (12 + (dnum/16))
#define VLN_EXEC_CYCLE(dnum)         (20 + (dnum/16))
#define VABS_EXEC_CYCLE(dnum)        (6 + (dnum/16))
#define VNEG_EXEC_CYCLE(dnum)        (6 + (dnum/16))
#define VSIGN_EXEC_CYCLE(dnum)       (6 + (dnum/16))
#define VSTEP_EXEC_CYCLE(dnum)       (6 + (dnum/16))
#define VCEIL_EXEC_CYCLE(dnum)       (6 + (dnum/16))
#define VTRUNC_EXEC_CYCLE(dnum)      (6 + (dnum/16))
#define VFLOOR_EXEC_CYCLE(dnum)      (6 + (dnum/16))
#define VRNE_EXEC_CYCLE(dnum)        (6 + (dnum/16))
#define ACCUM_EXEC_CYCLE(dnum)       (20 + (dnum/16))
#define CLAMP_EXEC_CYCLE(dnum)       (6 + (dnum/16))
#define ARANGE_EXEC_CYCLE(dnum)      (6 + (dnum/16))
#define VSUM_IMM_EXEC_CYCLE          13
#define VSQRT_IMM_EXEC_CYCLE         11
#define VEXP_IMM_EXEC_CYCLE          9
#define VLN_IMM_EXEC_CYCLE           20
#define VABS_IMM_EXEC_CYCLE          6
#define VNEG_IMM_EXEC_CYCLE          6
#define VSIGN_IMM_EXEC_CYCLE         6
#define VSTEP_IMM_EXEC_CYCLE         6
#define VCEIL_IMM_EXEC_CYCLE         6
#define VTRUNC_IMM_EXEC_CYCLE        6
#define VFLOOR_IMM_EXEC_CYCLE        6
#define VRNE_IMM_EXEC_CYCLE          6
#define AND_IMM_EXEC_CYCLE           6
#define OR_IMM_EXEC_CYCLE            6
#define NOT_IMM_EXEC_CYCLE           6
#define SHIFT_IMM_EXEC_CYCLE         6


#define SCVT_EXEC_CYCLE(dnum)        (12 + (dnum/16)*6)

#define TANH_EXEC_CYCLE(dnum)        (9 + (dnum/16))
#define RELU_EXEC_CYCLE(dnum)        (2 + (dnum/16))
#define PRELU_EXEC_CYCLE(dnum)       (5 + (dnum/16))

#define TANH_IMM_CYCLE               9
#define RELU_IMM_CYCLE               2
#define PRELU_IMM_CYCLE              5

#define ESUM_EXEC_CYCLE(dnum)        (21 + (dnum/16))
#define SOFTMAX_EXEC_CYCLE(dnum)     (12 + (dnum/16))
#define ESUM_IMM_CYCLE               21
#define SOFTMAX_IMM_CYCLE            12

#define MPOOL_EXEC_CYCLE(row_k, col_k, row_in, col_in, row_s, col_s)  (row_k * col_k  + col_in* row_in)
#define APOOL_EXEC_CYCLE(row_k, col_k, row_in, col_in, row_s, col_s)  (row_k * col_k  + col_in* row_in)

#define DMAC_EXEC_CYCLE(length, height)              ((length/256)*height + 2)

#define LQW_EXEC_CYCLE              5
 

#define WRSPR_EXEC_CYCLE            3
#define RDSPR_EXEC_CYCLE            3





//
#endif

