#include "GTX_perf_param.h"


uint64_t IM2COL_EXEC_CYCLE(uint16_t ksize, uint16_t stride, uint16_t A_row, uint16_t A_col){
    int hout, wout,p;
    hout = ((A_row - ksize)/stride) + 1;
    wout = ((A_col - ksize)/stride) + 1;
    p = hout * wout;
    float r;
    r = exp(0.700051f+(1.103271f*log(p)) + (0.231216f*log(ksize*ksize)) - (0.006564f*(log(p)*log(p))) + (0.071693f*(log(ksize))*log(ksize)) - (0.000971f*log(p)*log(ksize*ksize)));
    return (uint64_t)r;
} 