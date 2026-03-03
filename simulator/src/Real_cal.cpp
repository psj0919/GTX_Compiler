#include "Real_cal.h"
int highest_bit_pos_u16(uint16_t x) {
    if (x == 0) return -1;    
    int leadingZeros = __builtin_clz(x) - (32 - 16);
    return 15 - leadingZeros;   
}

uint16_t HW_SM_LN(uint16_t input, uint16_t max_val){
    uint16_t exp_part, lut_addr, mant_part;
    uint16_t slope, int_val, exp_lut_val;
    exp_part = (input>>10) & 0x1f;
    switch(exp_part){
        case 0xf : exp_lut_val = 0;      break;
        case 0x10: exp_lut_val = 0x398c; break;
        case 0x11: exp_lut_val = 0x3d8c; break;
        case 0x12: exp_lut_val = 0x4029; break;
        case 0x13: exp_lut_val = 0x418c; break;
        case 0x14: exp_lut_val = 0x42ee; break;
        case 0x15: exp_lut_val = 0x4429; break;
        case 0x16: exp_lut_val = 0x44da; break;
        case 0x17: exp_lut_val = 0x458c; break;
        case 0x18: exp_lut_val = 0x463d; break;
        case 0x19: exp_lut_val = 0x46ee; break;
        case 0x1a: exp_lut_val = 0x47a0; break;
        case 0x1b: exp_lut_val = 0x4829; break;
        case 0x1c: exp_lut_val = 0x4881; break;
        case 0x1d: exp_lut_val = 0x48da; break;
        case 0x1e: exp_lut_val = 0x4933; break;
        case 0x1f: exp_lut_val = 0x498c; break;
        default  : exp_lut_val = 0x0;    break;
    }
    lut_addr = (input>>4) & 0x3F;
    switch(lut_addr){
        case 0 : slope = 0x3bf0; int_val = 0xbbf0; break;
        case 1 : slope = 0x3bd2; int_val = 0xbbd2; break;
        case 2 : slope = 0x3bb2; int_val = 0xbbb1; break;
        case 3 : slope = 0x3b98; int_val = 0xbb95; break;
        case 4 : slope = 0x3b74; int_val = 0xbb6f; break;
        case 5 : slope = 0x3b60; int_val = 0xbb5a; break;
        case 6 : slope = 0x3b48; int_val = 0xbb3f; break;
        case 7 : slope = 0x3b28; int_val = 0xbb1c; break;
        case 8 : slope = 0x3b10; int_val = 0xbb01; break;
        case 9 : slope = 0x3af0; int_val = 0xbadc; break;
        case 10 : slope = 0x3ae0; int_val = 0xbaca; break;
        case 11 : slope = 0x3ad0; int_val = 0xbab7; break;
        case 12 : slope = 0x3ab0; int_val = 0xba91; break;
        case 13 : slope = 0x3aa0; int_val = 0xba7e; break;
        case 14 : slope = 0x3a80; int_val = 0xba57; break;
        case 15 : slope = 0x3a70; int_val = 0xba43; break;
        case 16 : slope = 0x3a60; int_val = 0xba2f; break;
        case 17 : slope = 0x3a40; int_val = 0xba06; break;
        case 18 : slope = 0x3a40; int_val = 0xba06; break;
        case 19 : slope = 0x3a20; int_val = 0xb9dd; break;
        case 20 : slope = 0x3a00; int_val = 0xb9b3; break;
        case 21 : slope = 0x3a00; int_val = 0xb9b3; break;
        case 22 : slope = 0x3a00; int_val = 0xb9b3; break;
        case 23 : slope = 0x39c0; int_val = 0xb95c; break;
        case 24 : slope = 0x39e0; int_val = 0xb988; break;
        case 25 : slope = 0x39a0; int_val = 0xb92f; break;
        case 26 : slope = 0x39c0; int_val = 0xb95c; break;
        case 27 : slope = 0x3980; int_val = 0xb901; break;
        case 28 : slope = 0x39a0; int_val = 0xb92f; break;
        case 29 : slope = 0x3980; int_val = 0xb900; break;
        case 30 : slope = 0x3960; int_val = 0xb8d2; break;
        case 31 : slope = 0x3960; int_val = 0xb8d2; break;
        case 32 : slope = 0x3940; int_val = 0xb8a2; break;
        case 33 : slope = 0x3940; int_val = 0xb8a2; break;
        case 34 : slope = 0x3940; int_val = 0xb8a2; break;
        case 35 : slope = 0x3920; int_val = 0xb870; break;
        case 36 : slope = 0x3920; int_val = 0xb870; break;
        case 37 : slope = 0x3900; int_val = 0xb83e; break;
        case 38 : slope = 0x3900; int_val = 0xb83e; break;
        case 39 : slope = 0x3900; int_val = 0xb83e; break;
        case 40 : slope = 0x38e0; int_val = 0xb80a; break;
        case 41 : slope = 0x38c0; int_val = 0xb7aa; break;
        case 42 : slope = 0x3900; int_val = 0xb83f; break;
        case 43 : slope = 0x38c0; int_val = 0xb7a8; break;
        case 44 : slope = 0x3880; int_val = 0xb6d0; break;
        case 45 : slope = 0x38c0; int_val = 0xb7aa; break;
        case 46 : slope = 0x38c0; int_val = 0xb7aa; break;
        case 47 : slope = 0x3880; int_val = 0xb6cc; break;
        case 48 : slope = 0x3880; int_val = 0xb6cc; break;
        case 49 : slope = 0x3880; int_val = 0xb6cc; break;
        case 50 : slope = 0x3880; int_val = 0xb6cc; break;
        case 51 : slope = 0x3880; int_val = 0xb6cc; break;
        case 52 : slope = 0x3880; int_val = 0xb6cc; break;
        case 53 : slope = 0x3840; int_val = 0xb5e2; break;
        case 54 : slope = 0x3840; int_val = 0xb5e2; break;
        case 55 : slope = 0x3840; int_val = 0xb5e2; break;
        case 56 : slope = 0x3840; int_val = 0xb5e2; break;
        case 57 : slope = 0x3840; int_val = 0xb5e2; break;
        case 58 : slope = 0x3840; int_val = 0xb5e2; break;
        case 59 : slope = 0x3840; int_val = 0xb5e2; break;
        case 60 : slope = 0x3800; int_val = 0xb4ea; break;
        case 61 : slope = 0x3800; int_val = 0xb4ea; break;
        case 62 : slope = 0x3840; int_val = 0xb5e6; break;
        case 63 : slope = 0x3800; int_val = 0xb4e8; break;
    }
    mant_part = (input& 0x3FF)| (0x3C00);
    float mac_a, mac_b, mac_c;
    uint16_t datr;
    mac_a = fp16_to_32(mant_part);
    mac_b = fp16_to_32(slope);
    mac_c = fp16_to_32(int_val);
    mac_a = mac_a * mac_b + mac_c;
    datr = fp32_to_16(mac_a);

    mac_a = fp16_to_32(datr);
    mac_c = fp16_to_32(exp_lut_val);
    mac_a = mac_a + mac_c;
    datr = fp32_to_16(mac_a);

    mac_a = fp16_to_32(datr);
    mac_c = fp16_to_32(max_val);
    mac_a = mac_a + mac_c;
    datr = fp32_to_16(mac_a);


    return datr;
    
}

uint16_t HW_SM_EXP(uint16_t input){
    uint16_t exp_part, lut_addr;
    uint16_t slope, int_val;
    exp_part = (input>>10) & 0x1f;
    switch(exp_part){
        case 0x11: lut_addr = ((~input)>>5)&0x1f; break;
        case 0x10: lut_addr = (((~input)>>6)&0xf)|0x20; break;
        case 0xf : lut_addr = (((~input)>>6)&0xf)|0x30; break;
        case 0xe : lut_addr = (((~input)>>5)&0x1f)|0x40; break;
        case 0xd : lut_addr = (((~input)>>6)&0xf)|0x60; break;
        case 0xc : lut_addr = (((~input)>>7)&0x7)|0x70; break;
        case 0xb : lut_addr = (((~input)>>8)&0x3)|0x78; break;
        case 0xa : lut_addr = (((~input)>>9)&0x1)|0x7c; break;
        case 0x9 : lut_addr = 126; break;
        case 0x8 : lut_addr = 127; break;
        case 0x7 : 
        case 0x6 :
        case 0x5 :
        case 0x4 :
        case 0x3 :
        case 0x2 :
        case 0x1 : 
        case 0x0 : lut_addr = 127; break;
        default  : lut_addr = 0; break;
    }
    switch(lut_addr){
        case 0: slope = 0x0000; int_val = 0x0000; break;
        case 1: slope = 0x0ea2; int_val = 0x1b4f; break;
        case 2: slope = 0x0f84; int_val = 0x1c15; break;
        case 3: slope = 0x1042; int_val = 0x1c8f; break;
        case 4: slope = 0x10d4; int_val = 0x1d17; break;
        case 5: slope = 0x1178; int_val = 0x1daf; break;
        case 6: slope = 0x1232; int_val = 0x1e58; break;
        case 7: slope = 0x1306; int_val = 0x1f14; break;
        case 8: slope = 0x13f5; int_val = 0x1fe5; break;
        case 9: slope = 0x1482; int_val = 0x2067; break;
        case 10: slope = 0x151c; int_val = 0x20e9; break;
        case 11: slope = 0x15ca; int_val = 0x2179; break;
        case 12: slope = 0x168f; int_val = 0x2219; break;
        case 13: slope = 0x176f; int_val = 0x22cc; break;
        case 14: slope = 0x1836; int_val = 0x2392; break;
        case 15: slope = 0x18c6; int_val = 0x2437; break;
        case 16: slope = 0x1968; int_val = 0x24b1; break;
        case 17: slope = 0x1a21; int_val = 0x2538; break;
        case 18: slope = 0x1af1; int_val = 0x25ce; break;
        case 19: slope = 0x1bde; int_val = 0x2675; break;
        case 20: slope = 0x1c75; int_val = 0x272d; break;
        case 21: slope = 0x1d0d; int_val = 0x27f9; break;
        case 22: slope = 0x1db9; int_val = 0x286e; break;
        case 23: slope = 0x1e7c; int_val = 0x28eb; break;
        case 24: slope = 0x1f59; int_val = 0x2975; break;
        case 25: slope = 0x202a; int_val = 0x2a0d; break;
        case 26: slope = 0x20b8; int_val = 0x2ab6; break;
        case 27: slope = 0x2159; int_val = 0x2b70; break;
        case 28: slope = 0x220f; int_val = 0x2c1e; break;
        case 29: slope = 0x22de; int_val = 0x2c8f; break;
        case 30: slope = 0x23c8; int_val = 0x2d0c; break;
        case 31: slope = 0x2468; int_val = 0x2d94; break;
        case 32: slope = 0x24ff; int_val = 0x2e2b; break;
        case 33: slope = 0x25a9; int_val = 0x2ed0; break;
        case 34: slope = 0x266a; int_val = 0x2f84; break;
        case 35: slope = 0x2744; int_val = 0x3025; break;
        case 36: slope = 0x281e; int_val = 0x3092; break;
        case 37: slope = 0x28aa; int_val = 0x3108; break;
        case 38: slope = 0x2949; int_val = 0x3189; break;
        case 39: slope = 0x29fe; int_val = 0x3216; break;
        case 40: slope = 0x2aca; int_val = 0x32af; break;
        case 41: slope = 0x2bb1; int_val = 0x3355; break;
        case 42: slope = 0x2c5c; int_val = 0x3405; break;
        case 43: slope = 0x2cf0; int_val = 0x3466; break;
        case 44: slope = 0x2d99; int_val = 0x34d0; break;
        case 45: slope = 0x2e57; int_val = 0x3541; break;
        case 46: slope = 0x2f2f; int_val = 0x35ba; break;
        case 47: slope = 0x3012; int_val = 0x363d; break;
        case 48: slope = 0x3078; int_val = 0x36a2; break;
        case 49: slope = 0x30c2; int_val = 0x36ea; break;
        case 50: slope = 0x3110; int_val = 0x3733; break;
        case 51: slope = 0x3164; int_val = 0x377f; break;
        case 52: slope = 0x31bd; int_val = 0x37cd; break;
        case 53: slope = 0x321c; int_val = 0x380f; break;
        case 54: slope = 0x3281; int_val = 0x3838; break;
        case 55: slope = 0x32ec; int_val = 0x3861; break;
        case 56: slope = 0x335e; int_val = 0x388c; break;
        case 57: slope = 0x33d8; int_val = 0x38b8; break;
        case 58: slope = 0x342d; int_val = 0x38e5; break;
        case 59: slope = 0x3472; int_val = 0x3912; break;
        case 60: slope = 0x34bb; int_val = 0x3940; break;
        case 61: slope = 0x3509; int_val = 0x396e; break;
        case 62: slope = 0x355c; int_val = 0x399d; break;
        case 63: slope = 0x35b5; int_val = 0x39cc; break;
        case 64: slope = 0x35ef; int_val = 0x39e9; break;
        case 65: slope = 0x3607; int_val = 0x39f5; break;
        case 66: slope = 0x361f; int_val = 0x3a00; break;
        case 67: slope = 0x3638; int_val = 0x3a0c; break;
        case 68: slope = 0x3651; int_val = 0x3a18; break;
        case 69: slope = 0x366a; int_val = 0x3a24; break;
        case 70: slope = 0x3684; int_val = 0x3a2f; break;
        case 71: slope = 0x369e; int_val = 0x3a3b; break;
        case 72: slope = 0x36b9; int_val = 0x3a47; break;
        case 73: slope = 0x36d4; int_val = 0x3a52; break;
        case 74: slope = 0x36f0; int_val = 0x3a5e; break;
        case 75: slope = 0x370b; int_val = 0x3a69; break;
        case 76: slope = 0x3728; int_val = 0x3a75; break;
        case 77: slope = 0x3745; int_val = 0x3a80; break;
        case 78: slope = 0x3762; int_val = 0x3a8c; break;
        case 79: slope = 0x3780; int_val = 0x3a97; break;
        case 80: slope = 0x379e; int_val = 0x3aa3; break;
        case 81: slope = 0x37bd; int_val = 0x3aae; break;
        case 82: slope = 0x37dc; int_val = 0x3ab9; break;
        case 83: slope = 0x37fc; int_val = 0x3ac4; break;
        case 84: slope = 0x380e; int_val = 0x3acf; break;
        case 85: slope = 0x381e; int_val = 0x3ada; break;
        case 86: slope = 0x382f; int_val = 0x3ae5; break;
        case 87: slope = 0x3840; int_val = 0x3af0; break;
        case 88: slope = 0x3851; int_val = 0x3afb; break;
        case 89: slope = 0x3862; int_val = 0x3b05; break;
        case 90: slope = 0x3874; int_val = 0x3b10; break;
        case 91: slope = 0x3886; int_val = 0x3b1a; break;
        case 92: slope = 0x3898; int_val = 0x3b24; break;
        case 93: slope = 0x38ab; int_val = 0x3b2f; break;
        case 94: slope = 0x38bd; int_val = 0x3b39; break;
        case 95: slope = 0x38d1; int_val = 0x3b42; break;
        case 96: slope = 0x38e4; int_val = 0x3b4c; break;
        case 97: slope = 0x38f8; int_val = 0x3b56; break;
        case 98: slope = 0x390c; int_val = 0x3b5f; break;
        case 99: slope = 0x3920; int_val = 0x3b68; break;
        case 100: slope = 0x3935; int_val = 0x3b71; break;
        case 101: slope = 0x394a; int_val = 0x3b7a; break;
        case 102: slope = 0x395f; int_val = 0x3b83; break;
        case 103: slope = 0x3975; int_val = 0x3b8b; break;
        case 104: slope = 0x398b; int_val = 0x3b94; break;
        case 105: slope = 0x39a1; int_val = 0x3b9c; break;
        case 106: slope = 0x39b8; int_val = 0x3ba3; break;
        case 107: slope = 0x39cf; int_val = 0x3bab; break;
        case 108: slope = 0x39e6; int_val = 0x3bb2; break;
        case 109: slope = 0x39fe; int_val = 0x3bb9; break;
        case 110: slope = 0x3a16; int_val = 0x3bc0; break;
        case 111: slope = 0x3a2f; int_val = 0x3bc7; break;
        case 112: slope = 0x3a48; int_val = 0x3bcd; break;
        case 113: slope = 0x3a61; int_val = 0x3bd3; break;
        case 114: slope = 0x3a7b; int_val = 0x3bd8; break;
        case 115: slope = 0x3a95; int_val = 0x3bde; break;
        case 116: slope = 0x3aaf; int_val = 0x3be3; break;
        case 117: slope = 0x3aca; int_val = 0x3be7; break;
        case 118: slope = 0x3ae6; int_val = 0x3bec; break;
        case 119: slope = 0x3b01; int_val = 0x3bf0; break;
        case 120: slope = 0x3b1e; int_val = 0x3bf3; break;
        case 121: slope = 0x3b3a; int_val = 0x3bf6; break;
        case 122: slope = 0x3b57; int_val = 0x3bf9; break;
        case 123: slope = 0x3b75; int_val = 0x3bfb; break;
        case 124: slope = 0x3b93; int_val = 0x3bfd; break;
        case 125: slope = 0x3bb2; int_val = 0x3bff; break;
        case 126: slope = 0x3bd1; int_val = 0x3c00; break;
        default : slope = 0x3bf0; int_val = 0x3c00; break;
    }
    float mac_a, mac_b, mac_c;
    mac_a = fp16_to_32(input);
    mac_b = fp16_to_32(slope);
    mac_c = fp16_to_32(int_val);
    mac_a = mac_a * mac_b + mac_c;
    return fp32_to_16(mac_a);
}

uint16_t HW_SIGM(uint16_t input){
    uint16_t exp_part, lut_addr;
    uint16_t slope, int_val;
    exp_part = (input>>10) & 0x1f;
    if(exp_part >= 0x12){
        lut_addr = 0;
    }else{
        switch(exp_part){
            case 0x11: lut_addr = (((~input)>>6)& 0xF); break;
            case 0x10: lut_addr = (((~input)>>7)& 0x7) | 0x10; break;
            case 0xf : lut_addr = (((~input)>>8)& 0x3) | 0x18; break;
            case 0xe : lut_addr = (((~input)>>9)& 0x1) | 0x1c; break;
            case 0xc : lut_addr = (((~input)>>10)& 0x1) | 0x1e; break;
            case 0xd : lut_addr = (((~input)>>10)& 0x1) | 0x1e; break;
            default  : lut_addr = 0x1f; break;
        }
    }
    lut_addr = ((input & 0x8000) == 0x8000)? lut_addr : (((~lut_addr)&0x1f)| 0x20);
    switch(lut_addr){
        case 0  : slope = 0x0000; int_val = 0x0000; break;
        case 1  : slope = 0x1001; int_val = 0x1c52; break;
        case 2  : slope = 0x1124; int_val = 0x1d63; break;
        case 3  : slope = 0x1299; int_val = 0x1eb5; break;
        case 4  : slope = 0x143c; int_val = 0x202c; break;
        case 5  : slope = 0x156f; int_val = 0x212f; break;
        case 6  : slope = 0x16f9; int_val = 0x226f; break;
        case 7  : slope = 0x1879; int_val = 0x23f9; break;
        case 8  : slope = 0x19bc; int_val = 0x24ef; break;
        case 9  : slope = 0x1b5a; int_val = 0x2619; break;
        case 10 : slope = 0x1cb6; int_val = 0x2785; break;
        case 11 : slope = 0x1e08; int_val = 0x28a0; break;
        case 12 : slope = 0x1fb8; int_val = 0x29ae; break;
        case 13 : slope = 0x20ef; int_val = 0x2af5; break;
        case 14 : slope = 0x224d; int_val = 0x2c3f; break;
        case 15 : slope = 0x2404; int_val = 0x2d2b; break;
        case 16 : slope = 0x251c; int_val = 0x2e43; break;
        case 17 : slope = 0x267d; int_val = 0x2f8d; break;
        case 18 : slope = 0x281a; int_val = 0x3087; break;
        case 19 : slope = 0x292c; int_val = 0x3165; break;
        case 20 : slope = 0x2a7b; int_val = 0x3261; break;
        case 21 : slope = 0x2c0a; int_val = 0x3379; break;
        case 22 : slope = 0x2cfd; int_val = 0x3455; break;
        case 23 : slope = 0x2e1b; int_val = 0x34f6; break;
        case 24 : slope = 0x2f62; int_val = 0x3599; break;
        case 25 : slope = 0x3067; int_val = 0x3638; break;
        case 26 : slope = 0x3128; int_val = 0x36c9; break;
        case 27 : slope = 0x31eb; int_val = 0x3743; break;
        case 28 : slope = 0x32a4; int_val = 0x37a0; break;
        case 29 : slope = 0x3343; int_val = 0x37db; break;
        case 30 : slope = 0x33b7; int_val = 0x37f8; break;
        case 31 : slope = 0x33f5; int_val = 0x3800; break;
        case 32 : slope = 0x33f5; int_val = 0x3800; break;
        case 33 : slope = 0x33b7; int_val = 0x3804; break;
        case 34 : slope = 0x3343; int_val = 0x3812; break;
        case 35 : slope = 0x32a4; int_val = 0x3830; break;
        case 36 : slope = 0x31eb; int_val = 0x385e; break;
        case 37 : slope = 0x3128; int_val = 0x389b; break;
        case 38 : slope = 0x3067; int_val = 0x38e4; break;
        case 39 : slope = 0x2f62; int_val = 0x3933; break;
        case 40 : slope = 0x2e1b; int_val = 0x3985; break;
        case 41 : slope = 0x2cfd; int_val = 0x39d5; break;
        case 42 : slope = 0x2c0a; int_val = 0x3a22; break;
        case 43 : slope = 0x2a7b; int_val = 0x3a68; break;
        case 44 : slope = 0x292c; int_val = 0x3aa7; break;
        case 45 : slope = 0x281a; int_val = 0x3ade; break;
        case 46 : slope = 0x267d; int_val = 0x3b0e; break;
        case 47 : slope = 0x251c; int_val = 0x3b38; break;
        case 48 : slope = 0x2404; int_val = 0x3b5b; break;
        case 49 : slope = 0x224d; int_val = 0x3b78; break;
        case 50 : slope = 0x20ef; int_val = 0x3b91; break;
        case 51 : slope = 0x1fb8; int_val = 0x3ba5; break;
        case 52 : slope = 0x1e08; int_val = 0x3bb6; break;
        case 53 : slope = 0x1cb6; int_val = 0x3bc4; break;
        case 54 : slope = 0x1b5a; int_val = 0x3bcf; break;
        case 55 : slope = 0x19bc; int_val = 0x3bd9; break;
        case 56 : slope = 0x1879; int_val = 0x3be0; break;
        case 57 : slope = 0x16f9; int_val = 0x3be6; break;
        case 58 : slope = 0x156f; int_val = 0x3beb; break;
        case 59 : slope = 0x143c; int_val = 0x3bef; break;
        case 60 : slope = 0x1299; int_val = 0x3bf3; break;
        case 61 : slope = 0x1124; int_val = 0x3bf5; break;
        case 62 : slope = 0x1002; int_val = 0x3bf7; break;
        default : slope = 0x0000; int_val = 0x3c00; break;
    }

    float mac_a, mac_b, mac_c;
    mac_a = fp16_to_32(input);
    mac_b = fp16_to_32(slope);
    mac_c = fp16_to_32(int_val);
    mac_a = mac_a * mac_b + mac_c;
    return fp32_to_16(mac_a);
}

uint16_t HW_TANH(uint16_t input){
    uint16_t exp_part, lut_addr;
    uint16_t slope, int_val;
    exp_part = (input>>10) & 0x1f;
    if(exp_part <= 0xa){
        lut_addr = 0;
    }else{
        switch(exp_part){
            case 0xb: lut_addr = 0x1; break;
            case 0xc: lut_addr = 0x2 | ((input>>9)& 0x1); break;
            case 0xd: lut_addr = 0x4 | ((input>>8)& 0x3); break;
            case 0xe: lut_addr = 0x8 | ((input>>7)& 0x7); break;
            case 0xf: lut_addr = 0x10 |((input>>6)& 0xf); break;
            case 0x10:lut_addr = 0x20 |((input>>5)& 0x1f); break;
            default : lut_addr = 63;
        }
    }
    switch(lut_addr){
        case 0 : slope = 0x3bfd; int_val = 0x0000; break;
        case 1 : slope = 0x3bed; int_val = 0x1000; break;
        case 2 : slope = 0x3bce; int_val = 0x18e0; break;
        case 3 : slope = 0x3ba0; int_val = 0x1ec0; break;
        case 4 : slope = 0x3b68; int_val = 0x22e0; break;
        case 5 : slope = 0x3b20; int_val = 0x2640; break;
        case 6 : slope = 0x3ad0; int_val = 0x2900; break;
        case 7 : slope = 0x3a78; int_val = 0x2b68; break;
        case 8 : slope = 0x3a18; int_val = 0x2d34; break;
        case 9 : slope = 0x39c0; int_val = 0x2ec0; break;
        case 10: slope = 0x3950; int_val = 0x3078; break;
        case 11: slope = 0x3900; int_val = 0x3154; break;
        case 12: slope = 0x3890; int_val = 0x32a4; break;
        case 13: slope = 0x3840; int_val = 0x33a8; break;
        case 14: slope = 0x37a0; int_val = 0x3498; break;
        case 15: slope = 0x3720; int_val = 0x3510; break;
        case 16: slope = 0x3660; int_val = 0x35d0; break;
        case 17: slope = 0x35c0; int_val = 0x367a; break;
        case 18: slope = 0x3540; int_val = 0x370a; break;
        case 19: slope = 0x34c0; int_val = 0x37a2; break;
        case 20: slope = 0x3440; int_val = 0x3821; break;
        case 21: slope = 0x33c0; int_val = 0x3860; break;
        case 22: slope = 0x32c0; int_val = 0x38b8; break;
        case 23: slope = 0x3240; int_val = 0x38e6; break;
        case 24: slope = 0x3180; int_val = 0x392e; break;
        case 25: slope = 0x30c0; int_val = 0x3979; break;
        case 26: slope = 0x3040; int_val = 0x39ad; break;
        case 27: slope = 0x3000; int_val = 0x39c8; break;
        case 28: slope = 0x2f00; int_val = 0x3a00; break;
        case 29: slope = 0x2e00; int_val = 0x3a3a; break;
        case 30: slope = 0x2d80; int_val = 0x3a58; break;
        case 31: slope = 0x2c80; int_val = 0x3a96; break;
        case 32: slope = 0x2c80; int_val = 0x3a96; break;
        case 33: slope = 0x2b00; int_val = 0x3ad8; break;
        case 34: slope = 0x2b00; int_val = 0x3ad8; break;
        case 35: slope = 0x2a00; int_val = 0x3afb; break;
        case 36: slope = 0x2900; int_val = 0x3b1f; break;
        case 37: slope = 0x2900; int_val = 0x3b1f; break;
        case 38: slope = 0x2800; int_val = 0x3b45; break;
        case 39: slope = 0x2800; int_val = 0x3b45; break;
        case 40: slope = 0x2600; int_val = 0x3b6d; break;
        case 41: slope = 0x2600; int_val = 0x3b6d; break;
        case 42: slope = 0x2400; int_val = 0x3b97; break;
        case 43: slope = 0x2400; int_val = 0x3b97; break;
        case 44: slope = 0x2400; int_val = 0x3b97; break;
        case 45: slope = 0x2400; int_val = 0x3b97; break;
        case 46: slope = 0x2400; int_val = 0x3b97; break;
        case 47: slope = 0x2000; int_val = 0x3bc6; break;
        case 48: slope = 0x2000; int_val = 0x3bc6; break;
        case 49: slope = 0x2000; int_val = 0x3bc6; break;
        case 50: slope = 0x2000; int_val = 0x3bc6; break;
        case 51: slope = 0x2000; int_val = 0x3bc6; break;
        case 52: slope = 0x2000; int_val = 0x3bc6; break;
        case 53: slope = 0x0000; int_val = 0x3bfb; break;
        case 54: slope = 0x2000; int_val = 0x3bc5; break;
        case 55: slope = 0x0000; int_val = 0x3bfc; break;
        case 56: slope = 0x2000; int_val = 0x3bc4; break;
        case 57: slope = 0x0000; int_val = 0x3bfd; break;
        case 58: slope = 0x0000; int_val = 0x3bfd; break;
        case 59: slope = 0x2000; int_val = 0x3bc2; break;
        case 60: slope = 0x0000; int_val = 0x3bfe; break;
        case 61: slope = 0x0000; int_val = 0x3bfe; break;
        case 62: slope = 0x0000; int_val = 0x3bfe; break;
        case 63: slope = 0x0000; int_val = 0x3c00; break;
    }
    int_val = (input &0x8000) | int_val;
    float mac_a, mac_b, mac_c;
    mac_a = fp16_to_32(input);
    mac_b = fp16_to_32(slope);
    mac_c = fp16_to_32(int_val);
    mac_a = mac_a * mac_b + mac_c;
    return fp32_to_16(mac_a);

}

uint16_t HW_GELU(uint16_t input){
    uint16_t exp_part, lut_addr;
    uint16_t slope, int_val;
    exp_part = (input>>10)& 0x3F;
    switch(exp_part){
        case 0x30: lut_addr = (((~input)>>7) & 0x7); break;
        case 0x2f: lut_addr = ((((~input)>>7) & 0x7)) | 0x8; break;
        case 0x2e: lut_addr = ((((~input)>>7) & 0x7)) | 0x10; break;
        case 0x2d: lut_addr = ((((~input)>>8) & 0x3)) | 0x18; break;
        case 0x2c: lut_addr = ((((~input)>>9) & 0x1)) | 0x1c; break;
        case 0x2b: lut_addr = 0x1e; break;
        case 0x2a: lut_addr = 0x1f; break;
        case 0x28: lut_addr = 0x1f; break;
        case 0x29: lut_addr = 0x1f; break;
        case 0x20: lut_addr = 0x1f; break;
        case 0x21: lut_addr = 0x1f; break;
        case 0x22: lut_addr = 0x1f; break;
        case 0x23: lut_addr = 0x1f; break;
        case 0x24: lut_addr = 0x1f; break;
        case 0x25: lut_addr = 0x1f; break;
        case 0x26: lut_addr = 0x1f; break;
        case 0x27: lut_addr = 0x1f; break;
        case 0x0 : lut_addr = 0x20; break;
        case 0x1 : lut_addr = 0x20; break;
        case 0x2 : lut_addr = 0x20; break;
        case 0x3 : lut_addr = 0x20; break;
        case 0x4 : lut_addr = 0x20; break;
        case 0x5 : lut_addr = 0x20; break;
        case 0x6 : lut_addr = 0x20; break;
        case 0x7 : lut_addr = 0x20; break;
        case 0x8 : lut_addr = 0x20; break;
        case 0x9 : lut_addr = 0x20; break;
        case 0xa : lut_addr = 0x20; break;
        case 0xb : lut_addr = 0x21; break;
        case 0xc : lut_addr = 0x22 | ((input>>9)& 0x1); break;
        case 0xd : lut_addr = 0x24 | ((input>>8) & 0x3); break;
        case 0xe : lut_addr = 0x28 | ((input>>7)& 0x7); break;
        case 0xf : lut_addr = 0x30 | ((input>>7)& 0x7); break;
        case 0x10: lut_addr = 0x38 | ((input>>7)& 0x7); break;
        default :  lut_addr = (((input>>15)& 0x1) == 0x1)?  0 : 63;
    }

    switch(lut_addr){
        case 0: slope = 0x0000; int_val = 0x0000; break;
        case 1: slope = 0x0000; int_val = 0x0000; break;
        case 2: slope = 0xa0e0; int_val = 0xa844; break;
        case 3: slope = 0x9880; int_val = 0xa0e0; break;
        case 4: slope = 0xa3c0; int_val = 0xaa30; break;
        case 5: slope = 0xa860; int_val = 0xae72; break;
        case 6: slope = 0xa9f8; int_val = 0xb038; break;
        case 7: slope = 0xacbc; int_val = 0xb230; break;
        case 8: slope = 0xadd0; int_val = 0xb344; break;
        case 9: slope = 0xae98; int_val = 0xb400; break;
        case 10: slope = 0xafd0; int_val = 0xb488; break;
        case 11: slope = 0xafe8; int_val = 0xb492; break;
        case 12: slope = 0xb038; int_val = 0xb4c5; break;
        case 13: slope = 0xafd0; int_val = 0xb48e; break;
        case 14: slope = 0xaf80; int_val = 0xb475; break;
        case 15: slope = 0xae20; int_val = 0xb412; break;
        case 16: slope = 0xace0; int_val = 0xb384; break;
        case 17: slope = 0xabc0; int_val = 0xb30c; break;
        case 18: slope = 0xa880; int_val = 0xb256; break;
        case 19: slope = 0xa000; int_val = 0xb1a0; break;
        case 20: slope = 0x2200; int_val = 0xb128; break;
        case 21: slope = 0x2a00; int_val = 0xb062; break;
        case 22: slope = 0x2ce0; int_val = 0xaf98; break;
        case 23: slope = 0x2f60; int_val = 0xae30; break;
        case 24: slope = 0x30e0; int_val = 0xad00; break;
        case 25: slope = 0x3220; int_val = 0xabd0; break;
        case 26: slope = 0x33a0; int_val = 0xa990; break;
        case 27: slope = 0x3480; int_val = 0xa7b0; break;
        case 28: slope = 0x3540; int_val = 0xa4b0; break;
        case 29: slope = 0x3608; int_val = 0xa0b0; break;
        case 30: slope = 0x36ce; int_val = 0x9a60; break;
        case 31: slope = 0x379a; int_val = 0x0000; break;
        case 32: slope = 0x3833; int_val = 0x0000; break;
        case 33: slope = 0x3899; int_val = 0x9a60; break;
        case 34: slope = 0x38fc; int_val = 0xa0b0; break;
        case 35: slope = 0x3960; int_val = 0xa4b0; break;
        case 36: slope = 0x39c0; int_val = 0xa7b0; break;
        case 37: slope = 0x3a18; int_val = 0xa990; break;
        case 38: slope = 0x3a78; int_val = 0xabd0; break;
        case 39: slope = 0x3ac8; int_val = 0xad00; break;
        case 40: slope = 0x3b10; int_val = 0xae20; break;
        case 41: slope = 0x3b68; int_val = 0xafac; break;
        case 42: slope = 0x3ba8; int_val = 0xb076; break;
        case 43: slope = 0x3be0; int_val = 0xb110; break;
        case 44: slope = 0x3c08; int_val = 0xb1a0; break;
        case 45: slope = 0x3c20; int_val = 0xb23c; break;
        case 46: slope = 0x3c40; int_val = 0xb31c; break;
        case 47: slope = 0x3c50; int_val = 0xb394; break;
        case 48: slope = 0x3c64; int_val = 0xb41a; break;
        case 49: slope = 0x3c78; int_val = 0xb474; break;
        case 50: slope = 0x3c78; int_val = 0xb474; break;
        case 51: slope = 0x3c88; int_val = 0xb4cc; break;
        case 52: slope = 0x3c80; int_val = 0xb49c; break;
        case 53: slope = 0x3c78; int_val = 0xb468; break;
        case 54: slope = 0x3c70; int_val = 0xb430; break;
        case 55: slope = 0x3c60; int_val = 0xb370; break;
        case 56: slope = 0x3c48; int_val = 0xb1f0; break;
        case 57: slope = 0x3c30; int_val = 0xb040; break;
        case 58: slope = 0x3c28; int_val = 0xaf40; break;
        case 59: slope = 0x3c08; int_val = 0xa700; break;
        case 60: slope = 0x3c00; int_val = 0x9c00; break;
        case 61: slope = 0x3c10; int_val = 0xab00; break;
        case 62: slope = 0x3c00; int_val = 0x0000; break;
        default: slope = 0x3c00; int_val = 0x0000; break;
    }
    float mac_a, mac_b, mac_c;
    mac_a = fp16_to_32(input);
    mac_b = fp16_to_32(slope);
    mac_c = fp16_to_32(int_val);
    mac_a = mac_a * mac_b + mac_c;
    return fp32_to_16(mac_a);
}

uint16_t HW_LN(uint16_t input, uint16_t op2_sel){
    uint16_t exp_part, mant_part, k;
    int16_t p;
    bool over_sqrt2, sub_flag, k_is_neg;
    exp_part = (input>>10)& 0x1F;
    mant_part = (input)& 0x3FF;
    p = highest_bit_pos_u16(mant_part);
    mant_part = (exp_part != 0) ? mant_part : mant_part << (10- p); 
    over_sqrt2 = (mant_part >= 0x1A9);
    sub_flag = (exp_part == 0);
    k_is_neg = (sub_flag)? true : (over_sqrt2)? (14 > exp_part) : (15> exp_part);
    if(!sub_flag){
        if(!over_sqrt2){
            k = (exp_part >= 15) ? (exp_part - 15) : (15 - exp_part);
        }else{
            k = (exp_part >= 14) ? (exp_part - 14) : (14 - exp_part);
        }
    }else{
        if(!over_sqrt2){
            k = 24 - p;
        }else{
            k = 23 - p;
        }
    }
    uint16_t const_k, m;
    m = over_sqrt2 ? (0x3800|mant_part) : (0x3c00|mant_part);
    switch(k){
        case 0 : const_k = (k_is_neg)? 0x8000 : 0x0000; break;
        case 1 : const_k = (k_is_neg)? 0xbc00 : 0x3c00; break;
        case 2 : const_k = (k_is_neg)? 0xc000 : 0x4000; break;
        case 3 : const_k = (k_is_neg)? 0xc200 : 0x4200; break;
        case 4 : const_k = (k_is_neg)? 0xc400 : 0x4400; break;
        case 5 : const_k = (k_is_neg)? 0xc500 : 0x4500; break;
        case 6 : const_k = (k_is_neg)? 0xc600 : 0x4600; break;
        case 7 : const_k = (k_is_neg)? 0xc700 : 0x4700; break;
        case 8 : const_k = (k_is_neg)? 0xc800 : 0x4800; break;
        case 9 : const_k = (k_is_neg)? 0xc880 : 0x4880; break;
        case 10: const_k = (k_is_neg)? 0xc900 : 0x4900; break;
        case 11: const_k = (k_is_neg)? 0xc980 : 0x4980; break;
        case 12: const_k = (k_is_neg)? 0xca00 : 0x4a00; break;
        case 13: const_k = (k_is_neg)? 0xca80 : 0x4a80; break;
        case 14: const_k = (k_is_neg)? 0xcb00 : 0x4b00; break;
        case 15: const_k = (k_is_neg)? 0xcb80 : 0x4b80; break;
        case 16: const_k = (k_is_neg)? 0xcc00 : 0x4c00; break;
        case 17: const_k = (k_is_neg)? 0xcc40 : 0x4c40; break;
        case 18: const_k = (k_is_neg)? 0xcc80 : 0x4c80; break;
        case 19: const_k = (k_is_neg)? 0xccc0 : 0x4cc0; break;
        case 20: const_k = (k_is_neg)? 0xcd00 : 0x4d00; break;
        case 21: const_k = (k_is_neg)? 0xcd40 : 0x4d40; break;
        case 22: const_k = (k_is_neg)? 0xcd80 : 0x4d80; break;
        case 23: const_k = (k_is_neg)? 0xcdc0 : 0x4dc0; break;
        case 24: const_k = (k_is_neg)? 0xce00 : 0x4e00; break;
        default: const_k = (k_is_neg)? 0x0000 : 0x0000; break;
    }
    uint16_t log_mode;
    switch(op2_sel){
        case 0 : log_mode = 0x3c00; break;
        case 1 : log_mode = 0x3DC5; break;
        case 2 : log_mode = 0x3B48; break;
        default: log_mode = 0x36F3; break;
    }

    float mac_a, mac_b, mac_c, mac_r,delta;
    uint16_t mac_r0, mac_r1;
    //IDLE
    mac_a = fp16_to_32(m);
    mac_r = mac_a -1.0f;
    mac_r0 = fp32_to_16(mac_r);
    //DELTA
    delta = fp16_to_32(mac_r0);
    mac_r = delta* delta;
    mac_r0 = fp32_to_16(mac_r);
    //STEP1
    mac_a = fp16_to_32(mac_r0);
    mac_b = fp16_to_32(0xB800);
    mac_r = mac_a * mac_b + delta;
    mac_r0 = fp32_to_16(mac_r);
    mac_r = mac_a * delta;
    mac_r1 = fp32_to_16(mac_r);

    //STEP2
    mac_a = fp16_to_32(mac_r1);
    mac_b = fp16_to_32(0x3555);
    mac_c = fp16_to_32(mac_r0);
    mac_r = mac_a*mac_b + mac_c;
    mac_r0 = fp32_to_16(mac_r);
    mac_r = mac_a * delta;
    mac_r1 = fp32_to_16(mac_r);

    //STEP3
    mac_a = fp16_to_32(mac_r1);
    mac_b = fp16_to_32(0xb400);
    mac_c = fp16_to_32(mac_r0);
    mac_r = mac_a * mac_b + mac_c;
    mac_r0 = fp32_to_16(mac_r);
    mac_r = mac_a * delta;
    mac_r1 = fp32_to_16(mac_r);
    
    //STEP4
    mac_a = fp16_to_32(mac_r1);
    mac_b = fp16_to_32(0x3266);
    mac_c = fp16_to_32(mac_r0);
    mac_r = mac_a * mac_b + mac_c;
    mac_r0 = fp32_to_16(mac_r);

    //STEP5
    mac_a = fp16_to_32(const_k);
    mac_b = fp16_to_32(0x398b);
    mac_c = fp16_to_32(mac_r0);
    mac_r = mac_a * mac_b  + mac_c;
    mac_r0 = fp32_to_16(mac_r);
    //STEP6

    mac_a = fp16_to_32(mac_r0);
    mac_b = fp16_to_32(log_mode);
    mac_r = mac_a * mac_b;
    mac_r0  = fp32_to_16(mac_r);

    return mac_r0;

}

uint16_t HW_EXP(uint16_t input, uint16_t op2_sel){
    uint16_t data_15_13 , taylor, exp_a, data_12_6;
    data_12_6  = (input>>6) &0x7f;
    data_15_13 = (input>>13)&0x7;
    switch(data_15_13){
    case 6: switch(data_12_6){
            case 47: taylor = 0x4c00; exp_a = 0x0002; break;
            case 46: taylor = 0x4b80; exp_a = 0x0005; break;
            case 45: taylor = 0x4b80; exp_a = 0x0005; break;
            case 44: taylor = 0x4b00; exp_a = 0x000E; break;
            case 43: taylor = 0x4b00; exp_a = 0x000E; break;
            case 42: taylor = 0x4a80; exp_a = 0x0026; break;
            case 41: taylor = 0x4a80; exp_a = 0x0026; break;
            case 40: taylor = 0x4a00; exp_a = 0x0067; break;
            case 39: taylor = 0x4a00; exp_a = 0x0067; break;
            case 38: taylor = 0x4980; exp_a = 0x0118; break;
            case 37: taylor = 0x4980; exp_a = 0x0118; break;
            case 36: taylor = 0x4900; exp_a = 0x02FA; break;
            case 35: taylor = 0x4900; exp_a = 0x02FA; break;
            case 34: taylor = 0x4880; exp_a = 0x080B; break;
            case 33: taylor = 0x4880; exp_a = 0x080B; break;
            case 32: taylor = 0x4800; exp_a = 0x0D7F; break;
            case 31:
            case 30: taylor = 0x4800; exp_a = 0x0D7F; break;
            case 29:
            case 28: taylor = 0x4700; exp_a = 0x1378; break;
            case 27:
            case 26: taylor = 0x4700; exp_a = 0x1378; break;
            case 25: 
            case 24: taylor = 0x4600; exp_a = 0x1914; break;
            case 23:
            case 22: taylor = 0x4600; exp_a = 0x1914; break;
            case 21:
            case 20: taylor = 0x4500; exp_a = 0x1EE6; break;
            case 19:
            case 18: taylor = 0x4500; exp_a = 0x1EE6; break;
            case 17:
            case 16: taylor = 0x4400; exp_a = 0x24B0; break;

            case 15:
            case 14:
            case 13:
            case 12: taylor = 0x4400; exp_a = 0x24B0; break;
            case 11:
            case 10:
            case  9:
            case  8: taylor = 0x4200; exp_a = 0x2A5F; break;
            case  7:
            case  6:
            case  5:
            case  4: taylor = 0x4200; exp_a = 0x2A5F; break;
            case  3:
            case  2:
            case  1:
            case  0: taylor = 0x4000; exp_a = 0x3054; break;
            default: taylor = ((data_15_13 & 0x4) == 0x4)? input & 0x7FFF : 0x7c00; exp_a = ((data_15_13 & 0x4) == 0x4)? 0x0 : 0x7c00; break;
            }
            break;
    case 5: if(data_12_6 > 119){
                taylor = 0x4000; exp_a = 0x3054; 
            }else if(data_12_6 > 111){
                taylor = 0x3c00; exp_a = 0x35E2;
            }else if(data_12_6 > 95){
                taylor = 0x3C00; exp_a = 0x35E2;
            }else if(data_12_6 >63){
                taylor = 0x0000; exp_a = 0x3c00;
            }else{
                taylor = 0x0000; exp_a = 0x3c00;
            }
            break;
    case 4: taylor = 0x0000; exp_a = 0x3c00; break;
    case 0: taylor = 0x0000; exp_a = 0x3c00; break;
    case 1: if(data_12_6 < 64){
                taylor = 0x0000; exp_a = 0x3c00;
            }else if(data_12_6 < 96){          
                taylor = 0x0000; exp_a = 0x3c00;
            }else if(data_12_6 < 112){
                taylor = 0xBC00; exp_a = 0x4170;
            }else if(data_12_6 < 120){
                taylor = 0xBC00; exp_a = 0x4170;
            }else{
                taylor = 0xC000; exp_a = 0x4764;
            }    
            break;
    case 2: switch(data_12_6){
            case 0:
            case 1:
            case 2:
            case 3: taylor = 0xC000; exp_a = 0x4764; break;
            case 4:
            case 5:
            case 6:
            case 7: taylor = 0xC200; exp_a = 0x4D05; break;
            case 8:
            case 9:
            case 10:
            case 11: taylor = 0xC200; exp_a = 0x4D05; break;
            case 12:
            case 13:
            case 14:
            case 15: taylor = 0xC400; exp_a = 0x52D3; break;
            case 16:
            case 17: taylor = 0xC400; exp_a = 0x52D3; break;
            case 18:
            case 19: taylor = 0xC500; exp_a = 0x58A3; break;
            case 20:
            case 21: taylor = 0xC500; exp_a = 0x58A3; break;
            case 22:
            case 23: taylor = 0xC600; exp_a = 0x5E4E; break;
            case 24:
            case 25: taylor = 0xC600; exp_a = 0x5E4E; break;
            case 26:
            case 27: taylor = 0xC700; exp_a = 0x6449; break;
            case 28:
            case 29: taylor = 0xC700; exp_a = 0x6449; break;
            case 30:
            case 31: taylor = 0xC800; exp_a = 0x69D2; break;
            case 32: taylor = 0xC800; exp_a = 0x69D2; break;
            case 33: taylor = 0xC880; exp_a = 0x6FEA; break;
            case 34: taylor = 0xC880; exp_a = 0x6FEA; break;
            case 35: taylor = 0xC900; exp_a = 0x7561; break;
            case 36: taylor = 0xC900; exp_a = 0x7561; break;
            case 37: taylor = 0xC980; exp_a = 0x7B4F; break;
            case 38: taylor = 0xC980; exp_a = 0x7B4F; break;
            case 39: taylor = 0x7c00; exp_a = 0x7c00; break;
            default: taylor = ((data_15_13 & 0x4) == 0x4)? input & 0x7FFF : 0x7c00; exp_a = ((data_15_13 & 0x4) == 0x4)? 0x0 : 0x7c00; break;
        }
        break;
    default: taylor = ((data_15_13 & 0x4) == 0x4)? input & 0x7FFF : 0x7c00; exp_a = ((data_15_13 & 0x4) == 0x4)? 0x0 : 0x7c00; break;
    }
    float mac_a, mac_b, mac_c, mac_r, delta;
    uint16_t mid_r0, mid_r1;
    uint16_t exp_mode;
    switch(op2_sel){
        case  0: exp_mode = 0x3c00; break;
        case  1: exp_mode = 0x398c; break;
        case  2: exp_mode = 0x3c65; break;
        default: exp_mode = 0x3e70; break;
    }

    //idle
    mac_a = fp16_to_32(input);
    mac_b = fp16_to_32(exp_mode);
    mac_r = mac_a * mac_b;
    mid_r0 = fp32_to_16(mac_r);
    //idle2
    mac_a = fp16_to_32(mid_r0);
    mac_b = 1.0f;
    mac_c = fp16_to_32(taylor);
    mac_r = mac_a * mac_b + mac_c;
    mid_r0 = fp32_to_16(mac_r);
    delta = fp16_to_32(mid_r0);
    //delta
    mac_a = fp16_to_32(mid_r0);
    mid_r0 = fp32_to_16(mac_a + 1.0f);
    mac_b = fp16_to_32(mid_r0);
    mac_r = mac_a * mac_a;
    mid_r1 = fp32_to_16(mac_r);
    //step1
    mac_a = fp16_to_32(mid_r1);
    mac_b = fp16_to_32(0x3800);
    mac_c = fp16_to_32(mid_r0);
    mac_r = mac_a * mac_b + mac_c;
    mid_r0 = fp32_to_16(mac_r);
    mac_r = delta * mac_a;
    mid_r1 = fp32_to_16(mac_r);
    //step2
    mac_a = fp16_to_32(mid_r1);
    mac_b = fp16_to_32(0x3155);
    mac_c = fp16_to_32(mid_r0); 
    mac_r = mac_a * mac_b + mac_c;
    mid_r0 = fp32_to_16(mac_r);
    mac_r = delta * mac_a;
    mid_r1 = fp32_to_16(mac_r);

    //step3
    mac_a = fp16_to_32(mid_r1);
    mac_b = fp16_to_32(0x2956);
    mac_c = fp16_to_32(mid_r0);
    mac_r = mac_a * mac_b + mac_c;
    mid_r0 = fp32_to_16(mac_r);
    mac_r = delta * mac_a;
    mid_r1 = fp32_to_16(mac_r);

    //step4
    mac_a = fp16_to_32(mid_r0);
    mac_b = fp16_to_32(exp_a);
    mac_r = mac_a * mac_b;
    mid_r1 = fp32_to_16(mac_r);
    return mid_r1;

}

uint16_t HW_SQRT(uint16_t input){
    uint16_t output, slope , intp_val,sqrt_exp , mac_input;
    uint16_t exp_part, mant_part;
    exp_part = (input>>10) & 0x1F;
    mant_part = (input & 0x3FF);
    bool sqrt_zero, sqrt_nan, sqrt_infinite, sqrt_subnorm;
    sqrt_subnorm = (exp_part == 0x0) && (mant_part != 0);
    sqrt_zero = (input ==0);
    sqrt_nan = (((exp_part) == 0x1F) && ((mant_part) != 0)) || ((input & 0x8000) == 0x8000);
    sqrt_infinite = ((input&  0xFFFF) == 0x7c00);

    uint16_t sp_slope;
    if(sqrt_nan){
        slope = 0x7E00;
    }else if(sqrt_infinite){
        slope = 0x7c00;
    }else if(sqrt_zero){
        slope = 0;
    }else if(sqrt_subnorm){
        switch(((input>>4)& 0x3F)){
            case 0 : if((input & 0x8) == 0x8){
                        slope = 0x60af; intp_val =0x0ea2; 
                    } else if((input & 0x4) == 0x4){
                        slope = 0x629f; intp_val =0x0cb1;
                    } else if((input & 0x2) == 0x2){
                        slope = 0x64af; intp_val =0x0aa2; 
                    } else if((input & 0x1) == 0x1){
                        slope = 0x669f; intp_val =0x08b1;
                    }
                     //sub_slope = 0x63fe; sub_intp_val =0x0000; //not use?
                     break;
            case 1 : slope = 0x5e9f; intp_val =0x10b1; break;
            case 2 : slope = 0x5d15; intp_val =0x123c; break;
            case 3 : slope = 0x5c49; intp_val =0x136e; break;
            case 4 : slope = 0x5b8c; intp_val =0x143a; break;
            case 5 : slope = 0x5ad3; intp_val =0x14ae; break;
            case 6 : slope = 0x5a46; intp_val =0x1517; break;
            case 7 : slope = 0x59d7; intp_val =0x1579; break;
            case 8 : slope = 0x597c; intp_val =0x15d4; break;
            case 9 : slope = 0x5930; intp_val =0x162a; break;
            case 10: slope = 0x58ef; intp_val =0x167b; break;
            case 11: slope = 0x58b7; intp_val =0x16c8; break;
            case 12: slope = 0x5886; intp_val =0x1713; break;
            case 13: slope = 0x585a; intp_val =0x175a; break;
            case 14: slope = 0x5833; intp_val =0x179e; break;
            case 15: slope = 0x5810; intp_val =0x17e1; break;
            case 16: slope = 0x57df; intp_val =0x1810; break;
            case 17: slope = 0x57a5; intp_val =0x182f; break;
            case 18: slope = 0x576f; intp_val =0x184e; break;
            case 19: slope = 0x573e; intp_val =0x186b; break;
            case 20: slope = 0x5710; intp_val =0x1888; break;
            case 21: slope = 0x56e5; intp_val =0x18a4; break;
            case 22: slope = 0x56be; intp_val =0x18bf; break;
            case 23: slope = 0x5699; intp_val =0x18da; break;
            case 24: slope = 0x5676; intp_val =0x18f4; break;
            case 25: slope = 0x5655; intp_val =0x190e; break;
            case 26: slope = 0x5636; intp_val =0x1927; break;
            case 27: slope = 0x5619; intp_val =0x193f; break;
            case 28: slope = 0x55fd; intp_val =0x1958; break;
            case 29: slope = 0x55e3; intp_val =0x196f; break;
            case 30: slope = 0x55ca; intp_val =0x1987; break;
            case 31: slope = 0x55b2; intp_val =0x199e; break;
            case 32: slope = 0x559c; intp_val =0x19b4; break;
            case 33: slope = 0x5586; intp_val =0x19cb; break;
            case 34: slope = 0x5572; intp_val =0x19e1; break;
            case 35: slope = 0x555e; intp_val =0x19f6; break;
            case 36: slope = 0x554b; intp_val =0x1a0c; break;
            case 37: slope = 0x5539; intp_val =0x1a21; break;
            case 38: slope = 0x5527; intp_val =0x1a36; break;
            case 39: slope = 0x5516; intp_val =0x1a4a; break;
            case 40: slope = 0x5506; intp_val =0x1a5e; break;
            case 41: slope = 0x54f7; intp_val =0x1a72; break;
            case 42: slope = 0x54e8; intp_val =0x1a86; break;
            case 43: slope = 0x54d9; intp_val =0x1a9a; break;
            case 44: slope = 0x54cb; intp_val =0x1aad; break;
            case 45: slope = 0x54be; intp_val =0x1ac0; break;
            case 46: slope = 0x54b0; intp_val =0x1ad3; break;
            case 47: slope = 0x54a4; intp_val =0x1ae6; break;
            case 48: slope = 0x5497; intp_val =0x1af8; break;
            case 49: slope = 0x548b; intp_val =0x1b0a; break;
            case 50: slope = 0x5480; intp_val =0x1b1d; break;
            case 51: slope = 0x5475; intp_val =0x1b2f; break;
            case 52: slope = 0x546a; intp_val =0x1b40; break;
            case 53: slope = 0x545f; intp_val =0x1b52; break;
            case 54: slope = 0x5455; intp_val =0x1b63; break;
            case 55: slope = 0x544b; intp_val =0x1b75; break;
            case 56: slope = 0x5441; intp_val =0x1b86; break;
            case 57: slope = 0x5437; intp_val =0x1b97; break;
            case 58: slope = 0x542e; intp_val =0x1ba8; break;
            case 59: slope = 0x5425; intp_val =0x1bb8; break;
            case 60: slope = 0x541c; intp_val =0x1bc9; break;
            case 61: slope = 0x5414; intp_val =0x1bd9; break;
            case 62: slope = 0x540b; intp_val =0x1be9; break;
            case 63: slope = 0x5403; intp_val =0x1bfa; break;
        }
    }else{
        switch((input>>5)& 0x1F){
            case 0 : slope = 0x37f0; intp_val = 0x3808; break;
            case 1 : slope = 0x37d2; intp_val = 0x3818; break;
            case 2 : slope = 0x37b4; intp_val = 0x3827; break;
            case 3 : slope = 0x3798; intp_val = 0x3836; break;
            case 4 : slope = 0x377e; intp_val = 0x3846; break;
            case 5 : slope = 0x3764; intp_val = 0x3854; break;
            case 6 : slope = 0x374b; intp_val = 0x3863; break;
            case 7 : slope = 0x3733; intp_val = 0x3872; break;
            case 8 : slope = 0x371c; intp_val = 0x3880; break;
            case 9 : slope = 0x3706; intp_val = 0x388e; break;
            case 10: slope = 0x36f1; intp_val = 0x389c; break;
            case 11: slope = 0x36dd; intp_val = 0x38aa; break;
            case 12: slope = 0x36c9; intp_val = 0x38b7; break;
            case 13: slope = 0x36b6; intp_val = 0x38c5; break;
            case 14: slope = 0x36a3; intp_val = 0x38d2; break;
            case 15: slope = 0x3691; intp_val = 0x38e0; break;
            case 16: slope = 0x3680; intp_val = 0x38ed; break;
            case 17: slope = 0x366f; intp_val = 0x38fa; break;
            case 18: slope = 0x365e; intp_val = 0x3906; break;
            case 19: slope = 0x364e; intp_val = 0x3913; break;
            case 20: slope = 0x363f; intp_val = 0x3920; break;
            case 21: slope = 0x3630; intp_val = 0x392c; break;
            case 22: slope = 0x3621; intp_val = 0x3938; break;
            case 23: slope = 0x3613; intp_val = 0x3945; break;
            case 24: slope = 0x3605; intp_val = 0x3951; break;
            case 25: slope = 0x35f8; intp_val = 0x395d; break;
            case 26: slope = 0x35eb; intp_val = 0x3968; break;
            case 27: slope = 0x35de; intp_val = 0x3974; break;
            case 28: slope = 0x35d1; intp_val = 0x3980; break;
            case 29: slope = 0x35c5; intp_val = 0x398c; break;
            case 30: slope = 0x35b9; intp_val = 0x3997; break;
            case 31: slope = 0x35ae; intp_val = 0x39a2; break;
        }
    }
    switch(exp_part){
        case 0: sqrt_exp = 0; break;
        case 1: 
        case 2: sqrt_exp = 0x8; break;
        case 3: 
        case 4: sqrt_exp = 0x9; break;
        case 5: 
        case 6: sqrt_exp = 0xa; break;
        case 7:
        case 8: sqrt_exp = 0xb; break;
        case 9:
        case 10: sqrt_exp = 0xc; break;
        case 11:
        case 12: sqrt_exp = 0xd; break;
        case 13:
        case 14: sqrt_exp = 0xe; break;
        case 15:
        case 16: sqrt_exp = 0xf; break;
        case 17:
        case 18: sqrt_exp = 0x10; break;
        case 19:
        case 20: sqrt_exp = 0x11; break;
        case 21:
        case 22: sqrt_exp = 0x12; break;
        case 23:
        case 24: sqrt_exp = 0x13; break;
        case 25:
        case 26: sqrt_exp = 0x14; break;
        case 27:
        case 28: sqrt_exp = 0x15; break;
        case 29:
        case 30: sqrt_exp = 0x16; break;
        case 31: sqrt_exp = 0x1F; break; 
    }
    float f_t, s, intp;
    //half h = half_from_bits(0xB019);
    //uint16_t raw = half_to_bits(h);
    //std::cout << h << "  r wa = " <<std::hex << raw << std::endl;
    bool sqrt_sp_mac;
    sqrt_sp_mac = (exp_part == 0) || (exp_part == 0x1F);
    if(sqrt_sp_mac){
        mac_input = input;
    }else{
        mac_input = (0x3C00)| (mant_part);
    }
    f_t = fp16_to_32(mac_input);
    s = fp16_to_32(slope);
    intp = fp16_to_32(intp_val);
    f_t = (f_t*s) + intp ;
    s = (((input & 0x400) == 0x400) || (exp_part == 0))? 1.0f : 1.4140625;
    output = fp32_to_16(f_t);
    output = (sqrt_sp_mac)? output: (output & 0x8000) | (sqrt_exp << 10) | (output & 0x3FF);
    f_t = fp16_to_32(output);

    f_t = f_t * s;
    output = fp32_to_16(f_t);
    

    return output;
}
