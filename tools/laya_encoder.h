#pragma once

#include "visp/ml.h"

namespace g2c::laya {

struct encoder_params {
    int hidden_size = 768;
    int attention_heads = 12;
    int block_count = 22;
    int sliding_window = 128;
    int sliding_pattern = 3;
    int context_length = 8192;
    float rope_theta = 160000.0f;
};

encoder_params detect_encoder_params(visp::model_file const& file);

// ids/positions: I32[L], sliding_mask: F32[L,L], output: F32[hidden,L].
visp::tensor encoder_forward(
    visp::model_ref m,
    visp::tensor ids,
    visp::tensor positions,
    visp::tensor sliding_mask,
    encoder_params const& params);

} // namespace g2c::laya
