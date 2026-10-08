#pragma once

#include "visp/ml.h"

namespace g2c::laya {

struct head_params {
    int hidden_size = 768;
    int attention_heads = 12;
    int block_count = 2;
};

head_params detect_head_params(visp::model_file const& file);

// hidden: [hidden_size, sequence_length], qtype: I32[1], markers: I32[option_count].
// Returns raw, uncalibrated option logits [1, option_count].
visp::tensor head_forward(
    visp::model_ref m,
    visp::tensor hidden,
    visp::tensor qtype,
    visp::tensor markers,
    head_params const& params);

} // namespace g2c::laya
