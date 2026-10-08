#include "laya_head.h"

#include "visp/nn.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace g2c::laya {

using namespace visp;

head_params detect_head_params(model_file const& file) {
    if (file.arch() != "laya-head") {
        throw std::runtime_error(
            "Expected general.architecture='laya-head', got '" + std::string(file.arch()) + "'");
    }
    return {};
}

static tensor transformer_block(model_ref m, tensor x, int n_heads) {
    tensor residual = x;
    tensor norm = layer_norm(m["norm1"], x, 1e-5f);
    auto [q, k, v] = split_qkv(m["self_attn"], norm, n_heads, 2);
    float scale = 1.0f / std::sqrt(float(x->ne[0] / n_heads));
    x = attention(m, q, k, v, nullptr, scale, m["self_attn"]["out_proj"]);
    x = ggml_add(m, residual, x);

    residual = x;
    x = layer_norm(m["norm2"], x, 1e-5f);
    x = linear(m["linear1"], x);
    x = ggml_relu(m, x);
    x = linear(m["linear2"], x);
    return named(m, ggml_add(m, residual, x));
}

tensor head_forward(
    model_ref m,
    tensor hidden,
    tensor qtype,
    tensor markers,
    head_params const& params) {
    GGML_ASSERT(hidden->ne[0] == params.hidden_size);
    GGML_ASSERT(qtype->type == GGML_TYPE_I32 && ggml_nelements(qtype) == 1);
    GGML_ASSERT(markers->type == GGML_TYPE_I32);

    tensor type = ggml_get_rows(m, m.weights("type_emb.weight"), qtype);
    type = ggml_repeat(m, type, hidden);
    tensor x = ggml_add(m, hidden, type);
    for (int i = 0; i < params.block_count; ++i) {
        x = transformer_block(m["head"]["layers"][i], x, params.attention_heads);
    }

    // torch.gather(h, sequence_axis, marker_pos) is a row gather in ggml's
    // [hidden, sequence] layout.
    x = ggml_get_rows(m, x, markers);
    x = layer_norm(m["scorer"][0], x, 1e-5f);
    x = linear(m["scorer"][1], x);
    x = ggml_gelu(m, x);
    x = linear(m["scorer"][3], x);
    return compute_graph_output(m, x, "logits");
}

} // namespace g2c::laya
