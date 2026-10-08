#include "laya_encoder.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace g2c::laya {

using namespace visp;

encoder_params detect_encoder_params(model_file const& file) {
    if (file.arch() != "modern-bert") {
        throw std::runtime_error(
            "Expected general.architecture='modern-bert', got '" + std::string(file.arch()) + "'");
    }
    // The exporter already validates these values against the pinned mmBERT
    // config. Keep the runner compatible with both vision.cpp metadata APIs
    // (older model_file has only signed get_int; current has get_uint32).
    return {};
}

static tensor norm_weight(model_ref m, tensor x, char const* name) {
    x = ggml_norm(m, x, 1e-5f);
    return ggml_mul(m, x, m.weights(name));
}

static tensor linear_weight(model_ref m, tensor x, char const* name) {
    return ggml_mul_mat(m, m.weights(name), x);
}

static tensor attention_block(
    model_ref m,
    tensor x,
    tensor positions,
    tensor mask,
    encoder_params const& params,
    int layer) {
    int64_t sequence = x->ne[1];
    int head_dim = params.hidden_size / params.attention_heads;
    model_ref block = m["blk"][layer];
    tensor residual = x;
    if (layer != 0) {
        x = norm_weight(block, x, "attn_norm.weight");
    }

    tensor qkv = linear_weight(block, x, "attn_qkv.weight");
    size_t element = ggml_type_size(qkv->type);
    tensor q = ggml_view_3d(
        m, qkv, head_dim, params.attention_heads, sequence,
        head_dim * element, qkv->nb[1], 0);
    tensor k = ggml_view_3d(
        m, qkv, head_dim, params.attention_heads, sequence,
        head_dim * element, qkv->nb[1], size_t(params.hidden_size) * element);
    tensor v = ggml_view_3d(
        m, qkv, head_dim, params.attention_heads, sequence,
        head_dim * element, qkv->nb[1], size_t(2 * params.hidden_size) * element);

    q = ggml_rope_ext(
        m, q, positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
        params.context_length, params.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(
        m, k, positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
        params.context_length, params.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

    q = ggml_permute(m, q, 0, 2, 1, 3);
    k = ggml_permute(m, k, 0, 2, 1, 3);
    v = ggml_cont(m, ggml_permute(m, v, 1, 2, 0, 3));
    tensor scores = ggml_mul_mat(m, k, q);
    scores = ggml_soft_max_ext(m, scores, mask, 1.0f / std::sqrt(float(head_dim)), 0.0f);
    tensor attended = ggml_mul_mat(m, v, scores);
    attended = ggml_cont(m, ggml_permute(m, attended, 0, 2, 1, 3));
    attended = ggml_reshape_3d(m, attended, params.hidden_size, sequence, 1);
    attended = linear_weight(block, attended, "attn_output.weight");
    x = ggml_add(m, residual, attended);

    residual = x;
    x = norm_weight(block, x, "ffn_norm.weight");
    x = linear_weight(block, x, "ffn_up.weight");
    int intermediate = int(x->ne[0] / 2);
    tensor input = ggml_view_2d(m, x, intermediate, sequence, x->nb[1], 0);
    tensor gate = ggml_view_2d(
        m, x, intermediate, sequence, x->nb[1], size_t(intermediate) * ggml_type_size(x->type));
    x = ggml_mul(m, ggml_gelu(m, input), gate);
    x = linear_weight(block, x, "ffn_down.weight");
    return ggml_add(m, residual, x);
}

tensor encoder_forward(
    model_ref m,
    tensor ids,
    tensor positions,
    tensor sliding_mask,
    encoder_params const& params) {
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && positions->type == GGML_TYPE_I32);
    GGML_ASSERT(ids->ne[0] == positions->ne[0]);
    tensor x = ggml_get_rows(m, m.weights("token_embd.weight"), ids);
    x = norm_weight(m, x, "token_embd_norm.weight");
    for (int layer = 0; layer < params.block_count; ++layer) {
        // HF mmBERT layer_types starts with full attention, followed by two
        // sliding layers. llama.cpp's generic ModernBERT loader uses the
        // opposite phase, so keep the checkpoint's explicit phase here.
        tensor mask = layer % params.sliding_pattern == 0 ? nullptr : sliding_mask;
        x = attention_block(m, x, positions, mask, params, layer);
    }
    x = norm_weight(m, x, "output_norm.weight");
    return compute_graph_output(m, x, "hidden");
}

} // namespace g2c::laya
