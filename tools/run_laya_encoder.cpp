#include "laya_encoder.h"

#include "visp/ml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <vector>

using namespace visp;

template <typename T>
static std::vector<T> load_binary(char const* path, size_t count) {
    std::vector<T> data(count);
    FILE* file = std::fopen(path, "rb");
    if (!file) {
        std::fprintf(stderr, "cannot open %s\n", path);
        std::exit(1);
    }
    size_t read = std::fread(data.data(), sizeof(T), count, file);
    std::fclose(file);
    if (read != count) {
        std::fprintf(stderr, "short read %s: %zu/%zu\n", path, read, count);
        std::exit(1);
    }
    return data;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: %s <encoder.gguf> <tokens.i32> <token-count> <hidden.f32>\n", argv[0]);
        return 1;
    }
    int token_count = std::atoi(argv[3]);
    if (token_count < 1 || token_count > 1024) {
        std::fprintf(stderr, "token-count must be in [1, 1024]\n");
        return 1;
    }

    backend_device backend = backend_init();
    model_file file = model_load(argv[1]);
    model_weights weights = model_init(file.n_tensors());
    // Keep checkpoint F16 weights resident. Expanding the 614 MB encoder to
    // F32 on CPU doubles load memory and is unnecessary for ggml_mul_mat.
    model_transfer(file, weights, backend, GGML_TYPE_COUNT, file.tensor_layout());
    auto params = g2c::laya::detect_encoder_params(file);

    compute_graph graph = compute_graph_init(2048);
    model_ref model(weights, graph);
    tensor ids = compute_graph_input(model, GGML_TYPE_I32, {token_count, 1, 1, 1}, "ids");
    tensor positions = compute_graph_input(model, GGML_TYPE_I32, {token_count, 1, 1, 1}, "positions");
    tensor mask = compute_graph_input(model, GGML_TYPE_F32, {token_count, token_count, 1, 1}, "sliding_mask");
    tensor hidden = g2c::laya::encoder_forward(model, ids, positions, mask, params);
    ggml_build_forward_expand(graph, hidden);
    compute_graph_allocate(graph, backend);

    auto token_data = load_binary<int32_t>(argv[2], token_count);
    std::vector<int32_t> position_data(token_count);
    std::vector<float> mask_data(size_t(token_count) * token_count);
    int radius = params.sliding_window / 2;
    for (int query = 0; query < token_count; ++query) {
        position_data[query] = query;
        for (int key = 0; key < token_count; ++key) {
            mask_data[size_t(query) * token_count + key] =
                std::abs(query - key) <= radius ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    transfer_to_backend(ids, std::as_bytes(std::span<const int32_t>(token_data)));
    transfer_to_backend(positions, std::as_bytes(std::span<const int32_t>(position_data)));
    transfer_to_backend(mask, std::span<const float>(mask_data));
    compute(graph, backend);

    std::vector<float> output(size_t(token_count) * params.hidden_size);
    transfer_from_backend(hidden, std::span<float>(output));
    FILE* out = std::fopen(argv[4], "wb");
    if (!out) {
        std::fprintf(stderr, "cannot open %s\n", argv[4]);
        return 1;
    }
    std::fwrite(output.data(), sizeof(float), output.size(), out);
    std::fclose(out);
    return 0;
}
