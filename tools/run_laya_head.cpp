#include "laya_head.h"

#include "visp/ml.h"

#include <cstdio>
#include <cstdlib>
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
    if (argc != 9) {
        std::fprintf(
            stderr,
            "usage: %s <head.gguf> <hidden.f32> <token-count> <qtype> "
            "<markers.i32> <marker-count> <logits.f32> <hidden-size>\n",
            argv[0]);
        return 1;
    }
    char const* gguf_path = argv[1];
    char const* hidden_path = argv[2];
    int token_count = std::atoi(argv[3]);
    int qtype_value = std::atoi(argv[4]);
    char const* marker_path = argv[5];
    int marker_count = std::atoi(argv[6]);
    char const* output_path = argv[7];
    int hidden_size = std::atoi(argv[8]);
    if (token_count < 1 || marker_count < 1 || hidden_size < 1 || qtype_value < 0 || qtype_value > 2) {
        std::fprintf(stderr, "invalid shape or qtype argument\n");
        return 1;
    }

    backend_device backend = backend_init();
    model_file file = model_load(gguf_path);
    model_weights weights = model_init(file.n_tensors());
    // LayerNorm weights are exported with the compact head as F16. The CPU
    // backend cannot multiply F32 activations by an F16 vector, so promote the
    // small (29 MB) head on backends that request F32.
    model_transfer(file, weights, backend, backend.preferred_float_type(), file.tensor_layout());

    compute_graph graph = compute_graph_init();
    model_ref model(weights, graph);
    auto params = g2c::laya::detect_head_params(file);
    if (params.hidden_size != hidden_size) {
        std::fprintf(stderr, "hidden-size mismatch: model=%d input=%d\n", params.hidden_size, hidden_size);
        return 1;
    }

    tensor hidden = compute_graph_input(model, GGML_TYPE_F32, {hidden_size, token_count, 1, 1}, "hidden");
    tensor qtype = compute_graph_input(model, GGML_TYPE_I32, {1, 1, 1, 1}, "qtype");
    tensor markers = compute_graph_input(model, GGML_TYPE_I32, {marker_count, 1, 1, 1}, "markers");
    tensor logits = g2c::laya::head_forward(model, hidden, qtype, markers, params);
    ggml_build_forward_expand(graph, logits);
    compute_graph_allocate(graph, backend);

    auto hidden_data = load_binary<float>(hidden_path, size_t(hidden_size) * token_count);
    auto marker_data = load_binary<int32_t>(marker_path, marker_count);
    int32_t qtype_data = qtype_value;
    transfer_to_backend(hidden, std::span<const float>(hidden_data));
    transfer_to_backend(qtype, std::as_bytes(std::span<const int32_t>(&qtype_data, 1)));
    transfer_to_backend(markers, std::as_bytes(std::span<const int32_t>(marker_data)));

    compute(graph, backend);
    std::vector<float> result(marker_count);
    transfer_from_backend(logits, std::span<float>(result));
    FILE* output = std::fopen(output_path, "wb");
    if (!output) {
        std::fprintf(stderr, "cannot open %s\n", output_path);
        return 1;
    }
    std::fwrite(result.data(), sizeof(float), result.size(), output);
    std::fclose(output);
    return 0;
}
