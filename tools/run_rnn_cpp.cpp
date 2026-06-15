// 생성 RNN(LSTM/GRU) arch .cpp 를 vision.cpp 빌드로 실행하는 검증 하네스.
// 입력은 시퀀스 [feat, seq, batch] (ggml ne). 컴파일 시 -DARCH=<클래스명>.
//   usage: run_rnn_cpp <gguf> <input.bin> <out.bin> <feat> <seq> <batch> [tapdir]
#include VISP_ARCH_HEADER
#include "visp/ml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

using namespace visp;

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)
#define FWD CAT(ARCH, _forward)
#define PARAMS_T CAT(ARCH, _params)
#define DETECT_PARAMS CAT(ARCH, _detect_params)

static std::vector<float> load_bin(const char* path, size_t n) {
    std::vector<float> v(n);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    size_t got = fread(v.data(), sizeof(float), n, f);
    fclose(f);
    if (got != n) { fprintf(stderr, "short read %s: %zu/%zu\n", path, got, n); exit(1); }
    return v;
}

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s <gguf> <input.bin> <out.bin> <feat> <seq> <batch> [tapdir]\n", argv[0]);
        return 1;
    }
    const char* gguf = argv[1];
    const char* inp  = argv[2];
    const char* outp = argv[3];
    const int FEAT  = atoi(argv[4]);
    const int SEQ   = atoi(argv[5]);
    const int BATCH = atoi(argv[6]);
    const std::string tapdir = argc > 7 ? argv[7] : "";

    backend_device backend = backend_init();
    model_file file = model_load(gguf);
    model_weights weights = model_init(file.n_tensors());
    model_transfer(file, weights, backend, backend.preferred_float_type(), file.tensor_layout());

    compute_graph graph = compute_graph_init();
    model_ref m(weights, graph);
    PARAMS_T p = DETECT_PARAMS(file);

    // RNN 시퀀스 입력: ne = [feat, seq, batch, 1]
    tensor input = compute_graph_input(m, GGML_TYPE_F32, {FEAT, SEQ, BATCH, 1}, "x");
    ggml_build_forward_expand(*m.graph, input);

    tensor out = FWD(m, input, p);
    ggml_build_forward_expand(*m.graph, out);

    compute_graph_allocate(graph, backend);

    auto in_data = load_bin(inp, (size_t)FEAT * SEQ * BATCH);
    transfer_to_backend(input, std::span<const float>(in_data.data(), in_data.size()));

    compute(graph, backend);

    if (!tapdir.empty()) {
        int nn = ggml_graph_n_nodes(*m.graph);
        std::string manpath = tapdir + "/manifest.txt";
        FILE* man = fopen(manpath.c_str(), "w");
        int dumped = 0;
        for (int i = 0; i < nn; ++i) {
            tensor t = ggml_graph_node(*m.graph, i);
            const char* nm = ggml_get_name(t);
            if (std::strncmp(nm, "tap", 3) != 0) continue;
            int64_t ne = ggml_nelements(t);
            std::vector<float> d((size_t)ne);
            transfer_from_backend(t, std::span<float>(d.data(), d.size()));
            if (man) fprintf(man, "%s %lld %lld %lld %lld\n", nm,
                             (long long)t->ne[0], (long long)t->ne[1],
                             (long long)t->ne[2], (long long)t->ne[3]);
            std::string bp = tapdir + "/" + nm + ".bin";
            FILE* f = fopen(bp.c_str(), "wb");
            if (f) { fwrite(d.data(), sizeof(float), (size_t)ne, f); fclose(f); }
            ++dumped;
        }
        if (man) fclose(man);
        printf("dumped %d taps → %s\n", dumped, tapdir.c_str());
    }

    int64_t n = ggml_nelements(out);
    printf("output ne = [%lld, %lld, %lld, %lld] nelements=%lld\n",
           (long long)out->ne[0], (long long)out->ne[1],
           (long long)out->ne[2], (long long)out->ne[3], (long long)n);
    std::vector<float> od((size_t)n);
    transfer_from_backend(out, std::span<float>(od.data(), od.size()));

    FILE* f = fopen(outp, "wb");
    fwrite(od.data(), sizeof(float), od.size(), f);
    fclose(f);
    printf("wrote %s (%zu floats)\n", outp, od.size());
    return 0;
}
