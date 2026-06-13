// 생성 yolo 계열 arch .cpp 를 vision.cpp 빌드로 실행하는 범용 검증 하네스.
// 컴파일 시 -DARCH=<클래스명> -DVISP_ARCH_HEADER='"visp/arch/<클래스명>.h"' 로 지정.
//   예: -DARCH=DetectionModel -DVISP_ARCH_HEADER='"visp/arch/DetectionModel.h"'
#include VISP_ARCH_HEADER
#include "visp/ml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <span>
#include <string>
#include <vector>

using namespace visp;

// ---------------------------------------------------------------------------
// v10 NMS-free CPU 후처리 (ultralytics v10Detect.get_topk_index 와 동치).
// dense: ggml ne=[1,na,C,1] flat=[a + na*ch] (ch 0..3=xyxy, 4..C-1=class sigmoid).
// 반환: (kk,6) [x1,y1,x2,y2,conf,label], score 내림차순. (vision.cpp 의 IoU NMS 는
// 전통 모델용 내부 함수 — v10 은 NMS-free 라 top-k 가 ultralytics 와 정합.)
static std::vector<float> v10_topk(const std::vector<float>& dense,
                                   int64_t na, int C, int max_det) {
    int nc = C - 4;
    auto cls = [&](int c, int64_t a) { return dense[(size_t)a + (size_t)na * (4 + c)]; };
    auto box = [&](int k, int64_t a) { return dense[(size_t)a + (size_t)na * k]; };

    std::vector<float> maxsc(na);
    for (int64_t a = 0; a < na; ++a) {
        float mx = -1e30f;
        for (int c = 0; c < nc; ++c) mx = std::max(mx, cls(c, a));
        maxsc[a] = mx;
    }
    int k = (int)std::min<int64_t>(max_det, na);
    std::vector<int64_t> ori(na);
    std::iota(ori.begin(), ori.end(), (int64_t)0);
    std::partial_sort(ori.begin(), ori.begin() + k, ori.end(),
                      [&](int64_t x, int64_t y) { return maxsc[x] > maxsc[y]; });
    ori.resize(k);

    int kn = k * nc;
    auto sc2 = [&](int f) { return cls(f % nc, ori[f / nc]); };
    std::vector<int> flat(kn);
    std::iota(flat.begin(), flat.end(), 0);
    int kk = std::min(max_det, kn);
    std::partial_sort(flat.begin(), flat.begin() + kk, flat.end(),
                      [&](int x, int y) { return sc2(x) > sc2(y); });

    std::vector<float> out((size_t)kk * 6);
    for (int i = 0; i < kk; ++i) {
        int f = flat[i];
        int64_t a = ori[f / nc];
        out[i * 6 + 0] = box(0, a);
        out[i * 6 + 1] = box(1, a);
        out[i * 6 + 2] = box(2, a);
        out[i * 6 + 3] = box(3, a);
        out[i * 6 + 4] = sc2(f);
        out[i * 6 + 5] = (float)(f % nc);
    }
    return out;
}

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
    if (argc < 4) {
        fprintf(stderr, "usage: %s <gguf> <input_cwhn.bin> <out.bin> [size=640] [postproc=none|v10]\n", argv[0]);
        return 1;
    }
    const char* gguf = argv[1];
    const char* inp  = argv[2];
    const char* outp = argv[3];
    const int SZ = argc > 4 ? atoi(argv[4]) : 640;
    const std::string postproc = argc > 5 ? argv[5] : "none";
    const std::string tapdir = argc > 6 ? argv[6] : "";

    backend_device backend = backend_init();
    model_file file = model_load(gguf);
    model_weights weights = model_init(file.n_tensors());
    model_transfer(file, weights, backend, backend.preferred_float_type(), file.tensor_layout());

    compute_graph graph = compute_graph_init();
    model_ref m(weights, graph);
    PARAMS_T p = DETECT_PARAMS(file);

    tensor input = compute_graph_input(m, GGML_TYPE_F32, {3, SZ, SZ, 1}, "x");
    ggml_build_forward_expand(m.graph, input);

    tensor out = FWD(m, input, p);
    ggml_build_forward_expand(m.graph, out);

    compute_graph_allocate(graph, backend);

    auto in_data = load_bin(inp, (size_t)3 * SZ * SZ);
    transfer_to_backend(input, std::span<const float>(in_data.data(), in_data.size()));

    compute(graph, backend);

    // 중간 텐서 탭 dump: cgraph 노드 중 이름이 "tap" 으로 시작하는 것을 파일로.
    if (!tapdir.empty()) {
        int nn = ggml_graph_n_nodes(graph.graph);
        std::string manpath = tapdir + "/manifest.txt";
        FILE* man = fopen(manpath.c_str(), "w");
        int dumped = 0;
        for (int i = 0; i < nn; ++i) {
            tensor t = ggml_graph_node(graph.graph, i);
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

    std::vector<float> result = od;
    if (postproc == "v10") {
        // dense ne=[1,na,C,1] → CPU top-k → (kk,6)
        int64_t na = out->ne[1];
        int C = (int)out->ne[2];
        result = v10_topk(od, na, C, 300);
        printf("v10 CPU top-k: na=%lld C=%d → %zu detections (6 each)\n",
               (long long)na, C, result.size() / 6);
    }

    FILE* f = fopen(outp, "wb");
    fwrite(result.data(), sizeof(float), result.size(), f);
    fclose(f);
    printf("wrote %s (%zu floats)\n", outp, result.size());
    return 0;
}
