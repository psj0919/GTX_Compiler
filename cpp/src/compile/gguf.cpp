// GGUF writer (fp16) — Python pipeline.generate_gguf 의 C++ 포트.
// 텐서명 = state_dict 키, dims reversed(ggml ne), arch=general.architecture.
#include "gtxc/ir.hpp"
#include "gtxc/parser.hpp"
#include "parse/parse_internal.hpp"

#include <torch/script.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace gtxc {
// ── GGUF writer (fp16) — Python pipeline.generate_gguf 의 C++ 포트(미fold 버전) ──
// 텐서명 = state_dict 키(<scope>:: 제거). conv weight 는 param_map 의 원본이 이미 OIHW
// 라 transpose 불필요(IR 은 shape 만 OHWI). GGUF 바이너리 포맷(v3) 직접 작성(의존성 X).
namespace {

void w_u32(std::ofstream& f, uint32_t v) { f.write(reinterpret_cast<char*>(&v), 4); }
void w_u64(std::ofstream& f, uint64_t v) { f.write(reinterpret_cast<char*>(&v), 8); }
void w_str(std::ofstream& f, const std::string& s) {
  w_u64(f, s.size());
  f.write(s.data(), (std::streamsize)s.size());
}

struct GgufTensor {
  std::string name;
  std::vector<int64_t> shape;  // numpy 순서(예: [OC,IC,KH,KW])
  std::vector<uint8_t> data;   // raw 바이트(f16 또는 q8_0 블록)
  uint32_t ggml_type = 1;      // 1=F16, 8=Q8_0
};

// f32 텐서 → fp16 GgufTensor(type=1).
GgufTensor make_f16(const std::string& name, at::Tensor a) {
  a = a.to(at::kFloat).contiguous();
  at::Tensor h = a.to(at::kHalf).contiguous();
  GgufTensor gt;
  gt.name = name;
  gt.ggml_type = 1;
  gt.shape.assign(a.sizes().begin(), a.sizes().end());
  gt.data.resize((size_t)h.numel() * 2);
  std::memcpy(gt.data.data(), h.data_ptr(), gt.data.size());
  return gt;
}

// 2D [out,in] (in%32==0) → q8_0 GgufTensor(type=8). 블록 34B = fp16 scale + 32 int8.
// ggml_mul_mat 이 on-the-fly dequant → linear/1x1conv 가 codegen 변경 없이 양자 가중치 사용.
GgufTensor make_q8_0(const std::string& name, at::Tensor a) {
  a = a.to(at::kFloat).contiguous();
  int64_t out = a.size(0), in = a.size(1), nb = in / 32;
  at::Tensor x = a.reshape({out, nb, 32});
  at::Tensor amax = std::get<0>(x.abs().max(2));         // [out,nb]
  at::Tensor d = amax / 127.0f;                          // [out,nb]
  at::Tensor id = at::where(d > 0, 1.0f / d, at::zeros_like(d));
  at::Tensor q = (x * id.unsqueeze(2)).round().clamp(-127, 127).to(at::kChar).contiguous();
  at::Tensor dh = d.to(at::kHalf).contiguous();          // [out,nb] fp16
  GgufTensor gt;
  gt.name = name;
  gt.ggml_type = 8;
  gt.shape.assign(a.sizes().begin(), a.sizes().end());
  int64_t nblocks = out * nb;
  gt.data.resize((size_t)nblocks * 34);
  const uint16_t* dp = reinterpret_cast<const uint16_t*>(dh.data_ptr());
  const int8_t* qp = reinterpret_cast<const int8_t*>(q.data_ptr());
  for (int64_t b = 0; b < nblocks; ++b) {
    std::memcpy(&gt.data[(size_t)b * 34], &dp[b], 2);
    std::memcpy(&gt.data[(size_t)b * 34 + 2], &qp[b * 32], 32);
  }
  return gt;
}

}  // namespace

// 파싱된 graph + param 데이터로 GGUF(fp16) 작성(재사용 코어).
// extra: const-fold baked 텐서(키=GGUF 텐서명 그대로). state_dict param 과 함께 수록.
void write_gguf_core(const Graph& g,
                            const std::unordered_map<std::string, at::Tensor>& params,
                            const std::string& out_path, const std::string& arch,
                            const std::unordered_map<std::string, at::Tensor>* extra,
                            const std::string& quant) {
  bool q8 = (quant == "q8_0");
  int nquant = 0;
  // 그래프 파라미터 수집(이름 dedup). 데이터는 param_map(원본 PyTorch 레이아웃)에서.
  std::vector<GgufTensor> tensors;
  std::set<std::string> seen;
  for (const auto& node : g.nodes) {
    // linear/dense 의 weight 만 양자화 자격(ggml_mul_mat 경유). conv 는 im2col 경로라 제외.
    bool linear_node = (node->op.type == "dense" || node->op.type == "linear");
    for (const auto& p : node->op.params) {
      bool is_weight = (p.key == "weight");
      for (const auto& t : p.tensors) {
        std::string key = t->name;
        auto pos = key.rfind("::");
        if (pos != std::string::npos) key = key.substr(pos + 2);
        if (key.size() >= 19 &&
            key.compare(key.size() - 19, 19, "num_batches_tracked") == 0) continue;
        if (seen.count(key)) continue;
        auto it = params.find(key);
        if (it == params.end()) continue;
        seen.insert(key);
        at::Tensor a = it->second.to(at::kFloat).contiguous();
        if (q8 && linear_node && is_weight && a.dim() == 2 && a.size(1) % 32 == 0) {
          tensors.push_back(make_q8_0(key, a));
          ++nquant;
        } else {
          tensors.push_back(make_f16(key, a));
        }
      }
    }
  }
  // const-fold baked 텐서(키=그대로). scope strip 없이 그대로 수록(fp16 유지).
  if (extra) {
    for (const auto& kv : *extra) {
      if (seen.count(kv.first)) continue;
      seen.insert(kv.first);
      at::Tensor a = kv.second.to(at::kFloat).contiguous();
      if (a.dim() == 0) a = a.reshape({1});  // 0-dim 스칼라(2.0) → [1]
      tensors.push_back(make_f16(kv.first, a));
    }
  }
  if (q8)
    std::fprintf(stderr, "[gtxc-gguf] quantized %d linear weight(s) → q8_0\n", nquant);

  std::ofstream f(out_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open gguf out: " + out_path);
  const uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" LE
  const uint32_t TYPE_STRING = 8;
  const uint64_t ALIGN = 32;

  w_u32(f, GGUF_MAGIC);
  w_u32(f, 3);                       // version
  w_u64(f, tensors.size());          // tensor count
  w_u64(f, 1);                       // kv count: general.architecture
  // kv: general.architecture = arch
  w_str(f, "general.architecture");
  w_u32(f, TYPE_STRING);
  w_str(f, arch);

  // 텐서 데이터 offset 계산(각 텐서 ALIGN 정렬)
  std::vector<uint64_t> offsets(tensors.size());
  uint64_t off = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    offsets[i] = off;
    off += (uint64_t)tensors[i].data.size();
    off = (off + ALIGN - 1) / ALIGN * ALIGN;
  }
  // tensor infos
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    w_str(f, t.name);
    w_u32(f, (uint32_t)t.shape.size());
    for (auto it = t.shape.rbegin(); it != t.shape.rend(); ++it)  // ggml ne = reversed
      w_u64(f, (uint64_t)*it);
    w_u32(f, t.ggml_type);
    w_u64(f, offsets[i]);
  }
  // data 섹션 시작 정렬
  uint64_t pos = (uint64_t)f.tellp();
  uint64_t pad = (pos + ALIGN - 1) / ALIGN * ALIGN - pos;
  for (uint64_t k = 0; k < pad; ++k) f.put(0);
  // tensor data (각 텐서 뒤 ALIGN 패딩)
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    f.write(reinterpret_cast<const char*>(t.data.data()), (std::streamsize)t.data.size());
    uint64_t nbytes = (uint64_t)t.data.size();
    uint64_t p2 = (nbytes + ALIGN - 1) / ALIGN * ALIGN - nbytes;
    for (uint64_t k = 0; k < p2; ++k) f.put(0);
  }
  f.close();
  std::fprintf(stderr, "[gtxc-gguf] %zu tensors → %s (arch=%s)\n", tensors.size(),
               out_path.c_str(), arch.c_str());
}

void write_gguf(const std::string& pt_path, const std::string& graph_name,
                const std::vector<int64_t>& input_shape, const std::string& out_path,
                const std::string& arch) {
  std::unordered_map<std::string, at::Tensor> params;
  Graph g = do_parse(pt_path, graph_name, input_shape, params);
  write_gguf_core(g, params, out_path, arch);
}

}  // namespace gtxc
