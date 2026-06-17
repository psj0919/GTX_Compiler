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
  std::vector<uint16_t> f16;   // row-major fp16 데이터
};

}  // namespace

// 파싱된 graph + param 데이터로 GGUF(fp16) 작성(재사용 코어).
void write_gguf_core(const Graph& g,
                            const std::unordered_map<std::string, at::Tensor>& params,
                            const std::string& out_path, const std::string& arch) {
  // 그래프 파라미터 수집(이름 dedup). 데이터는 param_map(원본 PyTorch 레이아웃)에서.
  std::vector<GgufTensor> tensors;
  std::set<std::string> seen;
  for (const auto& node : g.nodes) {
    for (const auto& p : node->op.params) {
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
        at::Tensor h = a.to(at::kHalf).contiguous();
        GgufTensor gt;
        gt.name = key;
        gt.shape.assign(a.sizes().begin(), a.sizes().end());
        gt.f16.resize(h.numel());
        std::memcpy(gt.f16.data(), h.data_ptr(), (size_t)h.numel() * 2);
        tensors.push_back(std::move(gt));
      }
    }
  }

  std::ofstream f(out_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open gguf out: " + out_path);
  const uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" LE
  const uint32_t TYPE_STRING = 8, GGML_F16 = 1;
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
    uint64_t nbytes = (uint64_t)tensors[i].f16.size() * 2;
    off += nbytes;
    off = (off + ALIGN - 1) / ALIGN * ALIGN;
  }
  // tensor infos
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    w_str(f, t.name);
    w_u32(f, (uint32_t)t.shape.size());
    for (auto it = t.shape.rbegin(); it != t.shape.rend(); ++it)  // ggml ne = reversed
      w_u64(f, (uint64_t)*it);
    w_u32(f, GGML_F16);
    w_u64(f, offsets[i]);
  }
  // data 섹션 시작 정렬
  uint64_t pos = (uint64_t)f.tellp();
  uint64_t pad = (pos + ALIGN - 1) / ALIGN * ALIGN - pos;
  for (uint64_t k = 0; k < pad; ++k) f.put(0);
  // tensor data (각 텐서 뒤 ALIGN 패딩)
  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    f.write(reinterpret_cast<const char*>(t.f16.data()), (std::streamsize)t.f16.size() * 2);
    uint64_t nbytes = (uint64_t)t.f16.size() * 2;
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
