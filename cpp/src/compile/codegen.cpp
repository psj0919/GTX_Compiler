// Conv-BN fold + ggml/vision.cpp arch C++ codegen (VispCodeGenerator 포트).
#include "gtxc/ir.hpp"
#include "gtxc/parser.hpp"
#include "parse/parse_internal.hpp"

#include <torch/script.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace gtxc {
// ── Conv-BN fold + ggml-arch C++ codegen (VispCodeGenerator 포트, resnet op셋) ──
namespace {

// scope:: 제거 + ".weight" 등 제거 → conv_2d(m["<key>"]) 의 weight key.
std::string weight_key_of(const Node& n) {
  for (const auto& p : n.op.params) {
    if (p.key == "weight" && !p.tensors.empty()) {
      std::string s = p.tensors[0]->name;
      auto pos = s.rfind("::");
      if (pos != std::string::npos) s = s.substr(pos + 2);
      if (s.size() > 7 && s.compare(s.size() - 7, 7, ".weight") == 0)
        s = s.substr(0, s.size() - 7);
      return s;
    }
  }
  return "";
}

std::string param_full(const Node& n, const std::string& key) {
  for (const auto& p : n.op.params)
    if (p.key == key && !p.tensors.empty()) return p.tensors[0]->name;
  return "";
}

double config_double(const Node& n, const std::string& key, double def) {
  for (const auto& kv : n.op.configs)
    if (kv.first == key) {
      if (kv.second.type == json::Value::Double) return kv.second.d;
      if (kv.second.type == json::Value::Int) return (double)kv.second.i;
    }
  return def;
}

int64_t config_int_elem(const Node& n, const std::string& key, size_t i, int64_t def) {
  for (const auto& kv : n.op.configs)
    if (kv.first == key && kv.second.type == json::Value::Arr && kv.second.arr->size() > i) {
      const auto& e = (*kv.second.arr)[i];
      if (e.type == json::Value::Int) return e.i;
    }
  return def;
}

}  // namespace

// Conv-BN fold: BN 을 직전 conv 의 weight/bias 로 흡수(vision.cpp 는 BN fused 강제).
// params(at::Tensor) 와 graph(IR) 를 함께 갱신. alias: bn 출력 텐서명 → conv 출력 텐서명.
static void fold_conv_bn(Graph& g, std::unordered_map<std::string, at::Tensor>& params,
                         std::unordered_map<std::string, std::string>& alias) {
  // 노드 이름 → Node*
  std::unordered_map<std::string, Node*> by_name;
  for (auto& n : g.nodes) by_name[n->name] = n.get();

  auto stripscope = [](std::string s) {
    auto p = s.rfind("::"); return p == std::string::npos ? s : s.substr(p + 2);
  };

  std::vector<size_t> remove_idx;
  for (size_t i = 0; i < g.nodes.size(); ++i) {
    Node* bn = g.nodes[i].get();
    if (bn->op.type != "batch_norm" || bn->in_tensors.empty()) continue;
    Node* conv = by_name.count(bn->in_tensors[0]->producer)
                     ? by_name[bn->in_tensors[0]->producer] : nullptr;
    if (!conv || (conv->op.type != "conv2d" && conv->op.type != "depthwise_conv2d")) continue;

    std::string wk = stripscope(param_full(*conv, "weight"));
    std::string gk = stripscope(param_full(*bn, "weight"));
    std::string bk = stripscope(param_full(*bn, "bias"));
    std::string mk = stripscope(param_full(*bn, "mean"));
    std::string vk = stripscope(param_full(*bn, "var"));
    if (!params.count(wk) || !params.count(gk) || !params.count(mk) || !params.count(vk))
      continue;
    double eps = config_double(*bn, "eps", 1e-5);
    at::Tensor W = params[wk].to(at::kFloat);      // OIHW [OC,IC,KH,KW]
    int64_t OC = W.size(0);
    at::Tensor gamma = params[gk].to(at::kFloat), beta = params[bk].to(at::kFloat);
    at::Tensor mean = params[mk].to(at::kFloat), var = params[vk].to(at::kFloat);
    at::Tensor scale = gamma / (var + eps).sqrt();        // [OC]
    at::Tensor Wf = W * scale.view({OC, 1, 1, 1});
    std::string conv_bk = stripscope(param_full(*conv, "bias"));
    at::Tensor b = conv_bk.size() && params.count(conv_bk) ? params[conv_bk].to(at::kFloat)
                                                           : at::zeros({OC});
    at::Tensor bf = (b - mean) * scale + beta;

    params[wk] = Wf;
    std::string newbk = wk.substr(0, wk.size() - 7) + ".bias";  // conv1.weight → conv1.bias
    params[newbk] = bf;
    // conv IR 에 bias param 추가(GGUF 가 수록하도록)
    auto bt = std::make_shared<Tensor>();
    bt->name = g.name + "::" + newbk;
    bt->shape = {OC}; bt->shape_known = true; bt->dtype = "float32";
    bool hasb = false;
    for (auto& p : conv->op.params) if (p.key == "bias") hasb = true;
    if (!hasb) conv->op.params.push_back(Param{"bias", {bt}, false});

    alias[bn->out_tensors[0]->name] = conv->out_tensors[0]->name;
    remove_idx.push_back(i);
  }
  // BN 노드 제거(역순)
  for (auto it = remove_idx.rbegin(); it != remove_idx.rend(); ++it)
    g.nodes.erase(g.nodes.begin() + *it);
}

namespace {

// alias 체인 따라 최종 텐서명 해석.
std::string resolve_alias(const std::unordered_map<std::string, std::string>& a, std::string n) {
  while (a.count(n)) n = a.at(n);
  return n;
}

std::string var_prefix(const std::string& op) {
  if (op == "conv2d" || op == "depthwise_conv2d") return "conv";
  if (op == "relu") return "relu";
  if (op == "aten::silu_" || op == "silu") return "silu";
  if (op == "elemwise_add") return "add";
  if (op == "elemwise_mul") return "mul";
  if (op == "elementwise_sub") return "sub";
  if (op == "elemwise_div") return "div";
  if (op == "sigmoid") return "sig";
  if (op == "concat") return "cat";
  if (op == "chunk" || op == "split") return "sl";
  if (op == "resize") return "up";
  if (op == "reshape") return "rs";
  if (op == "transpose") return "t";
  if (op == "permute") return "t";
  if (op == "matmul") return "mm";
  if (op == "softmax") return "sm";
  if (op == "maxpool") return "pool";
  if (op == "adaptive_avg_pool2d") return "pool";
  if (op == "flatten") return "flat";
  if (op == "dense") return "fc";
  return "t";
}

// torch dim → ggml ne 축(역순). rank 4, torch dim1(채널) → ggml 2.
int64_t ggml_dim(int64_t torch_dim, int64_t rank) {
  if (torch_dim < 0) torch_dim += rank;
  return rank - 1 - torch_dim;
}

int64_t config_int(const Node& n, const std::string& key, int64_t def) {
  for (const auto& kv : n.op.configs)
    if (kv.first == key && kv.second.type == json::Value::Int) return kv.second.i;
  return def;
}

// N-D permute(order) → 블록 축약. 인접·동순서 입력축을 한 블록으로 병합(평탄 재해석).
// order: 출력축 i ← 입력축 order[i]. (Python head_render._fold_permute 포트)
void fold_permute(const std::vector<int64_t>& in_shape, const std::vector<int>& order,
                  std::vector<int64_t>& block_sizes, std::vector<int>& order_blocks) {
  int n = (int)in_shape.size();
  std::vector<int> inv(n, 0);
  for (int op = 0; op < n; ++op) { int ia = ((order[op] % n) + n) % n; inv[ia] = op; }
  std::vector<std::vector<int>> blocks;
  blocks.push_back({0});
  for (int ax = 1; ax < n; ++ax) {
    if (inv[ax] == inv[ax - 1] + 1) blocks.back().push_back(ax);
    else blocks.push_back({ax});
  }
  block_sizes.clear();
  for (auto& b : blocks) { int64_t s = 1; for (int ax : b) s *= in_shape[ax]; block_sizes.push_back(s); }
  int K = (int)blocks.size();
  order_blocks.resize(K);
  for (int i = 0; i < K; ++i) order_blocks[i] = i;
  std::sort(order_blocks.begin(), order_blocks.end(),
            [&](int a, int b) { return inv[blocks[a][0]] < inv[blocks[b][0]]; });
}

// torch permute order → ggml_permute(p0..3) 인자(블록 K≤4).
void perm_args4(const std::vector<int>& order, int ndim, int p[4]) {
  for (int i = 0; i < 4; ++i) p[i] = i;
  for (int i = 0; i < ndim; ++i) { int o = ((order[i] % ndim) + ndim) % ndim; p[ndim - 1 - o] = ndim - 1 - i; }
}

// >4D permute → 블록 축약 ≤4D permute(K≤4) 또는 인접 전치 시퀀스(K>4). 정확.
std::string render_perm_general(const std::vector<int64_t>& in_shape, const std::vector<int>& order,
                                const std::string& iv, int& counter, std::string& out_var) {
  std::vector<int64_t> bs; std::vector<int> ob;
  fold_permute(in_shape, order, bs, ob);
  int K = (int)bs.size();
  std::string body; char b[640];
  if (K <= 4) {
    std::string ne;
    for (int i = K - 1; i >= 0; --i) ne += std::to_string((long long)bs[i]) + (i ? ", " : "");
    std::string r = "fld" + std::to_string(++counter);
    std::snprintf(b, sizeof(b), "    tensor %s = ggml_reshape_%dd(m, ggml_cont(m, %s), %s);\n",
                  r.c_str(), K, iv.c_str(), ne.c_str());
    body += b;
    int p[4]; perm_args4(ob, K, p);
    out_var = "fld" + std::to_string(++counter);
    std::snprintf(b, sizeof(b), "    tensor %s = ggml_cont(m, ggml_permute(m, %s, %d, %d, %d, %d));\n",
                  out_var.c_str(), r.c_str(), p[0], p[1], p[2], p[3]);
    body += b;
  } else {
    std::string v = "fld" + std::to_string(++counter);
    std::snprintf(b, sizeof(b), "    tensor %s = ggml_cont(m, %s);\n", v.c_str(), iv.c_str());
    body += b;
    std::vector<int> cur(K); for (int i = 0; i < K; ++i) cur[i] = i;
    std::vector<int64_t> cs = bs;
    for (int i = 0; i < K; ++i) {
      int j = (int)(std::find(cur.begin(), cur.end(), ob[i]) - cur.begin());
      while (j > i) {
        int64_t left = 1; for (int t = 0; t < j - 1; ++t) left *= cs[t];
        int64_t bp = cs[j - 1], bp1 = cs[j], right = 1; for (int t = j + 1; t < K; ++t) right *= cs[t];
        std::string r2 = "fld" + std::to_string(++counter);
        std::snprintf(b, sizeof(b), "    tensor %s = ggml_reshape_4d(m, %s, %lld, %lld, %lld, %lld);\n",
                      r2.c_str(), v.c_str(), (long long)right, (long long)bp1, (long long)bp, (long long)left);
        body += b;
        std::string s2 = "fld" + std::to_string(++counter);
        std::snprintf(b, sizeof(b), "    tensor %s = ggml_cont(m, ggml_permute(m, %s, 0, 2, 1, 3));\n",
                      s2.c_str(), r2.c_str());
        body += b;
        v = s2;
        std::swap(cur[j - 1], cur[j]); std::swap(cs[j - 1], cs[j]);
        --j;
      }
    }
    out_var = v;
  }
  return body;
}

}  // namespace

// RNN(lstm/gru) 시퀀스 정적 unroll render — Python nn/modules/lstm_render·gru_render 포트.
// 입력 x ne=[feat,seq,batch]. 게이트는 timestep 마다 ggml primitive 로 emit(C++ 루프 아님).
// 단/양방향·멀티레이어 지원. weight 는 GGUF 텐서명(weight_ih_l{n}[_reverse]) 으로 접근.
static std::string render_rnn(const Node& n, bool is_gru, const std::string& xvar,
                              int& counter, std::string& last_out) {
  std::string body;
  char buf[700];
  auto nv = [&](const char* p) { return std::string(p) + std::to_string(++counter); };
  auto L = [&](const std::string& s) { body += "    " + s + "\n"; };

  const auto& xs = n.in_tensors[0]->shape;  // torch [batch, seq, feat]
  int64_t batch = xs.size() > 0 ? xs[0] : 1;
  int64_t seq = xs.size() > 1 ? xs[1] : 1;
  int64_t feat0 = xs.size() > 2 ? xs[2] : 1;

  // weight 파라미터: key → shape. layer/dir/H 유도.
  std::unordered_map<std::string, const Tensor*> w;
  for (const auto& p : n.op.params)
    if (!p.tensors.empty()) w[p.key] = p.tensors[0].get();
  bool bidir = false;
  int num_layers = 1;
  for (const auto& kv : w) {
    if (kv.first.find("_reverse") != std::string::npos) bidir = true;
    if (kv.first.rfind("weight_ih_l", 0) == 0) {
      int lid = std::atoi(kv.first.c_str() + 11);  // after "weight_ih_l"
      if (lid + 1 > num_layers) num_layers = lid + 1;
    }
  }
  int ndir = bidir ? 2 : 1;
  int64_t H = w.count("weight_hh_l0") && w["weight_hh_l0"]->shape.size() > 1
                  ? w["weight_hh_l0"]->shape[1] : 0;

  std::string layer_in = xvar;
  int64_t in_size = feat0;
  for (int layer = 0; layer < num_layers; ++layer) {
    std::vector<std::string> dir_outs;
    for (int d = 0; d < ndir; ++d) {
      std::string suf = d == 1 ? "_reverse" : "";
      std::string ls = std::to_string(layer) + suf;
      std::string wih = "m.weights(\"weight_ih_l" + ls + "\")";
      std::string whh = "m.weights(\"weight_hh_l" + ls + "\")";
      std::string bih = "m.weights(\"bias_ih_l" + ls + "\")";
      std::string bhh = "m.weights(\"bias_hh_l" + ls + "\")";
      std::vector<std::string> h_at(seq);
      std::string h_var, c_var;
      bool first = true;
      for (int64_t step = 0; step < seq; ++step) {
        int64_t t = d == 1 ? seq - 1 - step : step;  // reverse 방향
        std::string xt = nv("xt");
        std::snprintf(buf, sizeof(buf),
            "tensor %s = ggml_cont(m, ggml_view_2d(m, %s, %lld, %lld, %s->nb[2], (size_t)%lld*%s->nb[1]));",
            xt.c_str(), layer_in.c_str(), (long long)in_size, (long long)batch,
            layer_in.c_str(), (long long)t, layer_in.c_str());
        L(buf);
        auto chunk = [&](const std::string& src, int64_t k, const char* name) {
          std::string v = nv(name);
          std::snprintf(buf, sizeof(buf),
              "tensor %s = ggml_cont(m, ggml_view_2d(m, %s, %lld, %lld, %s->nb[1], (size_t)%lld*%lld*sizeof(float)));",
              v.c_str(), src.c_str(), (long long)H, (long long)batch, src.c_str(),
              (long long)k, (long long)H);
          L(buf);
          return v;
        };
        if (!is_gru) {
          // LSTM: gate chunk 순서 i,f,g,o. c=f*c+i*g; h=o*tanh(c). h0=c0=0 첫 step 특수.
          std::string g = nv("g");
          std::snprintf(buf, sizeof(buf),
              "tensor %s = ggml_add(m, ggml_mul_mat(m, %s, %s), ggml_add(m, %s, %s));",
              g.c_str(), wih.c_str(), xt.c_str(), bih.c_str(), bhh.c_str());
          L(buf);
          if (!first) {
            std::snprintf(buf, sizeof(buf),
                "%s = ggml_add(m, %s, ggml_mul_mat(m, %s, %s));",
                g.c_str(), g.c_str(), whh.c_str(), h_var.c_str());
            L(buf);
          }
          std::string si = nv("si");
          L("tensor " + si + " = ggml_sigmoid(m, " + chunk(g, 0, "ig") + ");");
          std::string tg = nv("tg");
          L("tensor " + tg + " = ggml_tanh(m, " + chunk(g, 2, "gg") + ");");
          std::string so = nv("so");
          L("tensor " + so + " = ggml_sigmoid(m, " + chunk(g, 3, "og") + ");");
          std::string c_new = nv("c");
          if (first) {
            L("tensor " + c_new + " = ggml_mul(m, " + si + ", " + tg + ");");
          } else {
            std::string sf = nv("sf");
            L("tensor " + sf + " = ggml_sigmoid(m, " + chunk(g, 1, "fg") + ");");
            std::snprintf(buf, sizeof(buf),
                "tensor %s = ggml_add(m, ggml_mul(m, %s, %s), ggml_mul(m, %s, %s));",
                c_new.c_str(), sf.c_str(), c_var.c_str(), si.c_str(), tg.c_str());
            L(buf);
          }
          std::string h_new = nv("h");
          L("tensor " + h_new + " = ggml_mul(m, " + so + ", ggml_tanh(m, " + c_new + "));");
          c_var = c_new; h_var = h_new; first = false;
        } else {
          // GRU: gate chunk 순서 r,z,n. h=(1-z)*n+z*h_prev. b_hh 첫 step 도 유효→h0=0 일반경로.
          std::string gx = nv("gx");
          std::snprintf(buf, sizeof(buf),
              "tensor %s = ggml_add(m, ggml_mul_mat(m, %s, %s), %s);",
              gx.c_str(), wih.c_str(), xt.c_str(), bih.c_str());
          L(buf);
          if (first) {
            h_var = nv("h0");
            std::snprintf(buf, sizeof(buf),
                "tensor %s = ggml_scale(m, ggml_cont(m, ggml_view_2d(m, %s, %lld, %lld, %s->nb[1], (size_t)0)), 0.0f);",
                h_var.c_str(), gx.c_str(), (long long)H, (long long)batch, gx.c_str());
            L(buf);
          }
          std::string gh = nv("gh");
          std::snprintf(buf, sizeof(buf),
              "tensor %s = ggml_add(m, ggml_mul_mat(m, %s, %s), %s);",
              gh.c_str(), whh.c_str(), h_var.c_str(), bhh.c_str());
          L(buf);
          std::string gxr = chunk(gx, 0, "gxr"), gxz = chunk(gx, 1, "gxz"), gxn = chunk(gx, 2, "gxn");
          std::string ghr = chunk(gh, 0, "ghr"), ghz = chunk(gh, 1, "ghz"), ghn = chunk(gh, 2, "ghn");
          std::string r = nv("r");
          L("tensor " + r + " = ggml_sigmoid(m, ggml_add(m, " + gxr + ", " + ghr + "));");
          std::string z = nv("z");
          L("tensor " + z + " = ggml_sigmoid(m, ggml_add(m, " + gxz + ", " + ghz + "));");
          std::string nn_ = nv("n");
          L("tensor " + nn_ + " = ggml_tanh(m, ggml_add(m, " + gxn + ", ggml_mul(m, " + r + ", " + ghn + ")));");
          std::string h_new = nv("h");
          L("tensor " + h_new + " = ggml_add(m, " + nn_ + ", ggml_mul(m, " + z + ", ggml_sub(m, " + h_var + ", " + nn_ + ")));");
          h_var = h_new; first = false;
        }
        std::string hr = nv("hr");
        std::snprintf(buf, sizeof(buf), "tensor %s = ggml_reshape_3d(m, %s, %lld, 1, %lld);",
                      hr.c_str(), h_var.c_str(), (long long)H, (long long)batch);
        L(buf);
        h_at[t] = hr;
      }
      std::string out = h_at[0];
      for (int64_t t = 1; t < seq; ++t) {
        std::string nx = nv("seqcat");
        L("tensor " + nx + " = ggml_concat(m, " + out + ", " + h_at[t] + ", 1);");
        out = nx;
      }
      dir_outs.push_back(out);
    }
    if (ndir == 2) {
      std::string lo = nv("dircat");
      L("tensor " + lo + " = ggml_concat(m, " + dir_outs[0] + ", " + dir_outs[1] + ", 0);");
      layer_in = lo;
    } else {
      layer_in = dir_outs[0];
    }
    in_size = H * ndir;
  }
  // 마지막 layer_in 을 반환변수로(호출측이 last 로 사용).
  last_out = layer_in;
  return body;
}

// resnet op 체인 → vision.cpp arch 스타일 forward C++.
// constants: trace 가 상수로 fold 한 head anchor/stride/scalar(name→데이터).
// baked_out: codegen 이 const.foldN 키로 baking 한 텐서(gguf writer 가 수록).
static std::string generate_arch_cpp(const Graph& g, const std::string& model,
                                     const std::unordered_map<std::string, std::string>& alias,
                                     const std::unordered_map<std::string, at::Tensor>& constants,
                                     std::unordered_map<std::string, at::Tensor>& baked_out) {
  std::string body;
  std::unordered_map<std::string, std::string> tvar;  // 텐서명 → C++ var
  std::unordered_map<std::string, std::string> const_key;  // 상수 텐서명 → const.foldN
  int counter = 0, const_counter = 0;
  std::string last;
  char buf[512];

  // RNN(lstm/gru) 모델: 시퀀스 입력 [feat,seq,batch] — cwhn 변환 생략.
  bool is_rnn = false;
  for (const auto& np : g.nodes)
    if (np->op.type == "aten::lstm" || np->op.type == "aten::gru") is_rnn = true;

  auto invar = [&](const std::shared_ptr<Tensor>& t) -> std::string {
    std::string nm = resolve_alias(alias, t->name);
    auto it = tvar.find(nm);
    if (it != tvar.end()) return it->second;
    // const-fold: trace 가 접은 head 상수(anchor/stride/2.0)는 const.foldN GGUF weight 로
    // baking 해 m.weights 로 로드(Python const_fold.py 와 동등). 입력만 x 로 남는다.
    auto ci = constants.find(nm);
    if (ci != constants.end()) {
      auto bk = const_key.find(nm);
      std::string key;
      if (bk != const_key.end()) key = bk->second;
      else {
        key = "const.fold" + std::to_string(++const_counter);
        const_key[nm] = key;
        baked_out[key] = ci->second;
      }
      return "m.weights(\"" + key + "\")";
    }
    return "x";  // 미정의(입력) → x
  };

  for (const auto& np : g.nodes) {
    const Node& n = *np;
    if (n.op.type == "input" || n.op.type == "return") continue;
    if (n.op.type == "batch_norm") continue;  // folded
    if (n.op.type == "shape") continue;       // 스칼라(reshape dim) — 텐서 op 아님
    if (n.op.type == "zeros") continue;       // RNN h0/c0 — render_rnn 내부 처리

    // RNN(lstm/gru): 시퀀스 정적 unroll(cwhn 변환 없음).
    if (n.op.type == "aten::lstm" || n.op.type == "aten::gru") {
      std::string rnn_last;
      body += render_rnn(n, n.op.type == "aten::gru", invar(n.in_tensors[0]), counter,
                         rnn_last);
      if (!n.out_tensors.empty()) tvar[n.out_tensors[0]->name] = rnn_last;
      last = rnn_last;
      continue;
    }

    // >4D transpose/permute(전이 5D, 예: ShuffleNet channel-shuffle) → 블록 축약 ≤4D.
    // reshape↑/↓ 는 평탄 재해석(물리 ≤4D 유지)이라, 데이터 이동인 permute 만 표현하면 정확.
    if ((n.op.type == "transpose" || n.op.type == "permute") &&
        n.in_tensors[0]->shape_known && n.in_tensors[0]->shape.size() > 4) {
      const auto& insh = n.in_tensors[0]->shape;
      int rank = (int)insh.size();
      std::vector<int> order(rank);
      for (int i = 0; i < rank; ++i) order[i] = i;
      if (n.op.type == "transpose") {
        int d0 = (int)config_int(n, "dim0", 0), d1 = (int)config_int(n, "dim1", 1);
        if (d0 < 0) d0 += rank; if (d1 < 0) d1 += rank;
        if (d0 >= 0 && d0 < rank && d1 >= 0 && d1 < rank) std::swap(order[d0], order[d1]);
      } else {  // permute: order config(int list)
        for (const auto& kv : n.op.configs)
          if ((kv.first == "dims" || kv.first == "order") &&
              kv.second.type == json::Value::Arr) {
            order.clear();
            for (const auto& e : *kv.second.arr)
              if (e.type == json::Value::Int) order.push_back((int)e.i);
            break;
          }
      }
      std::string ov;
      body += render_perm_general(insh, order, invar(n.in_tensors[0]), counter, ov);
      if (!n.out_tensors.empty()) tvar[n.out_tensors[0]->name] = ov;
      last = ov;
      continue;
    }

    // chunk/split: 다중 출력 → 각 조각을 ggml_view_<rank>d 로(분할 축 오프셋 누적).
    // rank-general: ne=reversed(torch shape), 분할축 gd=ggml_dim(tdim,rank),
    // offset=accum*nb[gd]. 3d(DFL distance) / 4d(backbone) 모두 정합.
    if (n.op.type == "chunk" || n.op.type == "split") {
      std::string iv = invar(n.in_tensors[0]);
      int64_t rank = n.in_tensors[0]->shape_known ? (int64_t)n.in_tensors[0]->shape.size() : 4;
      int64_t tdim = config_int(n, "dim", 1);
      int64_t gd = ggml_dim(tdim, rank);
      int64_t accum = 0;
      for (const auto& ot : n.out_tensors) {
        std::string var = "sl" + std::to_string(++counter);
        auto& s = ot->shape;
        int64_t R = (int64_t)s.size();
        if (R < 1 || R > 4) R = 4;
        // ne = reversed torch shape (R dims)
        std::vector<int64_t> ne(R, 1);
        for (int64_t k = 0; k < R; ++k) ne[k] = s[(size_t)(R - 1 - k)];
        // strides nb[1..R-1] + offset
        std::string strides;
        for (int64_t k = 1; k < R; ++k)
          strides += iv + "->nb[" + std::to_string(k) + "], ";
        std::string nelist;
        for (int64_t k = 0; k < R; ++k)
          nelist += std::to_string((long long)ne[k]) + ", ";
        std::snprintf(buf, sizeof(buf),
                      "    tensor %s = ggml_view_%lldd(m, %s, %s%s%lld*%s->nb[%lld]);\n",
                      var.c_str(), (long long)R, iv.c_str(), nelist.c_str(),
                      strides.c_str(), (long long)accum, iv.c_str(), (long long)gd);
        body += buf;
        tvar[ot->name] = var;
        last = var;
        // 분할 축(torch dim)의 누적 크기.
        int64_t tdim_pos = tdim < 0 ? tdim + R : tdim;
        accum += (tdim_pos >= 0 && tdim_pos < R) ? s[(size_t)tdim_pos] : 0;
      }
      continue;
    }

    std::string vp = var_prefix(n.op.type);
    std::string var = vp + std::to_string(++counter);
    std::string line;

    if (n.op.type == "conv2d" || n.op.type == "depthwise_conv2d") {
      int64_t st = config_int_elem(n, "stride", 0, 1);
      int64_t pd = config_int_elem(n, "padding", 0, 0);
      const char* fn = n.op.type == "depthwise_conv2d" ? "conv_2d_depthwise" : "conv_2d";
      std::snprintf(buf, sizeof(buf), "    tensor %s = %s(m[\"%s\"], %s, %lld, %lld);\n",
                    var.c_str(), fn, weight_key_of(n).c_str(), invar(n.in_tensors[0]).c_str(),
                    (long long)st, (long long)pd);
      line = buf;
    } else if (n.op.type == "relu") {
      std::snprintf(buf, sizeof(buf), "    tensor %s = ggml_relu(m, %s);\n",
                    var.c_str(), invar(n.in_tensors[0]).c_str());
      line = buf;
    } else if (n.op.type == "aten::silu_" || n.op.type == "silu") {
      std::snprintf(buf, sizeof(buf), "    tensor %s = ggml_silu_inplace(m, %s);\n",
                    var.c_str(), invar(n.in_tensors[0]).c_str());
      line = buf;
    } else if (n.op.type == "sigmoid") {
      std::snprintf(buf, sizeof(buf), "    tensor %s = ggml_sigmoid(m, %s);\n",
                    var.c_str(), invar(n.in_tensors[0]).c_str());
      line = buf;
    } else if (n.op.type == "elemwise_add" || n.op.type == "elemwise_mul" ||
               n.op.type == "elementwise_sub" || n.op.type == "elemwise_div") {
      const char* fn = n.op.type == "elemwise_mul" ? "ggml_mul"
                       : n.op.type == "elementwise_sub" ? "ggml_sub"
                       : n.op.type == "elemwise_div" ? "ggml_div" : "ggml_add";
      std::string b1 = n.in_tensors.size() > 1 ? invar(n.in_tensors[1]) : invar(n.in_tensors[0]);
      std::snprintf(buf, sizeof(buf), "    tensor %s = %s(m, %s, %s);\n", var.c_str(), fn,
                    invar(n.in_tensors[0]).c_str(), b1.c_str());
      line = buf;
    } else if (n.op.type == "concat") {
      // N-입력 concat → ggml_concat 체인. torch dim → ggml 축.
      int64_t rank = n.out_tensors[0]->shape_known ? (int64_t)n.out_tensors[0]->shape.size() : 4;
      int64_t gd = ggml_dim(config_int(n, "dim", 1), rank);
      std::string acc = invar(n.in_tensors[0]);
      for (size_t k = 1; k < n.in_tensors.size(); ++k) {
        std::snprintf(buf, sizeof(buf), "ggml_concat(m, %s, %s, %lld)", acc.c_str(),
                      invar(n.in_tensors[k]).c_str(), (long long)gd);
        acc = buf;
      }
      line = "    tensor " + var + " = " + acc + ";\n";
    } else if (n.op.type == "resize") {
      std::snprintf(buf, sizeof(buf),
                    "    tensor %s = ggml_upscale(m, %s, 2, GGML_SCALE_MODE_NEAREST);\n",
                    var.c_str(), invar(n.in_tensors[0]).c_str());
      line = buf;
    } else if (n.op.type == "reshape") {
      // ggml_reshape_Nd(m, in, ne...) — ne = out shape 역순(ggml).
      auto& s = n.out_tensors[0]->shape;
      int64_t r = (int64_t)s.size();
      std::string args;
      for (auto it = s.rbegin(); it != s.rend(); ++it)
        args += ", " + std::to_string(*it);
      if (r >= 1 && r <= 4)
        std::snprintf(buf, sizeof(buf), "    tensor %s = ggml_reshape_%lldd(m, ggml_cont(m, %s)%s);\n",
                      var.c_str(), (long long)r, invar(n.in_tensors[0]).c_str(), args.c_str());
      else
        // >4D: 물리버퍼를 ≤4D 로 유지(cont passthrough). 전이 5D 의 그 사이 permute 는
        // render_perm_general 이 ≤4D 화하고, 마지막 ≤4D reshape↓ 가 정합한다.
        std::snprintf(buf, sizeof(buf), "    tensor %s = ggml_cont(m, %s); /* reshape rank %lld >4D → ≤4D 유지 */\n",
                      var.c_str(), invar(n.in_tensors[0]).c_str(), (long long)r);
      line = buf;
    } else if (n.op.type == "transpose" || n.op.type == "permute") {
      // torch transpose(dim0,dim1) → ggml_permute 축 swap + cont.
      int64_t rank = n.in_tensors[0]->shape_known ? (int64_t)n.in_tensors[0]->shape.size() : 4;
      int p[4] = {0, 1, 2, 3};
      if (n.op.type == "transpose") {
        int64_t g0 = ggml_dim(config_int(n, "dim0", 0), rank);
        int64_t g1 = ggml_dim(config_int(n, "dim1", 1), rank);
        if (g0 >= 0 && g0 < 4 && g1 >= 0 && g1 < 4) std::swap(p[g0], p[g1]);
      }
      std::snprintf(buf, sizeof(buf),
                    "    tensor %s = ggml_cont(m, ggml_permute(m, %s, %d, %d, %d, %d));\n",
                    var.c_str(), invar(n.in_tensors[0]).c_str(), p[0], p[1], p[2], p[3]);
      line = buf;
    } else if (n.op.type == "matmul") {
      // torch A@B → ggml_mul_mat(cont(permute(B,1,0,2,3)), A). ggml ne 는 torch 역순이라
      // 공유차원(K)을 ne0 로 맞추려면 B 를 permute. C++ IR 은 transpose 노드를 별도 렌더
      // (q^T/attn^T)하므로 in[0]=A, in[1]=B 에 이 규칙을 적용 → Python head_render 와 일치.
      std::string A = invar(n.in_tensors[0]);
      std::string B = n.in_tensors.size() > 1 ? invar(n.in_tensors[1]) : A;
      std::snprintf(buf, sizeof(buf),
                    "    tensor %s = ggml_mul_mat(m, ggml_cont(m, ggml_permute(m, %s, 1, 0, 2, 3)), %s);\n",
                    var.c_str(), B.c_str(), A.c_str());
      line = buf;
    } else if (n.op.type == "softmax") {
      // ggml_soft_max 는 축0 전용. torch dim → ggml 축 gd. gd==0 이면 그대로,
      // 아니면 gd↔0 permute 후 soft_max, 다시 permute(DFL: dim=1→ggml축2).
      int64_t rank = n.in_tensors[0]->shape_known ? (int64_t)n.in_tensors[0]->shape.size() : 4;
      int64_t gd = ggml_dim(config_int(n, "dim", -1), rank);
      std::string iv = invar(n.in_tensors[0]);
      if (gd <= 0 || gd > 3) {
        std::snprintf(buf, sizeof(buf), "    tensor %s = ggml_soft_max(m, %s);\n",
                      var.c_str(), iv.c_str());
      } else {
        int p[4] = {0, 1, 2, 3};
        std::swap(p[0], p[gd]);
        std::snprintf(buf, sizeof(buf),
                      "    tensor %s = ggml_cont(m, ggml_permute(m, ggml_soft_max(m, "
                      "ggml_cont(m, ggml_permute(m, %s, %d, %d, %d, %d))), %d, %d, %d, %d));\n",
                      var.c_str(), iv.c_str(), p[0], p[1], p[2], p[3], p[0], p[1], p[2], p[3]);
      }
      line = buf;
    } else if (n.op.type == "maxpool") {
      int64_t kh = config_int_elem(n, "kernel_size", 0, 1), kw = config_int_elem(n, "kernel_size", 1, 1);
      int64_t sh = config_int_elem(n, "stride", 0, 1), sw = config_int_elem(n, "stride", 1, 1);
      int64_t ph = config_int_elem(n, "padding", 0, 0), pw = config_int_elem(n, "padding", 1, 0);
      std::snprintf(buf, sizeof(buf),
                    "    tensor %s = ggml_pool_2d(m, %s, GGML_OP_POOL_MAX, %lld, %lld, %lld, %lld, %lld, %lld);\n",
                    var.c_str(), invar(n.in_tensors[0]).c_str(), (long long)kh, (long long)kw,
                    (long long)sh, (long long)sw, (long long)ph, (long long)pw);
      line = buf;
    } else if (n.op.type == "aten::mean" || n.op.type == "mean") {
      // mean([2,3]) (= [B,C,H,W] 공간평균 = global avg pool) → ggml AVG pool(ne0,ne1) → [1,1,C,B]
      // → keepdim 에 맞춰 reshape. (그 외 dim 은 best-effort ggml_mean(ne0)).
      std::string iv = invar(n.in_tensors[0]);
      std::vector<int64_t> dims;
      for (const auto& kv : n.op.configs)
        if (kv.first == "dim" && kv.second.type == json::Value::Arr)
          for (const auto& e : *kv.second.arr)
            if (e.type == json::Value::Int) dims.push_back(e.i);
      const auto& insh = n.in_tensors[0]->shape;
      bool spatial = dims.size() == 2 && insh.size() == 4 &&
                     ((dims[0] == 2 && dims[1] == 3) || (dims[0] == 3 && dims[1] == 2));
      if (spatial) {
        int64_t B = insh[0], C = insh[1], H = insh[2], W = insh[3];
        std::string g = "gap" + std::to_string(++counter);
        std::snprintf(buf, sizeof(buf),
                      "    tensor %s = ggml_pool_2d(m, %s, GGML_OP_POOL_AVG, %lld, %lld, %lld, %lld, 0, 0);\n",
                      g.c_str(), iv.c_str(), (long long)W, (long long)H, (long long)W, (long long)H);
        line = buf;
        bool keepdim = false;
        for (const auto& kv : n.op.configs)
          if (kv.first == "keepdim" && kv.second.type == json::Value::Bool) keepdim = kv.second.b;
        if (keepdim)
          std::snprintf(buf, sizeof(buf), "    tensor %s = ggml_cont(m, %s);\n", var.c_str(), g.c_str());
        else
          std::snprintf(buf, sizeof(buf),
                        "    tensor %s = ggml_reshape_2d(m, ggml_cont(m, %s), %lld, %lld);\n",
                        var.c_str(), g.c_str(), (long long)C, (long long)B);
        line += buf;
      } else {
        std::snprintf(buf, sizeof(buf), "    tensor %s = ggml_mean(m, %s); // TODO: mean dim\n",
                      var.c_str(), iv.c_str());
        line = buf;
      }
    } else if (n.op.type == "adaptive_avg_pool2d") {
      // 입력 (Hi,Wi) 가 출력 (Ho,Wo) 의 배수면 고정커널 avg pool 로 tiled pooling.
      // (1,1) global 은 특수해. 비-배수는 global 폴백.
      std::string iv = invar(n.in_tensors[0]);
      const auto& ish = n.in_tensors[0]->shape;
      const auto& osh = n.out_tensors[0]->shape;
      bool tiled = ish.size() >= 2 && osh.size() >= 2 &&
                   osh[osh.size() - 2] > 0 && osh[osh.size() - 1] > 0 &&
                   ish[ish.size() - 2] % osh[osh.size() - 2] == 0 &&
                   ish[ish.size() - 1] % osh[osh.size() - 1] == 0;
      if (tiled) {
        int64_t k0 = ish[ish.size() - 1] / osh[osh.size() - 1];
        int64_t k1 = ish[ish.size() - 2] / osh[osh.size() - 2];
        std::snprintf(buf, sizeof(buf),
                      "    tensor %s = ggml_pool_2d(m, %s, GGML_OP_POOL_AVG, %lld, %lld, %lld, %lld, 0, 0);\n",
                      var.c_str(), iv.c_str(), (long long)k0, (long long)k1, (long long)k0, (long long)k1);
      } else {
        std::snprintf(buf, sizeof(buf),
                      "    tensor %s = ggml_pool_2d(m, %s, GGML_OP_POOL_AVG, %s->ne[0], %s->ne[1], %s->ne[0], %s->ne[1], 0, 0);\n",
                      var.c_str(), iv.c_str(), iv.c_str(), iv.c_str(), iv.c_str(), iv.c_str());
      }
      line = buf;
    } else if (n.op.type == "flatten") {
      int64_t C = 1;
      if (n.out_tensors[0]->shape_known && n.out_tensors[0]->shape.size() >= 2)
        C = n.out_tensors[0]->shape.back();
      std::snprintf(buf, sizeof(buf),
                    "    tensor %s = ggml_reshape_2d(m, ggml_cont(m, %s), %lld, 1);\n",
                    var.c_str(), invar(n.in_tensors[0]).c_str(), (long long)C);
      line = buf;
    } else if (n.op.type == "dense") {
      std::snprintf(buf, sizeof(buf), "    tensor %s = linear(m[\"%s\"], %s);\n",
                    var.c_str(), weight_key_of(n).c_str(), invar(n.in_tensors[0]).c_str());
      line = buf;
    } else {
      // 미지원(헤드: reshape/transpose/matmul/softmax/strided_slice 등) — passthrough +
      // TODO 로 var 정의(데이터플로우 유지). codegen 헤드 렌더러는 향후 작업.
      std::string iv = n.in_tensors.empty() ? std::string("x") : invar(n.in_tensors[0]);
      std::snprintf(buf, sizeof(buf), "    tensor %s = %s; // TODO(head): %s\n",
                    var.c_str(), iv.c_str(), n.op.type.c_str());
      line = buf;
    }
    body += line;
    if (!n.out_tensors.empty()) tvar[n.out_tensors[0]->name] = var;
    last = var;
  }

  std::string out;
  // arch_id = 소문자 model (GGUF general.architecture 는 write_gguf 가 graph_name 그대로 기록).
  std::string arch_id = model;
  for (auto& c : arch_id) c = (char)std::tolower((unsigned char)c);

  out += "// GENERATED BY GTX Compiler (C++/libtorch port, ggml/vision.cpp backend), DO NOT EDIT!\n";
  out += "#include \"visp/arch/" + model + ".h\"\n#include \"visp/ml.h\"\n";
  out += "#include \"visp/nn.h\"\n#include \"visp/vision.h\"\n";
  out += "#include \"util/string.h\"\n\n#include <string_view>\n\nnamespace visp {\n\n";
  out += "tensor " + model + "_forward(model_ref m, tensor x, " + model + "_params const& p) {\n";
  if (is_rnn) {
    // RNN: 입력 [feat,seq,batch] 그대로 unroll, cwhn 변환 없음.
    out += "    (void)p;\n    // RNN: 시퀀스 입력 [feat,seq,batch] — cwhn 변환 생략\n\n";
    out += body;
    out += "\n    return compute_graph_output(m, " + last + ", \"result\");\n}\n\n";
  } else {
    out += "    (void)p;\n    x = cwhn_to_contiguous_2d(m, x);\n\n";
    out += body;
    out += "\n    x = contiguous_2d_to_cwhn(m, " + last + ");\n";
    out += "    return compute_graph_output(m, x, \"result\");\n}\n\n";
  }
  // detect_params: GGUF general.architecture 검증 (정적 unroll → 추가 하이퍼파라미터 없음).
  out += model + "_params " + model + "_detect_params(model_file const& f) {\n";
  out += "    " + model + "_params p{};\n";
  out += "    if (std::string_view arch = f.arch(); arch != \"" + arch_id + "\") {\n";
  out += "        throw except(\n";
  out += "            \"Architecture expected to be '" + arch_id + "', but was '{}' ({})\",\n";
  out += "            arch, f.path);\n";
  out += "    }\n    return p;\n}\n\n} // namespace visp\n";
  return out;
}

static std::string generate_arch_h(const std::string& model) {
  std::string o;
  o += "// GENERATED BY GTX Compiler (C++/libtorch port), DO NOT EDIT!\n#pragma once\n";
  o += "#include \"visp/ml.h\"\n\nnamespace visp {\n\n";
  o += "struct " + model + "_params {};\n\n";
  o += "tensor " + model + "_forward(model_ref m, tensor x, " + model + "_params const& p);\n";
  o += model + "_params " + model + "_detect_params(model_file const& f);\n";
  o += "\n} // namespace visp\n";
  return o;
}

void compile_model(const std::string& pt_path, const std::string& graph_name,
                   const std::vector<int64_t>& input_shape, const std::string& out_dir,
                   const std::string& quant) {
  std::unordered_map<std::string, at::Tensor> params;
  std::unordered_map<std::string, at::Tensor> constants;
  Graph g = do_parse(pt_path, graph_name, input_shape, params, &constants);
  std::unordered_map<std::string, std::string> alias;
  fold_conv_bn(g, params, alias);

  std::unordered_map<std::string, at::Tensor> baked;  // const.foldN → 텐서
  std::string cpp = generate_arch_cpp(g, graph_name, alias, constants, baked);
  std::string h = generate_arch_h(graph_name);
  { std::ofstream f(out_dir + "/" + graph_name + ".cpp"); f << cpp; }
  { std::ofstream f(out_dir + "/" + graph_name + ".h"); f << h; }
  // baked const.foldN 텐서를 params 와 함께 GGUF 에 수록(키=그대로, scope 없음).
  write_gguf_core(g, params, out_dir + "/" + graph_name + ".gguf", graph_name, &baked, quant);
  if (!baked.empty())
    std::fprintf(stderr, "[gtxc-compile] baked %zu const-fold tensor(s)\n", baked.size());
  std::fprintf(stderr, "[gtxc-compile] %s.{cpp,h,gguf} → %s/ (Conv-BN folded)\n",
               graph_name.c_str(), out_dir.c_str());
}


}  // namespace gtxc
