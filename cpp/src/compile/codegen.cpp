// Conv-BN fold + ggml/vision.cpp arch C++ codegen (VispCodeGenerator 포트).
#include "gtxc/ir.hpp"
#include "gtxc/parser.hpp"
#include "parse/parse_internal.hpp"

#include <torch/script.h>

#include <algorithm>
#include <cstdio>
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

}  // namespace

// resnet op 체인 → vision.cpp arch 스타일 forward C++.
static std::string generate_arch_cpp(const Graph& g, const std::string& model,
                                     const std::unordered_map<std::string, std::string>& alias) {
  std::string body;
  std::unordered_map<std::string, std::string> tvar;  // 텐서명 → C++ var
  int counter = 0;
  std::string last;
  char buf[512];

  auto invar = [&](const std::shared_ptr<Tensor>& t) -> std::string {
    std::string nm = resolve_alias(alias, t->name);
    auto it = tvar.find(nm);
    return it != tvar.end() ? it->second : "x";  // 미정의(입력) → x
  };

  for (const auto& np : g.nodes) {
    const Node& n = *np;
    if (n.op.type == "input" || n.op.type == "return") continue;
    if (n.op.type == "batch_norm") continue;  // folded
    if (n.op.type == "shape") continue;       // 스칼라(reshape dim) — 텐서 op 아님

    // chunk/split: 다중 출력 → 각 조각을 ggml_view_4d 로(채널 오프셋 누적).
    if (n.op.type == "chunk" || n.op.type == "split") {
      std::string iv = invar(n.in_tensors[0]);
      int64_t rank = n.in_tensors[0]->shape_known ? (int64_t)n.in_tensors[0]->shape.size() : 4;
      int64_t tdim = config_int(n, "dim", 1);
      int64_t gd = ggml_dim(tdim, rank);
      int64_t accum = 0;
      for (const auto& ot : n.out_tensors) {
        std::string var = "sl" + std::to_string(++counter);
        auto& s = ot->shape;  // [N,C,H,W]
        int64_t ne0 = s.size() > 3 ? s[3] : 1, ne1 = s.size() > 2 ? s[2] : 1;
        int64_t ne2 = s.size() > 1 ? s[1] : 1, ne3 = s.size() > 0 ? s[0] : 1;
        std::snprintf(buf, sizeof(buf),
                      "    tensor %s = ggml_view_4d(m, %s, %lld, %lld, %lld, %lld, %s->nb[1], %s->nb[2], %s->nb[3], %lld*%s->nb[%lld]);\n",
                      var.c_str(), iv.c_str(), (long long)ne0, (long long)ne1, (long long)ne2,
                      (long long)ne3, iv.c_str(), iv.c_str(), iv.c_str(), (long long)accum,
                      iv.c_str(), (long long)gd);
        body += buf;
        tvar[ot->name] = var;
        last = var;
        accum += (s.size() > 1 ? s[1] : 0);  // 채널 누적(채널 split 가정)
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
      std::snprintf(buf, sizeof(buf), "    tensor %s = ggml_upscale(m, %s, 2);\n",
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
        std::snprintf(buf, sizeof(buf), "    tensor %s = %s; // TODO(head): reshape rank %lld\n",
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
      std::snprintf(buf, sizeof(buf),
                    "    tensor %s = ggml_mul_mat(m, %s, %s); // TODO(ggml): operand order\n",
                    var.c_str(), invar(n.in_tensors[0]).c_str(),
                    n.in_tensors.size() > 1 ? invar(n.in_tensors[1]).c_str()
                                            : invar(n.in_tensors[0]).c_str());
      line = buf;
    } else if (n.op.type == "softmax") {
      std::snprintf(buf, sizeof(buf), "    tensor %s = ggml_soft_max(m, %s);\n",
                    var.c_str(), invar(n.in_tensors[0]).c_str());
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
    } else if (n.op.type == "adaptive_avg_pool2d") {
      std::string iv = invar(n.in_tensors[0]);
      std::snprintf(buf, sizeof(buf),
                    "    tensor %s = ggml_pool_2d(m, %s, GGML_OP_POOL_AVG, %s->ne[0], %s->ne[1], %s->ne[0], %s->ne[1], 0, 0);\n",
                    var.c_str(), iv.c_str(), iv.c_str(), iv.c_str(), iv.c_str(), iv.c_str());
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
  out += "// GENERATED BY GTX Compiler (C++/libtorch port, ggml/vision.cpp backend), DO NOT EDIT!\n";
  out += "#include \"visp/arch/" + model + ".h\"\n#include \"visp/ml.h\"\n";
  out += "#include \"visp/nn.h\"\n#include \"visp/vision.h\"\n\nnamespace visp {\n\n";
  out += "tensor " + model + "_forward(model_ref m, tensor x, " + model + "_params const& p) {\n";
  out += "    (void)p;\n    x = cwhn_to_contiguous_2d(m, x);\n\n";
  out += body;
  out += "\n    x = contiguous_2d_to_cwhn(m, " + last + ");\n";
  out += "    return compute_graph_output(m, x, \"result\");\n}\n\n} // namespace visp\n";
  return out;
}

static std::string generate_arch_h(const std::string& model) {
  std::string o;
  o += "// GENERATED BY GTX Compiler (C++/libtorch port), DO NOT EDIT!\n#pragma once\n";
  o += "#include \"visp/ml.h\"\n\nnamespace visp {\n\n";
  o += "struct " + model + "_params {};\n\n";
  o += "tensor " + model + "_forward(model_ref m, tensor x, " + model + "_params const& p);\n";
  o += "\n} // namespace visp\n";
  return o;
}

void compile_model(const std::string& pt_path, const std::string& graph_name,
                   const std::vector<int64_t>& input_shape, const std::string& out_dir) {
  std::unordered_map<std::string, at::Tensor> params;
  Graph g = do_parse(pt_path, graph_name, input_shape, params);
  std::unordered_map<std::string, std::string> alias;
  fold_conv_bn(g, params, alias);

  std::string cpp = generate_arch_cpp(g, graph_name, alias);
  std::string h = generate_arch_h(graph_name);
  { std::ofstream f(out_dir + "/" + graph_name + ".cpp"); f << cpp; }
  { std::ofstream f(out_dir + "/" + graph_name + ".h"); f << h; }
  write_gguf_core(g, params, out_dir + "/" + graph_name + ".gguf", graph_name);
  std::fprintf(stderr, "[gtxc-compile] %s.{cpp,h,gguf} → %s/ (Conv-BN folded)\n",
               graph_name.c_str(), out_dir.c_str());
}


}  // namespace gtxc
