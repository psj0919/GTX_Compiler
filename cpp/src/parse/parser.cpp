// parse/ TorchParser + op_dispatcher 의 C++ 포트 (resnet18 op 셋).
//
// 흐름: load → freeze_module(BN 보존) → optimize_graph_19 → 그래프 순회.
//   prim::Constant(Tensor)  → 파라미터 텐서(consumer op 의 param). ndim4 weight 는
//                             OIHW→OHWI(Python _load_data convert_parameter_tensor_format).
//   prim::Constant(scalar)  → config 인자(지연 해석).
//   aten::*                 → op_dispatcher 핸들러 → Operation + 출력 텐서(+shape 추론).
//
// shape: 저장된 trace 는 shape 가 소실되므로 per-op 로 추론(Python 트레이서와 동일 결과).
#include "gtxc/parser.hpp"

#include <torch/script.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/freeze_module.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

// namespace gtxc 내부에서 Node/Value 는 gtxc::Node 로 가려지므로 jit 타입은 별칭 사용.
using JNode = torch::jit::Node;
using JValue = torch::jit::Value;
namespace J = gtxc::json;

namespace gtxc {
namespace {

void optimize_graph_19(std::shared_ptr<torch::jit::Graph>& g) {
  torch::jit::Inline(*g);
  torch::jit::LowerAllTuples(g);
  // ConstantPropagation 은 생략: freeze 로 상수화된 파라미터를 Tensor[] 로 묶는
  // ListConstruct(예: lstm params)를 익명 TensorList 상수로 fold 해 weight_ih_l0 등
  // state_dict 이름을 잃는다. Python 은 저수준 트레이서(params=named inputs)라 fold 안 됨.
  // 이 패스 없이도 resnet 포함 parity 유지(상수 인자는 trace 단계에서 이미 Constant).
  torch::jit::EliminateDeadCode(g);
  torch::jit::PeepholeOptimize(g, /*addmm_fusion_enabled=*/true);
  torch::jit::FuseLinear(g);
  g = torch::jit::Canonicalize(g);
}

std::string scalar_type_str(at::ScalarType st) {
  switch (st) {
    case at::kFloat: return "float32";
    case at::kDouble: return "float64";
    case at::kHalf: return "float16";
    case at::kLong: return "int64";
    case at::kInt: return "int32";
    case at::kShort: return "int16";
    case at::kChar: return "int8";
    case at::kByte: return "uint8";
    case at::kBool: return "bool";
    default: return "float32";
  }
}

// 그래프 입력에서 state_dict 키 복원: "self.conv1.weight" → "conv1.weight".
std::string strip_self(const std::string& dbg) {
  if (dbg.rfind("self.", 0) == 0) return dbg.substr(5);
  return dbg;
}

// op 입력 해석 결과(텐서 또는 상수 스칼라/리스트).
struct Arg {
  enum K { TensorK, IntK, IntListK, DoubleK, BoolK, NoneK, StrK } k = NoneK;
  TensorPtr t;
  int64_t i = 0;
  std::vector<int64_t> il;
  double d = 0;
  bool b = false;
  std::string s;
};

class Parser {
 public:
  Parser(std::string graph_name, std::vector<int64_t> input_shape,
         std::unordered_map<std::string, at::Tensor> params)
      : input_shape_(std::move(input_shape)), param_map_(std::move(params)) {
    g_.name = std::move(graph_name);
    scope_ = g_.name + "::";
  }

  Graph run(std::shared_ptr<torch::jit::Graph> jit) {
    // 입력 노드(데이터 입력 = TensorType 인 그래프 입력)
    JValue* data_in = nullptr;
    for (JValue* in : jit->inputs()) {
      if (in->type()->kind() == c10::TypeKind::TensorType) { data_in = in; break; }
    }
    if (!data_in) throw std::runtime_error("no tensor input in graph");
    emit_input(data_in);

    // return 노드는 Python 과 동일하게 두 번째(idx=1)로 자리 예약, 본문 뒤 채움
    auto ret = std::make_shared<Node>();
    ret->idx = idx_++;
    ret->name = scope_ + "return_0";
    ret->op.type = "return";
    g_.nodes.push_back(ret);

    for (JNode* n : jit->nodes()) {
      if (n->kind() == c10::prim::Constant) {
        register_constant(n);
        continue;
      }
      // 스칼라 정수 연산(NumToTensor/Int/floordiv/mul/...)은 val2int_ 로 추적 후 노드 생략
      // (Python remove_simple_op 대응 — reshape 의 동적 dim 해석에 필요).
      // prim::NumToTensor 도 추적하므로 prim 스킵보다 먼저 호출해야 한다.
      if (track_scalar_int(n)) continue;
      // prim::ListConstruct/TupleConstruct 등은 IR 노드가 아니라 소비 시점에 inline 처리
      if (std::string(n->kind().toQualString()).rfind("prim::", 0) == 0) continue;
      try {
        dispatch(n, n->kind().toUnqualString());
      } catch (const std::exception& e) {
        // 미지원/예외 op → 그래프 진행을 막지 않도록 placeholder 노드(견고성).
        emit_placeholder(n, e.what());
      }
    }

    // return 채우기: 그래프 출력 텐서들
    J::Array ref_list;
    for (JValue* out : jit->outputs()) {
      auto t = lookup(out);
      if (t) { ret->in_tensors.push_back(t); ref_list.push_back(t->ref()); }
    }
    ret->op.configs.emplace_back("input", J::Value::array(std::move(ref_list)));
    if (unparsed_)
      std::fprintf(stderr, "[gtxc-parse] %d op(s) emitted as UNPARSED placeholder\n", unparsed_);
    return std::move(g_);
  }

 private:
  Graph g_;
  std::string scope_;
  std::vector<int64_t> input_shape_;
  int64_t idx_ = 0;
  std::unordered_map<JValue*, TensorPtr> val2tensor_;
  std::unordered_map<JValue*, int64_t> val2int_;  // aten::size 등 정수 산출값 추적
  std::unordered_map<std::string, at::Tensor> param_map_;  // full name → 파라미터 텐서
 public:
  std::unordered_map<std::string, at::Tensor> constants_;  // const tensor name → 데이터(baking)
 private:

  // prim::GetAttr 체인 → state_dict full name (예: weight / conv1.weight).
  static std::string getattr_name(JNode* gn) {
    std::string nm = gn->s(c10::attr::name);
    JValue* obj = gn->input(0);
    if (obj->node()->kind() == c10::prim::GetAttr)
      return getattr_name(obj->node()) + "." + nm;
    return nm;  // obj == %self
  }

  TensorPtr lookup(JValue* v) {
    auto it = val2tensor_.find(v);
    if (it != val2tensor_.end()) return it->second;
    // prim::GetAttr → 파라미터 텐서 지연 생성(이름은 모듈에서, ndim4 weight 는 OHWI).
    if (v->node()->kind() == c10::prim::GetAttr) {
      std::string full = getattr_name(v->node());
      auto pit = param_map_.find(full);
      if (pit == param_map_.end()) return nullptr;
      at::Tensor t = pit->second;
      std::vector<int64_t> shape(t.sizes().begin(), t.sizes().end());
      if (shape.size() == 4) shape = {shape[0], shape[2], shape[3], shape[1]};  // OIHW→OHWI
      auto tp = make_tensor(scope_ + full, shape, true, scalar_type_str(t.scalar_type()), "");
      val2tensor_[v] = tp;
      return tp;
    }
    return nullptr;
  }

  TensorPtr make_tensor(const std::string& name, std::vector<int64_t> shape,
                        bool known, std::string dtype, std::string producer) {
    auto t = std::make_shared<Tensor>();
    t->name = name;
    t->shape = std::move(shape);
    t->shape_known = known;
    t->dtype = std::move(dtype);
    t->producer = std::move(producer);
    g_.tensors[name] = t;
    return t;
  }

  void emit_input(JValue* v) {
    auto node = std::make_shared<Node>();
    node->idx = idx_++;
    node->name = scope_ + "input_0";
    node->dtype = "float32";
    node->op.type = "input";
    auto t = make_tensor(scope_ + v->debugName(), input_shape_, true, "float32", node->name);
    node->out_tensors.push_back(t);
    val2tensor_[v] = t;
    node->op.configs.emplace_back("input", J::Value::str("args[0]"));
    node->op.attrs.emplace_back("input", J::Value::str("args[0]"));
    g_.nodes.push_back(node);
  }

  // 파라미터 상수 등록(텐서). ndim4 weight 는 OHWI 로 shape 변환.
  void register_constant(JNode* n) {
    auto iv = torch::jit::toIValue(n->output());
    if (!iv || !iv->isTensor()) return;  // 스칼라/리스트/None 은 지연 해석
    at::Tensor t = iv->toTensor();
    std::vector<int64_t> shape(t.sizes().begin(), t.sizes().end());
    if (shape.size() == 4) {  // OIHW → OHWI (conv weight)
      shape = {shape[0], shape[2], shape[3], shape[1]};
    }
    std::string name = scope_ + strip_self(n->output()->debugName());
    auto tp = make_tensor(name, shape, true, scalar_type_str(t.scalar_type()), "");
    val2tensor_[n->output()] = tp;
    // const-fold baking 대상: trace 가 head anchor/stride/scalar 를 단일 상수 텐서로
    // 접었으므로(Python const_fold.py 와 동등) 그 데이터를 캡처한다. conv weight(ndim4)
    // 는 param 경로로 별도 바인딩되므로 제외(중복/대용량 회피).
    if (t.dim() != 4)
      constants_[name] = t.detach().to(at::kFloat).contiguous().clone();
  }

  // 스칼라 정수 산출 op 추적(both 입력이 정수일 때만). 추적 시 true → 노드 생략.
  bool track_scalar_int(JNode* n) {
    const std::string k = n->kind().toUnqualString();
    auto geti = [&](int idx, bool& ok) -> int64_t {
      if (idx >= (int)n->inputs().size()) { ok = false; return 0; }
      JValue* v = n->input(idx);
      auto it = val2int_.find(v);
      if (it != val2int_.end()) { ok = true; return it->second; }
      auto iv = torch::jit::toIValue(v);
      if (iv && iv->isInt()) { ok = true; return iv->toInt(); }
      ok = false; return 0;
    };
    if (n->outputs().size() != 1) return false;
    if (k == "Int" || k == "ScalarImplicit" || k == "NumToTensor" || k == "IntImplicit") {
      bool ok; int64_t a = geti(0, ok);
      if (ok) { val2int_[n->output(0)] = a; return true; }
    } else if (k == "mul" || k == "add" || k == "sub" || k == "floordiv" ||
               k == "floor_divide" || k == "div") {
      bool oa, ob; int64_t a = geti(0, oa), b = geti(1, ob);
      if (oa && ob) {
        int64_t r = k == "mul" ? a * b : k == "add" ? a + b : k == "sub" ? a - b
                    : (b != 0 ? a / b : 0);
        val2int_[n->output(0)] = r;
        return true;
      }
    }
    return false;
  }

  Arg resolve(JValue* v) {
    Arg a;
    if (auto t = lookup(v)) { a.k = Arg::TensorK; a.t = t; return a; }
    auto iv = torch::jit::toIValue(v);
    if (!iv) { a.k = Arg::NoneK; return a; }
    if (iv->isNone()) a.k = Arg::NoneK;
    else if (iv->isInt()) { a.k = Arg::IntK; a.i = iv->toInt(); }
    else if (iv->isBool()) { a.k = Arg::BoolK; a.b = iv->toBool(); }
    else if (iv->isDouble()) { a.k = Arg::DoubleK; a.d = iv->toDouble(); }
    else if (iv->isIntList()) {
      a.k = Arg::IntListK;
      for (auto e : iv->toIntList()) a.il.push_back(e);
    } else if (iv->isString()) { a.k = Arg::StrK; a.s = iv->toStringRef(); }
    else a.k = Arg::NoneK;
    return a;
  }

  static std::vector<int64_t> rev(std::vector<int64_t> v) {
    return {v.rbegin(), v.rend()};
  }

  // 노드 골격 생성(idx/name/scope/op.type), out 텐서 1개 추가.
  std::shared_ptr<Node> begin_node(JNode* jn, const std::string& op_type,
                                   const std::vector<int64_t>& out_shape,
                                   bool shape_known = true) {
    auto node = std::make_shared<Node>();
    node->idx = idx_++;
    JValue* out0 = jn->output(0);
    node->name = scope_ + out0->debugName();
    node->scope_name = jn->scopeName();
    node->dtype = "float32";
    node->op.type = op_type;
    auto t = make_tensor(scope_ + out0->debugName(), out_shape, shape_known, "float32",
                         node->name);
    node->out_tensors.push_back(t);
    val2tensor_[out0] = t;
    return node;
  }

  void dispatch(JNode* n, const std::string& kind) {
    if (kind == std::string("_convolution")) return conv(n);
    if (kind == "batch_norm") return batch_norm(n);
    if (kind == "relu" || kind == "relu_") return relu(n, kind == "relu_");
    if (kind == "max_pool2d") return max_pool2d(n);
    if (kind == "add" || kind == "add_") return add(n);
    if (kind == "adaptive_avg_pool2d") return adaptive_avg_pool2d(n);
    if (kind == "flatten") return flatten(n);
    if (kind == "linear") return linear(n);
    if (kind == "size") return size_op(n);
    if (kind == "zeros") return zeros_op(n);
    if (kind == "slice") return slice_op(n);
    if (kind == "select") return select_op(n);
    return default_op(n);  // op_dispatcher auto_infer_op (schema 기반) 포트
  }

  // aten::slice(self, dim, start, end, step) → op_type "strided_slice".
  // Python OptPass slice_to_strided_slice 대응(begin/end/strides + dim shape).
  void slice_op(JNode* n) {
    auto in = resolve(n->input(0));
    if (!in.t || in.t->shape.empty())
      throw std::runtime_error("slice: shapeless input");
    int64_t rank = (int64_t)in.t->shape.size();
    int64_t dim = in_at(n, 1).i;
    if (dim < 0) dim += rank;
    int64_t sz = (dim >= 0 && dim < rank) ? in.t->shape[dim] : 0;
    auto sa = in_at(n, 2), ea = in_at(n, 3), st = in_at(n, 4);
    int64_t start = sa.k == Arg::IntK ? sa.i : 0;
    int64_t end = ea.k == Arg::IntK ? ea.i : sz;
    int64_t step = st.k == Arg::IntK ? st.i : 1;
    if (start < 0) start += sz;
    if (end < 0) end += sz;
    end = std::min(end, sz);
    start = std::max<int64_t>(0, start);
    if (step <= 0) step = 1;
    int64_t out_sz = (end - start + step - 1) / step;
    out_sz = std::max<int64_t>(0, out_sz);
    auto out = in.t->shape;
    if (dim >= 0 && dim < rank) out[dim] = out_sz;

    auto node = begin_node(n, "strided_slice", out);
    node->in_tensors = {in.t};
    auto& c = node->op.configs;
    c.emplace_back("begin", J::Value::int_list({start}));
    c.emplace_back("end", J::Value::int_list({end}));
    c.emplace_back("strides", J::Value::int_list({step}));
    c.emplace_back("dim", J::Value::integer(dim));
    g_.nodes.push_back(node);
  }

  // aten::select(self, dim, index) → 해당 dim 제거한 strided_slice.
  void select_op(JNode* n) {
    auto in = resolve(n->input(0));
    if (!in.t || in.t->shape.empty())
      throw std::runtime_error("select: shapeless input");
    int64_t rank = (int64_t)in.t->shape.size();
    int64_t dim = in_at(n, 1).i;
    int64_t index = in_at(n, 2).i;
    if (dim < 0) dim += rank;
    auto out = in.t->shape;
    if (dim >= 0 && dim < rank) out.erase(out.begin() + dim);
    auto node = begin_node(n, "strided_slice", out);
    node->in_tensors = {in.t};
    auto& c = node->op.configs;
    c.emplace_back("dim", J::Value::integer(dim));
    c.emplace_back("index", J::Value::integer(index));
    g_.nodes.push_back(node);
  }

  // ── 제네릭 default (auto_infer_op): schema 인자 → config, free param 바인딩 ──

  // state_dict 키 → formal name: "::" 뒤 부분에서 '.'→'_' (Python get_formal_name).
  static std::string formal_name(const std::string& tname) {
    auto pos = tname.rfind("::");
    std::string s = pos == std::string::npos ? tname : tname.substr(pos + 2);
    for (auto& c : s) if (c == '.') c = '_';
    return s;
  }

  static std::string scalar_type_pytorch(int64_t st) {
    switch (st) {
      case 0: return "torch.uint8";
      case 1: return "torch.int8";
      case 2: return "torch.int16";
      case 3: return "torch.int32";
      case 4: return "torch.int64";
      case 5: return "torch.float16";
      case 6: return "torch.float";
      case 7: return "torch.double";
      case 11: return "torch.bool";
      default: return "torch.float";
    }
  }

  // 리스트 원소 → json (텐서는 ref, 스칼라는 값).
  J::Value elem_json(JValue* e) {
    if (auto t = lookup(e)) return t->ref();
    auto iv = torch::jit::toIValue(e);
    if (!iv || iv->isNone()) return J::Value::null();
    if (iv->isBool()) return J::Value::boolean(iv->toBool());
    if (iv->isInt()) return J::Value::integer(iv->toInt());
    if (iv->isDouble()) return J::Value::real(iv->toDouble());
    if (iv->isString()) return J::Value::str(iv->toStringRef());
    return J::Value::null();
  }

  // schema 인자값 → config json (auto_infer_op 변환 규칙).
  J::Value config_json(JValue* v, const c10::Argument* arg) {
    if (v->node()->kind() == c10::prim::ListConstruct) {
      J::Array a;
      for (JValue* e : v->node()->inputs()) a.push_back(elem_json(e));
      return J::Value::array(std::move(a));
    }
    if (auto t = lookup(v)) return t->ref();
    auto iv = torch::jit::toIValue(v);
    if (!iv || iv->isNone()) return J::Value::null();
    const std::string an = arg ? arg->name() : "";
    if (an == "dtype" && iv->isInt()) return J::Value::str(scalar_type_pytorch(iv->toInt()));
    if (iv->isBool()) return J::Value::boolean(iv->toBool());
    if (iv->isInt()) return J::Value::integer(iv->toInt());
    if (iv->isDouble()) return J::Value::real(iv->toDouble());
    if (iv->isIntList()) {
      J::Array a;
      for (auto e : iv->toIntList()) a.push_back(J::Value::integer(e));
      return J::Value::array(std::move(a));
    }
    if (iv->isDevice())
      return J::Value::str("torch.device('" + iv->toDevice().str() + "')");
    if (iv->isString()) {
      bool is_dev = arg && arg->type()->kind() == c10::TypeKind::DeviceObjType;
      std::string s = iv->toStringRef();
      return J::Value::str(is_dev ? "torch.device('" + s + "')" : "'" + s + "'");
    }
    return J::Value::null();
  }

  // in_tensors 수집(ListConstruct 평탄화) + free param 바인딩.
  void collect_inputs(JNode* n, std::shared_ptr<Node>& node) {
    auto add = [&](JValue* v) {
      if (auto t = lookup(v)) {
        node->in_tensors.push_back(t);
        if (t->producer.empty())  // 파라미터 → op.param 바인딩(formal name)
          node->op.params.push_back(Param{formal_name(t->name), {t}, false});
      }
    };
    for (JValue* v : n->inputs()) {
      if (v->node()->kind() == c10::prim::ListConstruct)
        for (JValue* e : v->node()->inputs()) add(e);
      else
        add(v);
    }
  }

  int unparsed_ = 0;
  // 미지원/예외 op → 그래프 연결 유지용 placeholder (op_type="UNPARSED:..").
  void emit_placeholder(JNode* n, const std::string& /*err*/) {
    ++unparsed_;
    auto node = std::make_shared<Node>();
    node->idx = idx_++;
    node->name = scope_ + (n->outputs().size() ? n->output(0)->debugName() : "unparsed");
    node->scope_name = n->scopeName();
    node->op.type = std::string("UNPARSED:") + n->kind().toQualString();
    try { collect_inputs(n, node); } catch (...) {}
    for (JValue* o : n->outputs()) {
      auto t = make_tensor(scope_ + o->debugName(), {}, false, "float32", node->name);
      node->out_tensors.push_back(t);
      val2tensor_[o] = t;
    }
    g_.nodes.push_back(node);
  }

  // aten kind → 깔끔한 op_type(OP.*). 미정의면 빈 문자열(→ schema 이름 유지).
  static std::string clean_optype(const std::string& kind) {
    static const std::map<std::string, std::string> m = {
        {"view", "reshape"}, {"reshape", "reshape"}, {"transpose", "transpose"},
        {"permute", "permute"}, {"matmul", "matmul"}, {"cat", "concat"},
        {"sub", "elementwise_sub"}, {"sub_", "elementwise_sub"},
        {"mul", "elemwise_mul"}, {"mul_", "elemwise_mul"},
        {"div", "elemwise_div"}, {"div_", "elemwise_div"},
        {"softmax", "softmax"}, {"sigmoid", "sigmoid"}, {"sigmoid_", "sigmoid"},
        {"unsqueeze", "unsqueeze"}, {"stack", "stack"}, {"arange", "arange"},
        {"split_with_sizes", "split"}, {"chunk", "chunk"}, {"gather", "gather"},
        {"meshgrid", "meshgrid"}, {"full", "full"}, {"floor_divide", "floor_divide"},
        {"upsample_nearest2d", "resize"}, {"index", "index"},
    };
    auto it = m.find(kind);
    return it == m.end() ? "" : it->second;
  }

  void default_op(JNode* n) {
    auto node = std::make_shared<Node>();
    node->idx = idx_++;
    node->name = scope_ + n->output(0)->debugName();
    node->scope_name = n->scopeName();
    node->dtype = "float32";
    std::string cl = clean_optype(n->kind().toUnqualString());
    node->op.type = !cl.empty() ? cl
                    : n->maybeSchema() ? n->maybeSchema()->name()
                                       : std::string(n->kind().toQualString());

    if (auto* sch = n->maybeSchema()) {
      const auto& sargs = sch->arguments();
      for (size_t i = 0; i < sargs.size() && i < n->inputs().size(); ++i)
        node->op.configs.emplace_back(sargs[i].name(), config_json(n->input(i), &sargs[i]));
    }
    collect_inputs(n, node);
    // chunk/split 은 Tensor[] 1개를 내고 prim::ListUnpack 이 N개로 분해한다.
    // Python OptPass unpack_ListUnpack_op 처럼 unpack 결과를 producer 의 출력으로 병합.
    std::vector<JValue*> outs;
    if (n->outputs().size() == 1 &&
        n->output(0)->type()->kind() == c10::TypeKind::ListType &&
        n->output(0)->uses().size() == 1 &&
        n->output(0)->uses()[0].user->kind() == c10::prim::ListUnpack) {
      for (JValue* o : n->output(0)->uses()[0].user->outputs()) outs.push_back(o);
    } else {
      for (JValue* o : n->outputs()) outs.push_back(o);
    }
    for (JValue* o : outs) {
      auto t = make_tensor(scope_ + o->debugName(), {}, false, "float32", node->name);
      node->out_tensors.push_back(t);
      val2tensor_[o] = t;
    }
    propagate_same_shape(n, node);        // activation 류 same-shape 전파
    propagate_structural_shape(n, node);  // cat/chunk/split/upsample (백본 연결)
    set_rnn_shapes(n, node);
    g_.nodes.push_back(node);
  }

  // n->input(i) 안전 접근(개수 초과 시 default Arg).
  Arg in_at(JNode* n, size_t i) {
    if (i >= n->inputs().size()) return Arg{};
    return resolve(n->input(i));
  }

  // 구조적 op shape 추론(yolo 백본 C2f/C3k2: concat/chunk/split/upsample 연쇄).
  void propagate_structural_shape(JNode* n, std::shared_ptr<Node>& node) {
    const std::string k = n->kind().toUnqualString();
    auto& ins = node->in_tensors;
    auto valid_dim = [](int64_t& dim, int64_t rank) {
      if (dim < 0) dim += rank;
      return dim >= 0 && dim < rank;
    };
    auto setout = [](TensorPtr& t, std::vector<int64_t> s) {
      t->shape = std::move(s); t->shape_known = true;
    };
    if (ins.empty() || !ins[0]->shape_known || node->out_tensors.empty()) return;
    auto shp = ins[0]->shape;
    int64_t rank = (int64_t)shp.size();

    if (k == "cat" || k == "concat") {
      int64_t dim = in_at(n, 1).i;
      if (!valid_dim(dim, rank)) return;
      int64_t sum = 0;
      for (auto& t : ins) {
        if (!t->shape_known || (int64_t)t->shape.size() != rank) return;
        sum += t->shape[dim];
      }
      shp[dim] = sum;
      setout(node->out_tensors[0], shp);
    } else if (k == "chunk") {
      int64_t chunks = in_at(n, 1).i;
      int64_t dim = in_at(n, 2).i;
      if (chunks <= 0 || !valid_dim(dim, rank)) return;
      int64_t size = shp[dim], cs = (size + chunks - 1) / chunks;
      for (size_t i = 0; i < node->out_tensors.size(); ++i) {
        auto s = shp;
        int64_t rem = size - cs * (int64_t)i;
        s[dim] = std::max<int64_t>(0, std::min(cs, rem));
        setout(node->out_tensors[i], s);
      }
    } else if (k == "split_with_sizes" || k == "split") {
      auto sizes = in_at(n, 1).il;
      int64_t dim = in_at(n, 2).i;
      if (sizes.empty() || !valid_dim(dim, rank)) return;
      for (size_t i = 0; i < node->out_tensors.size() && i < sizes.size(); ++i) {
        auto s = shp;
        s[dim] = sizes[i];
        setout(node->out_tensors[i], s);
      }
    } else if (k == "upsample_nearest2d") {
      if (rank != 4) return;
      auto os = in_at(n, 1);
      if (os.k == Arg::IntListK && os.il.size() == 2) { shp[2] = os.il[0]; shp[3] = os.il[1]; }
      else { shp[2] *= 2; shp[3] *= 2; }  // yolo Upsample scale=2
      setout(node->out_tensors[0], shp);
    } else if (k == "view" || k == "reshape") {
      // input(1) = 목표 shape(ListConstruct, -1/size-op 정수 포함 가능)
      JValue* sv = n->input(1);
      std::vector<int64_t> tgt;
      bool ok = true;
      if (sv->node()->kind() == c10::prim::ListConstruct) {
        for (JValue* e : sv->node()->inputs()) {
          if (val2int_.count(e)) tgt.push_back(val2int_[e]);
          else {
            auto iv = torch::jit::toIValue(e);
            if (iv && iv->isInt()) tgt.push_back(iv->toInt());
            else {
              if (getenv("GTX_DBG"))
                std::fprintf(stderr, "[reshape-fail] %s elem %s <- %s\n",
                             n->output(0)->debugName().c_str(), e->debugName().c_str(),
                             e->node()->kind().toQualString());
              ok = false; break;
            }
          }
        }
      } else {
        auto a = resolve(sv);
        if (a.k == Arg::IntListK) tgt = a.il; else ok = false;
      }
      if (!ok) return;
      int64_t total = 1;
      for (auto d : shp) total *= d;
      int64_t prod = 1, neg = -1;
      for (size_t i = 0; i < tgt.size(); ++i) { if (tgt[i] == -1) neg = (int64_t)i; else prod *= tgt[i]; }
      if (neg >= 0 && prod > 0) tgt[neg] = total / prod;
      setout(node->out_tensors[0], tgt);
    } else if (k == "transpose") {
      int64_t d0 = in_at(n, 1).i, d1 = in_at(n, 2).i;
      if (!valid_dim(d0, rank) || !valid_dim(d1, rank)) return;
      auto s = shp; std::swap(s[d0], s[d1]); setout(node->out_tensors[0], s);
    } else if (k == "permute") {
      auto perm = in_at(n, 1).il;
      if ((int64_t)perm.size() != rank) return;
      std::vector<int64_t> s;
      for (auto p : perm) { if (!valid_dim(p, rank)) return; s.push_back(shp[p]); }
      setout(node->out_tensors[0], s);
    } else if (k == "unsqueeze") {
      int64_t d = in_at(n, 1).i;
      if (d < 0) d += rank + 1;
      if (d < 0 || d > rank) return;
      auto s = shp; s.insert(s.begin() + d, 1); setout(node->out_tensors[0], s);
    } else if (k == "matmul") {
      if (ins.size() < 2 || !ins[1]->shape_known || ins[1]->shape.empty()) return;
      auto s = shp; s.back() = ins[1]->shape.back(); setout(node->out_tensors[0], s);
    } else if (k == "sub" || k == "sub_" || k == "mul" || k == "mul_" ||
               k == "div" || k == "div_" || k == "rsub") {
      setout(node->out_tensors[0], shp);  // elementwise broadcast → 첫 입력 shape
    }
  }

  // shape 보존 단항 op(activation 등): 출력 shape = 첫 텐서 입력 shape.
  void propagate_same_shape(JNode* n, std::shared_ptr<Node>& node) {
    static const std::set<std::string> same = {
        "silu", "silu_", "sigmoid", "sigmoid_", "tanh", "tanh_", "gelu",
        "hardswish", "hardswish_", "hardsigmoid", "mish", "elu", "elu_",
        "leaky_relu", "leaky_relu_", "relu6", "clamp", "clamp_", "neg",
        "exp", "log", "sqrt", "abs", "softmax", "log_softmax", "dropout"};
    const std::string k = n->kind().toUnqualString();
    if (!same.count(k)) return;
    if (node->in_tensors.empty() || !node->in_tensors[0]->shape_known) return;
    if (node->out_tensors.size() == 1) {
      node->out_tensors[0]->shape = node->in_tensors[0]->shape;
      node->out_tensors[0]->shape_known = true;
    }
  }

  // lstm/gru 출력 shape: y=[B,T,H*dir], h_n/c_n=[L*dir,B,H]. h0=in_tensors[1].
  void set_rnn_shapes(JNode* n, std::shared_ptr<Node>& node) {
    const std::string k = n->kind().toUnqualString();
    if (k != "lstm" && k != "gru") return;
    if (node->in_tensors.size() < 2 || !node->in_tensors[0]->shape_known) return;
    auto in = node->in_tensors[0]->shape;   // [B,T,in] (batch_first)
    auto h0 = node->in_tensors[1]->shape;   // [L*dir, B, H]
    if (h0.size() != 3) return;
    int64_t B = in[0], T = in[1], Ldir = h0[0], H = h0[2];
    bool bidir = false;
    for (auto& kv : node->op.configs)
      if (kv.first == "bidirectional" && kv.second.type == J::Value::Bool) bidir = kv.second.b;
    int64_t dir = bidir ? 2 : 1;
    auto set = [](TensorPtr& t, std::vector<int64_t> s) { t->shape = std::move(s); t->shape_known = true; };
    set(node->out_tensors[0], {B, T, H * dir});                 // y
    if (node->out_tensors.size() > 1) set(node->out_tensors[1], {Ldir, B, H});  // h_n
    if (node->out_tensors.size() > 2) set(node->out_tensors[2], {Ldir, B, H});  // c_n
  }

  // aten::size(input, dim) → op_type "shape". 정수값 추적(zeros size 해석용).
  void size_op(JNode* n) {
    auto in = resolve(n->input(0));
    int64_t dim = resolve(n->input(1)).i;
    auto node = std::make_shared<Node>();
    node->idx = idx_++;
    node->name = scope_ + n->output(0)->debugName();
    node->scope_name = n->scopeName();
    node->op.type = "shape";
    if (in.t) node->in_tensors = {in.t};
    node->op.configs.emplace_back("dim", J::Value::integer(dim));
    node->op.attrs.emplace_back("axis", J::Value::integer(dim));
    auto t = make_tensor(scope_ + n->output(0)->debugName(), {}, false, "", node->name);
    node->out_tensors.push_back(t);
    val2tensor_[n->output(0)] = t;
    int64_t val = (in.t && in.t->shape_known && dim >= 0 && dim < (int64_t)in.t->shape.size())
                      ? in.t->shape[dim] : 0;
    val2int_[n->output(0)] = val;
    g_.nodes.push_back(node);
  }

  // aten::zeros(size, dtype, layout, device, pin_memory) → op_type "zeros".
  void zeros_op(JNode* n) {
    JValue* size_v = n->input(0);
    J::Array size_cfg;
    std::vector<int64_t> out_shape;
    auto node = std::make_shared<Node>();
    auto push = [&](JValue* e) {
      if (auto t = lookup(e)) {
        size_cfg.push_back(t->ref());
        out_shape.push_back(val2int_.count(e) ? val2int_[e] : 0);
        node->in_tensors.push_back(t);  // size 의 텐서 원소(aten::size 출력)는 in_tensor
      } else {
        auto iv = torch::jit::toIValue(e);
        int64_t x = iv && iv->isInt() ? iv->toInt() : 0;
        size_cfg.push_back(J::Value::integer(x));
        out_shape.push_back(x);
      }
    };
    if (size_v->node()->kind() == c10::prim::ListConstruct)
      for (JValue* e : size_v->node()->inputs()) push(e);

    node->idx = idx_++;
    node->name = scope_ + n->output(0)->debugName();
    node->scope_name = n->scopeName();
    node->op.type = "zeros";
    auto dt = resolve(n->input(1));
    std::string dtype_s = dt.k == Arg::IntK ? scalar_type_pytorch(dt.i) : "torch.float";
    std::string dev = "cpu";
    auto dv = torch::jit::toIValue(n->input(3));
    if (dv && dv->isDevice()) dev = dv->toDevice().str();
    else if (dv && dv->isString()) dev = dv->toStringRef();
    node->op.configs.emplace_back("size", J::Value::array(size_cfg));
    node->op.configs.emplace_back("dtype", J::Value::str(dtype_s));
    node->op.configs.emplace_back("device", J::Value::str("'" + dev + "'"));
    node->op.attrs.emplace_back("shape", J::Value::array(size_cfg));
    auto t = make_tensor(scope_ + n->output(0)->debugName(), out_shape, true, "float32",
                         node->name);
    node->out_tensors.push_back(t);
    val2tensor_[n->output(0)] = t;
    g_.nodes.push_back(node);
  }

  // ── op_dispatcher 핸들러 포트 ───────────────────────────────────

  void conv(JNode* n) {
    auto in = resolve(n->input(0));
    auto w = resolve(n->input(1));
    auto bias = resolve(n->input(2));
    auto stride = resolve(n->input(3)).il;
    auto padding = resolve(n->input(4)).il;
    auto dilation = resolve(n->input(5)).il;
    auto groups = resolve(n->input(8)).i;
    if (!in.t || !w.t || in.t->shape.size() < 4 || w.t->shape.size() < 4)
      throw std::runtime_error("conv: unresolved/shapeless input");
    // weight (OHWI): OC=s0, KH=s1, KW=s2, IC=s3
    auto ws = w.t->shape;
    int64_t OC = ws[0], KH = ws[1], KW = ws[2], IC = ws[3];
    int64_t in_ch = IC * groups, out_ch = OC;
    int64_t ph = padding[0], pw = padding[1];
    // depthwise: 그룹당 입력채널 1 (IC==1, groups>1) → op_dispatcher 와 동일하게 분리.
    bool depthwise = groups > 1 && IC == 1;

    auto in_shape = in.t->shape;
    int64_t H = in_shape[2], W = in_shape[3];
    int64_t Ho = (H + 2 * ph - dilation[0] * (KH - 1) - 1) / stride[0] + 1;
    int64_t Wo = (W + 2 * pw - dilation[1] * (KW - 1) - 1) / stride[1] + 1;

    auto node = begin_node(n, depthwise ? "depthwise_conv2d" : "conv2d",
                           {in_shape[0], out_ch, Ho, Wo});
    node->in_tensors = {in.t, w.t};
    node->op.params.push_back(Param{"weight", {w.t}, false});
    bool has_bias = bias.k != Arg::NoneK;
    if (has_bias) { node->in_tensors.push_back(bias.t); node->op.params.push_back(Param{"bias", {bias.t}, false}); }

    auto& c = node->op.configs;
    c.emplace_back("in_channels", J::Value::integer(in_ch));
    c.emplace_back("out_channels", J::Value::integer(out_ch));
    c.emplace_back("bias", J::Value::boolean(has_bias));
    c.emplace_back("dilation", J::Value::int_list(dilation));
    c.emplace_back("kernel_size", J::Value::int_list({KH, KW}));
    c.emplace_back("stride", J::Value::int_list(stride));
    c.emplace_back("groups", J::Value::integer(groups));
    c.emplace_back("padding", J::Value::int_list(padding));

    auto& a = node->op.attrs;
    a.emplace_back("kernel", J::Value::int_list(rev({KH, KW})));
    a.emplace_back("stride", J::Value::int_list(rev(stride)));
    a.emplace_back("dilation", J::Value::int_list(rev(dilation)));
    a.emplace_back("pad_mode", J::Value::integer(0));
    a.emplace_back("pad", J::Value::int_list({pw, pw, ph, ph}));
    a.emplace_back("group", J::Value::integer(groups));
    a.emplace_back("bias_term", J::Value::boolean(has_bias));
    a.emplace_back("in_dim", J::Value::integer(in_ch));
    a.emplace_back("out_dim", J::Value::integer(out_ch));
    g_.nodes.push_back(node);
  }

  void batch_norm(JNode* n) {
    auto in = resolve(n->input(0));
    auto weight = resolve(n->input(1));
    auto bias = resolve(n->input(2));
    auto mean = resolve(n->input(3));
    auto var = resolve(n->input(4));
    double momentum = resolve(n->input(6)).d;
    double eps = resolve(n->input(7)).d;
    int64_t nf = weight.t->shape[0];

    auto node = begin_node(n, "batch_norm", in.t->shape);
    node->in_tensors = {in.t, weight.t, bias.t, mean.t, var.t};
    node->op.params.push_back(Param{"weight", {weight.t}, false});
    node->op.params.push_back(Param{"bias", {bias.t}, false});
    node->op.params.push_back(Param{"mean", {mean.t}, false});
    node->op.params.push_back(Param{"var", {var.t}, false});

    auto& c = node->op.configs;
    c.emplace_back("num_features", J::Value::integer(nf));
    c.emplace_back("eps", J::Value::real(eps));
    c.emplace_back("momentum", J::Value::real(momentum));
    auto& a = node->op.attrs;
    a.emplace_back("epsilon", J::Value::real(eps));
    a.emplace_back("scale", J::Value::boolean(true));
    a.emplace_back("center", J::Value::boolean(true));
    a.emplace_back("out_dim", J::Value::integer(nf));
    a.emplace_back("axis", J::Value::null());
    g_.nodes.push_back(node);
  }

  void relu(JNode* n, bool inplace) {
    auto in = resolve(n->input(0));
    if (!in.t) throw std::runtime_error("relu: unresolved input");
    auto node = begin_node(n, "relu", in.t->shape);
    node->in_tensors = {in.t};
    node->op.configs.emplace_back("inplace", J::Value::boolean(inplace));
    g_.nodes.push_back(node);
  }

  void max_pool2d(JNode* n) {
    auto in = resolve(n->input(0));
    auto kernel = resolve(n->input(1)).il;
    auto stride = resolve(n->input(2)).il;
    auto padding = resolve(n->input(3)).il;
    auto dilation = resolve(n->input(4)).il;
    bool ceil_mode = resolve(n->input(5)).b;
    if (!in.t || in.t->shape.size() < 4)
      throw std::runtime_error("maxpool: unresolved/shapeless input");
    if (stride.empty()) stride = kernel;
    int64_t ph = padding[0], pw = padding[1];
    int64_t KH = kernel[0], KW = kernel[1];

    auto in_shape = in.t->shape;
    int64_t H = in_shape[2], W = in_shape[3];
    auto pool_dim = [&](int64_t sz, int64_t k, int64_t p, int64_t s, int64_t d) {
      double num = sz + 2.0 * p - d * (k - 1) - 1;
      double q = num / s + 1;
      return ceil_mode ? (int64_t)std::ceil(q) : (int64_t)std::floor(q);
    };
    int64_t Ho = pool_dim(H, KH, ph, stride[0], dilation[0]);
    int64_t Wo = pool_dim(W, KW, pw, stride[1], dilation[1]);

    auto node = begin_node(n, "maxpool", {in_shape[0], in_shape[1], Ho, Wo});
    node->in_tensors = {in.t};
    auto& c = node->op.configs;
    c.emplace_back("kernel_size", J::Value::int_list(kernel));
    c.emplace_back("stride", J::Value::int_list(stride));
    c.emplace_back("ceil_mode", J::Value::boolean(ceil_mode));
    c.emplace_back("padding", J::Value::int_list(padding));
    c.emplace_back("dilation", J::Value::int_list(dilation));
    auto& a = node->op.attrs;
    a.emplace_back("kernel", J::Value::int_list(rev(kernel)));
    a.emplace_back("stride", J::Value::int_list(rev(stride)));
    a.emplace_back("pad_mode", J::Value::integer((int64_t)ceil_mode));
    a.emplace_back("pad", J::Value::int_list({pw, pw, ph, ph}));
    a.emplace_back("global", J::Value::boolean(false));
    g_.nodes.push_back(node);
  }

  void add(JNode* n) {
    auto in = resolve(n->input(0));
    auto other = resolve(n->input(1));
    if (!in.t || !other.t) throw std::runtime_error("add: non-tensor operand");
    Arg alpha;
    if (n->inputs().size() > 2) alpha = resolve(n->input(2));

    auto node = begin_node(n, "elemwise_add", in.t->shape);
    node->in_tensors = {in.t, other.t};
    auto& c = node->op.configs;
    c.emplace_back("input", in.t->ref());
    c.emplace_back("other", other.t->ref());
    if (alpha.k == Arg::IntK) c.emplace_back("alpha", J::Value::integer(alpha.i));
    else if (alpha.k == Arg::DoubleK) c.emplace_back("alpha", J::Value::real(alpha.d));
    auto& a = node->op.attrs;
    a.emplace_back("input", in.t->ref());
    a.emplace_back("other", other.t->ref());
    g_.nodes.push_back(node);
  }

  void adaptive_avg_pool2d(JNode* n) {
    auto in = resolve(n->input(0));
    auto out_size = resolve(n->input(1)).il;
    auto s = in.t->shape;
    auto node = begin_node(n, "adaptive_avg_pool2d", {s[0], s[1], out_size[0], out_size[1]});
    node->in_tensors = {in.t};
    node->op.configs.emplace_back("output_size", J::Value::int_list(out_size));
    node->op.attrs.emplace_back("input", J::Value::null());
    g_.nodes.push_back(node);
  }

  void flatten(JNode* n) {
    auto in = resolve(n->input(0));
    int64_t start = resolve(n->input(1)).i;
    int64_t end = resolve(n->input(2)).i;
    auto s = in.t->shape;
    int64_t nd = (int64_t)s.size();
    int64_t st = start < 0 ? start + nd : start;
    int64_t en = end < 0 ? end + nd : end;
    std::vector<int64_t> out;
    int64_t merged = 1;
    for (int64_t k = 0; k < nd; ++k) {
      if (k < st || k > en) out.push_back(s[k]);
      else { merged *= s[k]; if (k == en) out.push_back(merged); }
    }
    auto node = begin_node(n, "flatten", out);
    node->in_tensors = {in.t};
    auto& c = node->op.configs;
    c.emplace_back("input", in.t->ref());
    c.emplace_back("start_dim", J::Value::integer(start));
    c.emplace_back("end_dim", J::Value::integer(end));
    auto& a = node->op.attrs;
    a.emplace_back("start_axis", J::Value::integer(start));
    a.emplace_back("end_axis", J::Value::integer(end));
    g_.nodes.push_back(node);
  }

  void linear(JNode* n) {
    auto in = resolve(n->input(0));
    auto w = resolve(n->input(1));
    auto bias = resolve(n->input(2));
    int64_t out_f = w.t->shape[0];
    int64_t in_f = w.t->shape[1];
    // 입력 shape 미상(예: 앞 reduce op 의 shape 추론 누락)일 때 segfault 방지:
    // 알려진 shape 면 마지막 축을 out_f 로, 아니면 [1, out_f] 로 best-effort.
    std::vector<int64_t> out;
    if (in.t && in.t->shape_known && !in.t->shape.empty()) {
      out.assign(in.t->shape.begin(), in.t->shape.end());
      out.back() = out_f;
    } else {
      out = {1, out_f};
    }

    auto node = begin_node(n, "dense", out);
    node->in_tensors = {in.t, w.t};
    node->op.params.push_back(Param{"weight", {w.t}, false});
    bool has_bias = bias.k != Arg::NoneK;
    if (has_bias) { node->in_tensors.push_back(bias.t); node->op.params.push_back(Param{"bias", {bias.t}, false}); }

    auto& c = node->op.configs;
    c.emplace_back("bias", J::Value::boolean(has_bias));
    c.emplace_back("out_features", J::Value::integer(out_f));
    c.emplace_back("in_features", J::Value::integer(in_f));
    auto& a = node->op.attrs;
    a.emplace_back("bias_term", J::Value::boolean(has_bias));
    a.emplace_back("in_dim", J::Value::integer(in_f));
    a.emplace_back("out_dim", J::Value::integer(out_f));
    g_.nodes.push_back(node);
  }
};

}  // namespace

// load→freeze(preserveParameters)→optimize→parse. params(name→tensor)도 함께 반환.
Graph do_parse(const std::string& pt_path, const std::string& graph_name,
                      const std::vector<int64_t>& input_shape,
                      std::unordered_map<std::string, at::Tensor>& params,
                      std::unordered_map<std::string, at::Tensor>* constants_out) {
  torch::jit::Module module = torch::jit::load(pt_path);
  module.eval();
  // preserveParameters=true: 파라미터를 상수로 fold 하지 않고 prim::GetAttr 로 유지 →
  // weight_ih_l0 / conv1.weight 등 state_dict 이름 보존. fold 하면 lstm 의 Tensor[] 가
  // 익명 상수로 묶여 이름을 잃는다.
  torch::jit::Module frozen =
      torch::jit::freeze_module(module, /*preservedAttrs=*/{},
                                /*freezeInterfaces=*/true, /*preserveParameters=*/true);
  auto graph = frozen.get_method("forward").graph();
  optimize_graph_19(graph);

  for (const auto& p : frozen.named_parameters(/*recurse=*/true)) params[p.name] = p.value;
  for (const auto& b : frozen.named_buffers(/*recurse=*/true)) params[b.name] = b.value;

  Parser parser(graph_name, input_shape, params);
  Graph g = parser.run(graph);
  if (constants_out) *constants_out = std::move(parser.constants_);
  return g;
}

Graph do_parse(const std::string& pt_path, const std::string& graph_name,
               const std::vector<int64_t>& input_shape,
               std::unordered_map<std::string, at::Tensor>& params) {
  return do_parse(pt_path, graph_name, input_shape, params, nullptr);
}

Graph parse_traced(const std::string& pt_path, const std::string& graph_name,
                   const std::vector<int64_t>& input_shape) {
  std::unordered_map<std::string, at::Tensor> params;
  return do_parse(pt_path, graph_name, input_shape, params);
}

}  // namespace gtxc
