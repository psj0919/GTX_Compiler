// GTX Compiler Graph IR — shared/graph/ (Tensor/Operation/Node/Graph) 의 C++ 미러.
//
// parity 정책: 관찰 가능한 IR 표면(op type, params 이름·shape·dtype, configs/attrs 값,
// 연결성)을 Python 과 일치시킨다. Python 의 IrAttr typed-memory·use-def 추적은 최적화
// 패스 전용이라 파서 단계에서는 json::Value 기반으로 단순화(동작 동일, idiomatic).
#pragma once
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtxc/json.hpp"

namespace gtxc {

struct Node;

// np.ndarray 래퍼(shared/graph/base_tensor.py Tensor). op 출력 또는 파라미터.
struct Tensor {
  std::string name;
  std::vector<int64_t> shape;
  bool shape_known = false;
  std::string dtype;            // "float32" 등
  std::string producer;         // 생산 노드 이름("" = 파라미터/입력)

  json::Value desc() const {
    json::Object o;
    o.emplace_back("name", json::Value::str(name));
    o.emplace_back("shape", shape_known ? json::Value::int_list(shape) : json::Value::null());
    o.emplace_back("dtype", dtype.empty() ? json::Value::null() : json::Value::str(dtype));
    return json::Value::object(std::move(o));
  }
  // configs/attrs 안에 텐서가 들어갈 때의 표현({"__tensor__":..,"shape":..}).
  json::Value ref() const {
    json::Object o;
    o.emplace_back("__tensor__", json::Value::str(name));
    o.emplace_back("shape", shape_known ? json::Value::int_list(shape) : json::Value::null());
    return json::Value::object(std::move(o));
  }
};
using TensorPtr = std::shared_ptr<Tensor>;

// shared/graph/base_operator.py Operation.
struct Param {
  std::string key;                 // "weight","bias","gamma"... (Python ParamName.value)
  std::vector<TensorPtr> tensors;  // 단일이면 size 1, 리스트면 N
  bool is_list = false;
  Param() = default;
  Param(std::string k, std::vector<TensorPtr> t, bool l = false)
      : key(std::move(k)), tensors(std::move(t)), is_list(l) {}
};
struct Operation {
  std::string type;                // OP.* 문자열
  std::vector<Param> params;       // 순서 보존
  json::Object configs;            // torch-native attr (stride/padding/...)
  json::Object attrs;              // IR attr (kernel/pad/...)
};

// shared/graph/base_node.py Node.
struct Node {
  int64_t idx = 0;
  std::string name;
  std::string scope_name;
  std::string dtype;
  std::vector<TensorPtr> in_tensors;
  std::vector<TensorPtr> out_tensors;
  Operation op;
};
using NodePtr = std::shared_ptr<Node>;

// shared/graph/base_graph.py Graph.
struct Graph {
  std::string name;
  std::vector<NodePtr> nodes;
  std::unordered_map<std::string, TensorPtr> tensors;

  // in_nodes/out_nodes 는 텐서 producer/consumer 로부터 계산(Python 과 동일 의미).
  void dump_json(std::ostream& os, const std::vector<int64_t>& input_shape) const;
};

}  // namespace gtxc
