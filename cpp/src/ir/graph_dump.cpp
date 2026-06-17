// Graph::dump_json — Python cpp/tools/dump_ir.py 와 동일한 스키마로 IR 출력.
#include "gtxc/ir.hpp"

#include <unordered_map>

namespace gtxc {

void Graph::dump_json(std::ostream& os, const std::vector<int64_t>& input_shape) const {
  // consumer 맵: tensor name → 소비 노드 이름들(노드 topo 순, dedup 없음 — Python 과 동일).
  std::unordered_map<std::string, std::vector<std::string>> consumers;
  for (const auto& n : nodes) {
    for (const auto& it : n->in_tensors) {
      consumers[it->name].push_back(n->name);
    }
  }

  json::Array node_arr;
  for (const auto& n : nodes) {
    json::Object jn;
    jn.emplace_back("idx", json::Value::integer(n->idx));
    jn.emplace_back("name", json::Value::str(n->name));
    jn.emplace_back("scope_name", json::Value::str(n->scope_name));
    jn.emplace_back("op_type", json::Value::str(n->op.type));

    // in_nodes: 각 in_tensor 의 producer(파라미터/입력 제외), dedup 없음
    json::Array in_nodes;
    for (const auto& it : n->in_tensors) {
      if (!it->producer.empty()) in_nodes.push_back(json::Value::str(it->producer));
    }
    jn.emplace_back("in_nodes", json::Value::array(std::move(in_nodes)));

    // out_nodes: 각 out_tensor 의 consumer 들
    json::Array out_nodes;
    for (const auto& ot : n->out_tensors) {
      auto it = consumers.find(ot->name);
      if (it != consumers.end())
        for (const auto& c : it->second) out_nodes.push_back(json::Value::str(c));
    }
    jn.emplace_back("out_nodes", json::Value::array(std::move(out_nodes)));

    json::Array in_t, out_t;
    for (const auto& t : n->in_tensors) in_t.push_back(t->desc());
    for (const auto& t : n->out_tensors) out_t.push_back(t->desc());
    jn.emplace_back("in_tensors", json::Value::array(std::move(in_t)));
    jn.emplace_back("out_tensors", json::Value::array(std::move(out_t)));

    // params
    json::Object jp;
    for (const auto& p : n->op.params) {
      if (p.is_list) {
        json::Array a;
        for (const auto& t : p.tensors) a.push_back(t->desc());
        jp.emplace_back(p.key, json::Value::array(std::move(a)));
      } else {
        jp.emplace_back(p.key, p.tensors.empty() ? json::Value::null() : p.tensors[0]->desc());
      }
    }
    jn.emplace_back("params", json::Value::object(std::move(jp)));

    jn.emplace_back("configs", json::Value::object(n->op.configs));
    jn.emplace_back("attrs", json::Value::object(n->op.attrs));
    node_arr.push_back(json::Value::object(std::move(jn)));
  }

  json::Object doc;
  doc.emplace_back("model", json::Value::str(name));
  doc.emplace_back("input_shape", json::Value::int_list(input_shape));
  doc.emplace_back("nodes", json::Value::array(std::move(node_arr)));
  json::Value(json::Value::object(std::move(doc))).dump(os, 2, 0);
  os << "\n";
}

}  // namespace gtxc
