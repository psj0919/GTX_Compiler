// 최소 JSON 값 + writer (의존성 없이 IR 덤프 전용 — 파싱 불필요).
#pragma once
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace gtxc::json {

struct Value;
using Array = std::vector<Value>;
using Object = std::vector<std::pair<std::string, Value>>;  // 순서 보존

struct Value {
  enum Type { Null, Bool, Int, Double, Str, Arr, Obj } type = Null;
  bool b = false;
  int64_t i = 0;
  double d = 0;
  std::string s;
  std::shared_ptr<Array> arr;
  std::shared_ptr<Object> obj;

  Value() = default;
  static Value null() { return Value{}; }
  static Value boolean(bool v) { Value x; x.type = Bool; x.b = v; return x; }
  static Value integer(int64_t v) { Value x; x.type = Int; x.i = v; return x; }
  static Value real(double v) { Value x; x.type = Double; x.d = v; return x; }
  static Value str(std::string v) { Value x; x.type = Str; x.s = std::move(v); return x; }
  static Value array(Array v = {}) {
    Value x; x.type = Arr; x.arr = std::make_shared<Array>(std::move(v)); return x;
  }
  static Value object(Object v = {}) {
    Value x; x.type = Obj; x.obj = std::make_shared<Object>(std::move(v)); return x;
  }
  static Value int_list(const std::vector<int64_t>& v) {
    Array a; a.reserve(v.size());
    for (auto e : v) a.push_back(integer(e));
    return array(std::move(a));
  }

  void dump(std::ostream& os, int indent = 2, int level = 0) const;
};

inline void dump_string(std::ostream& os, const std::string& s) {
  os << '"';
  for (char c : s) {
    switch (c) {
      case '"': os << "\\\""; break;
      case '\\': os << "\\\\"; break;
      case '\n': os << "\\n"; break;
      case '\t': os << "\\t"; break;
      case '\r': os << "\\r"; break;
      default: os << c;
    }
  }
  os << '"';
}

inline void Value::dump(std::ostream& os, int indent, int level) const {
  const std::string pad(static_cast<size_t>(indent) * (level + 1), ' ');
  const std::string pad0(static_cast<size_t>(indent) * level, ' ');
  switch (type) {
    case Null: os << "null"; break;
    case Bool: os << (b ? "true" : "false"); break;
    case Int: os << i; break;
    case Double: {
      // 가독성: 정수값이면 .0 유지하지 않고 일반 표기. JSON 호환 위해 기본 ostream.
      os << d; break;
    }
    case Str: dump_string(os, s); break;
    case Arr: {
      if (arr->empty()) { os << "[]"; break; }
      os << "[\n";
      for (size_t k = 0; k < arr->size(); ++k) {
        os << pad;
        (*arr)[k].dump(os, indent, level + 1);
        if (k + 1 < arr->size()) os << ",";
        os << "\n";
      }
      os << pad0 << "]";
      break;
    }
    case Obj: {
      if (obj->empty()) { os << "{}"; break; }
      os << "{\n";
      for (size_t k = 0; k < obj->size(); ++k) {
        os << pad;
        dump_string(os, (*obj)[k].first);
        os << ": ";
        (*obj)[k].second.dump(os, indent, level + 1);
        if (k + 1 < obj->size()) os << ",";
        os << "\n";
      }
      os << pad0 << "}";
      break;
    }
  }
}

}  // namespace gtxc::json
