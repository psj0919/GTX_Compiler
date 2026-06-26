// gtxc-parse — traced TorchScript(.pt) → GTX Graph IR JSON.
//
//   gtxc-parse <model.pt> --graph-name resnet18 --input-shape 1,3,224,224 [--out ir.json]
//
// 출력 스키마는 cpp/tools/dump_ir.py(Python TorchParser) 와 동일 → 1:1 대조.

#include "gtxc/parser.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<int64_t> parse_shape(const std::string& s) {
  std::vector<int64_t> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(std::stoll(tok));
  return out;
}

std::string basename_noext(const std::string& path) {
  size_t slash = path.find_last_of("/\\");
  std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
  size_t dot = base.find_last_of('.');
  return dot == std::string::npos ? base : base.substr(0, dot);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: gtxc-parse <model.pt> [--graph-name N] "
                 "[--input-shape 1,3,224,224] [--out ir.json]\n";
    return 2;
  }
  std::string pt_path = argv[1];
  std::string graph_name = basename_noext(pt_path);
  std::vector<int64_t> input_shape = {1, 3, 224, 224};
  std::string out_path;
  std::string gguf_path;
  std::string compile_dir;
  std::string quant;

  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--graph-name" && i + 1 < argc) graph_name = argv[++i];
    else if (a == "--input-shape" && i + 1 < argc) input_shape = parse_shape(argv[++i]);
    else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
    else if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
    else if (a == "--compile" && i + 1 < argc) compile_dir = argv[++i];
    else if (a == "--quantize" && i + 1 < argc) quant = argv[++i];
  }

  if (!compile_dir.empty()) {
    try {
      gtxc::compile_model(pt_path, graph_name, input_shape, compile_dir, quant);
      return 0;
    } catch (const std::exception& e) {
      std::cerr << "[gtxc-compile] error: " << e.what() << "\n";
      return 1;
    }
  }

  if (!gguf_path.empty()) {
    try {
      gtxc::write_gguf(pt_path, graph_name, input_shape, gguf_path, graph_name);
    } catch (const std::exception& e) {
      std::cerr << "[gtxc-gguf] error: " << e.what() << "\n";
      return 1;
    }
  }

  gtxc::Graph graph;
  try {
    graph = gtxc::parse_traced(pt_path, graph_name, input_shape);
  } catch (const std::exception& e) {
    std::cerr << "[gtxc-parse] error: " << e.what() << "\n";
    return 1;
  }

  if (out_path.empty()) {
    graph.dump_json(std::cout, input_shape);
  } else {
    std::ofstream ofs(out_path);
    graph.dump_json(ofs, input_shape);
    std::cerr << "[gtxc-parse] " << graph.nodes.size() << " nodes → " << out_path << "\n";
  }
  return 0;
}
