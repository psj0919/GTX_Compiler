// torch::jit::Graph (traced .pt) → GTX Graph IR. parse/ TorchParser 의 C++ 포트.
#pragma once
#include <string>
#include <vector>

#include "gtxc/ir.hpp"

namespace gtxc {

// pt_path: torch.jit.trace(...).save() 결과. graph_name: IR scope(예 "resnet18").
// input_shape: 데이터 입력 shape(예 {1,3,224,224}).
Graph parse_traced(const std::string& pt_path, const std::string& graph_name,
                   const std::vector<int64_t>& input_shape);

// 그래프 파라미터를 GGUF(fp16)로 직렬화. 텐서명 = state_dict 키, arch=general.architecture.
void write_gguf(const std::string& pt_path, const std::string& graph_name,
                const std::vector<int64_t>& input_shape, const std::string& out_path,
                const std::string& arch);

// E2E 컴파일: parse → Conv-BN fold → vision.cpp arch C++(.cpp/.h) + folded GGUF 출력.
void compile_model(const std::string& pt_path, const std::string& graph_name,
                   const std::vector<int64_t>& input_shape, const std::string& out_dir);

}  // namespace gtxc
