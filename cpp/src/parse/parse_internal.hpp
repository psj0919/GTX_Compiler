// 내부 헤더: parser.cpp 의 파싱 진입점을 compile 모듈(gguf/codegen)과 공유.
// (공개 API 는 include/gtxc/parser.hpp. 이건 torch 의존 구현 공유용.)
#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/script.h>

#include "gtxc/ir.hpp"

namespace gtxc {

// load→freeze(preserveParameters)→optimize→parse. params(full name→tensor) 도 채운다.
Graph do_parse(const std::string& pt_path, const std::string& graph_name,
               const std::vector<int64_t>& input_shape,
               std::unordered_map<std::string, at::Tensor>& params);

// 파싱된 graph + param 데이터로 GGUF(fp16) 작성(gguf.cpp). codegen 도 사용.
void write_gguf_core(const Graph& g,
                     const std::unordered_map<std::string, at::Tensor>& params,
                     const std::string& out_path, const std::string& arch);

}  // namespace gtxc
