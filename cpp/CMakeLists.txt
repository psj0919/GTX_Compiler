# GTX Compiler — C++ / libtorch port
#
# 빌드:
#   cmake -S cpp -B cpp/build -DCMAKE_PREFIX_PATH="$(.venv/bin/python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
#   cmake --build cpp/build -j
#
# 또는 helper:  cpp/build.sh
#
# 주의: pip torch(2.6.0+cu124)는 pre-cxx11 ABI(_GLIBCXX_USE_CXX11_ABI=0).
#       find_package(Torch) 가 ${TORCH_CXX_FLAGS} 로 이 플래그를 주입하므로 그대로 사용.

cmake_minimum_required(VERSION 3.18)
project(gtx_compiler_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# CUDA 12.x 는 nvToolsExt 를 header-only(nvtx3)로 대체 → torch CMake config 가
# 참조하는 CUDA::nvToolsExt IMPORTED 타깃이 없어 generate 단계가 깨진다.
# find_package(Torch) 전에 빈 INTERFACE 스텁을 제공하여 우회(파서는 CPU만 사용).
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()

find_package(Torch REQUIRED)

# libtorch ABI 플래그(예: -D_GLIBCXX_USE_CXX11_ABI=0)를 전역 적용
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ── gtxc 코어 라이브러리 (IR + parser). 소스가 생기면 자동 포함 ────────
file(GLOB GTXC_CORE_SRC CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ir/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/parse/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/compile/*.cpp
)

# ── CLI: gtxc-parse (traced .pt → Graph IR JSON / GGUF / C++) ──────
add_executable(gtxc-parse ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${GTXC_CORE_SRC})
target_include_directories(gtxc-parse PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src)  # 내부 헤더 parse/parse_internal.hpp
target_link_libraries(gtxc-parse PRIVATE ${TORCH_LIBRARIES})

# 실행 시 LD_LIBRARY_PATH 없이도 .venv torch/lib 의 .so 를 찾도록 RPATH 주입
set_target_properties(gtxc-parse PROPERTIES
  BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")
