cmake_minimum_required(VERSION 3.24)
project(woodcock VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(woodcock
  src/core/json.cpp
  src/core/weights.cpp
  src/core/qwen3.cpp
  src/core/drafter.cpp
  src/core/spec.cpp
  src/backends/cpu_ops.cpp
  src/cache/kv_cache.cpp
  src/sampling/sampling.cpp)
target_include_directories(woodcock PUBLIC include)
target_compile_options(woodcock PRIVATE -Wall -Wextra -Wno-unused-parameter)

if(APPLE)
  find_library(ACCELERATE Accelerate REQUIRED)
  target_link_libraries(woodcock PUBLIC ${ACCELERATE})
  target_compile_definitions(woodcock PRIVATE WOODCOCK_ACCELERATE ACCELERATE_NEW_LAPACK)
endif()

add_executable(woodcock-generate tools/woodcock_generate.cpp)
target_link_libraries(woodcock-generate PRIVATE woodcock)

enable_testing()
add_executable(woodcock_tests tests/cpp/test_main.cpp)
target_link_libraries(woodcock_tests PRIVATE woodcock)
add_test(NAME unit COMMAND woodcock_tests)
