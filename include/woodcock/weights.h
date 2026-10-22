// Safetensors weights loader. Tensors are memory-mapped and converted to fp32
// on access (bf16 / f16 / f32 sources supported).
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace woodcock {

struct Tensor {
  std::vector<int64_t> shape;
  std::vector<float> data;  // row-major, fp32
  int64_t numel() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
  int64_t rows() const { return shape.at(0); }
  int64_t cols() const { return shape.size() > 1 ? shape[1] : 1; }
};

class WeightFile;

class Weights {
 public:
  // Loads every *.safetensors file in `dir` (or a single file).
  static Weights open(const std::filesystem::path& path);
  bool has(const std::string& name) const { return index_.count(name) > 0; }
  Tensor get(const std::string& name) const;  // converted to fp32
  std::vector<std::string> names() const;

 private:
  struct Entry {
    std::shared_ptr<WeightFile> file;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t begin = 0, end = 0;
  };
  std::map<std::string, Entry> index_;
  std::vector<std::shared_ptr<WeightFile>> files_;
};

float bf16_to_f32(uint16_t v);
float f16_to_f32(uint16_t v);

}  // namespace woodcock
