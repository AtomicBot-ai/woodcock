#include "woodcock/weights.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "woodcock/json.h"

namespace fs = std::filesystem;

namespace woodcock {

class WeightFile {
 public:
  explicit WeightFile(const fs::path& p) {
    fd_ = ::open(p.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open " + p.string());
    struct stat st {};
    fstat(fd_, &st);
    size_ = static_cast<size_t>(st.st_size);
    base_ = static_cast<const uint8_t*>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if (base_ == MAP_FAILED) throw std::runtime_error("mmap failed " + p.string());
    uint64_t hlen = 0;
    std::memcpy(&hlen, base_, 8);
    header_ = std::string(reinterpret_cast<const char*>(base_ + 8), hlen);
    data_ = base_ + 8 + hlen;
  }
  ~WeightFile() {
    if (base_ && base_ != MAP_FAILED) munmap(const_cast<uint8_t*>(base_), size_);
    if (fd_ >= 0) ::close(fd_);
  }
  const std::string& header() const { return header_; }
  const uint8_t* data() const { return data_; }

 private:
  int fd_ = -1;
  size_t size_ = 0;
  const uint8_t* base_ = nullptr;
  const uint8_t* data_ = nullptr;
  std::string header_;
};

float bf16_to_f32(uint16_t v) {
  uint32_t u = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

float f16_to_f32(uint16_t h) {
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
  float v;
  if (e == 0) v = std::ldexp(float(m), -24);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = std::ldexp(float(m | 0x400), int(e) - 25);
  return s ? -v : v;
}

Weights Weights::open(const fs::path& path) {
  Weights w;
  std::vector<fs::path> files;
  if (fs::is_directory(path)) {
    for (auto& e : fs::directory_iterator(path))
      if (e.path().extension() == ".safetensors") files.push_back(e.path());
  } else {
    files.push_back(path);
  }
  if (files.empty()) throw std::runtime_error("no .safetensors in " + path.string());
  for (auto& f : files) {
    // follow Hugging Face cache symlinks
    auto real = fs::is_symlink(f) ? fs::canonical(f) : f;
    auto wf = std::make_shared<WeightFile>(real);
    w.files_.push_back(wf);
    Json h = Json::parse(wf->header());
    for (auto& [name, t] : h.obj()) {
      if (name == "__metadata__") continue;
      Entry e;
      e.file = wf;
      e.dtype = t["dtype"].str();
      for (auto& s : t["shape"].arr()) e.shape.push_back(s.i64());
      e.begin = size_t(t["data_offsets"][0].i64());
      e.end = size_t(t["data_offsets"][1].i64());
      w.index_[name] = e;
    }
  }
  return w;
}

std::vector<std::string> Weights::names() const {
  std::vector<std::string> n;
  for (auto& [k, _] : index_) n.push_back(k);
  return n;
}

Tensor Weights::get(const std::string& name) const {
  auto it = index_.find(name);
  if (it == index_.end()) throw std::runtime_error("missing tensor " + name);
  const Entry& e = it->second;
  Tensor t;
  t.shape = e.shape;
  int64_t n = t.numel();
  t.data.resize(static_cast<size_t>(n));
  const uint8_t* p = e.file->data() + e.begin;
  if (e.dtype == "F32") {
    std::memcpy(t.data.data(), p, size_t(n) * 4);
  } else if (e.dtype == "BF16" || e.dtype == "F16") {
    const bool bf = e.dtype == "BF16";
    for (int64_t i = 0; i < n; ++i) {
      uint16_t v;
      std::memcpy(&v, p + 2 * i, 2);
      t.data[size_t(i)] = bf ? bf16_to_f32(v) : f16_to_f32(v);
    }
  } else {
    throw std::runtime_error("unsupported dtype " + e.dtype + " for " + name);
  }
  return t;
}

}  // namespace woodcock
