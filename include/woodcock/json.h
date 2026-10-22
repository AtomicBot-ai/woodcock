// Minimal JSON reader for config / manifest / safetensors headers.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace woodcock {

class Json {
 public:
  using Array = std::vector<Json>;
  using Object = std::map<std::string, Json>;

  Json() = default;
  static Json parse(std::string_view text);

  bool is_null() const { return std::holds_alternative<std::monostate>(v_); }
  bool is_object() const { return std::holds_alternative<std::shared_ptr<Object>>(v_); }
  bool is_array() const { return std::holds_alternative<std::shared_ptr<Array>>(v_); }
  bool is_number() const { return std::holds_alternative<double>(v_); }
  bool is_string() const { return std::holds_alternative<std::string>(v_); }
  bool is_bool() const { return std::holds_alternative<bool>(v_); }

  double num() const;
  int64_t i64() const { return static_cast<int64_t>(num()); }
  bool boolean() const;
  const std::string& str() const;
  const Array& arr() const;
  const Object& obj() const;

  bool has(const std::string& k) const;
  const Json& operator[](const std::string& k) const;
  const Json& operator[](size_t i) const { return arr().at(i); }
  // value or default when the key is missing or null
  double num_or(const std::string& k, double d) const;
  bool bool_or(const std::string& k, bool d) const;
  std::string str_or(const std::string& k, const std::string& d) const;

 private:
  std::variant<std::monostate, bool, double, std::string, std::shared_ptr<Array>, std::shared_ptr<Object>> v_;
  friend class JsonParser;
};

}  // namespace woodcock
