#include "woodcock/json.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace woodcock {

class JsonParser {
 public:
  explicit JsonParser(std::string_view s) : s_(s) {}

  Json value() {
    ws();
    if (i_ >= s_.size()) fail("unexpected end");
    char c = s_[i_];
    Json j;
    if (c == '{') {
      auto o = std::make_shared<Json::Object>();
      ++i_;
      ws();
      if (peek() == '}') { ++i_; j.v_ = o; return j; }
      while (true) {
        ws();
        std::string k = string();
        ws();
        expect(':');
        (*o)[k] = value();
        ws();
        if (peek() == ',') { ++i_; continue; }
        expect('}');
        break;
      }
      j.v_ = o;
    } else if (c == '[') {
      auto a = std::make_shared<Json::Array>();
      ++i_;
      ws();
      if (peek() == ']') { ++i_; j.v_ = a; return j; }
      while (true) {
        a->push_back(value());
        ws();
        if (peek() == ',') { ++i_; continue; }
        expect(']');
        break;
      }
      j.v_ = a;
    } else if (c == '"') {
      j.v_ = string();
    } else if (s_.compare(i_, 4, "true") == 0) {
      i_ += 4; j.v_ = true;
    } else if (s_.compare(i_, 5, "false") == 0) {
      i_ += 5; j.v_ = false;
    } else if (s_.compare(i_, 4, "null") == 0) {
      i_ += 4;
    } else if (s_.compare(i_, 3, "NaN") == 0) {
      i_ += 3; j.v_ = std::numeric_limits<double>::quiet_NaN();
    } else {
      size_t st = i_;
      while (i_ < s_.size() && (std::isdigit((unsigned char)s_[i_]) || std::strchr("+-.eE", s_[i_]))) ++i_;
      if (st == i_) fail("bad token");
      j.v_ = std::strtod(std::string(s_.substr(st, i_ - st)).c_str(), nullptr);
    }
    return j;
  }

  void finish() {
    ws();
    if (i_ != s_.size()) fail("trailing data");
  }

 private:
  std::string_view s_;
  size_t i_ = 0;

  [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("json: ") + m + " at " + std::to_string(i_)); }
  char peek() { return i_ < s_.size() ? s_[i_] : '\0'; }
  void ws() { while (i_ < s_.size() && std::isspace((unsigned char)s_[i_])) ++i_; }
  void expect(char c) { if (peek() != c) fail("unexpected char"); ++i_; }

  static void utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += char(cp);
    else if (cp < 0x800) { o += char(0xC0 | (cp >> 6)); o += char(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { o += char(0xE0 | (cp >> 12)); o += char(0x80 | ((cp >> 6) & 0x3F)); o += char(0x80 | (cp & 0x3F)); }
    else { o += char(0xF0 | (cp >> 18)); o += char(0x80 | ((cp >> 12) & 0x3F)); o += char(0x80 | ((cp >> 6) & 0x3F)); o += char(0x80 | (cp & 0x3F)); }
  }

  uint32_t hex4() {
    if (i_ + 4 > s_.size()) fail("bad escape");
    uint32_t v = std::stoul(std::string(s_.substr(i_, 4)), nullptr, 16);
    i_ += 4;
    return v;
  }

  std::string string() {
    expect('"');
    std::string o;
    while (true) {
      if (i_ >= s_.size()) fail("unterminated string");
      char c = s_[i_++];
      if (c == '"') break;
      if (c != '\\') { o += c; continue; }
      char e = s_[i_++];
      switch (e) {
        case 'n': o += '\n'; break;
        case 't': o += '\t'; break;
        case 'r': o += '\r'; break;
        case 'b': o += '\b'; break;
        case 'f': o += '\f'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00 && s_.compare(i_, 2, "\\u") == 0) {
            i_ += 2;
            uint32_t lo = hex4();
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          utf8(o, cp);
          break;
        }
        default: o += e;
      }
    }
    return o;
  }
};

Json Json::parse(std::string_view text) {
  JsonParser p(text);
  Json j = p.value();
  p.finish();
  return j;
}

double Json::num() const {
  if (auto* b = std::get_if<bool>(&v_)) return *b ? 1.0 : 0.0;
  return std::get<double>(v_);
}
bool Json::boolean() const { return std::get<bool>(v_); }
const std::string& Json::str() const { return std::get<std::string>(v_); }
const Json::Array& Json::arr() const { return *std::get<std::shared_ptr<Array>>(v_); }
const Json::Object& Json::obj() const { return *std::get<std::shared_ptr<Object>>(v_); }
bool Json::has(const std::string& k) const { return is_object() && obj().count(k) > 0; }
const Json& Json::operator[](const std::string& k) const {
  auto& o = obj();
  auto it = o.find(k);
  if (it == o.end()) throw std::runtime_error("json: missing key " + k);
  return it->second;
}
double Json::num_or(const std::string& k, double d) const { return has(k) && !(*this)[k].is_null() ? (*this)[k].num() : d; }
bool Json::bool_or(const std::string& k, bool d) const { return has(k) && (*this)[k].is_bool() ? (*this)[k].boolean() : d; }
std::string Json::str_or(const std::string& k, const std::string& d) const {
  return has(k) && (*this)[k].is_string() ? (*this)[k].str() : d;
}

}  // namespace woodcock
