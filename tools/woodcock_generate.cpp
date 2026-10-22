// woodcock-generate: speculative decoding from token ids, no Python / MLX.
//   woodcock-generate --target DIR --drafter DIR --ids 1,2,3 [--max 128]
//                     [--policy chain:5|tree:D,W,N] [--temp 0] [--ar]
// Tokenization is the host application's job (v1 runtime takes token ids).
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>

#include "woodcock/spec.h"

using namespace woodcock;

static std::vector<int> parse_ids(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stoi(tok));
  return v;
}

int main(int argc, char** argv) {
  std::string target, drafter, ids, policy = "chain:5";
  int max_new = 128;
  float temp = 0.f;
  bool ar = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&] { return std::string(argv[++i]); };
    if (a == "--target") target = next();
    else if (a == "--drafter") drafter = next();
    else if (a == "--ids") ids = next();
    else if (a == "--max") max_new = std::stoi(next());
    else if (a == "--policy") policy = next();
    else if (a == "--temp") temp = std::stof(next());
    else if (a == "--ar") ar = true;
  }
  if (target.empty() || ids.empty() || (!ar && drafter.empty())) {
    std::fprintf(stderr, "usage: woodcock-generate --target DIR --drafter DIR --ids 1,2,3 [--max N] [--policy P] [--temp T] [--ar]\n");
    return 2;
  }
  Qwen3 t(target);
  const std::vector<int> eos{151645, 151643};  // Qwen3 <|im_end|>, <|endoftext|>
  SamplingParams sp;
  sp.temperature = temp;
  GenStats st;
  std::vector<int> out;
  if (ar) {
    out = generate_autoregressive(t, parse_ids(ids), max_new, sp, &st, eos);
  } else {
    Drafter d(drafter);
    std::unique_ptr<Policy> pol;
    auto colon = policy.find(':');
    std::string kind = policy.substr(0, colon), arg = colon == std::string::npos ? "" : policy.substr(colon + 1);
    if (kind == "chain") pol = std::make_unique<StaticPolicy>(std::stoi(arg));
    else if (kind == "tree") {
      auto v = parse_ids(arg);
      pol = std::make_unique<StaticPolicy>(v.at(0), v.at(1), v.at(2));
    } else {
      std::fprintf(stderr, "unknown policy %s\n", policy.c_str());
      return 2;
    }
    SpeculativeDecoder dec(t, d, *pol);
    out = dec.generate(parse_ids(ids), max_new, sp, &st, eos);
  }
  for (size_t i = 0; i < out.size(); ++i) std::printf("%s%d", i ? "," : "", out[i]);
  std::printf("\n{\"new_tokens\": %d, \"rounds\": %d, \"tokens_per_round\": %.3f, \"decode_tps\": %.2f, "
              "\"t_prefill\": %.3f, \"t_draft\": %.3f, \"t_verify\": %.3f, \"t_update\": %.3f}\n",
              st.new_tokens, st.rounds, st.tokens_per_round(), st.decode_tps(), st.t_prefill, st.t_draft, st.t_verify,
              st.t_update);
  return 0;
}
