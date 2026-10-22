"""Target-regenerated responses on a CUDA GPU with vLLM (Apache-2.0).

Same output as `woodcock regen` (python/woodcock/data.py::regenerate), so
`woodcock adapt` reads it unchanged:

  {"id", "domain", "prompt_ids", "response_ids", "max_new_tokens", "generator"}

* prompt_ids: Qwen3 chat template, one user turn, non-thinking mode, encoded
  without extra special tokens (data.chat_ids);
* sampling: T = 0.7, top-p 0.8, top-k 20, at most 512 new tokens (data.regenerate);
* response_ids never contain the stop token (mlx-lm's batch_generate drops it;
  SFTStream appends EOS itself to responses shorter than max_new_tokens);
* prompts are shuffled with random.Random(seed) and cut at --limit exactly like
  data.regenerate; ids already in --out are skipped, so the script resumes.

It does not import woodcock (no MLX on the CUDA box's vLLM environment).

  python scripts/cuda/regen_vllm.py --model Qwen/Qwen3-1.7B \
      --prompts data/prompts-v2/train.jsonl --out data/regen/qwen3-1.7b-v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def chat_ids(tok, prompt: str) -> list[int]:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok.encode(text, add_special_tokens=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4096, help="prompts per generate call (one flush each)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    out = Path(a.out)
    done = {r["id"] for r in read_jsonl(out)}
    todo = [r for r in read_jsonl(Path(a.prompts)) if r["id"] not in done]
    random.Random(a.seed).shuffle(todo)  # identical to data.regenerate
    if a.limit is not None:
        todo = todo[: max(0, a.limit - len(done))]
    print(f"[regen-vllm] {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(a.model)
    eos = {tok.eos_token_id}
    llm = LLM(model=a.model, dtype="bfloat16", seed=a.seed, gpu_memory_utilization=a.gpu_memory_utilization)
    # Qwen3: <|im_end|> ends a turn, <|endoftext|> ends a document; both stop generation
    eos |= {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    eos.discard(None)
    print(f"[regen-vllm] stop ids {sorted(eos)}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    t0, n_tok = time.perf_counter(), 0
    with out.open("a") as f:
        for s in range(0, len(todo), a.chunk):
            chunk = todo[s : s + a.chunk]
            pids = [chat_ids(tok, r["prompt"]) for r in chunk]
            # per-request seeds: a resumed or re-chunked run draws the same samples
            sps = [SamplingParams(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                                  max_tokens=a.max_new_tokens, seed=a.seed * 1_000_003 + hash_id(r["id"]),
                                  stop_token_ids=sorted(eos))
                   for r in chunk]
            res = llm.generate([TokensPrompt(prompt_token_ids=p) for p in pids], sps, use_tqdm=False)
            for r, p, o in zip(chunk, pids, res):
                c = o.outputs[0]
                resp = list(c.token_ids)
                if resp and resp[-1] in eos:  # keep the mlx-lm convention: no stop token
                    resp = resp[:-1]
                n_tok += len(resp)
                rec = {"id": r["id"], "domain": r["domain"], "prompt_ids": p, "response_ids": resp,
                       "max_new_tokens": a.max_new_tokens, "generator": "vllm"}
                f.write(json.dumps(rec) + "\n")
            f.flush()
            dt = time.perf_counter() - t0
            print(f"[regen-vllm] {s + len(chunk)}/{len(todo)}  {n_tok / dt:.0f} gen tok/s  "
                  f"{n_tok / 1e6:.1f}M tokens  {dt / 60:.1f} min", flush=True)


def hash_id(s: str) -> int:
    """Stable 31-bit hash (Python's hash() is salted per process)."""
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h & 0x7FFFFFFF


if __name__ == "__main__":
    main()
