"""Datasets: prompt pools, target-regenerated responses, SFT batches.

All sources are permissively licensed (see docs/provenance.md). Training and
evaluation prompts come from disjoint sources or disjoint splits.

Files are JSONL so every stage streams and is restartable: regeneration appends
one record per finished prompt and skips ids already present.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np

# domain -> [(hf repo, config, split)], used in order until the domain's count is reached
# (a larger count therefore extends the same pool: earlier prompts stay a prefix)
TRAIN_SOURCES = {
    "chat": [("HuggingFaceH4/ultrachat_200k", None, "train_sft")],
    "code": [("ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train")],
    # GSM8K train has 7.5k problems; NuminaMath-CoT (Apache-2.0) extends math, without its
    # gsm8k subset so the GSM8K-test eval prompts stay disjoint from training
    "math": [("openai/gsm8k", "main", "train"), ("AI-MO/NuminaMath-CoT", None, "train")],
    "multilingual": [("CohereLabs/aya_dataset", None, "train")],
}
EVAL_SOURCES = {
    "chat": [("HuggingFaceH4/mt_bench_prompts", None, "train")],
    "code": [("openai/openai_humaneval", None, "test")],
    "math": [("openai/gsm8k", "main", "test")],
    "multilingual": [("CohereLabs/aya_dataset", None, "test")],
}
AYA_LANGS = {"Russian", "German", "French", "Spanish", "Japanese", "Chinese", "Arabic", "Portuguese", "Italian", "Korean"}


def _extract(domain: str, row: dict) -> str | None:
    if domain == "chat":
        if "prompt" in row and isinstance(row["prompt"], str):
            return row["prompt"]
        if "prompt" in row and isinstance(row["prompt"], list):
            return row["prompt"][0]
        return None
    if domain == "code":
        return row.get("problem") or row.get("prompt")
    if domain == "math":
        if row.get("source") == "gsm8k":  # NuminaMath's GSM8K subset
            return None
        return row.get("question") or row.get("problem")
    if domain == "multilingual":
        if row.get("language") not in AYA_LANGS:
            return None
        return row.get("inputs")
    return None


def build_prompts(out: Path, counts: dict[str, int], split: str = "train", seed: int = 0, max_chars: int = 4000):
    from datasets import load_dataset

    srcs = TRAIN_SOURCES if split == "train" else EVAL_SOURCES
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for domain, n in counts.items():
            got = 0
            for repo, cfg, sp in srcs[domain]:
                if got >= n:
                    break
                ds = load_dataset(repo, cfg, split=sp, streaming=True)
                ds = ds.shuffle(seed=seed, buffer_size=10_000)
                start = got
                for row in ds:
                    text = _extract(domain, row)
                    if not text or len(text) > max_chars:
                        continue
                    if domain == "code" and split == "eval":
                        text = "Complete the following Python function.\n\n```python\n" + text + "```"
                    rec = {"id": f"{split}-{domain}-{got}", "domain": domain, "prompt": text.strip(), "source": repo}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    got += 1
                    if got >= n:
                        break
                print(f"[prompts] {split}/{domain}: {got - start} from {repo}")


def read_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def chat_ids(tokenizer, prompt: str) -> list[int]:
    """Qwen3 chat template, non-thinking mode (Woodcock v1 trains and benchmarks without <think>)."""
    msgs = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer.encode(text, add_special_tokens=False)


def regenerate(
    target_path: str,
    prompts: Path,
    out: Path,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    batch: int = 64,
    limit: int | None = None,
    seed: int = 0,
):
    """Target-regenerated responses (as in EAGLE-3 / EAGLE-3.1 training data)."""
    import mlx.core as mx
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(seed)
    model, tok = load(target_path)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [r for r in read_jsonl(prompts) if r["id"] not in done]
    # prompts files are grouped by domain; a seeded shuffle keeps any prefix (limit)
    # domain-balanced, and is identical across restarts
    random.Random(seed).shuffle(todo)
    if limit is not None:
        todo = todo[: max(0, limit - len(done))]
    sampler = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
    print(f"[regen] {len(done)} done, {len(todo)} to go")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for s in range(0, len(todo), batch):
            chunk = todo[s : s + batch]
            pids = [chat_ids(tok, r["prompt"]) for r in chunk]
            res = batch_generate(model, tok, pids, max_tokens=max_new_tokens, sampler=sampler,
                                 return_token_ids=True, completion_batch_size=batch, prefill_batch_size=8)
            for r, p, resp in zip(chunk, pids, res.token_ids):
                # batch_generate drops the stop token; a response shorter than the cap stopped on EOS
                rec = {"id": r["id"], "domain": r["domain"], "prompt_ids": p, "response_ids": list(resp),
                       "max_new_tokens": max_new_tokens}
                f.write(json.dumps(rec) + "\n")
            f.flush()
            st = res.stats
            print(f"[regen] {s + len(chunk)}/{len(todo)}  gen {st.generation_tps:.0f} tok/s  "
                  f"prefill {st.prompt_tps:.0f} tok/s  peak {st.peak_memory:.1f} GB", flush=True)


class SFTStream:
    """Streams (ids, loss_mask) examples, loss only on response tokens
    (EAGLE-3.1 §5.1: "the loss is computed only on assistant tokens")."""

    def __init__(self, path: Path, max_len: int, eos_id: int, seed: int = 0, holdout: int = 0):
        self.path, self.max_len, self.eos_id, self.seed, self.holdout = path, max_len, eos_id, seed, holdout

    def records(self, train=True):
        recs = list(read_jsonl(self.path))
        random.Random(self.seed).shuffle(recs)
        return recs[self.holdout :] if train else recs[: self.holdout]

    def examples(self, train=True, epoch=0):
        recs = self.records(train)
        random.Random(self.seed + epoch).shuffle(recs)
        for r in recs:
            # EOS only for responses that ended naturally: a response that hit the generation
            # cap was cut mid-text (records from before the field existed used a 512 cap)
            ended = len(r["response_ids"]) < r.get("max_new_tokens", 512)
            p, resp = r["prompt_ids"], r["response_ids"] + ([self.eos_id] if ended else [])
            ids = (p + resp)[: self.max_len]
            mask = ([0] * len(p) + [1] * len(resp))[: self.max_len]
            if sum(mask) < 8:
                continue
            yield np.array(ids, np.int32), np.array(mask, np.float32)


def batches(examples, token_budget: int, max_batch: int = 16):
    """Length-bucketed padded batches with at most `token_budget` padded tokens."""
    buf = []
    for ex in examples:
        buf.append(ex)
        if len(buf) >= 256:
            yield from _flush(buf, token_budget, max_batch)
            buf = []
    if buf:
        yield from _flush(buf, token_budget, max_batch)


def _flush(buf, token_budget, max_batch):
    buf.sort(key=lambda e: len(e[0]))
    cur = []
    for ex in buf:
        L = max([len(e[0]) for e in cur] + [len(ex[0])])
        if cur and (L * (len(cur) + 1) > token_budget or len(cur) >= max_batch):
            yield _pad(cur)
            cur = []
        cur.append(ex)
    if cur:
        yield _pad(cur)


def _pad(exs):
    L = max(len(e[0]) for e in exs)
    ids = np.zeros((len(exs), L), np.int32)
    mask = np.zeros((len(exs), L), np.float32)
    for i, (a, m) in enumerate(exs):
        ids[i, : len(a)] = a
        mask[i, : len(m)] = m
    return ids, mask
