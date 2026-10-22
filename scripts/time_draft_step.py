"""Latency of one draft step and of target verification (batch 1, bf16).

Measures, at a fixed context length:
  head     drafter LM head + argmax on one row (full or K-row head)
  step     one fused-chain draft step: head + argmax + drafter layer on the drafted token
  verify   target forward on n tokens (n = 1..max_n), as in a chain verification
Each number is the median over `--iters` evaluations, after warm-up.

Usage:
  python scripts/time_draft_step.py --target Qwen/Qwen3-1.7B \
      --drafter runs/adapt/qwen3-1.7b/final --vocab runs/vocab/qwen3-1.7b.npz --sizes 16384 32768 49152
"""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np

from woodcock.drafter import Drafter
from woodcock.target import Target
from woodcock.vocab import load_ids


def median_ms(fn, iters, warmup=10):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


def time_drafter(d: Drafter, target: Target, ctx: int, iters: int):
    toks = mx.random.randint(0, target.vocab_size, (1, ctx))
    o = target(toks, target.make_cache())
    cache = d.make_cache()
    _, out = d(toks, d.fuse(o.features), cache)
    row = out[:, -1:]
    mx.eval(row)
    base = cache[0].offset

    def head():
        return d.token_ids(mx.argmax(d.logits(row[0]), axis=-1, keepdims=True))

    def step():
        tok = d.token_ids(mx.argmax(d.logits(row[0]), axis=-1, keepdims=True)).astype(mx.int32)
        hid, out2 = d(tok, row, cache)
        for c in cache:
            c.offset = base  # discard the speculative entry, like a chain round does
        return out2

    return {"head_ms": median_ms(head, iters), "step_ms": median_ms(step, iters)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--vocab", help="counts file from `woodcock vocab`")
    ap.add_argument("--sizes", type=int, nargs="*", default=[])
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--max-n", type=int, default=7)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out")
    a = ap.parse_args()
    mx.random.seed(0)
    t = Target.load(a.target)
    res = {"ctx": a.ctx, "drafter": {}, "verify_ms": {}}
    res["drafter"]["full"] = time_drafter(Drafter.load(a.drafter), t, a.ctx, a.iters)
    print("full", res["drafter"]["full"], flush=True)
    for k in a.sizes:
        d = Drafter.load(a.drafter)
        d.restrict_vocab(load_ids(f"{a.vocab}:{k}"))
        res["drafter"][str(k)] = time_drafter(d, t, a.ctx, a.iters)
        print(k, res["drafter"][str(k)], flush=True)
    cache = t.make_cache()
    mx.eval(t(mx.random.randint(0, t.vocab_size, (1, a.ctx)), cache).logits)
    for n in range(1, a.max_n + 1):
        seq = mx.random.randint(0, t.vocab_size, (1, n))

        def verify():
            o = t(seq, cache)
            for c in cache:
                c.offset = a.ctx
            return mx.argmax(o.logits[0], axis=-1)

        res["verify_ms"][n] = median_ms(verify, a.iters)
        print("verify", n, round(res["verify_ms"][n], 3), flush=True)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
