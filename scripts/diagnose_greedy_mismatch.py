"""Is a bf16 greedy mismatch between speculative and autoregressive decoding a
near-tie (numerics) or a verifier bug?

For every benchmark row whose speculative output diverged from autoregressive
greedy, recompute the target logits at the first divergent position with an fp32
copy of the target and report the top-1/top-2 logit gap and which token each
decoding chose. A verifier bug shows up as a large gap where speculative decoding
picked a non-argmax token."""

import argparse
import json

import mlx.core as mx
import numpy as np

from woodcock.data import chat_ids, read_jsonl
from woodcock.drafter import Drafter
from woodcock.policies import StaticPolicy
from woodcock.spec import SpeculativeDecoder, generate_autoregressive
from woodcock.target import Target

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)
ap.add_argument("--target", default="Qwen/Qwen3-1.7B")
ap.add_argument("--drafter", required=True)
ap.add_argument("--name", default="e4")
ap.add_argument("--max", type=int, default=8)
a = ap.parse_args()

res = json.loads(open(a.bench).read())
prompts = {r["id"]: r for r in read_jsonl(res["config"]["prompts"])}
t = Target.load(a.target)
dec = SpeculativeDecoder(t, Drafter.load(a.drafter), StaticPolicy(5))
t32 = Target.load(a.target)
t32.model.set_dtype(mx.float32)
n_new = res["config"]["max_new_tokens"]
rows = [r for r in res["rows"] if r[a.name].get("first_mismatch") is not None][: a.max]
gaps = []
for r in rows:
    ids = chat_ids(t.tokenizer, prompts[r["id"]]["prompt"])
    ar, _ = generate_autoregressive(t, ids, n_new)
    sd, _ = dec.generate(ids, n_new)
    k = next(i for i in range(min(len(ar), len(sd))) if ar[i] != sd[i])
    z = t32(mx.array([ids + ar[:k]]), capture=False).logits[0, -1].astype(mx.float32)
    top = np.argsort(-np.array(z))[:2]
    zz = np.array(z)
    gap = float(zz[top[0]] - zz[top[1]])
    gaps.append(gap)
    print(json.dumps({"id": r["id"], "pos": k, "ar_tok": ar[k], "spec_tok": sd[k], "fp32_top2": top.tolist(),
                      "fp32_gap": round(gap, 4), "ar_is_fp32_argmax": int(ar[k] == top[0]),
                      "spec_is_fp32_argmax": int(sd[k] == top[0]), "both_in_fp32_top2": int({ar[k], sd[k]} <= set(top.tolist()))}))
print(json.dumps({"n": len(gaps), "max_gap": max(gaps) if gaps else None, "median_gap": float(np.median(gaps)) if gaps else None}))
