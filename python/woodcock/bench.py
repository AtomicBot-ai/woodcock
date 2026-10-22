"""End-to-end benchmarks on real models (batch size 1).

Baselines:
  ar-woodcock   plain decoding through the same Target forward the verifier uses
  ar-mlxlm      mlx_lm.stream_generate (async-evaluated, the best AR path available)
Speedup is reported against the FASTER of the two, so a slow baseline cannot
inflate it.

Greedy outputs of the speculative decoder are compared with ar-woodcock token by
token; mismatches are counted and reported, never hidden (in bf16 a near-tie can
flip between a 1-token and a k-token forward, which is not a verifier bug).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from .data import chat_ids, read_jsonl
from .sampling import SamplingParams
from .spec import SpeculativeDecoder, generate_autoregressive


@dataclass
class BenchConfig:
    prompts: str = "data/prompts/eval.jsonl"
    per_domain: int = 10
    max_new_tokens: int = 256
    temperature: float = 0.0
    warmup: int = 2
    seed: int = 0
    domains: list[str] = field(default_factory=lambda: ["chat", "code", "math", "multilingual"])


def load_eval_prompts(cfg: BenchConfig):
    by = {d: [] for d in cfg.domains}
    for r in read_jsonl(Path(cfg.prompts)):
        if r["domain"] in by and len(by[r["domain"]]) < cfg.per_domain:
            by[r["domain"]].append(r)
    return by


def _mlxlm_tps(target, ids, n):
    from mlx_lm import stream_generate

    last = None
    t0 = time.perf_counter()
    first = None
    for r in stream_generate(target.model, target.tokenizer, ids, max_tokens=n):
        if first is None:
            first = time.perf_counter() - t0
        last = r
    return last.generation_tps, last.generation_tokens, first


def run_bench(target, decoders: dict[str, SpeculativeDecoder], cfg: BenchConfig) -> dict:
    sp = SamplingParams(temperature=cfg.temperature, seed=cfg.seed)
    prompts = load_eval_prompts(cfg)
    flat = [(d, r) for d, rs in prompts.items() for r in rs]
    # warm-up (kernel compilation, allocator)
    for _, r in flat[: cfg.warmup]:
        ids = chat_ids(target.tokenizer, r["prompt"])
        generate_autoregressive(target, ids, 32, sp)
        for dec in decoders.values():
            dec.generate(ids, 32, sp)
    rows = []
    mx.reset_peak_memory()
    for dom, r in flat:
        ids = chat_ids(target.tokenizer, r["prompt"])
        row = {"id": r["id"], "domain": dom, "prompt_tokens": len(ids)}
        ar_out, ar = generate_autoregressive(target, ids, cfg.max_new_tokens, sp)
        row["ar"] = {"tps": ar.decode_tps, "ttft": ar.t_prefill, "tokens": ar.new_tokens}
        tps, ntok, ttft = _mlxlm_tps(target, ids, cfg.max_new_tokens)
        row["mlxlm"] = {"tps": tps, "tokens": ntok, "ttft": ttft}
        for name, dec in decoders.items():
            out, st = dec.generate(ids, cfg.max_new_tokens, sp)
            d = st.as_dict()
            if sp.greedy:
                L = min(len(out), len(ar_out))
                mism = next((i for i in range(L) if out[i] != ar_out[i]), None)
                d["first_mismatch"] = mism
                d["exact"] = mism is None and len(out) == len(ar_out)
            row[name] = d
        rows.append(row)
        print("[bench]", dom, r["id"], json.dumps({k: (v["tps"] if k in ("ar", "mlxlm") else
              {"tps": round(v["decode_tps"], 1), "tau": round(v["tokens_per_round"], 2)})
              for k, v in row.items() if isinstance(v, dict)}), flush=True)
    peak = mx.get_peak_memory() / 1e9
    return {"config": cfg.__dict__, "rows": rows, "summary": summarize(rows, list(decoders)), "peak_gb": peak}


def summarize(rows, names):
    out = {}
    doms = sorted({r["domain"] for r in rows}) + ["all"]
    for dom in doms:
        rs = [r for r in rows if dom == "all" or r["domain"] == dom]
        ar = float(np.mean([r["ar"]["tps"] for r in rs]))
        ml = float(np.mean([r["mlxlm"]["tps"] for r in rs]))
        base = max(ar, ml)
        s = {"ar_tps": ar, "mlxlm_tps": ml, "baseline_tps": base, "ar_ttft": float(np.mean([r["ar"]["ttft"] for r in rs]))}
        for n in names:
            v = [r[n] for r in rs]
            tps = float(np.mean([x["decode_tps"] for x in v]))
            tot_t = sum(x["t_total"] - x["t_prefill"] for x in v)
            s[n] = {
                "tps": tps,
                "speedup": tps / base,
                "tokens_per_round": float(np.mean([x["tokens_per_round"] for x in v])),
                "acceptance_rate": float(np.mean([x["acceptance_rate"] for x in v])),
                "mean_verified": float(np.mean([x["mean_verified"] for x in v])),
                # fused (single-sync) rounds cannot split draft / verify time
                "draft_frac": None if v[0].get("fused") else sum(x["t_draft"] for x in v) / tot_t,
                "verify_frac": None if v[0].get("fused") else sum(x["t_verify"] for x in v) / tot_t,
                "update_frac": None if v[0].get("fused") else sum(x["t_update"] for x in v) / tot_t,
                "ttft": float(np.mean([x["t_prefill"] for x in v])),
                "exact_greedy": sum(bool(x.get("exact")) for x in v),
                "n": len(v),
            }
        out[dom] = s
    return out


def format_summary(res: dict) -> str:
    lines = []
    for dom, s in res["summary"].items():
        lines.append(f"## {dom}: baseline {s['baseline_tps']:.1f} tok/s (woodcock AR {s['ar_tps']:.1f}, mlx-lm {s['mlxlm_tps']:.1f})")
        lines.append("| decoder | tok/s | speedup | tokens/round | accept rate | verified/round | draft % | verify % | exact greedy |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for n, v in s.items():
            if isinstance(v, dict):
                pct = lambda x: "n/a" if x is None else f"{100 * x:.0f}"
                lines.append(f"| {n} | {v['tps']:.1f} | {v['speedup']:.2f}x | {v['tokens_per_round']:.2f} | "
                             f"{v['acceptance_rate']:.2f} | {v['mean_verified']:.1f} | {pct(v['draft_frac'])} | "
                             f"{pct(v['verify_frac'])} | {v['exact_greedy']}/{v['n']} |")
        lines.append("")
    lines.append(f"peak memory {res['peak_gb']:.1f} GB")
    return "\n".join(lines)
