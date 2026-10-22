"""`woodcock` command line: one subcommand per pipeline stage.

Every stage reads an optional YAML config (`--config`), whose sections map onto
the stage's dataclass; explicit `--set key=value` overrides win.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import yaml


def _load_cfg(path: str | None, section: str) -> dict:
    if not path:
        return {}
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    base = {}
    for inc in cfg.get("include", []):
        base.update(_load_cfg(str(Path(path).parent / inc), section))
    base.update(cfg.get(section, {}) or {})
    return base


def _overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        try:
            out[k] = yaml.safe_load(v)
        except yaml.YAMLError:
            out[k] = v
    return out


def _make(cls, args, section):
    d = _load_cfg(args.config, section)
    d.update(_overrides(args.set))
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise SystemExit(f"unknown {section} options: {sorted(unknown)}")
    return cls(**d)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="woodcock")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, help_):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config")
        p.add_argument("--set", nargs="*", default=[], help="key=value overrides")
        return p

    p = add("prompts", "build train/eval prompt pools from permissive datasets")
    p.add_argument("--out", default="data/prompts")
    p.add_argument("--only", choices=["train", "eval"], help="build one pool (keeps the other file untouched)")
    add("regen", "regenerate responses with the frozen target")
    add("adapt", "train a drafter against a frozen target (TTT distillation)")
    p = add("vocab", "count target-generated tokens for a truncated draft vocabulary")
    p.add_argument("--out", required=True, help="counts file, e.g. runs/vocab/qwen3-1.7b.npz")
    p = add("bench", "benchmark speculative vs autoregressive decoding")
    p.add_argument("--target", required=True)
    p.add_argument("--drafter", action="append", required=True,
                   help="name=path[@counts.npz:K] (K-token draft vocabulary, docs/design-vocab-trim.md)")
    p.add_argument("--policy", action="append",
                   help="repeatable; chain:K (single-sync fused) | chainsync:K (synced) | tree:D,W,N | "
                        "auto:<costs.json>[:K_max] (chain length from measured latencies)")
    p.add_argument("--out", required=True)
    p = add("generate", "generate text with speculative decoding")
    p.add_argument("--target", required=True)
    p.add_argument("--drafter", required=True, help="path[@counts.npz:K]")
    p.add_argument("--prompt", required=True)
    p.add_argument("--policy", default="chain:5")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p = add("export", "export a drafter to the Woodcock native format")
    p.add_argument("--drafter", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    from .memory import configure_from_env

    configure_from_env()

    if args.cmd == "prompts":
        from .data import build_prompts

        c = _load_cfg(args.config, "prompts")
        out = Path(args.out)
        if args.only != "eval":
            build_prompts(out / "train.jsonl", c.get("train", {"chat": 12000, "code": 5000, "math": 5000, "multilingual": 3000}), "train")
        if args.only != "train":
            build_prompts(out / "eval.jsonl", c.get("eval", {"chat": 80, "code": 80, "math": 80, "multilingual": 80}), "eval", seed=1)
    elif args.cmd == "regen":
        from .data import regenerate

        c = _load_cfg(args.config, "regen")
        c.update(_overrides(args.set))
        regenerate(c.pop("target"), Path(c.pop("prompts")), Path(c.pop("out")), **c)
    elif args.cmd == "adapt":
        from .train import AdaptConfig, adapt

        adapt(_make(AdaptConfig, args, "adapt"))
    elif args.cmd == "vocab":
        from .vocab import build_counts

        c = _load_cfg(args.config, "adapt")
        c.update(_overrides(args.set))
        build_counts(c["data"], c["target"], args.out, seed=c.get("seed", 0), holdout=c.get("holdout", 200))
    elif args.cmd in ("bench", "generate"):
        from .spec import SpeculativeDecoder
        from .target import Target

        t = Target.load(args.target)
        if args.cmd == "generate":
            from .data import chat_ids
            from .sampling import SamplingParams

            dr = load_drafter(args.drafter)
            dec = SpeculativeDecoder(t, dr, make_policy(args.policy, dr))
            out, st = dec.generate(chat_ids(t.tokenizer, args.prompt), args.max_tokens, SamplingParams(temperature=args.temperature))
            print(t.tokenizer.decode(out))
            print(json.dumps(st.as_dict(), indent=1))
            return
        from .bench import BenchConfig, format_summary, run_bench

        # every drafter x policy is measured on the same prompts in the same run, against
        # the same per-prompt baselines (run-to-run baseline spread is 3-5%)
        policies = args.policy or ["chain:5"]
        decs = {}
        for spec in args.drafter:
            name, path = spec.split("=", 1)
            dr = load_drafter(path)
            for pol in policies:
                fused = not pol.startswith("chainsync:")
                short = "auto" + (":" + pol.split(":")[2] if pol.count(":") > 1 else "") if pol.startswith("auto:") else pol
                key = name if len(policies) == 1 else f"{name} {short}"
                decs[key] = SpeculativeDecoder(t, dr, make_policy(pol.replace("chainsync:", "chain:"), dr), fused=fused)
        res = run_bench(t, decs, _make(BenchConfig, args, "bench"))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
        md = format_summary(res)
        Path(args.out).with_suffix(".md").write_text(md)
        print(md)
    elif args.cmd == "export":
        from .export import export_drafter

        export_drafter(args.drafter, args.out)


def load_drafter(spec: str):
    """'path' or 'path@counts.npz:K' (drafter with a K-token frequency-ranked head)."""
    from .drafter import Drafter

    path, _, vocab = spec.partition("@")
    dr = Drafter.load(path)
    if vocab:
        from .vocab import load_ids

        dr.restrict_vocab(load_ids(vocab))
    return dr


def make_policy(spec: str, drafter=None):
    from .policies import CostAwareChain, StaticPolicy

    kind, _, arg = spec.partition(":")
    if kind == "auto":  # auto:<costs.json>[:K_max], costs from scripts/time_draft_step.py
        path, _, k_max = arg.partition(":")
        vids = drafter.vocab_ids if drafter is not None else None
        head = "full" if vids is None else str(vids.size)
        return CostAwareChain.from_costs(json.loads(Path(path).read_text()), head, **({"k_max": int(k_max)} if k_max else {}))
    if kind == "chain":
        return StaticPolicy(depth=int(arg or 5))
    if kind == "tree":
        D, W, N = (int(x) for x in arg.split(","))
        return StaticPolicy(depth=D, width=W, budget=N)
    raise SystemExit(f"bad policy {spec}")


if __name__ == "__main__":
    main()
