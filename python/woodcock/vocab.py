"""Frequency-truncated draft vocabulary (docs/design-vocab-trim.md).

The drafter's LM head is restricted to the K token ids the target generates most
often (FR-Spec, arXiv:2502.14856 §3.2; VocabTrim, arXiv:2506.22694 §3, which
finds target-generated calibration text best). Counts come from the target's own
regenerated responses, training split only, so the hold-out records can measure
coverage.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .data import read_jsonl


def response_counts(path: Path, vocab_size: int, seed: int = 0, holdout: int = 200):
    """Token counts of the response ids of the training records, and of the hold-out
    records (same seeded shuffle and split as data.SFTStream)."""
    recs = list(read_jsonl(Path(path)))
    random.Random(seed).shuffle(recs)
    hold, train = recs[:holdout], recs[holdout:]

    def count(rs):
        ids = [np.asarray(r["response_ids"], np.int64) for r in rs]
        return np.bincount(np.concatenate(ids), minlength=vocab_size) if ids else np.zeros(vocab_size, np.int64)

    return count(train), hold


def top_ids(counts: np.ndarray, k: int, always: set[int] | frozenset[int] = frozenset()) -> np.ndarray:
    """The k most frequent ids (ties: smaller id first), always containing `always`,
    sorted ascending."""
    if not 0 < k <= len(counts):
        raise ValueError(f"vocabulary size {k} not in 1..{len(counts)}")
    always = sorted(set(always))
    if len(always) > k:
        raise ValueError("more forced ids than k")
    order = np.argsort(-counts, kind="stable")
    forced = np.zeros(len(counts), bool)
    forced[always] = True
    rest = order[~forced[order]][: k - len(always)]
    return np.sort(np.concatenate([np.array(always, np.int64), rest])).astype(np.int32)


def coverage(counts: np.ndarray, ids: np.ndarray) -> float:
    tot = counts.sum()
    return float(counts[ids].sum() / tot) if tot else 0.0


SIZES = (8192, 16384, 32768, 49152, 65536)


def build_counts(data: str, target: str, out: str, seed: int = 0, holdout: int = 200):
    """Write runs/vocab/<name>.npz: counts, plus a hold-out coverage table (.json)."""
    from mlx_lm.utils import load as mlx_load

    model, tok = mlx_load(target, lazy=True)  # only the config and tokenizer are needed
    V = model.args.vocab_size
    eos = sorted(set(getattr(tok, "eos_token_ids", []) or []) | {tok.eos_token_id})
    del model
    counts, hold = response_counts(Path(data), V, seed, holdout)
    table = {}
    for dom in sorted({r["domain"] for r in hold}) + ["all"]:
        rs = [r for r in hold if dom == "all" or r["domain"] == dom]
        h = np.bincount(np.concatenate([np.asarray(r["response_ids"], np.int64) for r in rs]), minlength=V)
        table[dom] = {str(k): coverage(h, top_ids(counts, k, eos)) for k in SIZES}
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_p, counts=counts, eos=np.array(eos, np.int64))
    meta = {"data": data, "target": target, "seed": seed, "holdout": holdout,
            "counted_tokens": int(counts.sum()), "distinct": int((counts > 0).sum()),
            "eos": eos, "holdout_coverage": table}
    out_p.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1))
    return out_p


def load_ids(spec: str) -> np.ndarray:
    """'<counts.npz>:<K>' -> draft vocabulary ids (EOS always included)."""
    path, _, k = spec.rpartition(":")
    z = np.load(path)
    return top_ids(z["counts"], int(k), set(int(e) for e in z["eos"]))
